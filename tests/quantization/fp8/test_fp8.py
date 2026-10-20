# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for block-FP8 (`Fp8Config`, `quant_method="fp8"`).

They cover the config, the quantize op, the checkpoint layouts it loads and saves back, and model-level loading.
"""

import tempfile
import unittest
from unittest import mock

import torch
from parameterized import parameterized

import transformers.integrations.finegrained.layers as fg
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.finegrained import FineGrainedExperts
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import (
    TestCasePlus,
    get_device_properties,
    require_accelerate,
    require_compressed_tensors,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils.quantization_config import (
    Fp8Config,
)

from ...integrations.finegrained.utils import (
    E,
    H,
    I,
    _Cfg,
    _ChainAssertions,
    _compressed_tensors,
    _conversion_model,
    _load_and_save,
    _per_expert,
    _quantizer_for,
)
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ..block_scaled_checkpoint_tester import TinyCheckpointTesterMixin
from ..block_scaled_parallel_tester import LoadPathEquivalenceTesterMixin
from ..block_scaled_tester import (
    FP8,
    BlockScaledFormatTesterMixin,
    _require_kernel_capability,
    no_accelerator,
)


@require_torch
class Fp8ConfigTest(unittest.TestCase):
    def test_a_passed_config_overrides_the_checkpoints_loading_attributes(self):
        from transformers.quantizers.auto import AutoHfQuantizer

        passed = Fp8Config(dequantize=True, modules_to_not_convert=["lm_head", "model.layers.0"])
        merged = AutoHfQuantizer.merge_quantization_configs(
            {"quant_method": "fp8", "modules_to_not_convert": ["lm_head"]}, passed
        )
        self.assertEqual((merged.dequantize, merged.modules_to_not_convert), (True, ["lm_head", "model.layers.0"]))

    def test_a_compressed_tensors_block_fp8_checkpoint_reads_as_block_fp8(self):
        block = {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]}
        fp8 = {"num_bits": 8, "type": "float", "strategy": "group", "group_size": 128, "dynamic": True}
        config = AutoQuantizationConfig.from_dict(_compressed_tensors(block, fp8, fmt="float-quantized"))
        self.assertEqual((type(config), config.weight_block_size), (Fp8Config, (128, 128)))

    def test_a_compressed_tensors_checkpoint_with_a_quantized_kv_cache_stays_with_compressed_tensors(self):
        from transformers.utils.quantization_config import read_compressed_tensors

        tensor = {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False}
        checkpoint = _compressed_tensors(tensor, tensor, fmt="float-quantized")
        self.assertIsInstance(read_compressed_tensors(checkpoint), Fp8Config)
        self.assertIsNone(read_compressed_tensors({**checkpoint, "kv_cache_scheme": tensor}))

    @require_compressed_tensors
    def test_a_passed_compressed_tensors_config_keeps_its_backend(self):
        from transformers import PreTrainedConfig
        from transformers.quantizers.auto import get_hf_quantizer
        from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
        from transformers.utils.quantization_config import CompressedTensorsConfig

        block = {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]}
        model_config = PreTrainedConfig()
        model_config.quantization_config = _compressed_tensors(block, fmt="float-quantized")
        quantizer, _, _ = get_hf_quantizer(model_config, CompressedTensorsConfig(), None, True, {})
        self.assertIsInstance(quantizer, CompressedTensorsHfQuantizer)

    def test_an_fp8_checkpoint_loads_as_block_fp8(self):
        from transformers import PreTrainedConfig
        from transformers.quantizers.auto import get_hf_quantizer
        from transformers.quantizers.quantizer_fp8 import Fp8HfQuantizer

        model_config = PreTrainedConfig()
        model_config.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        quantizer, _, _ = get_hf_quantizer(model_config, None, None, True, {})
        self.assertIsInstance(quantizer, Fp8HfQuantizer)
        self.assertIsInstance(quantizer.quantization_config, Fp8Config)


@require_torch
class Fp8QuantizeOpTest(unittest.TestCase):
    """`FineGrainedQuantize` on block-FP8, which quantizes in torch into the module's layout.

    The group formats use the kernels' own quantizers, which need a GPU.
    """

    def test_a_static_scheme_gets_its_activation_scales_written(self):
        """Check that static activation scales are written as the identity when the checkpoint has none.

        The loader fills a missing key with `torch.empty_like`, and `_init_weights` does not touch scales.
        A slot this op skips would reach the kernels as uninitialized memory.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        model = torch.nn.Module()
        model.experts = fg.FineGrainedExperts(
            cfg, quantization_config=Fp8Config(weight_block_size=(128, 128), activation_scheme="static")
        )
        model.proj = fg.FineGrainedLinear(
            quantization_config=Fp8Config(weight_block_size=(128, 128), activation_scheme="static"),
            in_features=256,
            out_features=256,
        )
        op = FineGrainedQuantize(hf_quantizer=None)
        for key, tensor, slot in (
            ("experts.gate_up_proj", torch.randn(4, 256, 256), "experts.gate_up_proj_activation_scale"),
            ("proj.weight", torch.randn(256, 256), "proj.activation_scale"),
        ):
            with self.subTest(key=key):
                out = op.convert({key: tensor}, model=model)
                self.assertIn(slot, out, f"{key}: the static activation scale was not written")
                held = model.get_parameter(slot)
                self.assertEqual((out[slot].shape, out[slot].dtype), (held.shape, held.dtype))
                torch.testing.assert_close(out[slot], torch.ones_like(out[slot]), rtol=0, atol=0)

    def test_an_expert_bias_is_passed_through_not_quantized(self):
        """Check that a GPT-OSS expert bias passes through unquantized.

        The bias is 2-D `(E, rows)` like a dense weight, so the rank guard alone lets it through.
        It has no scale slot. Emitting one would give the loader a `<proj>_bias_scale_inv` that no module holds.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        model = torch.nn.Module()
        model.experts = fg.FineGrainedExperts(
            cfg, quantization_config=Fp8Config(weight_block_size=(128, 128)), has_bias=True
        )
        bias = torch.randn(4, 256)
        out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.down_proj_bias": bias}, model=model)
        self.assertEqual(list(out), ["experts.down_proj_bias"])
        self.assertIs(out["experts.down_proj_bias"], bias)

    def test_a_partial_trailing_block_round_trips_against_the_configured_block(self):
        """Check that a partial trailing block round-trips against the configured block.

        DeepSeek-V3 ships `kv_a_proj_with_mqa` as `(576, 7168)` with a 128x128 block. The grid rounds up, and the
        short block quantizes on its own values. The block size cannot be recovered from the grid shape.
        A dequantize that guesses it reads the wrong rows per scale. The bound is tight enough to catch that.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        torch.manual_seed(0)
        quantizer = _quantizer_for(Fp8Config(weight_block_size=(128, 128)))
        for rows, cols in ((576, 256), (192, 256), (256, 256)):
            with self.subTest(shape=(rows, cols)):
                original = torch.randn(rows, cols)
                weight, scale = FineGrainedQuantize._quantize_block_fp8(original, (128, 128), ue8m0=False)
                self.assertEqual(weight.shape, (rows, cols))
                self.assertEqual(weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(scale.shape, (-(-rows // 128), -(-cols // 128)))

                recovered = FineGrainedDequantize(quantizer)._dequantize_one(
                    weight, scale.float(), output_dtype=torch.float32
                )
                rel = ((recovered - original).norm() / original.norm()).item()
                self.assertLess(rel, 3e-2, f"{(rows, cols)}: {rel:.4f}")

    def test_block_fp8_round_trips_within_its_floor(self):
        from transformers.integrations.finegrained.conversions import FineGrainedDequantize, FineGrainedQuantize

        torch.manual_seed(0)
        cfg = _Cfg()
        cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts = 256, 128, 4
        for scale_fmt in ("float", "ue8m0"):
            with self.subTest(scale_fmt=scale_fmt), mock.patch.object(fg, "is_sm100", return_value=False):
                experts = fg.FineGrainedExperts(
                    cfg, quantization_config=Fp8Config(weight_block_size=(128, 128), scale_fmt=scale_fmt)
                )
                model = torch.nn.Module()
                model.experts = experts
                original = torch.randn(4, 256, 256) * 0.02
                out = FineGrainedQuantize(hf_quantizer=None).convert({"experts.gate_up_proj": original}, model=model)
                weight, scale = out["experts.gate_up_proj"], out["experts.gate_up_proj_scale_inv"]
                self.assertEqual(
                    (weight.dtype, weight.shape), (experts.gate_up_proj.dtype, experts.gate_up_proj.shape)
                )
                self.assertEqual(
                    (scale.dtype, scale.shape),
                    (experts.gate_up_proj_scale_inv.dtype, experts.gate_up_proj_scale_inv.shape),
                )
                deq = FineGrainedDequantize(None)._dequantize_one(weight, scale.float(), output_dtype=torch.float32)
                rel = ((deq - original).norm() / original.norm()).item()
                self.assertLess(rel, 4e-2, f"{scale_fmt}: {rel:.3f}")

    def test_float_scale_fmt_stays_bit_identical_to_its_formula(self):
        """Check that the float scale format matches its formula bit for bit.

        The inverse scale is `amax / MAX` and the weight is divided by it. The algebraically equal
        `weight * (MAX / amax)` is not bit-equal. The difference only shows on some fp32 maxima, so the test
        runs many blocks.
        """
        from transformers.integrations.finegrained.conversions import FineGrainedQuantize
        from transformers.integrations.finegrained.formats import FP8_DTYPE, FP8_MAX, FP8_MIN

        layer = fg.FineGrainedLinear(128, 128, quantization_config=Fp8Config(weight_block_size=(128, 128)))
        for seed in range(100):
            torch.manual_seed(seed)
            weight = torch.randn(128, 128, dtype=torch.float32)
            out = FineGrainedQuantize(None)._quantize_one("layer.weight", weight, layer)
            ref_inv = (weight.abs().amax() / FP8_MAX).to(torch.float32).reshape(1, 1)
            ref_q = torch.clamp(weight / ref_inv, min=FP8_MIN, max=FP8_MAX).to(FP8_DTYPE)
            self.assertTrue(torch.equal(out["layer.weight"], ref_q), f"float weight diverged at seed {seed}")
            self.assertTrue(
                torch.equal(out["layer.weight_scale_inv"], ref_inv), f"float scale diverged at seed {seed}"
            )


@require_torch
class Fp8ChainTest(_ChainAssertions, unittest.TestCase):
    """Block-FP8 and per-tensor FP8, as DeepSeek-V3, DeepSeek-V4 and static Mistral checkpoints ship them."""

    def test_deepseek_v3_block_fp8(self):
        """Per-expert e4m3 weights with fp32 `weight_scale_inv` per block, merged onto the stacked experts."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randn(n, k).to(torch.float8_e4m3fn),
                "weight_scale_inv": torch.rand(-(-n // 16), -(-k // 16)) + 0.5,
            }
        )
        model, info, saved = _load_and_save(checkpoint, Fp8Config(weight_block_size=(16, 16)), sm100=True)
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        self.assertHoldsInterleaved(experts, checkpoint, "weight", experts.gate_up_proj)
        self.assertHoldsInterleaved(experts, checkpoint, "weight_scale_inv", experts.gate_up_proj_scale_inv)
        self.assertSavesBack(saved, checkpoint)

    def test_ue8m0_block_scales(self):
        """Check that E8M0 block scales load as `float8_e8m0fnu` and save back unchanged.

        DeepSeek-V4 ships them under `.scale`, which its own conversion mapping renames.
        """
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randn(n, k).to(torch.float8_e4m3fn),
                "weight_scale_inv": torch.randint(120, 130, (-(-n // 16), -(-k // 16)), dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
            }
        )
        config = Fp8Config(weight_block_size=(16, 16), scale_fmt="ue8m0")
        model, info, saved = _load_and_save(checkpoint, config)
        self.assertLoadsCleanly(info)
        self.assertEqual(model.model.layers[0].self_attn.q_proj.weight_scale_inv.dtype, torch.float8_e8m0fnu)
        self.assertSavesBack(saved, checkpoint)

    def test_float32_ue8m0_containers_save_as_native_ue8m0(self):
        """DeepSeek-V4-Base's power-of-two float32 scale containers normalize to native UE8M0."""
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randn(n, k).to(torch.float8_e4m3fn),
                "weight_scale_inv": torch.pow(2.0, torch.randint(-8, 8, (-(-n // 16), -(-k // 16))).float()),
            }
        )
        config = Fp8Config(weight_block_size=(16, 16), scale_fmt="ue8m0")
        model, info, saved = _load_and_save(checkpoint, config)
        self.assertLoadsCleanly(info)
        self.assertEqual(set(saved), set(checkpoint))
        for name, tensor in checkpoint.items():
            with self.subTest(name=name):
                self.assertEqual(
                    saved[name].dtype, torch.float8_e8m0fnu if name.endswith("scale_inv") else tensor.dtype
                )
                torch.testing.assert_close(saved[name].float(), tensor.float(), rtol=0, atol=0)
        _, info, resaved = _load_and_save(saved, config)
        self.assertLoadsCleanly(info)
        self.assertSavesBack(resaved, saved)

    def test_mistral4_static_per_tensor(self):
        """Mistral-Small-4 ships stacked e4m3 experts under the module's own names.

        Each expert has one BF16 weight scale `(E, 1, 1)` and one calibrated BF16 activation scale.
        Dense linears hold one 0-dim BF16 scale of each.
        """
        torch.manual_seed(0)
        checkpoint = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(H, H).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.q_proj.weight_scale_inv": (torch.rand(()) + 0.5).to(torch.bfloat16),
            "model.layers.0.self_attn.q_proj.activation_scale": (torch.rand(()) + 0.5).to(torch.bfloat16),
        }
        for proj, (rows, cols) in (("gate_up_proj", (2 * I, H)), ("down_proj", (H, I))):
            prefix = f"model.layers.0.mlp.experts.{proj}"
            checkpoint[prefix] = torch.randn(E, rows, cols).to(torch.float8_e4m3fn)
            checkpoint[f"{prefix}_scale_inv"] = (torch.rand(E, 1, 1) + 0.5).to(torch.bfloat16)
            checkpoint[f"{prefix}_activation_scale"] = (torch.rand(E) + 0.5).to(torch.bfloat16)
        config = Fp8Config(weight_block_size=None, activation_scheme="static")
        model, info, saved = _load_and_save(checkpoint, config)
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        torch.testing.assert_close(
            experts.gate_up_proj_activation_scale.float(),
            checkpoint["model.layers.0.mlp.experts.gate_up_proj_activation_scale"].float(),
        )
        # The loader uses the initialized FP32 dtype for a dense block scale.
        expected = dict(checkpoint)
        key = "model.layers.0.self_attn.q_proj.weight_scale_inv"
        expected[key] = expected[key].float()
        self.assertSavesBack(saved, expected)

    def test_a_calibrated_checkpoint_s_input_scale_reaches_the_module_s_slot(self):
        """Check that each calibrated `input_scale` reaches its module's slot, in that module's shape.

        A dense linear gets one value and the stacked experts get one per expert. The gate/up pair reduces to one per
        expert instead of flattening to 2E. An unrouted key keeps the default of 1.0, which silently uses the wrong
        scale.
        """
        from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

        quantizer = _quantizer_for(Fp8Config(activation_scheme="static"))
        quantizer.pre_quantized = True
        conversions = quantizer.update_weight_conversions([], _conversion_model(quantizer))
        renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
        converters = [c for c in conversions if isinstance(c, WeightConverter)]

        for key, expected in (
            ("model.layers.3.self_attn.q_proj.input_scale", "model.layers.3.self_attn.q_proj.activation_scale"),
            ("model.layers.3.mlp.experts.7.gate_proj.input_scale", "experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.7.up_proj.input_scale", "experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.7.down_proj.input_scale", "experts.down_proj_activation_scale"),
            ("model.layers.3.mlp.experts.gate_up_proj_input_scale", "experts.gate_up_proj_activation_scale"),
            ("model.layers.3.mlp.experts.down_proj_input_scale", "experts.down_proj_activation_scale"),
        ):
            with self.subTest(key=key.rsplit(".", 2)[-2]):
                # Route as the loader does. Renamings apply first, then the one matching converter.
                renamed, pattern = rename_source_key(key, renamings, converters)
                targets = (
                    [renamed]
                    if pattern is None
                    else [t for c in converters if pattern in c.source_patterns for t in c.target_patterns]
                )
                self.assertTrue(any(expected in t for t in targets), f"{key} -> {targets}")
        # A dynamic checkpoint has no such key, and a weight-only one ignores it
        dynamic = _quantizer_for(Fp8Config())
        dynamic.pre_quantized = True
        self.assertEqual(dynamic.get_weight_conversions(), [])


@slow
@require_accelerate
@require_torch_accelerator
@unittest.skipIf(
    get_device_properties()[0] == "cuda"
    and (get_device_properties()[1] < 8 or (get_device_properties()[1] == 8 and get_device_properties()[2] < 9)),
    "Skipping Fp8QuantizerTest because it is not supported on GPU with capability < 8.9",
)
class Fp8QuantizerTest(MemoryCleanupMixin, unittest.TestCase):
    # An ungated copy of meta-llama/Llama-3.2-1B with the same weights
    model_name = "unsloth/Llama-3.2-1B"
    quantized_model_name = "hf-internal-testing/Llama-3.2-1B-Instruct-fp8"
    input_text = "Once upon a time"
    max_new_tokens = 10
    EXPECTED_OUTPUTS = {
        "Once upon a time, there was a little girl who loved to play",
        "Once upon a time, there was a man who was very rich.",
    }
    EXPECTED_DEQUANTIZED_OUTPUT = "Once upon a time, in a small village nestled in the rolling hills"
    device_map = torch_device
    offload_device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": "cpu",
        "model.layers.8": "cpu",
        "model.layers.9": "cpu",
        "model.layers.10": "cpu",
        "model.layers.11": "cpu",
        "model.layers.12": "cpu",
        "model.layers.13": "cpu",
        "model.layers.14": "cpu",
        "model.layers.15": "cpu",
        "model.rotary_emb": "cpu",
        "model.norm": "cpu",
        "lm_head": 0,
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.quantization_config = Fp8Config()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, quantization_config=cls.quantization_config
        )

    def test_quantized_model(self):
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

        output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertIn(output_tokens, self.EXPECTED_OUTPUTS)

    def test_dequantized_model(self):
        quantization_config = Fp8Config(dequantize=True)
        dequantized_model = AutoModelForCausalLM.from_pretrained(
            self.quantized_model_name, device_map=self.device_map, quantization_config=quantization_config
        )
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        output = dequantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertEqual(output_tokens, self.EXPECTED_DEQUANTIZED_OUTPUT)
        del dequantized_model

    def test_dequantize_when_no_accelerator(self):
        with no_accelerator():
            dequantized_model = AutoModelForCausalLM.from_pretrained(self.quantized_model_name, device_map="cpu")
            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to("cpu")
            output = dequantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            output_tokens = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.assertEqual(output_tokens, self.EXPECTED_DEQUANTIZED_OUTPUT)
            del dequantized_model

    def test_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.device_map)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_weight_and_weight_scale_inv(self):
        weight = self.quantized_model.model.layers[0].self_attn.q_proj.weight
        weight_scale_inv = self.quantized_model.model.layers[0].self_attn.q_proj.weight_scale_inv
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(weight_scale_inv.dtype, torch.float32)
        self.assertEqual(weight.shape, (weight_scale_inv.shape[0] * 128, weight_scale_inv.shape[1] * 128))

    def test_block_size(self):
        self.assertEqual(self.quantized_model.config.quantization_config.weight_block_size, (128, 128))
        quantization_config = Fp8Config(weight_block_size=(32, 32))
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device_map, quantization_config=quantization_config
        )
        self.assertEqual(quantized_model.config.quantization_config.weight_block_size, (32, 32))

    @require_torch_multi_accelerator
    def test_quantized_model_multi_accelerators(self):
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)
        quantization_config = Fp8Config()
        # need to empty cache or set max_memory, otherwise we will use the reserved memory that was not allocated when computing max-memory
        # this will lead to put the entire model to device 0.
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
            max_memory={0: "1GB", 1: "10GB"},
        )
        self.assertTrue(set(quantized_model.hf_device_map.values()) == {0, 1})

        output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    @require_torch_multi_accelerator
    def test_save_pretrained_multi_accelerators(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            # need to empty cache or set max_memory, otherwise we will use the reserved memory that was not allocated when computing max-memory
            # this will lead to put the entire model to device 0.
            model = AutoModelForCausalLM.from_pretrained(
                tmpdirname, device_map="auto", max_memory={0: "1GB", 1: "10GB"}
            )
            self.assertTrue(set(model.hf_device_map.values()) == {0, 1})

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_quantized_model_offload(self):
        with self.assertRaisesRegex(
            ValueError,
            "You are attempting to load a fine-grained quantized model with a device_map that contains a cpu/disk",
        ):
            AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map=self.offload_device_map, quantization_config=self.quantization_config
            )

    def test_save_pretrained_offload(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device_map)

            quantized_model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map=self.offload_device_map)
            output = quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
            self.assertIn(self.tokenizer.decode(output[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_a_compressed_tensors_fp8_checkpoint_loads_as_fp8(self):
        """Check that a per-tensor FP8 compressed-tensors export runs on the fine-grained kernels."""
        from transformers.quantizers.quantizer_fp8 import Fp8HfQuantizer

        model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device_map)
        self.assertIsInstance(model.hf_quantizer, Fp8HfQuantizer)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer("The capital of France is", return_tensors="pt").to(self.device_map)
        output = model.generate(**input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.assertIn("Paris", tokenizer.decode(output[0], skip_special_tokens=True))


@slow
@require_accelerate
@require_torch_accelerator
class Fp8HubMoeTest(MemoryCleanupMixin, unittest.TestCase):
    """Tiny MoE checkpoints from the Hub, quantized on the fly or shipped in FP8."""

    def setUp(self):
        super().setUp()
        _require_kernel_capability(self)

    @parameterized.expand(
        [
            "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM",
            "hf-internal-testing/tiny-random-MixtralForCausalLM",
        ]
    )
    def test_moe_conversion(self, model_id):
        quantization_config = Fp8Config(weight_block_size=(32, 32))
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        self.assertTrue(any(isinstance(module, FineGrainedExperts) for module in model.modules()))

    @parameterized.expand(["eager", "batched_mm", "grouped_mm", "deepgemm"])
    def test_quantized_moe_forward(self, experts_implementation):
        """MiniMax-M2 routes indices past `top_k`. Every experts implementation must handle them."""
        if experts_implementation == "deepgemm" and torch_device != "cuda":
            self.skipTest("deepgemm is only supported on CUDA")
        if experts_implementation == "deepgemm" and torch.cuda.get_device_capability()[0] >= 10:
            self.skipTest("DeepGEMM's Blackwell experts read only UE8M0 scales; this checkpoint ships float32 ones")

        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/MiniMax-M2-Tiny-FP8",  # single layer version
            experts_implementation=experts_implementation,
            device_map=torch_device,
        )
        assert model.config._experts_implementation == experts_implementation

        tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What is your favourite condiment?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Do you have mayonnaise recipes?"}]},
        ]
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
            torch_device
        )

        # Only caring about this not crashing
        _ = model.generate(**model_inputs, max_new_tokens=24)


class Fp8Test(BlockScaledFormatTesterMixin, unittest.TestCase):
    spec = FP8


class Fp8TinyCheckpointTest(TinyCheckpointTesterMixin, TestCasePlus):
    spec = FP8


class Fp8LoadPathEquivalenceTest(LoadPathEquivalenceTesterMixin, TestCasePlus):
    spec = FP8
