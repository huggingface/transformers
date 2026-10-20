# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Tests for MXFP4 (`Mxfp4Config`).

They cover GPT-OSS's blocks layout, the packed per-expert layout, the generate path and the published GPT-OSS
checkpoints.
"""

import tempfile
import unittest
from unittest.mock import patch

import torch

from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM, Mxfp4Config
from transformers.testing_utils import (
    TestCasePlus,
    cleanup,
    require_kernels,
    require_torch,
    require_torch_accelerator,
    require_torch_accelerator_memory,
    require_torch_large_accelerator,
    require_triton,
    slow,
    torch_device,
)

from ...integrations.finegrained.utils import (
    E,
    H,
    I,
    _ChainAssertions,
    _load_and_save,
    _per_expert,
    _quantizer_for,
)
from ..block_scaled_checkpoint_tester import TinyCheckpointTesterMixin
from ..block_scaled_parallel_tester import LoadPathEquivalenceTesterMixin
from ..block_scaled_tester import (
    MXFP4,
    BlockScaledFormatTesterMixin,
    _cheap_tuning,
)


class Mxfp4EnvironmentTest(unittest.TestCase):
    def test_weight_only_mxfp4_keeps_existing_cuda_architectures(self):
        for capability in ((7, 5), (8, 0), (8, 6)):
            with self.subTest(capability=capability):
                config = Mxfp4Config()
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=capability),
                    patch("transformers.quantizers.block_scaled_utils.is_accelerate_available", return_value=True),
                ):
                    _quantizer_for(config).validate_environment(device_map="cuda")
                self.assertFalse(config.dequantize)

    def test_quantized_activations_keep_the_fp8_hardware_requirement(self):
        for activation_format in ("mxfp4", "mxfp8"):
            with self.subTest(activation_format=activation_format):
                config = Mxfp4Config(activation_format=activation_format)
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=(8, 0)),
                    patch("transformers.quantizers.block_scaled_utils.is_accelerate_available", return_value=True),
                ):
                    _quantizer_for(config).validate_environment(device_map="cuda")
                self.assertTrue(config.dequantize)

    def test_model_device_map_validation(self):
        # Quantizing on the fly takes a CPU device map, but not one that also offloads
        quantizer = _quantizer_for(Mxfp4Config())
        quantizer.pre_quantized = False
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch("transformers.quantizers.block_scaled_utils.is_kernels_available", return_value=True),
        ):
            quantizer.validate_environment(device_map={"": "cpu"})
            with self.assertRaises(ValueError):
                quantizer.validate_environment(device_map={"model": 0, "lm_head": "disk"})

    def test_hub_kernels_dequantize_the_experts(self):
        # GPT-OSS's hub kernel forward reads full-precision experts
        config = GptOssConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=2,
        )
        with torch.device("meta"):
            model = GptOssForCausalLM(config)
        quantizer = _quantizer_for(Mxfp4Config())
        quantizer.pre_quantized = True
        with patch(
            "transformers.quantizers.quantizer_mxfp4.get_current_accelerator", return_value=torch.device("cuda")
        ):
            quantizer._process_model_before_weight_loading(model, use_kernels=True)
        self.assertTrue(quantizer.quantization_config.dequantize)


class Mxfp4IntegrationTest(unittest.TestCase):
    def test_convert_moe_packed_tensors(self):
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        blocks = torch.randint(0, 255, (2, 4, 8, 16), dtype=torch.uint8)
        scales = torch.randint(100, 150, (2, 4, 8), dtype=torch.uint8)
        result = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
        self.assertEqual(result.shape, (2, 8 * 16 * 2, 4))
        self.assertEqual(result.dtype, torch.bfloat16)

    @require_triton(min_version="3.4.0")
    @require_kernels
    @require_torch_accelerator
    def test_quantize_to_mxfp4(self):
        from transformers.integrations.hub_kernels import get_kernel
        from transformers.integrations.mxfp4 import quantize_to_mxfp4

        w = torch.randn(32, 64, 128, dtype=torch.bfloat16, device=torch_device)
        quantized_w, _ = quantize_to_mxfp4(w, get_kernel("kernels-community/gpt-oss-triton-kernels", version=1))
        self.assertEqual(quantized_w.dtype, torch.uint8)


@require_torch_accelerator
@require_kernels
class Mxfp4GenerateTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(_cheap_tuning())

    def test_generate_matches_forward_logits(self):
        """Check that each `generate` step's logits match a plain forward over the same prefix.

        `generate` runs the batched expert kernels and a forward runs the grouped ones. The tiny W4A16 GPT-OSS has
        large expert biases, so any difference in bias handling shows up.
        """
        torch.manual_seed(0)
        config = GptOssConfig(
            vocab_size=512,
            hidden_size=256,
            intermediate_size=256,
            head_dim=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        model = GptOssForCausalLM(config).to(torch.bfloat16)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.experts.gate_up_proj_bias.normal_()
                layer.mlp.experts.down_proj_bias.normal_()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            quantized = GptOssForCausalLM.from_pretrained(
                tmp, quantization_config=Mxfp4Config(), dtype=torch.bfloat16, device_map=torch_device
            )

        prompt = torch.arange(1, 17, device=torch_device).unsqueeze(0)
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache), torch.no_grad():
                out = quantized.generate(
                    prompt,
                    max_new_tokens=3,
                    do_sample=False,
                    use_cache=use_cache,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
                for step, logits in enumerate(out.logits):
                    prefix = out.sequences[:, : prompt.shape[1] + step]
                    expected = quantized(prefix).logits[:, -1].float()
                    torch.testing.assert_close(logits.float(), expected, atol=5e-2, rtol=5e-2)


@require_torch_large_accelerator
@require_kernels
@slow
class Mxfp4ModelTest(unittest.TestCase):
    """The published GPT-OSS checkpoint, quantized and dequantized, and saved back."""

    model_name = "openai/gpt-oss-20b"
    input_text = "Once upon a time"
    EXPECTED_OUTPUTS = {"Once upon a time, in a small town, there lived a young"}
    # what the published checkpoint keeps in full precision
    MODULES_TO_NOT_CONVERT = ["model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def check_inference_correctness_quantized(self, model, tokenizer):
        encoded_input = tokenizer(self.input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_sequences = model.generate(
                **encoded_input,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        self.assertIn(generated_text, self.EXPECTED_OUTPUTS)

    def test_gpt_oss_model_loading_quantized_with_device_map(self):
        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.check_inference_correctness_quantized(model, tokenizer)

    # ~39 GiB for the dequantized bfloat16 weights.
    @require_torch_accelerator_memory(memory=48)
    def test_gpt_oss_model_loading_dequantized_with_device_map(self):
        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=Mxfp4Config(dequantize=True),
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.check_inference_correctness_quantized(model, tokenizer)

    # Holds the packed (~13 GiB) and the dequantized (~39 GiB) model at the same time, by design.
    @require_torch_accelerator_memory(memory=56)
    def test_memory_footprint_comparison(self):
        quantization_config = Mxfp4Config(dequantize=True)
        quantized_model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        dequantized_model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        quantized_mem = quantized_model.get_memory_footprint()
        dequantized_mem = dequantized_model.get_memory_footprint()
        self.assertLess(quantized_mem, dequantized_mem)

    # Keeps the packed model alive while reloading the saved copy, dequantized (~13 + ~39 GiB).
    @require_torch_accelerator_memory(memory=56)
    def test_save_mxfp4(self):
        model = GptOssForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            cleanup(torch_device, gc_collect=True)
            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                quantization_config=Mxfp4Config(dequantize=True),
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

    # Quantizes from a bfloat16 checkpoint, so it holds the bfloat16 source and the mxfp4 result at once.
    @require_torch_accelerator_memory(memory=56)
    def test_save_mxfp4_non_quantized(self):
        non_quantized_model_name = "hf-internal-testing/gpt-oss-20b-bf16"
        tokenizer = AutoTokenizer.from_pretrained(non_quantized_model_name)
        loaded_model = GptOssForCausalLM.from_pretrained(
            non_quantized_model_name,
            quantization_config=Mxfp4Config(modules_to_not_convert=self.MODULES_TO_NOT_CONVERT),
            dtype=torch.bfloat16,
            device_map="auto",
        )
        with tempfile.TemporaryDirectory() as tmp:
            loaded_model.save_pretrained(tmp)
            cleanup(torch_device, gc_collect=True)
            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

            loaded_model = GptOssForCausalLM.from_pretrained(
                tmp,
                quantization_config=Mxfp4Config(dequantize=True),
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self.check_inference_correctness_quantized(loaded_model, tokenizer)

    def test_compute_module_sizes(self):
        """Check that the replaced experts count 4 bits per value. The device map is sized before weights load."""
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.integrations.finegrained import FineGrainedExperts
        from transformers.modeling_utils import expand_device_map, get_total_byte_count
        from transformers.quantizers import AutoHfQuantizer

        with torch.device("meta"):
            config = AutoConfig.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
            model_size, _ = compute_module_sizes(model, only_modules=False)
            hf_quantizer = AutoHfQuantizer.from_config(Mxfp4Config(), pre_quantized=False)
            hf_quantizer.preprocess_model(model=model, config=model.config)
            quantized_model_size, _ = compute_module_sizes(model, hf_quantizer, only_modules=False)
            keys = [name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()]
            expanded_device_map = expand_device_map({"": torch_device}, keys)
            (quantized_total,) = get_total_byte_count(model, expanded_device_map, hf_quantizer).values()

        for name, module in model.named_modules():
            if isinstance(module, FineGrainedExperts):
                for proj in ("gate_up_proj", "down_proj"):
                    self.assertEqual(model_size[f"{name}.{proj}"] // 4, quantized_model_size[f"{name}.{proj}"])
        self.assertEqual(quantized_total, quantized_model_size[""])
        self.assertGreater(model_size[""], quantized_model_size[""] * 3)


@require_torch
class Mxfp4ChainTest(_ChainAssertions, unittest.TestCase):
    """MXFP4 as GPT-OSS ships it.

    Blocks are `(E, N, K/32, 16)` uint8 E2M1, low nibble first, with biased-127 exponent scales.
    Gate and up rows are interleaved.
    """

    FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    def _reference_dequant(self, blocks, scales):
        # Reference from mxfp4.py. Low and high nibbles interleave along K, scaled by 2^(scale-127).
        lut = torch.tensor(self.FP4_VALUES)
        lo = lut[(blocks & 0xF).long()]
        hi = lut[(blocks >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(*blocks.shape[:-1], -1)  # (E, N, K/32, 32)
        exp = (scales.long() - 127).unsqueeze(-1)
        return (vals * torch.pow(torch.tensor(2.0), exp)).reshape(*blocks.shape[:2], -1)

    def test_packed_experts(self):
        """Packed E2M1 per expert with an E8M0 scale per 32, as DeepSeek-V4 ships its experts.

        Two values share each `int8` byte along K. The experts merge onto the stacked module.
        """
        torch.manual_seed(0)
        checkpoint = _per_expert(
            lambda n, k: {
                "weight": torch.randint(-128, 128, (n, k // 2), dtype=torch.int8),
                "weight_scale_inv": torch.randint(120, 130, (n, k // 32), dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
            }
        )
        model, info, saved = _load_and_save(checkpoint, Mxfp4Config())
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        self.assertHoldsInterleaved(experts, checkpoint, "weight", experts.gate_up_proj)
        self.assertHoldsInterleaved(experts, checkpoint, "weight_scale_inv", experts.gate_up_proj_scale_inv)
        self.assertSavesBack(saved, checkpoint)

    def test_gpt_oss_blocks_and_scales(self):
        """GPT-OSS's `{proj}_blocks` `(E, N, K/32, 16)` hold the packed `(E, N, K/2)` bytes the kernels read.

        The scales are E8M0 exponent bytes. Both save back as they shipped.
        """
        torch.manual_seed(0)
        checkpoint = {"model.layers.0.self_attn.q_proj.weight": torch.randn(H, H)}
        for proj, (rows, cols) in (("gate_up_proj", (2 * I, H)), ("down_proj", (H, I))):
            prefix = f"model.layers.0.mlp.experts.{proj}"
            checkpoint[f"{prefix}_blocks"] = torch.randint(0, 256, (E, rows, cols // 32, 16), dtype=torch.uint8)
            checkpoint[f"{prefix}_scales"] = torch.randint(120, 130, (E, rows, cols // 32), dtype=torch.uint8)
        model, info, saved = _load_and_save(checkpoint, Mxfp4Config(modules_to_not_convert=["q_proj"]))
        self.assertLoadsCleanly(info)
        experts = model.model.layers[0].mlp.experts
        blocks = checkpoint["model.layers.0.mlp.experts.down_proj_blocks"]
        self.assertEqual(experts.down_proj.dtype, torch.int8)
        self.assertTrue(torch.equal(experts.down_proj.view(torch.uint8), blocks.reshape(E, H, -1)))
        self.assertSavesBack(saved, checkpoint)

    def test_blocks_dequantize_to_the_unquantized_experts_orientation(self):
        """Check that `dequantize=True` loads the blocks in the `(E, hidden, 2I)` orientation GPT-OSS's experts hold.

        The model then saves as an unquantized one.
        """
        from safetensors.torch import load_file, save_file

        torch.manual_seed(0)
        config = GptOssConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=2,
            num_experts_per_tok=1,
        )
        state_dict = GptOssForCausalLM(config).state_dict()
        unquantized_keys = set(state_dict)
        expected = {}
        for proj, (rows, cols) in (("gate_up_proj", (128, 64)), ("down_proj", (64, 64))):
            name = f"model.layers.0.mlp.experts.{proj}"
            blocks = torch.randint(0, 256, (2, rows, cols // 32, 16), dtype=torch.uint8)
            scales = torch.randint(120, 134, (2, rows, cols // 32), dtype=torch.uint8)
            expected[proj] = self._reference_dequant(blocks, scales).transpose(1, 2)
            del state_dict[name]
            state_dict[f"{name}_blocks"], state_dict[f"{name}_scales"] = blocks, scales
        config.quantization_config = {"quant_method": "mxfp4"}
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            save_file({k: v.contiguous() for k, v in state_dict.items()}, f"{tmp}/model.safetensors")
            model = GptOssForCausalLM.from_pretrained(tmp, quantization_config=Mxfp4Config(dequantize=True))
            model.save_pretrained(f"{tmp}/saved")
            saved_keys = set(load_file(f"{tmp}/saved/model.safetensors"))
        experts = model.model.layers[0].mlp.experts
        for proj, reference in expected.items():
            torch.testing.assert_close(getattr(experts, proj).float(), reference, rtol=0, atol=0)
        self.assertEqual(saved_keys, unquantized_keys)


class Mxfp4Test(BlockScaledFormatTesterMixin, unittest.TestCase):
    spec = MXFP4


class Mxfp4TinyCheckpointTest(TinyCheckpointTesterMixin, TestCasePlus):
    spec = MXFP4


class Mxfp4LoadPathEquivalenceTest(LoadPathEquivalenceTesterMixin, TestCasePlus):
    spec = MXFP4
