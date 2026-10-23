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
import json
import tempfile
import unittest
import unittest.mock
from types import SimpleNamespace

from parameterized import parameterized
from safetensors import safe_open

from transformers import FineGrainedFP8Config, MixtralConfig, NVFP4Config
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.testing_utils import require_kernels, require_torch, require_torch_accelerator, slow, torch_device
from transformers.utils import is_torch_available, logging
from transformers.utils.quantization_config import QuantizationMethod, read_modelopt


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, MixtralForCausalLM
    from transformers.integrations.nvfp4 import (
        NVFP4Dequantize,
        NVFP4ExpertGlobals,
        NVFP4Experts,
        NVFP4Linear,
        NVFP4SwizzleScales,
        replace_with_nvfp4,
        update_nvfp4_parallel_plan,
    )

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4(weight, amax=None):
    """Quantize a `(N, K)` weight like ModelOpt. Return `(packed uint8, E4M3 scales, global, dequantized)`."""
    weight = weight.float()
    values = torch.tensor(E2M1)
    global_scale = (weight.abs().amax() if amax is None else amax) / (6 * 448)
    blocks = weight.reshape(weight.shape[0], -1, 16)
    scales = (blocks.abs().amax(-1) / 6 / global_scale).clamp(min=2**-9).to(torch.float8_e4m3fn)
    scaled = blocks / (scales.float() * global_scale).unsqueeze(-1)
    index = (scaled.abs().unsqueeze(-1) - values).abs().argmin(-1)
    codes = (index | ((scaled < 0).long() << 3)).reshape(weight.shape[0], -1)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    dequantized = values[index] * scaled.sign() * scales.float().unsqueeze(-1) * global_scale
    return packed, scales, global_scale.reshape(()), dequantized.reshape(weight.shape)


def _fp8_block(weight):
    """Quantize a `(N, K)` weight to block FP8 with ModelOpt's `[N / 128, 1, K / 128, 1]` grid."""
    rows, cols = weight.shape
    blocks = weight.float().reshape(rows // 128, 128, cols // 128, 128)
    scales = blocks.abs().amax(dim=(1, 3), keepdim=True) / 448
    quantized = (blocks / scales).to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scales).reshape(rows, cols)
    return quantized.reshape(rows, cols), scales.reshape(rows // 128, 1, cols // 128, 1), dequantized


def _mxfp8(weight):
    """Quantize a `(N, K)` weight to MXFP8 with UE8M0 scales per 32. Return `(values, uint8 scales, dequantized)`."""
    blocks = weight.float().reshape(weight.shape[0], -1, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp(min=2**-100) / 448))
    values = (blocks / torch.exp2(exponent).unsqueeze(-1)).to(torch.float8_e4m3fn)
    dequantized = values.float() * torch.exp2(exponent).unsqueeze(-1)
    return values.reshape(weight.shape), (exponent + 127).to(torch.uint8), dequantized.reshape(weight.shape)


def _mixtral_config(**kwargs):
    options = {
        "vocab_size": 512,
        "hidden_size": 256,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
    }
    return MixtralConfig(**{**options, **kwargs})


def _cosine(got, expected):
    return torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0)


def _modelopt_checkpoint(path):
    """Save a tiny Mixtral as a ModelOpt export and return its dequantized reference model.

    The experts are NVFP4 per expert. The attention mixes NVFP4, weight-only NVFP4, block and per-tensor FP8. The
    `lm_head` is weight-only NVFP4.
    """
    torch.manual_seed(0)
    config = _mixtral_config()
    reference = MixtralForCausalLM(config).to(torch.bfloat16).eval()
    state = {name: param.clone() for name, param in reference.state_dict().items() if ".mlp.experts." not in name}
    layers = {}
    for i, layer in enumerate(reference.model.layers):
        experts = layer.mlp.experts
        for expert in range(config.num_local_experts):
            gate, up = experts.gate_up_proj.data[expert].chunk(2, dim=0)
            # ModelOpt calibrates w1 and w3 together, so they share one global
            shared = experts.gate_up_proj.data[expert].float().abs().amax()
            for proj, weight, amax in (
                ("w1", gate, shared),
                ("w3", up, shared),
                ("w2", experts.down_proj.data[expert], None),
            ):
                packed, scales, global_scale, dequantized = _nvfp4(weight, amax)
                key = f"model.layers.{i}.block_sparse_moe.experts.{expert}.{proj}"
                state[f"{key}.weight"], state[f"{key}.weight_scale"] = packed, scales
                state[f"{key}.weight_scale_2"], state[f"{key}.input_scale"] = global_scale, torch.tensor(4 / (6 * 448))
                weight.copy_(dequantized.to(weight.dtype))
        layers[f"model.layers.{i}.block_sparse_moe.experts"] = {"quant_algo": "NVFP4"}
        for proj, algo in (("q_proj", "NVFP4"), ("k_proj", "W4A16_NVFP4"), ("o_proj", "FP8_PB_WO"), ("v_proj", "FP8")):
            linear = getattr(layer.self_attn, proj)
            key = f"model.layers.{i}.self_attn.{proj}"
            if algo == "FP8":
                scale = linear.weight.data.float().abs().amax() / 448
                quantized = (linear.weight.data.float() / scale).to(torch.float8_e4m3fn)
                dequantized = quantized.float() * scale
                state[f"{key}.weight"], state[f"{key}.weight_scale"] = quantized, scale.reshape(())
            elif algo == "FP8_PB_WO":
                quantized, scales, dequantized = _fp8_block(linear.weight.data)
                state[f"{key}.weight"], state[f"{key}.weight_scale"] = quantized, scales
            else:
                packed, scales, global_scale, dequantized = _nvfp4(linear.weight.data)
                state[f"{key}.weight"], state[f"{key}.weight_scale"], state[f"{key}.weight_scale_2"] = (
                    packed,
                    scales,
                    global_scale,
                )
                if algo == "NVFP4":
                    state[f"{key}.input_scale"] = torch.tensor(8 / (6 * 448))
            linear.weight.data.copy_(dequantized.to(linear.weight.dtype))
            layers[key] = {"quant_algo": algo}
    packed, scales, global_scale, dequantized = _nvfp4(reference.lm_head.weight.data)
    state["lm_head.weight"], state["lm_head.weight_scale"], state["lm_head.weight_scale_2"] = (
        packed,
        scales,
        global_scale,
    )
    reference.lm_head.weight.data.copy_(dequantized.to(reference.lm_head.weight.dtype))
    layers["lm_head"] = {"quant_algo": "W4A16_NVFP4"}

    from safetensors.torch import save_file

    config.save_pretrained(path)
    with open(f"{path}/config.json", encoding="utf-8") as file:
        saved = json.load(file)
    saved["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": layers,
    }
    with open(f"{path}/config.json", "w", encoding="utf-8") as file:
        json.dump(saved, file)
    save_file({key: value.contiguous() for key, value in state.items()}, f"{path}/model.safetensors")
    return reference


def _mxfp8_checkpoint(path):
    """Save a tiny Mixtral as an MXFP8 checkpoint with NVFP4 experts and return its dequantized reference model.

    The experts are `nvfp4-pack-quantized`, with reciprocal globals. The attention is MXFP8. The routers and `lm_head`
    stay in full precision.
    """
    torch.manual_seed(0)
    config = _mixtral_config()
    reference = MixtralForCausalLM(config).to(torch.bfloat16).eval()
    state = {name: param.clone() for name, param in reference.state_dict().items() if ".mlp.experts." not in name}
    for i, layer in enumerate(reference.model.layers):
        experts = layer.mlp.experts
        for expert in range(config.num_local_experts):
            gate, up = experts.gate_up_proj.data[expert].chunk(2, dim=0)
            for proj, weight in (("w1", gate), ("w3", up), ("w2", experts.down_proj.data[expert])):
                packed, scales, global_scale, dequantized = _nvfp4(weight)
                key = f"model.layers.{i}.block_sparse_moe.experts.{expert}.{proj}"
                state[f"{key}.weight_packed"], state[f"{key}.weight_scale"] = packed, scales
                state[f"{key}.weight_global_scale"] = 1 / global_scale
                state[f"{key}.input_global_scale"] = torch.tensor(1.0)
                weight.copy_(dequantized.to(weight.dtype))
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            linear = getattr(layer.self_attn, proj)
            key = f"model.layers.{i}.self_attn.{proj}"
            state[f"{key}.weight"], state[f"{key}.weight_scale_inv"], dequantized = _mxfp8(linear.weight.data)
            linear.weight.data.copy_(dequantized.to(linear.weight.dtype))

    from safetensors.torch import save_file

    config.save_pretrained(path)
    with open(f"{path}/config.json", encoding="utf-8") as file:
        saved = json.load(file)
    saved["quantization_config"] = {
        "quant_method": "mxfp8",
        "weight_block_size": [1, 32],
        "ignored_layers": ["lm_head", *(f"model.layers.{i}.block_sparse_moe.gate" for i in range(2))],
        "moe_quant_algo": "NVFP4",
        "moe_quant_format": "nvfp4-pack-quantized",
    }
    with open(f"{path}/config.json", "w", encoding="utf-8") as file:
        json.dump(saved, file)
    save_file({key: value.contiguous() for key, value in state.items()}, f"{path}/model.safetensors")
    return reference


@require_torch
class NVFP4ConfigTest(unittest.TestCase):
    def test_a_modelopt_export_reads_its_formats_and_skip_list(self):
        shipped = {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "layers.0.experts": {"quant_algo": "NVFP4"},
                "layers.0.attention.wq": {"quant_algo": "FP8_PB_WO"},
            },
            "exclude_modules": ["lm_head"],
            # some exports write the list as its string
            "modules_to_not_convert": "['model.layers.0.mlp.gate']",
        }
        logging.warning_once.cache_clear()
        with self.assertLogs("transformers", level="WARNING") as logs:
            config = AutoQuantizationConfig.from_dict(shipped)
        self.assertIn("FP8_PB_WO", logs.output[0])
        self.assertIsInstance(config, NVFP4Config)
        self.assertEqual(config.quant_method, QuantizationMethod.NVFP4)
        self.assertEqual(config.quantized_layers, {"layers.0.experts": "nvfp4", "layers.0.attention.wq": "dequantize"})
        self.assertEqual(config.modules_to_not_convert, ["model.layers.0.mlp.gate", r".*lm_head"])
        self.assertIsNone(
            AutoQuantizationConfig.from_dict({"quant_method": "modelopt", "quant_algo": "NVFP4"}).quantized_layers
        )
        # a single weight-only algorithm makes the whole model weight-only
        config = read_modelopt({"quant_method": "modelopt", "quant_algo": "W4A16_NVFP4"})
        self.assertEqual(config.activation_format, "bf16")
        # a mixed export lists every quantized module, so the default skips (e.g. `lm_head`) do not apply
        layers = {"lm_head": {"quant_algo": "W4A16_NVFP4"}}
        config = read_modelopt(
            {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers}
        )
        self.assertEqual(config.modules_to_not_convert, [])

    def test_an_mxfp8_checkpoint_with_nvfp4_experts_reads_as_nvfp4(self):
        shipped = {"quant_method": "mxfp8", "weight_block_size": [1, 32], "ignored_layers": ["lm_head"]}
        self.assertIsInstance(AutoQuantizationConfig.from_dict(shipped), FineGrainedFP8Config)
        shipped.update({"moe_quant_algo": "NVFP4", "moe_quant_format": "nvfp4-pack-quantized"})
        config = AutoQuantizationConfig.from_dict(shipped)
        self.assertIsInstance(config, NVFP4Config)
        self.assertEqual(config.checkpoint_format, "nvfp4-pack-quantized")
        self.assertEqual(
            config.quantized_layers, {r"lm_head(\..*)?": None, r".*\.experts": "nvfp4", r".*": "dequantize"}
        )
        with self.assertRaises(ValueError):
            AutoQuantizationConfig.from_dict({**shipped, "moe_quant_format": "unknown"})

    @parameterized.expand([("without_nvfp4", ["FP8"]), ("with_another_format", ["NVFP4", "W4A8_AWQ"])])
    def test_a_modelopt_export_raises(self, _, algos):
        layers = {f"layer{i}": {"quant_algo": algo} for i, algo in enumerate(algos)}
        with self.assertRaises(ValueError):
            read_modelopt({"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers})


@require_torch
class NVFP4ModulesTest(unittest.TestCase):
    def _model(self):
        with torch.device("meta"):
            return MixtralForCausalLM(_mixtral_config(num_hidden_layers=1))

    def test_on_the_fly_quantizes_linears_and_experts_but_not_routers(self):
        model = self._model()
        replace_with_nvfp4(model, modules_to_not_convert=["lm_head"])
        layer = model.model.layers[0]
        self.assertIsInstance(layer.mlp.experts, NVFP4Experts)
        self.assertIsInstance(layer.self_attn.q_proj, NVFP4Linear)
        self.assertNotIsInstance(layer.mlp.gate, NVFP4Linear)
        # the weights start as full-precision placeholders, which the loader quantizes
        self.assertEqual(layer.self_attn.q_proj.weight.dtype, torch.float32)
        # without calibration, the activations quantize with a global of 1
        self.assertIsNotNone(layer.self_attn.q_proj.input_global_scale)

    def test_pre_quantized_modules_follow_the_formats(self):
        model = self._model()
        formats = {
            "model.layers.0.mlp.experts": "nvfp4",
            "model.layers.0.self_attn.q_proj": "nvfp4",
            "model.layers.0.self_attn.k_proj": "nvfp4_weight_only",
            "model.layers.0.self_attn.o_proj": "dequantize",
        }
        replace_with_nvfp4(model, formats=formats, pre_quantized=True)
        attention = model.model.layers[0].self_attn
        self.assertEqual(attention.q_proj.weight.dtype, torch.int8)
        self.assertIsNotNone(attention.q_proj.input_global_scale)
        self.assertEqual((attention.k_proj.activation_format, attention.k_proj.input_global_scale), ("bf16", None))
        # an FP8 layer keeps the model's own module, with a slot for its scale until it dequantizes
        self.assertIs(type(attention.o_proj), torch.nn.Linear)
        self.assertIn("weight_scale_inv", attention.o_proj._parameters)
        self.assertIs(type(attention.v_proj), torch.nn.Linear)

    def test_expert_parallelism_is_masked_and_tensor_parallelism_splits_inside_experts(self):
        model = self._model()
        replace_with_nvfp4(model)
        model.config.distributed_config = SimpleNamespace()
        plan = {
            "model.layers.*.mlp.experts": "ep_dispatch_experts",
            "model.layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "model.layers.*.mlp.experts.down_proj": "grouped_gemm",
        }
        update_nvfp4_parallel_plan(model, plan)
        self.assertEqual(plan["model.layers.*.mlp.experts"], "moe_tp_experts")
        self.assertEqual(plan["model.layers.*.mlp.gate"], "ep_router")
        self.assertEqual(plan["model.layers.*.mlp.experts.down_proj_weight_global_scale"], "grouped_gemm")
        # tensor parallelism alone splits each weight's scales like the weight
        plan = {
            "model.layers.*.mlp.experts": "moe_tp_experts",
            "model.layers.*.mlp.experts.gate_up_proj": "packed_colwise",
            "model.layers.*.mlp.experts.down_proj": "rowwise",
        }
        update_nvfp4_parallel_plan(model, plan)
        self.assertEqual(plan["model.layers.*.mlp.experts"], "moe_tp_experts")
        self.assertEqual(plan["model.layers.*.mlp.experts.gate_up_proj_scale_inv"], "packed_colwise")
        self.assertEqual(plan["model.layers.*.mlp.experts.down_proj_scale_inv"], "rowwise")
        self.assertNotIn("model.layers.*.mlp.experts.down_proj_weight_global_scale", plan)

    def test_experts_swizzle_gate_up_only_if_down_swizzles(self):
        # a swizzled gate_up writes swizzled activations for down, which only a swizzled down reads
        config = _mixtral_config(hidden_size=256, intermediate_size=512)
        sm100 = (
            unittest.mock.patch("torch.cuda.is_available", return_value=True),
            unittest.mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        )
        with torch.device("meta"), sm100[0], sm100[1]:
            experts = NVFP4Experts(config, pre_quantized=True)
            # tensor parallelism shrinks the intermediate size on each rank
            for split, swizzled in ((1, True), (2, True), (8, False)):
                intermediate = 512 // split
                with self.subTest(split=split):
                    self.assertEqual(experts.swizzles_scales("gate_up_proj_scale_inv", 2 * intermediate, 16), swizzled)
                    self.assertEqual(experts.swizzles_scales("down_proj_scale_inv", 256, intermediate // 16), swizzled)
            # without a gate, 192 up rows don't fit, but down still swizzles on its own
            experts = NVFP4Experts(_mixtral_config(hidden_size=256, intermediate_size=192), has_gate=False)
            self.assertFalse(experts.swizzles_scales("up_proj_scale_inv", 192, 16))
            self.assertTrue(experts.swizzles_scales("down_proj_scale_inv", 256, 12))
            # weight-only experts never swizzle
            experts = NVFP4Experts(config, pre_quantized=True, weight_only=True)
            self.assertFalse(experts.swizzles_scales("gate_up_proj_scale_inv", 1024, 16))


@require_torch
class NVFP4LoadingOpsTest(unittest.TestCase):
    def test_a_block_fp8_layer_dequantizes_from_the_modelopt_grid(self):
        weight = torch.randn(256, 256)
        quantized, scales, dequantized = _fp8_block(weight)
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
        out = NVFP4Dequantize(None).convert(
            {"weight": quantized, "weight_scale_inv": scales}, full_layer_name="proj.weight", model=model
        )
        torch.testing.assert_close(out["proj.weight"].float(), dequantized, rtol=1e-2, atol=1e-2)

    def test_the_globals_reduce_to_one_per_expert(self):
        gate, up = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 1.0])
        out = NVFP4ExpertGlobals(None, "max").convert({"w1": gate, "w3": up}, full_layer_name="x")
        self.assertEqual(out["x"].tolist(), [3.0, 2.0])
        with self.assertLogs("transformers", level="WARNING"):
            out = NVFP4ExpertGlobals(None, "first").convert({"w1": gate, "w3": up}, full_layer_name="y")
        self.assertEqual(out["y"].tolist(), [1.0, 2.0])

    def test_without_a_gpu_a_checkpoint_loads_dequantized(self):
        """Check that a ModelOpt checkpoint loads dequantized without a GPU, its NVFP4 globals included."""
        with tempfile.TemporaryDirectory() as path:
            reference = _modelopt_checkpoint(path)
            with (
                unittest.mock.patch("torch.cuda.is_available", return_value=False),
                unittest.mock.patch(
                    "transformers.quantizers.quantizer_nvfp4.is_torch_xpu_available", return_value=False
                ),
            ):
                model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu")
        self.assertIsNone(getattr(model.config, "quantization_config", None))
        self.assertFalse(any(isinstance(m, (NVFP4Linear, NVFP4Experts)) for m in model.modules()))
        ids = torch.randint(0, 512, (2, 12))
        with torch.no_grad():
            self.assertGreater(_cosine(model(ids).logits, reference(ids).logits), 0.999)

    def test_scales_that_cannot_be_swizzled_warn_on_blackwell(self):
        model = torch.nn.Module()
        with torch.device("meta"):
            model.linear = NVFP4Linear(64, 64, pre_quantized=True)
        # 4 scale columns are fewer than the 8 the swizzled layout needs
        scales = torch.zeros(64, 4, dtype=torch.float8_e4m3fn)
        logging.warning_once.cache_clear()
        with (
            unittest.mock.patch("torch.cuda.is_available", return_value=True),
            unittest.mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            unittest.mock.patch("transformers.integrations.nvfp4.load_finegrained_kernels"),
            self.assertLogs("transformers", level="WARNING") as logs,
        ):
            out = NVFP4SwizzleScales(None).convert(
                {"linear.weight_scale_inv": scales}, model=model, full_layer_name="linear.weight_scale_inv"
            )
        self.assertIs(out["linear.weight_scale_inv"], scales)
        self.assertIn("swizzled layout", logs.output[0])


def _requires_fp8_triton(test):
    # the NVFP4 kernels use Triton's FP8 type, which NVIDIA GPUs only have from compute capability 8.9
    capable = torch_device == "xpu" or (
        torch_device == "cuda" and (torch.version.hip is not None or torch.cuda.get_device_capability() >= (8, 9))
    )
    return unittest.skipUnless(capable, "NVFP4 needs a GPU with FP8 support")(test)


@require_kernels
@require_torch_accelerator
@_requires_fp8_triton
class NVFP4LoadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        reference = _modelopt_checkpoint(cls.directory.name).to(torch_device)
        cls.ids = torch.randint(0, 512, (2, 24), device=torch_device)
        cls.expected = cls._logits(reference)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    @classmethod
    def _logits(cls, model):
        with torch.no_grad():
            return model(cls.ids).logits.float()

    def _load(self, path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=torch_device, **kwargs)

    @parameterized.expand(["eager", "grouped_mm", "batched_mm"])
    def test_a_modelopt_checkpoint_matches_its_dequantized_model(self, implementation):
        model, info = self._load(self.directory.name, experts_implementation=implementation, output_loading_info=True)
        self.assertFalse(any(info.values()), info)
        self.assertIsInstance(model.model.layers[0].mlp.experts, NVFP4Experts)
        self.assertIs(type(model.model.layers[0].self_attn.o_proj), torch.nn.Linear)
        self.assertIsInstance(model.lm_head, NVFP4Linear)
        self.assertGreater(_cosine(self._logits(model), self.expected), 0.995)

    @parameterized.expand(
        [("weight_only", {"activation_format": "bf16"}, 0.998), ("dequantize", {"dequantize": True}, 0.999)]
    )
    def test_a_passed_config_overrides_the_checkpoint(self, _, options, threshold):
        model = self._load(self.directory.name, quantization_config=NVFP4Config(**options))
        if options.get("dequantize"):
            self.assertIs(type(model.model.layers[0].self_attn.q_proj), torch.nn.Linear)
        else:
            self.assertEqual(model.model.layers[0].mlp.experts.activation_format, "bf16")
        self.assertGreater(_cosine(self._logits(model), self.expected), threshold)

    def test_a_full_precision_checkpoint_quantizes_on_load(self):
        torch.manual_seed(0)
        reference = MixtralForCausalLM(_mixtral_config()).to(torch.bfloat16).eval()
        with tempfile.TemporaryDirectory() as path:
            reference.save_pretrained(path)
            model = self._load(path, quantization_config=NVFP4Config())
        self.assertIsInstance(model.model.layers[0].mlp.experts, NVFP4Experts)
        self.assertEqual(model.model.layers[0].self_attn.q_proj.weight.dtype, torch.int8)
        self.assertGreater(_cosine(self._logits(model), self._logits(reference.to(torch_device))), 0.95)

    def test_an_mxfp8_checkpoint_with_nvfp4_experts_matches_its_dequantized_model(self):
        with tempfile.TemporaryDirectory() as path, tempfile.TemporaryDirectory() as saved:
            expected = self._logits(_mxfp8_checkpoint(path).to(torch_device))
            model, info = self._load(path, output_loading_info=True)
            model.save_pretrained(saved)
            reloaded = self._load(saved)
        self.assertFalse(any(info.values()), info)
        self.assertIsInstance(model.model.layers[0].mlp.experts, NVFP4Experts)
        self.assertIs(type(model.model.layers[0].self_attn.q_proj), torch.nn.Linear)
        self.assertGreater(_cosine(self._logits(model), expected), 0.995)
        torch.testing.assert_close(self._logits(reloaded), self._logits(model))

    @parameterized.expand(["modelopt", "on_the_fly"])
    def test_a_saved_model_reloads_with_the_same_outputs(self, source):
        with tempfile.TemporaryDirectory() as full_precision, tempfile.TemporaryDirectory() as saved:
            if source == "modelopt":
                model = self._load(self.directory.name)
            else:
                MixtralForCausalLM(_mixtral_config()).to(torch.bfloat16).save_pretrained(full_precision)
                model = self._load(full_precision, quantization_config=NVFP4Config())
            model.save_pretrained(saved)
            with safe_open(f"{saved}/model.safetensors", "pt") as file:
                keys = list(file.keys())
            reloaded, info = self._load(saved, output_loading_info=True)
        # the scales split per expert, like their weights
        self.assertIn("model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv", keys)
        self.assertFalse(any(".experts.gate_up_proj" in key for key in keys))
        self.assertFalse(any(info.values()), info)
        torch.testing.assert_close(self._logits(reloaded), self._logits(model))

    @slow
    def test_muse_spark_loads_and_saves_its_fused_experts_and_post_expert_norm(self):
        from transformers import MuseSparkForConditionalGeneration

        repo = "IlyasMoutawwakil/tiny-random-Muse-Spark-NVFP4"
        load = {"dtype": torch.bfloat16, "device_map": torch_device}
        reference = MuseSparkForConditionalGeneration.from_pretrained(repo, subfolder="reference", **load)
        model = MuseSparkForConditionalGeneration.from_pretrained(repo, **load)
        experts = model.model.language_model.layers[1].mlp.experts
        self.assertIsInstance(experts, NVFP4Experts)
        self.assertIsNotNone(experts.post_expert_norm)
        ids = torch.randint(0, reference.config.get_text_config().vocab_size, (2, 16), device=torch_device)
        with torch.no_grad():
            got = model(ids).logits
            self.assertGreater(_cosine(got, reference(ids).logits), 0.99)
            with tempfile.TemporaryDirectory() as path:
                model.save_pretrained(path)
                reloaded = MuseSparkForConditionalGeneration.from_pretrained(path, **load)
            torch.testing.assert_close(reloaded(ids).logits, got)
