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
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import MixtralConfig, MixtralForCausalLM, NVFP4Config, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.integrations.mxfp4 import FP4_VALUES
from transformers.integrations.nvfp4 import (
    NVFP4Dequantize,
    NVFP4Deserialize,
    NVFP4Experts,
    NVFP4Linear,
    NVFP4Quantize,
    swizzle_sf,
)
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from transformers.quantizers.quantizer_nvfp4 import NVFP4HfQuantizer
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import QuantizationMethod


if is_torch_available():
    import torch
    from safetensors.torch import save_file
    from torch import nn


def _padded_sf_shape(rows, columns):
    return -(-rows // 128) * 128, -(-columns // 64)  # 128 x 4 tiles of fp8 scales, four per int32


def _unswizzle(sf, rows, columns):
    "Inverse of `swizzle_sf`: the row-major `(rows, columns)` block scales"
    row_tiles, col_tiles = -(-rows // 128), -(-columns // 4)
    tiles = sf.reshape(-1).view(torch.uint8).view(row_tiles, col_tiles, 32, 4, 4).transpose(2, 3)
    return tiles.reshape(row_tiles, col_tiles, 128, 4).transpose(1, 2).reshape(row_tiles * 128, -1)[:rows, :columns]


def _dequantize(packed, sf_rowmajor):
    "e2m1 values times their fp8 block scales"
    values = torch.tensor(FP4_VALUES)[torch.stack([packed & 0xF, packed >> 4], -1).int()].flatten(-2)
    scales = sf_rowmajor.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return values * scales


def _quantize(input, global_scale):
    "NVFP4 activations: `input * global_scale` as e2m1 values and row-major fp8 block scales"
    blocks = (input.float() * global_scale).view(input.shape[0], -1, 16)
    sf = (blocks.abs().amax(-1) / 6).to(torch.float8_e4m3fn)
    scaled = blocks / sf.float().clamp(min=1e-12).unsqueeze(-1)
    codes = (scaled.unsqueeze(-1) - torch.tensor(FP4_VALUES)).abs().argmin(-1).flatten(1)
    return (codes[:, 0::2] | codes[:, 1::2] << 4).to(torch.uint8), sf.view(torch.uint8)


def _fake_kernel():
    "The NVFP4 kernel's entry points, computing in float32 from the dequantized operands"

    def pack(weight, device):
        out_features, in_features = weight.shape
        return SimpleNamespace(
            qweight=torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
            sf=torch.empty(_padded_sf_shape(out_features, in_features), dtype=torch.int32, device=device),
            sf_rowmajor=torch.empty(out_features, in_features // 16, dtype=torch.uint8, device=device),
            global_scale=torch.ones(1, dtype=torch.float32, device=device),
        )

    def scaled_fp4_quant(input, global_scale, swizzled):
        qinput, sf = _quantize(input, global_scale)
        return qinput, swizzle_sf(sf).view(torch.int32).view(_padded_sf_shape(*input.shape)) if swizzled else sf

    def cutlass_scaled_fp4_mm(qinput, weight, input_sf, weight_sf, alpha):
        (rows, k), out_features = qinput.shape, weight.shape[0]
        input = _dequantize(qinput, _unswizzle(input_sf, rows, k // 8))
        weight = _dequantize(weight, _unswizzle(weight_sf, out_features, k // 8))
        return (input @ weight.T * alpha).to(torch.bfloat16)

    def gemm(packed_weight, input):  # W4A16
        weight = _dequantize(packed_weight.qweight, packed_weight.sf_rowmajor) / packed_weight.global_scale
        return (input.float() @ weight.T).to(input.dtype)

    return SimpleNamespace(
        PackedWeight=SimpleNamespace,
        gemm=gemm,
        pack=pack,
        swizzled_sf_shape=_padded_sf_shape,
        ops=SimpleNamespace(scaled_fp4_quant=scaled_fp4_quant, cutlass_scaled_fp4_mm=cutlass_scaled_fp4_mm),
    )


def _fake_scaled_grouped_mm(mat_a, mat_b, scale_a, recipe_a, scale_b, recipe_b, swizzle_a, swizzle_b, offs, **_):
    "`scaled_grouped_mm` for NVFP4 2D x 3D operands: each group's activation scales start on a 128-row tile"
    a, b = mat_a.view(torch.uint8), mat_b.transpose(-2, -1).view(torch.uint8)  # (rows, k / 2), (groups, n, k / 2)
    k_blocks = a.shape[1] // 8
    a_sf = _unswizzle(scale_a, scale_a.numel() // (-(-k_blocks // 4) * 4), k_blocks)
    output, start, padded_start = a.new_zeros(a.shape[0], b.shape[1], dtype=torch.float32), 0, 0
    for group, end in enumerate(offs.tolist()):
        input = _dequantize(a[start:end], a_sf[padded_start : padded_start + end - start])
        output[start:end] = input @ _dequantize(b[group], _unswizzle(scale_b[group], b.shape[1], k_blocks)).T
        start, padded_start = end, padded_start + -(-(end - start) // 128) * 128
    return output.to(torch.bfloat16)


def _modelopt_quantize(weight):
    "ModelOpt's NVFP4 (`NVFP4QTensor.quantize`): `weight ~= fp4 * weight_scale * weight_scale_2`"
    weight_scale_2 = weight.abs().amax().float() / (6 * 448)
    packed, weight_scale = _quantize(weight, 1 / weight_scale_2)
    return packed, weight_scale.view(torch.float8_e4m3fn), weight_scale_2


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
class NVFP4ConfigTest(unittest.TestCase):
    def test_config_uses_quantization_method_enum(self):
        config = NVFP4Config(modules_to_not_convert=["lm_head"])

        self.assertEqual(config.quant_method, QuantizationMethod.NVFP4)
        self.assertEqual(config.to_dict()["modules_to_not_convert"], ["lm_head"])


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
class NVFP4IntegrationTest(unittest.TestCase):
    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_replaces_only_eligible_linears(self, _):
        model = nn.ModuleDict(
            {
                "convert": nn.Linear(64, 32, bias=False),
                "skip": nn.Linear(64, 32, bias=False),
                "biased": nn.Linear(64, 32, bias=True),
                "unaligned": nn.Linear(63, 32, bias=False),
            }
        )

        quantizer = NVFP4HfQuantizer(NVFP4Config(), pre_quantized=False)
        quantizer.modules_to_not_convert = ["skip"]
        quantizer.replace_modules(model)

        self.assertIsInstance(model["convert"], NVFP4Linear)
        self.assertIsInstance(model["skip"], nn.Linear)
        self.assertIsInstance(model["biased"], nn.Linear)
        self.assertIsInstance(model["unaligned"], nn.Linear)
        self.assertEqual(model["convert"].weight_sf.dtype, torch.int32)
        self.assertEqual(model["convert"].weight_scale.dtype, torch.uint8)
        self.assertEqual(model["convert"].weight_scale_2.dtype, torch.float32)

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_linear_forward_uses_normal_function_entry_point(self, _):
        layer = NVFP4Linear(64, 32)
        layer.weight = torch.zeros(32, 32, dtype=torch.uint8)
        output = layer(torch.randn(2, 3, 64))

        self.assertEqual(output.shape, (2, 3, 32))

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_packed_tensors_are_registered_in_state_dict(self, _):
        layer = NVFP4Linear(64, 32)

        buffer_names = {"weight", "weight_sf", "weight_scale", "weight_scale_2", "input_scale"}
        for buffer_name in buffer_names:
            self.assertIsInstance(getattr(layer, buffer_name), nn.Buffer)
        self.assertEqual(set(layer.state_dict()), buffer_names)
        self.assertNotIn("nvfp4_kernel", layer.state_dict())

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_conversion_produces_all_registered_buffers(self, _):
        conversion = NVFP4Quantize(torch.device("cpu"))

        output = conversion.convert({"model.proj.weight": torch.randn(32, 64)})

        self.assertEqual(
            set(output),
            {
                "model.proj.weight",
                "model.proj.weight_sf",
                "model.proj.weight_scale",
                "model.proj.weight_scale_2",
                "model.proj.input_scale",
            },
        )
        self.assertEqual(output["model.proj.weight"].shape, (32, 32))

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_modelopt_checkpoint_round_trip(self, _):
        weight = torch.randn(128, 64)
        packed, weight_scale, weight_scale_2 = _modelopt_quantize(weight)
        # As ModelOpt saves them
        checkpoint = {
            "weight": packed,
            "weight_scale": weight_scale,
            "weight_scale_2": weight_scale_2,
            "input_scale": torch.tensor(1 / 448),
        }
        buffers = NVFP4Deserialize(None).convert(dict(checkpoint))
        layer = NVFP4Linear(64, 128)
        self.assertEqual(set(buffers), set(layer.state_dict()))

        model = SimpleNamespace(get_parameter_or_buffer=lambda name: torch.empty(0))
        dense = NVFP4Dequantize(None).convert(checkpoint, model=model, full_layer_name="proj.weight")["weight"][0]
        # Within half the widest e2m1 step (4 -> 6) of each block
        bound = (weight_scale.float() * weight_scale_2).unsqueeze(-1)
        self.assertTrue(((dense - weight).abs().view(128, -1, 16) <= bound + 1e-6).all())

        # Static `input_scale`: the W4A4 GEMM on the swizzled scales, the W4A16 GEMV on the row-major ones for one row.
        # Activations on the e2m1 grid with a block max of 6 are quantized exactly by `input_scale = 1 / 448`
        for name, buffer in buffers.items():
            setattr(layer, name, buffer)
        input = torch.tensor(FP4_VALUES)[torch.randint(0, 16, (3, 64))]
        input[:, ::16] = 6.0
        for rows in (input, input[:1]):
            torch.testing.assert_close(layer(rows.bfloat16()), (rows @ dense.float().T).bfloat16())


def _tiny_config(config_class, **kwargs):
    # Only the experts are quantized in these tests
    modules_to_not_convert = ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head", ".*shared_expert", ".*linear_attn"]
    return config_class(
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        quantization_config=NVFP4Config(modules_to_not_convert=modules_to_not_convert),
        **kwargs,
    )


def _modelopt_experts_checkpoint(model, names, rename):
    """
    The state dict of `model` as a ModelOpt NVFP4 export, renamed by `rename`: per-expert projections `names` (gate,
    up, down) with their scales, a different `weight_scale_2` for each.
    """
    checkpoint = {}
    for key, value in model.state_dict().items():
        if ".experts." not in key:
            checkpoint[rename(key)] = value.contiguous()
            continue
        layer = key.rsplit(".experts.", 1)[0]
        for expert, weight in enumerate(value):
            projections = zip(names, weight.chunk(2)) if key.endswith("gate_up_proj") else [(names[2], weight)]
            for name, projection in projections:
                name = rename(f"{layer}.experts.{expert}.{name}")
                packed, weight_scale, weight_scale_2 = _modelopt_quantize(projection.float())
                checkpoint[f"{name}.weight"], checkpoint[f"{name}.weight_scale"] = packed, weight_scale
                checkpoint[f"{name}.weight_scale_2"] = weight_scale_2
                checkpoint[f"{name}.input_scale"] = torch.tensor(1 / 448)
    return checkpoint


def _dequantized_expert(checkpoint, prefix, names, expert):
    "The dense `gate_up_proj` and `down_proj` of one expert, from its ModelOpt tensors"
    projections = []
    for name in names:
        name = f"{prefix}.{expert}.{name}"
        dense = _dequantize(checkpoint[f"{name}.weight"], checkpoint[f"{name}.weight_scale"].view(torch.uint8))
        projections.append(dense * checkpoint[f"{name}.weight_scale_2"])
    return torch.cat(projections[:2]), projections[2]


def _load(config, model_class, checkpoint, **kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        config.save_pretrained(tmp)
        save_file(checkpoint, f"{tmp}/model.safetensors", metadata={"format": "pt"})
        return model_class.from_pretrained(tmp, output_loading_info=True, **kwargs)


MIXTRAL_NAMES, MIXTRAL_EXPERTS = ("w1", "w3", "w2"), "model.layers.0.block_sparse_moe.experts"


def _mixtral_checkpoint():
    config = _tiny_config(MixtralConfig, intermediate_size=64, num_local_experts=4)
    rename = lambda key: key.replace(".mlp.", ".block_sparse_moe.")  # noqa: E731
    return config, _modelopt_experts_checkpoint(MixtralForCausalLM(config), MIXTRAL_NAMES, rename)


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
@patch.object(NVFP4HfQuantizer, "validate_environment")  # no CUDA GPU needed with the fake kernel
@patch.object(FineGrainedFP8HfQuantizer, "validate_environment")
@patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
class NVFP4ExpertsTest(unittest.TestCase):
    def test_load(self, *_):
        # Like nvidia/Qwen3.6-35B-A3B-NVFP4: `model.language_model.layers.N.mlp.experts.E.gate_proj`, ModelOpt's
        # per-layer algos with an FP8 attention listed before the NVFP4 experts
        qwen = _tiny_config(
            Qwen3_5MoeTextConfig,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=4,
            layer_types=["linear_attention", "full_attention"],
        )
        qwen_names, qwen_prefix = ("gate_proj", "up_proj", "down_proj"), "model.language_model."
        qwen_checkpoint = _modelopt_experts_checkpoint(
            Qwen3_5MoeForCausalLM(qwen), qwen_names, lambda key: key.replace("model.", qwen_prefix, 1)
        )
        o_proj = f"{qwen_prefix}layers.1.self_attn.o_proj"
        qwen_checkpoint[f"{o_proj}.weight"] = qwen_checkpoint[f"{o_proj}.weight"].to(torch.float8_e4m3fn)
        for name in ("weight_scale", "input_scale"):
            qwen_checkpoint[f"{o_proj}.{name}"] = torch.tensor(1.0)
        experts = {f"{qwen_prefix}layers.{i}.mlp.experts": {"quant_algo": "W4A16_NVFP4"} for i in range(2)}
        qwen.quantization_config = {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {o_proj: {"quant_algo": "FP8"}} | experts,
        }
        cases = [
            (MixtralForCausalLM, *_mixtral_checkpoint(), MIXTRAL_NAMES, MIXTRAL_EXPERTS),
            (Qwen3_5MoeForCausalLM, qwen, qwen_checkpoint, qwen_names, f"{qwen_prefix}layers.0.mlp.experts"),
        ]
        for model_class, config, checkpoint, names, prefix in cases:
            dequantize = {"dequantize": True} if model_class is Qwen3_5MoeForCausalLM else NVFP4Config(dequantize=True)
            quantized, info = _load(config, model_class, checkpoint, experts_implementation="eager")
            dense, dense_info = _load(config, model_class, checkpoint, quantization_config=dequantize)
            self.assertFalse(any(info.values()) or any(dense_info.values()), (info, dense_info))
            self.assertIsInstance(quantized.model.layers[0].mlp.experts, NVFP4Experts)
            # `dequantize=True`: the dense experts, each half with its own global scale
            gate_up, down = _dequantized_expert(checkpoint, prefix, names, 2)
            torch.testing.assert_close(dense.model.layers[0].mlp.experts.gate_up_proj[2], gate_up)
            torch.testing.assert_close(dense.model.layers[0].mlp.experts.down_proj[2], down)

    def test_forward(self, *_):
        config, checkpoint = _mixtral_checkpoint()
        quantized, _ = _load(config, MixtralForCausalLM, checkpoint, experts_implementation="eager")
        experts = quantized.model.layers[0].mlp.experts
        # Activations on the e2m1 grid with a block max of 6: `input_scale = 1 / 448` quantizes them exactly, so each
        # half's global scale must be applied exactly
        input = torch.tensor(FP4_VALUES)[torch.randint(0, 16, (5, 128))]
        input[:, ::16] = 6.0
        input, expert_ids, tokens = input.bfloat16(), torch.tensor([0, 0, 1, 3, 3]), torch.tensor([2, 1, 0, 2])
        with patch("torch.nn.functional.scaled_grouped_mm", _fake_scaled_grouped_mm):
            offsets = torch.cumsum(tokens, 0, dtype=torch.int32)
            grouped = experts.grouped_linear("gate_up_proj", input, expert_ids, tokens, offsets)
            for row, expert in enumerate(expert_ids.tolist()):
                gate_up, _ = _dequantized_expert(checkpoint, MIXTRAL_EXPERTS, MIXTRAL_NAMES, expert)
                reference = (input[row].float() @ gate_up.T).bfloat16()
                torch.testing.assert_close(grouped[row], reference)
                # Eager, one expert: the W4A4 GEMM for a few rows, the W4A16 GEMV for one
                for rows in (input[[row, row, row]], input[[row]]):
                    eager = experts.linear("gate_up_proj", torch.tensor(expert), rows)
                    torch.testing.assert_close(eager[0], reference)

            # The whole grouped forward matches the eager one
            hidden_states = torch.randn(7, 128, dtype=torch.bfloat16)
            top_k_index = torch.tensor([[0, 3], [3, 0], [1, 3], [3, 1], [0, 1], [3, 0], [1, 0]])  # expert 2 is not hit
            top_k_weights = torch.rand(7, 2)
            eager = experts.eager_forward(hidden_states, top_k_index, top_k_weights)
            experts.config._experts_implementation = "grouped_mm"
            grouped = experts(hidden_states, top_k_index, top_k_weights)
        # Same quantized operands, but the grouped path divides the activations by their scale in bf16 first
        torch.testing.assert_close(grouped, eager, rtol=0.05, atol=0.05)


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
class NVFP4QuantizerValidationTest(unittest.TestCase):
    def setUp(self):
        self.quantizer = NVFP4HfQuantizer(NVFP4Config(), pre_quantized=False)

    @patch("transformers.quantizers.quantizer_nvfp4.is_kernels_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(10, 0))
    def test_accepts_one_blackwell_cuda_device(self, _, __, ___):
        self.quantizer.validate_environment(device_map={"": "cuda:1"})

        self.assertEqual(self.quantizer.quantization_device, torch.device("cuda:1"))

    @patch("transformers.quantizers.quantizer_nvfp4.is_kernels_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_rejects_multi_device_map_until_tp_metadata_is_supported(self, _, __):
        with self.assertRaisesRegex(ValueError, "Tensor parallelism"):
            self.quantizer.validate_environment(device_map={"model.layers.0": 0, "model.layers.1": 1})

    def test_rejects_native_tp_until_scale_metadata_is_supported(self):
        model = SimpleNamespace(config=SimpleNamespace(distributed_config=SimpleNamespace(tp_size=2)))

        with self.assertRaisesRegex(ValueError, "scale metadata does not have a sharding plan"):
            self.quantizer.update_parallel_plan(model, {})

    def test_update_parallel_plan_keeps_non_tp_configs(self):
        configs = (
            SimpleNamespace(),
            SimpleNamespace(distributed_config=SimpleNamespace(tp_size=1)),
        )

        for config in configs:
            with self.subTest(config=config):
                self.quantizer.update_parallel_plan(SimpleNamespace(config=config), {})

    @patch("transformers.quantizers.quantizer_nvfp4.is_kernels_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_rejects_cpu_offload(self, _, __):
        with self.assertRaisesRegex(ValueError, "CPU or disk offload"):
            self.quantizer.validate_environment(device_map={"model": "cuda", "lm_head": "cpu"})
