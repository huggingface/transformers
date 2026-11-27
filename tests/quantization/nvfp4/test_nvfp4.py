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
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import NVFP4Config
from transformers.integrations.mxfp4 import FP4_VALUES
from transformers.integrations.nvfp4 import (
    NVFP4Dequantize,
    NVFP4Deserialize,
    NVFP4Linear,
    NVFP4Quantize,
    load_nvfp4_kernel,
    swizzle_sf,
    swizzled_sf_shape,
)
from transformers.quantizers.quantizer_nvfp4 import NVFP4HfQuantizer
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import QuantizationMethod


if is_torch_available():
    import torch
    from torch import nn


def _unswizzle(sf, rows, columns):
    """Inverse of `swizzle_sf`: the row-major `(rows, columns)` block scales"""
    row_tiles, col_tiles = -(-rows // 128), -(-columns // 4)
    tiles = sf.reshape(-1).view(torch.uint8).view(row_tiles, col_tiles, 32, 4, 4).transpose(2, 3)
    return tiles.reshape(row_tiles, col_tiles, 128, 4).transpose(1, 2).reshape(row_tiles * 128, -1)[:rows, :columns]


def _dequantize(packed, sf_rowmajor):
    """e2m1 values times their fp8 block scales"""
    values = torch.tensor(FP4_VALUES, device=packed.device)[
        torch.stack([packed & 0xF, packed >> 4], -1).int()
    ].flatten(-2)
    scales = sf_rowmajor.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return values * scales


def _quantize(input, global_scale):
    """NVFP4 activations: `input * global_scale` as e2m1 values and row-major fp8 block scales"""
    blocks = (input.float() * global_scale).view(input.shape[0], -1, 16)
    sf = (blocks.abs().amax(-1) / 6).to(torch.float8_e4m3fn)
    scaled = blocks / sf.float().clamp(min=1e-12).unsqueeze(-1)
    codes = (scaled.unsqueeze(-1) - torch.tensor(FP4_VALUES, device=input.device)).abs().argmin(-1).flatten(1)
    return (codes[:, 0::2] | codes[:, 1::2] << 4).to(torch.uint8), sf.view(torch.uint8)


def _fake_kernel():
    """The NVFP4 kernel's entry points, computing in float32 from the dequantized operands"""

    def pack(weight, device):
        out_features, in_features = weight.shape
        return SimpleNamespace(
            qweight=torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
            sf=torch.empty(swizzled_sf_shape(out_features, in_features), dtype=torch.int32, device=device),
            sf_rowmajor=torch.empty(out_features, in_features // 16, dtype=torch.uint8, device=device),
            global_scale=torch.ones(1, dtype=torch.float32, device=device),
        )

    def scaled_fp4_quant(input, global_scale, swizzled):
        qinput, sf = _quantize(input, global_scale)
        return qinput, swizzle_sf(sf).view(torch.int32).view(swizzled_sf_shape(*input.shape)) if swizzled else sf

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
        scaled_fp4_quant=scaled_fp4_quant,
        cutlass_scaled_fp4_mm=cutlass_scaled_fp4_mm,
    )


def _modelopt_quantize(weight):
    """ModelOpt's NVFP4 (`NVFP4QTensor.quantize`): `weight ~= fp4 * weight_scale * weight_scale_2`"""
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

    def test_kernel_missing_an_op_fails_at_load(self):
        kernel = _fake_kernel()
        hub_kernel = SimpleNamespace(
            PackedWeight=kernel.PackedWeight,
            gemm=kernel.gemm,
            pack=kernel.pack,
            _ops=SimpleNamespace(ops=SimpleNamespace(scaled_fp4_quant=kernel.scaled_fp4_quant)),
        )
        with (
            patch("transformers.integrations.nvfp4.is_kernels_available", return_value=True),
            patch("transformers.integrations.nvfp4.lazy_load_kernel", return_value=hub_kernel),
            self.assertRaisesRegex(ImportError, "cutlass_scaled_fp4_mm"),
        ):
            load_nvfp4_kernel.__wrapped__()

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
    def test_multi_device_map_only_for_pre_quantized_checkpoints(self, _, __):
        device_map = {"model.layers.0": 0, "model.layers.1": 1}
        # Quantizing on the fly packs every weight on one device
        with self.assertRaisesRegex(ValueError, "multi-device `device_map`"):
            self.quantizer.validate_environment(device_map=device_map)
        # A pre-quantized checkpoint spreads over several, each of them Blackwell
        quantizer = NVFP4HfQuantizer(NVFP4Config(), pre_quantized=True)
        capabilities = {0: (10, 0), 1: (9, 0)}
        with patch("torch.cuda.get_device_capability", lambda device: capabilities[device.index]):
            with self.assertRaisesRegex(RuntimeError, "sm90 on cuda:1"):
                quantizer.validate_environment(device_map=device_map)
            capabilities[1] = (10, 0)
            quantizer.validate_environment(device_map=device_map)

    @staticmethod
    def _parallel_model(**distributed):
        layer = nn.Module()
        layer.proj = NVFP4Linear(64, 32)
        model = nn.Module()
        model.layers = nn.ModuleList([layer])
        model.config = SimpleNamespace(distributed_config=SimpleNamespace(**distributed) if distributed else None)
        return model

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_rejects_tensor_parallelism(self, _):
        model = self._parallel_model(tp_size=2)
        with self.assertRaisesRegex(ValueError, "does not support tensor parallelism"):
            self.quantizer.update_parallel_plan(model, {})
        # Not distributed, or no tensor parallelism
        for distributed in ({}, {"tp_size": 1}):
            model = self._parallel_model(**distributed)
            self.quantizer.update_parallel_plan(model, {})

    @patch("transformers.quantizers.quantizer_nvfp4.is_kernels_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_rejects_cpu_offload(self, _, __):
        with self.assertRaisesRegex(ValueError, "CPU or disk offload"):
            self.quantizer.validate_environment(device_map={"model": "cuda", "lm_head": "cpu"})
