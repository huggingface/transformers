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
from unittest.mock import Mock, patch

from transformers import NVFP4Config
from transformers.integrations.mxfp4 import FP4_VALUES
from transformers.integrations.nvfp4 import (
    NVFP4Dequantize,
    NVFP4Deserialize,
    NVFP4Linear,
    NVFP4Quantize,
    swizzle_sf,
)
from transformers.quantizers.quantizer_nvfp4 import NVFP4HfQuantizer
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import QuantizationMethod


if is_torch_available():
    import torch
    from torch import nn


class _PackedWeight:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_kernel():
    def pack(weight, device):
        out_features, in_features = weight.shape
        return SimpleNamespace(
            qweight=torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
            sf=torch.empty(out_features, in_features // 64, dtype=torch.int32, device=device),
            sf_rowmajor=torch.empty(out_features, in_features // 16, dtype=torch.uint8, device=device),
            global_scale=torch.ones(1, dtype=torch.float32, device=device),
        )

    def scaled_fp4_quant(input, global_scale, swizzled):
        return input.new_zeros(input.shape[0], input.shape[1] // 2, dtype=torch.uint8), input.new_zeros(
            input.shape[0], input.shape[1] // 64, dtype=torch.int32
        )

    def cutlass_scaled_fp4_mm(qinput, weight, input_sf, weight_sf, alpha):
        return qinput.new_zeros(qinput.shape[0], weight.shape[0], dtype=torch.bfloat16)

    return SimpleNamespace(
        PackedWeight=_PackedWeight,
        gemm=lambda packed_weight, input: input.new_zeros((*input.shape[:-1], packed_weight.n)),
        pack=pack,
        swizzled_sf_shape=lambda rows, columns: (rows, columns // 64),
        ops=SimpleNamespace(scaled_fp4_quant=scaled_fp4_quant, cutlass_scaled_fp4_mm=cutlass_scaled_fp4_mm),
    )


def _modelopt_quantize(weight):
    """ModelOpt's NVFP4 (`NVFP4QTensor.quantize`): `weight ~= fp4 * weight_scale * weight_scale_2`."""
    weight_scale_2 = weight.abs().amax() / (6 * 448)
    blocks = weight.view(weight.shape[0], -1, 16)
    weight_scale = (blocks.abs().amax(-1) / (6 * weight_scale_2)).to(torch.float8_e4m3fn)
    scaled = blocks / (weight_scale.float() * weight_scale_2).unsqueeze(-1)
    codes = (scaled.unsqueeze(-1) - torch.tensor(FP4_VALUES)).abs().argmin(-1).flatten(1)  # nearest e2m1
    return (codes[:, 0::2] | codes[:, 1::2] << 4).to(torch.uint8), weight_scale, weight_scale_2


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
        self.assertEqual(model["convert"].weight_sf_rowmajor.dtype, torch.uint8)
        self.assertEqual(model["convert"].weight_global_scale.dtype, torch.float32)

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_linear_forward_uses_normal_function_entry_point(self, _):
        layer = NVFP4Linear(64, 32)
        output = layer(torch.randn(2, 3, 64))

        self.assertEqual(output.shape, (2, 3, 32))

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_packed_tensors_are_registered_in_state_dict(self, _):
        layer = NVFP4Linear(64, 32)

        buffer_names = {"weight", "weight_sf", "weight_sf_rowmajor", "weight_global_scale", "input_scale"}
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
                "model.proj.weight_sf_rowmajor",
                "model.proj.weight_global_scale",
                "model.proj.input_scale",
            },
        )
        self.assertEqual(output["model.proj.weight"].shape, (32, 32))

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel", side_effect=_fake_kernel)
    def test_modelopt_checkpoint_round_trip(self, _):
        weight = torch.randn(128, 64)
        packed, weight_scale, weight_scale_2 = _modelopt_quantize(weight)
        checkpoint = {
            "weight": packed,
            "weight_scale": weight_scale,
            "weight_scale_2": weight_scale_2,
            "input_scale": torch.tensor(0.01),
        }

        buffers = NVFP4Deserialize(None).convert(dict(checkpoint))
        layer = NVFP4Linear(64, 128)
        self.assertEqual(set(buffers), set(layer.state_dict()))
        for name in ("weight_sf", "weight_sf_rowmajor", "weight_global_scale"):  # `weight` is a bf16 placeholder
            self.assertEqual(
                (name, buffers[name].dtype, buffers[name].shape),
                (name, getattr(layer, name).dtype, getattr(layer, name).shape),
            )
        torch.testing.assert_close(buffers["weight_sf_rowmajor"], weight_scale.view(torch.uint8))
        torch.testing.assert_close(
            buffers["weight_sf"].view(-1), swizzle_sf(weight_scale.view(torch.uint8)).view(torch.int32)
        )
        torch.testing.assert_close(buffers["weight_global_scale"], (1 / weight_scale_2).reshape(1))
        torch.testing.assert_close(buffers["input_scale"], torch.tensor([0.01]))

        model = SimpleNamespace(get_parameter_or_buffer=lambda name: torch.empty(0))
        dense = NVFP4Dequantize(None).convert(checkpoint, model=model, full_layer_name="proj.weight")["weight"][0]
        # Within half the widest e2m1 step (4 -> 6) of each block
        bound = (weight_scale.float() * weight_scale_2).unsqueeze(-1)
        self.assertTrue(((dense - weight).abs().view(128, -1, 16) <= bound + 1e-6).all())
        # The transformers layout of the same scales dequantizes to the same weight
        ours = {key: buffers[key] for key in ("weight", "weight_sf_rowmajor", "weight_global_scale")}
        torch.testing.assert_close(
            NVFP4Dequantize(None).convert(ours, model=model, full_layer_name="proj.weight")["weight"][0], dense
        )

    @patch("transformers.integrations.nvfp4.load_nvfp4_kernel")
    def test_static_input_scale_replaces_the_dynamic_one(self, load_kernel):
        kernel = _fake_kernel()
        kernel.ops = SimpleNamespace(
            scaled_fp4_quant=Mock(wraps=kernel.ops.scaled_fp4_quant),
            cutlass_scaled_fp4_mm=Mock(wraps=kernel.ops.cutlass_scaled_fp4_mm),
        )
        kernel.gemm = Mock(wraps=kernel.gemm)
        load_kernel.return_value = kernel
        layer = NVFP4Linear(64, 32)
        layer.weight_global_scale = torch.tensor([2.0])

        layer(torch.randn(3, 64))  # no static scale: the kernel's dynamic GEMM
        self.assertEqual((kernel.gemm.call_count, kernel.ops.scaled_fp4_quant.call_count), (1, 0))

        layer.input_scale = torch.tensor([0.5])
        self.assertEqual(layer(torch.randn(2, 3, 64)).shape, (2, 3, 32))
        torch.testing.assert_close(kernel.ops.scaled_fp4_quant.call_args.args[1], torch.tensor([2.0]))
        torch.testing.assert_close(kernel.ops.cutlass_scaled_fp4_mm.call_args.args[4], torch.tensor([0.25]))
        layer(torch.randn(1, 64))  # decode keeps the W4A16 GEMV
        self.assertEqual((kernel.gemm.call_count, kernel.ops.scaled_fp4_quant.call_count), (2, 1))


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
