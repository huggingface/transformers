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
from transformers.integrations.nvfp4 import NVFP4Linear, NVFP4Quantize, replace_with_nvfp4_linear
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

    return SimpleNamespace(
        PackedWeight=_PackedWeight,
        gemm=lambda packed_weight, input: input.new_zeros((*input.shape[:-1], packed_weight.n)),
        pack=pack,
        swizzled_sf_shape=lambda rows, columns: (rows, columns // 64),
    )


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

        replace_with_nvfp4_linear(model, modules_to_not_convert=["skip"])

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

        for buffer_name in ("weight", "weight_sf", "weight_sf_rowmajor", "weight_global_scale"):
            self.assertIsInstance(getattr(layer, buffer_name), nn.Buffer)
        self.assertEqual(
            set(layer.state_dict()),
            {"weight", "weight_sf", "weight_sf_rowmajor", "weight_global_scale"},
        )
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
            },
        )
        self.assertEqual(output["model.proj.weight"].shape, (32, 32))


@unittest.skipUnless(is_torch_available(), "test requires PyTorch")
class NVFP4QuantizerValidationTest(unittest.TestCase):
    def setUp(self):
        self.quantizer = NVFP4HfQuantizer(NVFP4Config(), pre_quantized=False)

    def test_rejects_pre_quantized_checkpoints(self):
        quantizer = NVFP4HfQuantizer(NVFP4Config(), pre_quantized=True)

        with self.assertRaisesRegex(ValueError, "pre-quantized NVFP4"):
            quantizer.validate_environment(device_map={"": "cuda"})

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
        config = SimpleNamespace(distributed_config=SimpleNamespace(tp_size=2))

        with self.assertRaisesRegex(ValueError, "scale metadata does not have a sharding plan"):
            self.quantizer.update_tp_plan(config)

    def test_update_tp_plan_keeps_non_tp_configs(self):
        configs = (
            SimpleNamespace(),
            SimpleNamespace(distributed_config=SimpleNamespace(tp_size=1)),
        )

        for config in configs:
            with self.subTest(config=config):
                self.assertIs(self.quantizer.update_tp_plan(config), config)

    @patch("transformers.quantizers.quantizer_nvfp4.is_kernels_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_rejects_cpu_offload(self, _, __):
        with self.assertRaisesRegex(ValueError, "CPU or disk offload"):
            self.quantizer.validate_environment(device_map={"model": "cuda", "lm_head": "cpu"})
