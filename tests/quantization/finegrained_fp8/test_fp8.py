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
"""The deprecated `FineGrainedFP8Config`: passed explicitly, it loads through its own frozen integration; a checkpoint
declaring `quant_method="fp8"` loads as [`Fp8Config`]. The formats it spelled are tested in `fp8`."""

import tempfile
import unittest
import warnings
from unittest import mock

from transformers import PreTrainedConfig
from transformers.testing_utils import (
    get_device_properties,
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    torch_device,
)
from transformers.utils import is_torch_available
from transformers.utils.quantization_config import FineGrainedFP8Config, Fp8Config


if is_torch_available():
    import torch


def _legacy(**options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return FineGrainedFP8Config(**options)


def _quantizer(passed, checkpoint=None):
    """Return the quantizer `from_pretrained` builds for a passed config and an optional checkpoint config."""
    from transformers.quantizers.auto import get_hf_quantizer

    model_config = PreTrainedConfig()
    if checkpoint is not None:
        model_config.quantization_config = dict(checkpoint)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        quantizer, _, _ = get_hf_quantizer(model_config, passed, None, True, {})
    return quantizer


@require_torch
@require_accelerate
class FineGrainedFP8LegacyTest(unittest.TestCase):
    def test_building_one_is_deprecated(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FineGrainedFP8Config()
        self.assertTrue(any(issubclass(w.category, FutureWarning) for w in caught))

    def test_a_passed_deprecated_config_loads_through_its_frozen_integration(self):
        """Check both on the fly and over an `fp8` checkpoint. The checkpoint's config reads as the passed class."""
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

        for checkpoint in (None, {"quant_method": "fp8", "weight_block_size": [128, 128]}):
            with self.subTest(pre_quantized=checkpoint is not None):
                quantizer = _quantizer(_legacy(dequantize=True), checkpoint)
                self.assertIsInstance(quantizer, FineGrainedFP8HfQuantizer)
                self.assertIsInstance(quantizer.quantization_config, FineGrainedFP8Config)
                self.assertTrue(quantizer.quantization_config.dequantize)

    def test_megamoe_takes_over_the_experts_ep_rule(self):
        from transformers import DeepseekV4Config
        from transformers.distributed import DistributedConfig
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
        from transformers.quantizers.quantizer_fp8 import Fp8HfQuantizer

        # Cover the legacy integration and the default block-FP8 integration.
        for quantizer in (FineGrainedFP8HfQuantizer(_legacy()), Fp8HfQuantizer(Fp8Config())):
            with self.subTest(quantizer=type(quantizer).__name__):
                model = torch.nn.Module()
                model.config = DeepseekV4Config()
                model.config._experts_implementation = "deepgemm_megamoe"
                plan = dict(model.config.base_model_ep_plan)
                quantizer.update_parallel_plan(model, plan)
                self.assertEqual(plan["layers.*.mlp.experts"], "megamoe_experts")
                with self.assertRaisesRegex(ValueError, "Mega MoE expert parallelism requires `ep_size=tp_size`"):
                    DistributedConfig(tp_size=1, fsdp_size=8, ep_size=8)._validate_resolved_ep_plan(plan)

    @mock.patch(
        "transformers.quantizers.quantizer_finegrained_fp8.FineGrainedFP8HfQuantizer.validate_environment",
        lambda *args, **kwargs: None,
    )
    def test_float32_scale_container_loads_as_ue8m0(self):
        from safetensors.torch import load_file, save_file

        from transformers import MixtralConfig, MixtralForCausalLM

        if not hasattr(torch, "float8_e8m0fnu"):
            self.skipTest("needs torch.float8_e8m0fnu")
        config = MixtralConfig(
            vocab_size=64, hidden_size=128, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
        )  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            MixtralForCausalLM(config).save_pretrained(f"{tmp}/bf16")
            quantized = MixtralForCausalLM.from_pretrained(
                f"{tmp}/bf16", quantization_config=_legacy(scale_fmt="ue8m0"), device_map="cpu"
            )
            expected = {
                name: tensor.float() for name, tensor in quantized.state_dict().items() if name.endswith("scale_inv")
            }
            quantized.save_pretrained(f"{tmp}/fp8")
            path = f"{tmp}/fp8/model.safetensors"
            state_dict = {
                name.replace("proj.weight_scale_inv", "proj.scale"): tensor.float()
                if name.endswith("scale_inv")
                else tensor
                for name, tensor in load_file(path).items()
            }
            save_file(state_dict, path, metadata={"format": "pt"})
            loaded = MixtralForCausalLM.from_pretrained(
                f"{tmp}/fp8", quantization_config=_legacy(scale_fmt="ue8m0"), device_map="cpu"
            )
        scales = {name: tensor for name, tensor in loaded.state_dict().items() if name.endswith("scale_inv")}
        self.assertEqual(scales.keys(), expected.keys())
        for name, scale in scales.items():
            self.assertEqual(scale.dtype, torch.float8_e8m0fnu, name)
            torch.testing.assert_close(scale.float(), expected[name], rtol=0, atol=0)

    @mock.patch(
        "transformers.quantizers.quantizer_finegrained_fp8.FineGrainedFP8HfQuantizer.validate_environment",
        lambda *args, **kwargs: None,
    )
    def test_a_dequantized_load_saves_as_an_unquantized_model(self):
        """Check that a dequantized load saves the unquantized keys. Mixtral's fused-experts converter must reverse."""
        from safetensors.torch import load_file

        from transformers import MixtralConfig, MixtralForCausalLM

        config = MixtralConfig(
            vocab_size=64, hidden_size=128, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
        )  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            MixtralForCausalLM(config).save_pretrained(f"{tmp}/bf16")
            unquantized_keys = set(load_file(f"{tmp}/bf16/model.safetensors"))
            MixtralForCausalLM.from_pretrained(
                f"{tmp}/bf16", quantization_config=_legacy(), device_map="cpu"
            ).save_pretrained(f"{tmp}/fp8")

            dequantized = MixtralForCausalLM.from_pretrained(
                f"{tmp}/fp8", quantization_config=_legacy(dequantize=True), device_map="cpu"
            )
            dequantized.save_pretrained(f"{tmp}/dequantized")
            self.assertEqual(set(load_file(f"{tmp}/dequantized/model.safetensors")), unquantized_keys)

            reloaded = MixtralForCausalLM.from_pretrained(f"{tmp}/dequantized", device_map="cpu")
        expected = dequantized.state_dict()
        for name, tensor in reloaded.state_dict().items():
            torch.testing.assert_close(tensor.float(), expected[name].float(), rtol=0, atol=0, msg=name)


@require_torch_accelerator
@unittest.skipIf(
    get_device_properties()[0] == "cuda"
    and (get_device_properties()[1] < 8 or (get_device_properties()[1] == 8 and get_device_properties()[2] < 9)),
    "Skipping FP8LinearTest because it is not supported on GPU with capability < 8.9",
)
class FP8LinearTest(unittest.TestCase):
    def test_linear_preserves_shape(self):
        from transformers.integrations import FP8Linear

        linear = FP8Linear(256, 256, block_size=(128, 128)).to(torch_device)
        x = torch.rand((1, 5, 256)).to(torch_device)
        self.assertEqual(linear(x).shape, x.shape)

    def test_linear_with_diff_feature_size_preserves_shape(self):
        from transformers.integrations import FP8Linear

        linear = FP8Linear(128, 256, block_size=(128, 128)).to(torch_device)
        x = torch.rand((1, 5, 128)).to(torch_device)
        self.assertEqual(linear(x).shape, (1, 5, 256))
