# Copyright 2024 HuggingFace Inc.
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


import logging as stdlib_logging
import math
import unittest

from parameterized import parameterized

from transformers import Gemma3TextConfig, LlamaConfig
from transformers.testing_utils import is_torch_available, require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import ROPE_INIT_FUNCTIONS
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


@require_torch
class RopeTest(unittest.TestCase):
    def get_config_with_rope_parameters(
        self, rope_params: dict, is_nested: bool = False, same_rope_per_layer: bool = False
    ):
        if same_rope_per_layer and not is_nested:
            raise ValueError(
                "Cannot use same RoPE params per layer when the config doesn't have layer types. Set `is_nested=True`."
            )

        if is_nested:
            config = Gemma3TextConfig()
            config.layer_types = ["full_attention", "sliding_attention"]
            config.rope_parameters = {
                "full_attention": rope_params,
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
                if not same_rope_per_layer
                else rope_params,
            }
        else:
            config = LlamaConfig()
            config.layer_types = None
            config.rope_parameters = rope_params
        return config

    @parameterized.expand(
        [
            (True, True),
            (True, False),
            (False, False),
        ]
    )
    def test_rope_validation(self, is_nested: bool, same_rope_per_layer: bool):
        all_rope_types = ROPE_INIT_FUNCTIONS.keys()

        # The base config is always valid (default RoPE)
        config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0},
            is_nested=is_nested,
            same_rope_per_layer=same_rope_per_layer,
        )
        config.validate_rope()

        # If we explicitly set the other (non-default) RoPE types with only rope_theta,
        # validation should fail because required keys are missing (e.g. factor, short_factor)
        for rope_type in all_rope_types:
            # "default.proportional" is always valid with just rope_theta in keys
            if rope_type in ["default", "proportional"]:
                continue
            config = self.get_config_with_rope_parameters(
                rope_params={"rope_type": rope_type, "rope_theta": 10000.0},
                is_nested=is_nested,
                same_rope_per_layer=same_rope_per_layer,
            )
            with self.assertRaises(KeyError):
                config.validate_rope()

        # Parameters are exclusive to their own RoPE type, and should raise an exception if incorrectly passed
        valid_param_mapping = {
            "factor": ["linear", "dynamic", "yarn", "longrope"],
            "attention_factor": ["yarn", "longrope"],
            "beta_fast": ["yarn"],
            "beta_slow": ["yarn"],
            "short_factor": ["longrope"],
            "long_factor": ["longrope"],
        }
        for rope_type in all_rope_types:
            # "default.proportional" is always valid with just rope_theta in keys
            if rope_type in ["default", "proportional"]:
                continue
            for param, valid_rope_types in valid_param_mapping.items():
                # Set `param` with a dummy value -- we want to test the dict key
                config = self.get_config_with_rope_parameters(
                    rope_params={"rope_type": rope_type, "rope_theta": 10000.0, param: True},
                    is_nested=is_nested,
                    same_rope_per_layer=same_rope_per_layer,
                )
                if rope_type in valid_rope_types:
                    continue
                else:
                    with self.assertRaises(KeyError):
                        config.validate_rope()

        # Any other parameters passed to RoPE will raise a warning that a particular key is not used
        # But sometimes we can have model-specific RoPE kwargs and bypass warning with `ignore_keys`
        config.ignore_keys_at_rope_validation = {"mrope_sections"}  # e,g in Qwen2-VL
        config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0, "mrope_sections": True},
            is_nested=is_nested,
            same_rope_per_layer=same_rope_per_layer,
        )
        config.validate_rope()

        with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
            config.ignore_keys_at_rope_validation = set()
            config.validate_rope()
            self.assertEqual(len(logs.output), 1 if not same_rope_per_layer else 2)
            self.assertIn("mrope_sections", logs.output[0])
            # Raise a warning for each existing layer-type
            if same_rope_per_layer:
                self.assertIn("mrope_sections", logs.output[1])

    def test_rope_bad_layer_types(self):
        config = LlamaConfig()

        config.layer_types = ["full", "sliding"]
        config.rope_parameters = {"rope_type": "default", "rope_theta": 10000}
        # Never raises an error, prob for BC?
        # with self.assertRaises(ValueError):
        #     config.validate_rope()

    @parameterized.expand(
        [
            (True, True),
            (True, False),
            (False, False),
        ]
    )
    def test_yarn_original_original_max_position_embeddings_validation(
        self, is_nested: bool, same_rope_per_layer: bool
    ):
        """Tests that models with no/bad `original_max_position_embeddings` raise a warning"""
        # good rope config: has a factor AND original_max_position_embeddings -> no warnings
        config = Gemma3TextConfig() if is_nested else LlamaConfig()
        rope_params = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2.0,
            "original_max_position_embeddings": int(config.max_position_embeddings / 2.0),
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=is_nested,
            same_rope_per_layer=same_rope_per_layer,
        )
        with self.assertRaises(AssertionError):  # confirm that no warnings are thrown
            with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
                config.validate_rope()

        # bad rope config, no `original_max_position_embeddings` -> raise error
        rope_params = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2.0,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=is_nested,
            same_rope_per_layer=same_rope_per_layer,
        )
        with self.assertRaises(KeyError):
            config.validate_rope()

        # bad rope config, bad implicit factor -> warning
        rope_params = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2.0,
            "original_max_position_embeddings": 1,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=is_nested,
            same_rope_per_layer=same_rope_per_layer,
        )
        stdlib_logging.Logger.warning_once.cache_clear()
        with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
            config.validate_rope()
            self.assertGreaterEqual(len(logs.output), 1)
            self.assertIn("implicit factor", logs.output[0])

    @parameterized.expand(
        [
            (True, True),
            (True, False),
            (False, False),
        ]
    )
    def test_convert_rope_params_to_dict_with_list_ignore_keys(self, is_nested: bool, same_rope_per_layer: bool):
        # Regression test for #46121: `ignore_keys_at_rope_validation` becomes a list when loaded from a config.json
        # (JSON has no set type). `convert_rope_params_to_dict` used to do `list | set` and crash with
        # TypeError when `partial_rotary_factor` was also set.
        config = LlamaConfig(partial_rotary_factor=0.25)
        config.ignore_keys_at_rope_validation = ["mrope_section", "mrope_interleaved"]

        config.convert_rope_params_to_dict(partial_rotary_factor=0.25)

        self.assertIsInstance(config.ignore_keys_at_rope_validation, set)
        self.assertEqual(
            config.ignore_keys_at_rope_validation,
            {"mrope_section", "mrope_interleaved", "partial_rotary_factor"},
        )

        # Round-trip through from_dict to mimic the JSON-deserialized path that triggered this in production.
        cfg_dict = config.to_dict()
        cfg_dict["ignore_keys_at_rope_validation"] = ["mrope_section", "mrope_interleaved"]
        reloaded = LlamaConfig.from_dict(cfg_dict)
        reloaded.convert_rope_params_to_dict(partial_rotary_factor=0.25)
        self.assertIsInstance(reloaded.ignore_keys_at_rope_validation, set)

        # Also accept None (the class-level attribute can be cleared on an instance).
        config_none = LlamaConfig(partial_rotary_factor=0.25)
        config_none.ignore_keys_at_rope_validation = None
        config_none.convert_rope_params_to_dict(partial_rotary_factor=0.25)
        self.assertEqual(config_none.ignore_keys_at_rope_validation, {"partial_rotary_factor"})

    def test_default_rope_numerically(self):
        # Note: some RoPE scaling methods start off by calling the default RoPE frequencies. If this test fails, then
        # multiple RoPE strategies will fail.
        # fmt: off
        EXPECTED_INV_FREQ = torch.tensor(
            [
                1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
                4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
                1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 1.0000e-01, 8.6596e-02,
                7.4989e-02, 6.4938e-02, 5.6234e-02, 4.8697e-02, 4.2170e-02, 3.6517e-02,
                3.1623e-02, 2.7384e-02, 2.3714e-02, 2.0535e-02, 1.7783e-02, 1.5399e-02,
                1.3335e-02, 1.1548e-02, 1.0000e-02, 8.6596e-03, 7.4989e-03, 6.4938e-03,
                5.6234e-03, 4.8697e-03, 4.2170e-03, 3.6517e-03, 3.1623e-03, 2.7384e-03,
                2.3714e-03, 2.0535e-03, 1.7783e-03, 1.5399e-03, 1.3335e-03, 1.1548e-03,
                1.0000e-03, 8.6596e-04, 7.4989e-04, 6.4938e-04, 5.6234e-04, 4.8697e-04,
                4.2170e-04, 3.6517e-04, 3.1623e-04, 2.7384e-04, 2.3714e-04, 2.0535e-04,
                1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04
            ], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        config = LlamaConfig()
        config.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}

        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)

        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        inv_freq, attention_scale = LlamaRotaryEmbedding.compute_default_rope_parameters(
            config=config, device=torch_device
        )
        rope_module = LlamaRotaryEmbedding(config, device=torch_device)

        self.assertTrue(hasattr(rope_module, "inv_freq"))
        self.assertTrue(hasattr(rope_module, "attention_scaling"))
        self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for default RoPE
        self.assertEqual(rope_module.attention_scaling, attention_scale)
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)
        torch.testing.assert_close(rope_module.inv_freq, inv_freq)

    @parameterized.expand([True, False])
    def test_default_rope_numerically_nested(self, same_rope_per_layer: bool):
        # Note: some RoPE scaling methods start off by calling the default RoPE frequencies. If this test fails, then
        # multiple RoPE strategies will fail.
        # fmt: off
        EXPECTED_INV_FREQ = torch.tensor([
            1.0000e+00, 9.3057e-01, 8.6596e-01, 8.0584e-01, 7.4989e-01, 6.9783e-01,
            6.4938e-01, 6.0430e-01, 5.6234e-01, 5.2330e-01, 4.8697e-01, 4.5316e-01,
            4.2170e-01, 3.9242e-01, 3.6517e-01, 3.3982e-01, 3.1623e-01, 2.9427e-01,
            2.7384e-01, 2.5483e-01, 2.3714e-01, 2.2067e-01, 2.0535e-01, 1.9110e-01,
            1.7783e-01, 1.6548e-01, 1.5399e-01, 1.4330e-01, 1.3335e-01, 1.2409e-01,
            1.1548e-01, 1.0746e-01, 1.0000e-01, 9.3057e-02, 8.6596e-02, 8.0584e-02,
            7.4989e-02, 6.9783e-02, 6.4938e-02, 6.0430e-02, 5.6234e-02, 5.2330e-02,
            4.8697e-02, 4.5316e-02, 4.2170e-02, 3.9242e-02, 3.6517e-02, 3.3982e-02,
            3.1623e-02, 2.9427e-02, 2.7384e-02, 2.5483e-02, 2.3714e-02, 2.2067e-02,
            2.0535e-02, 1.9110e-02, 1.7783e-02, 1.6548e-02, 1.5399e-02, 1.4330e-02,
            1.3335e-02, 1.2409e-02, 1.1548e-02, 1.0746e-02, 1.0000e-02, 9.3057e-03,
            8.6596e-03, 8.0584e-03, 7.4989e-03, 6.9783e-03, 6.4938e-03, 6.0430e-03,
            5.6234e-03, 5.2330e-03, 4.8697e-03, 4.5316e-03, 4.2170e-03, 3.9242e-03,
            3.6517e-03, 3.3982e-03, 3.1623e-03, 2.9427e-03, 2.7384e-03, 2.5483e-03,
            2.3714e-03, 2.2067e-03, 2.0535e-03, 1.9110e-03, 1.7783e-03, 1.6548e-03,
            1.5399e-03, 1.4330e-03, 1.3335e-03, 1.2409e-03, 1.1548e-03, 1.0746e-03,
            1.0000e-03, 9.3057e-04, 8.6596e-04, 8.0584e-04, 7.4989e-04, 6.9783e-04,
            6.4938e-04, 6.0430e-04, 5.6234e-04, 5.2330e-04, 4.8697e-04, 4.5316e-04,
            4.2170e-04, 3.9242e-04, 3.6517e-04, 3.3982e-04, 3.1623e-04, 2.9427e-04,
            2.7384e-04, 2.5483e-04, 2.3714e-04, 2.2067e-04, 2.0535e-04, 1.9110e-04,
            1.7783e-04, 1.6548e-04, 1.5399e-04, 1.4330e-04, 1.3335e-04, 1.2409e-04,
            1.1548e-04, 1.0746e-04], device=torch_device)
        # fmt: on

        # input sanity checks: if these change, the output will also change
        config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0},
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )

        self.assertEqual(config.hidden_size, 2304)
        self.assertEqual(config.num_attention_heads, 8)
        for layer_type in config.layer_types:
            self.assertFalse(hasattr(config.rope_parameters[layer_type], "partial_rotary_factor"))
            self.assertEqual(config.rope_parameters[layer_type], {"rope_type": "default", "rope_theta": 10000.0})

        rope_module = Gemma3RotaryEmbedding(config, device=torch_device)
        for layer_type in config.layer_types:
            inv_freq, attention_scale = Gemma3RotaryEmbedding.compute_default_rope_parameters(
                config=config, layer_type=layer_type, device=torch_device
            )

            self.assertTrue(hasattr(rope_module, f"{layer_type}_inv_freq"))
            self.assertTrue(hasattr(rope_module, f"{layer_type}_attention_scaling"))
            self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for default RoPE
            self.assertEqual(getattr(rope_module, f"{layer_type}_attention_scaling"), attention_scale)
            torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)
            torch.testing.assert_close(getattr(rope_module, f"{layer_type}_inv_freq"), inv_freq)

    def test_linear_rope_numerically(self):
        # This is a linear scaling strategy, the **frequencies** are scaled linearly with respect to the default
        # frequencies (= the inverse frequencies are scaled **inversely**)
        config = LlamaConfig()
        default_inv_freq, _ = LlamaRotaryEmbedding.compute_default_rope_parameters(config=config, device=torch_device)

        rope_fn = ROPE_INIT_FUNCTIONS["linear"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {"rope_type": "linear", "rope_theta": 10000.0, "factor": factor}
            inv_freq, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for linear RoPE
            torch.testing.assert_close(inv_freq, default_inv_freq / factor)

            rope_module = LlamaRotaryEmbedding(config, device=torch_device)
            self.assertEqual(rope_module.attention_scaling, attention_scale)
            torch.testing.assert_close(rope_module.inv_freq, inv_freq)

    @parameterized.expand([True, False])
    def test_linear_rope_numerically_nested(self, same_rope_per_layer: bool):
        # This is a linear scaling strategy, the **frequencies** are scaled linearly with respect to the default
        # frequencies (= the inverse frequencies are scaled **inversely**)
        default_config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0},
            is_nested=True,
            same_rope_per_layer=True,
        )

        self.assertEqual(default_config.hidden_size, 2304)
        self.assertEqual(default_config.num_attention_heads, 8)
        expected_defaults = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        for layer_type in set(default_config.layer_types):
            self.assertFalse(hasattr(default_config.rope_parameters[layer_type], "partial_rotary_factor"))
            self.assertEqual(default_config.rope_parameters[layer_type], expected_defaults[layer_type])

        default_inv_freq, _ = Gemma3RotaryEmbedding.compute_default_rope_parameters(
            config=default_config, layer_type="full_attention", device=torch_device
        )
        layer_types = default_config.layer_types if same_rope_per_layer else ["full_attention"]
        rope_fn = ROPE_INIT_FUNCTIONS["linear"]
        for factor in (2.0, 10.0, 20.0):
            config = self.get_config_with_rope_parameters(
                rope_params={"rope_type": "linear", "rope_theta": 10000.0, "factor": factor},
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                inv_freq, attention_scale = rope_fn(config=config, layer_type=layer_type, device=torch_device)
                self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for linear RoPE
                torch.testing.assert_close(inv_freq, default_inv_freq / factor)

                rope_module = Gemma3RotaryEmbedding(config, device=torch_device)
                self.assertEqual(getattr(rope_module, f"{layer_type}_attention_scaling"), attention_scale)
                torch.testing.assert_close(getattr(rope_module, f"{layer_type}_inv_freq"), inv_freq)

    def test_dynamic_rope_numerically(self):
        # fmt: off
        EXPECTED_INV_FREQ = torch.tensor(
            [
                1.0000e+00, 8.0931e-01, 6.5498e-01, 5.3008e-01, 4.2900e-01, 3.4720e-01,
                2.8099e-01, 2.2741e-01, 1.8404e-01, 1.4895e-01, 1.2055e-01, 9.7558e-02,
                7.8955e-02, 6.3899e-02, 5.1714e-02, 4.1853e-02, 3.3872e-02, 2.7413e-02,
                2.2185e-02, 1.7955e-02, 1.4531e-02, 1.1760e-02, 9.5176e-03, 7.7027e-03,
                6.2339e-03, 5.0451e-03, 4.0831e-03, 3.3045e-03, 2.6744e-03, 2.1644e-03,
                1.7517e-03, 1.4176e-03, 1.1473e-03, 9.2852e-04, 7.5146e-04, 6.0817e-04,
                4.9220e-04, 3.9834e-04, 3.2238e-04, 2.6091e-04, 2.1115e-04, 1.7089e-04,
                1.3830e-04, 1.1193e-04, 9.0585e-05, 7.3312e-05, 5.9332e-05, 4.8018e-05,
                3.8861e-05, 3.1451e-05, 2.5453e-05, 2.0600e-05, 1.6672e-05, 1.3492e-05,
                1.0920e-05, 8.8374e-06, 7.1522e-06, 5.7883e-06, 4.6845e-06, 3.7912e-06,
                3.0683e-06, 2.4832e-06, 2.0097e-06, 1.6265e-06
            ], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        config = LlamaConfig()
        self.assertEqual(config.rope_parameters, {"rope_type": "default", "rope_theta": 10000.0})
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = rope_fn(config=config, device=torch_device)

        # Check 1: this is a dynamic scaling strategy, it will not scale unless we provide `seq_len` larger than the
        # model's original training sequence length
        rope_fn = ROPE_INIT_FUNCTIONS["dynamic"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": factor}
            inv_freq, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for dynamic RoPE
            torch.testing.assert_close(inv_freq, default_inv_freq)

            inv_freq, _ = rope_fn(config=config, device=torch_device, seq_len=1)
            torch.testing.assert_close(inv_freq, default_inv_freq)

            inv_freq, _ = rope_fn(config=config, device=torch_device, seq_len=torch.tensor(1, dtype=torch.int64))
            torch.testing.assert_close(inv_freq, default_inv_freq)

        # Check 2: if we provide `seq_len` larger than the model's original training sequence length, the frequencies
        # will scale up (i.e., the inverse frequencies will scale down).
        factor = 10.0
        config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": factor}
        inv_freq, _ = rope_fn(config=config, device=torch_device, seq_len=16384)
        with self.assertRaises(AssertionError):  # It is NOT a linear factor
            torch.testing.assert_close(inv_freq, default_inv_freq / factor)
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    @parameterized.expand([True, False])
    def test_dynamic_rope_numerically_nested(self, same_rope_per_layer: bool):
        # fmt: off
        # Gemma3: head_dim=256, rope_theta=10000.0, factor=10.0, seq_len=200000 (> max_position_embeddings=131072)
        EXPECTED_INV_FREQ = torch.tensor(
            [
                1.0000e+00, 9.1723e-01, 8.4131e-01, 7.7168e-01, 7.0781e-01, 6.4922e-01,
                5.9548e-01, 5.4620e-01, 5.0099e-01, 4.5952e-01, 4.2149e-01, 3.8660e-01,
                3.5460e-01, 3.2525e-01, 2.9833e-01, 2.7364e-01, 2.5099e-01, 2.3021e-01,
                2.1116e-01, 1.9368e-01, 1.7765e-01, 1.6295e-01, 1.4946e-01, 1.3709e-01,
                1.2574e-01, 1.1533e-01, 1.0579e-01, 9.7033e-02, 8.9001e-02, 8.1635e-02,
                7.4878e-02, 6.8680e-02, 6.2996e-02, 5.7781e-02, 5.2999e-02, 4.8612e-02,
                4.4589e-02, 4.0898e-02, 3.7513e-02, 3.4408e-02, 3.1560e-02, 2.8948e-02,
                2.6552e-02, 2.4354e-02, 2.2338e-02, 2.0489e-02, 1.8793e-02, 1.7238e-02,
                1.5811e-02, 1.4502e-02, 1.3302e-02, 1.2201e-02, 1.1191e-02, 1.0265e-02,
                9.4153e-03, 8.6360e-03, 7.9212e-03, 7.2656e-03, 6.6642e-03, 6.1126e-03,
                5.6067e-03, 5.1426e-03, 4.7170e-03, 4.3265e-03, 3.9684e-03, 3.6400e-03,
                3.3387e-03, 3.0623e-03, 2.8089e-03, 2.5764e-03, 2.3631e-03, 2.1675e-03,
                1.9881e-03, 1.8236e-03, 1.6726e-03, 1.5342e-03, 1.4072e-03, 1.2907e-03,
                1.1839e-03, 1.0859e-03, 9.9603e-04, 9.1359e-04, 8.3797e-04, 7.6862e-04,
                7.0500e-04, 6.4665e-04, 5.9312e-04, 5.4403e-04, 4.9900e-04, 4.5770e-04,
                4.1982e-04, 3.8507e-04, 3.5320e-04, 3.2396e-04, 2.9715e-04, 2.7255e-04,
                2.4999e-04, 2.2930e-04, 2.1032e-04, 1.9291e-04, 1.7695e-04, 1.6230e-04,
                1.4887e-04, 1.3655e-04, 1.2524e-04, 1.1488e-04, 1.0537e-04, 9.6648e-05,
                8.8648e-05, 8.1311e-05, 7.4581e-05, 6.8408e-05, 6.2746e-05, 5.7552e-05,
                5.2789e-05, 4.8419e-05, 4.4412e-05, 4.0736e-05, 3.7364e-05, 3.4271e-05,
                3.1435e-05, 2.8833e-05, 2.6446e-05, 2.4257e-05, 2.2250e-05, 2.0408e-05,
                1.8719e-05, 1.7170e-05
            ], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        default_config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0},
            is_nested=True,
            same_rope_per_layer=True,
        )
        self.assertEqual(default_config.hidden_size, 2304)
        self.assertEqual(default_config.num_attention_heads, 8)
        expected_defaults = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        for layer_type in set(default_config.layer_types):
            self.assertFalse(hasattr(default_config.rope_parameters[layer_type], "partial_rotary_factor"))
            self.assertEqual(default_config.rope_parameters[layer_type], expected_defaults[layer_type])

        default_inv_freq, _ = Gemma3RotaryEmbedding.compute_default_rope_parameters(
            config=default_config, layer_type="full_attention", device=torch_device
        )
        layer_types = default_config.layer_types if same_rope_per_layer else ["full_attention"]

        # Check 1: this is a dynamic scaling strategy, it will not scale unless we provide `seq_len` larger than the
        # model's original training sequence length
        rope_fn = ROPE_INIT_FUNCTIONS["dynamic"]
        for factor in (2.0, 10.0, 20.0):
            config = self.get_config_with_rope_parameters(
                rope_params={"rope_type": "dynamic", "rope_theta": 10000.0, "factor": factor},
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                inv_freq, attention_scale = rope_fn(config=config, layer_type=layer_type, device=torch_device)
                self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for dynamic RoPE
                torch.testing.assert_close(inv_freq, default_inv_freq)

                inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device, seq_len=1)
                torch.testing.assert_close(inv_freq, default_inv_freq)

                inv_freq, _ = rope_fn(
                    config=config,
                    layer_type=layer_type,
                    device=torch_device,
                    seq_len=torch.tensor(1, dtype=torch.int64),
                )
                torch.testing.assert_close(inv_freq, default_inv_freq)

        # Check 2: if we provide `seq_len` larger than the model's original training sequence length, the frequencies
        # will scale up (i.e., the inverse frequencies will scale down).
        # Use seq_len=200000 > max_position_embeddings=131072 to trigger dynamic scaling for Gemma3
        factor = 10.0
        config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "dynamic", "rope_theta": 10000.0, "factor": factor},
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device, seq_len=200000)
            with self.assertRaises(AssertionError):  # It is NOT a linear factor
                torch.testing.assert_close(inv_freq, default_inv_freq / factor)
            torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    def test_dynamic_rope_resets_after_long_sequence(self):
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
            rope_parameters={"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 4.0},
        )
        rotary_embedding = LlamaRotaryEmbedding(config)
        original_inv_freq = rotary_embedding.original_inv_freq.clone()

        long_position_ids = torch.arange(32).unsqueeze(0)
        long_input = torch.zeros(1, 32, 2, 8)
        rotary_embedding(long_input, long_position_ids)

        self.assertEqual(int(rotary_embedding.max_seq_len_cached), 32)
        self.assertFalse(hasattr(rotary_embedding, "None_max_seq_len_cached"))
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(rotary_embedding.inv_freq, original_inv_freq)

        short_position_ids = torch.arange(4).unsqueeze(0)
        short_input = torch.zeros(1, 4, 2, 8)
        rotary_embedding(short_input, short_position_ids)

        self.assertEqual(int(rotary_embedding.max_seq_len_cached), config.max_position_embeddings)
        self.assertFalse(hasattr(rotary_embedding, "None_max_seq_len_cached"))
        torch.testing.assert_close(rotary_embedding.inv_freq, original_inv_freq)

    @parameterized.expand([True, False])
    def test_dynamic_rope_resets_after_long_sequence_nested(self, same_rope_per_layer: bool):
        config = Gemma3TextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
        )
        config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 4.0},
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        config.max_position_embeddings = 8  # restore small max_position_embeddings for dynamic scaling test

        layer_types = config.layer_types if same_rope_per_layer else ["full_attention"]
        rotary_embedding = Gemma3RotaryEmbedding(config, device=torch_device)

        for layer_type in layer_types:
            original_inv_freq = getattr(rotary_embedding, f"{layer_type}_original_inv_freq").clone()
            long_position_ids = torch.arange(32).unsqueeze(0).to(torch_device)
            long_input = torch.zeros(1, 32, 2, 8).to(torch_device)
            rotary_embedding(long_input, long_position_ids, layer_type=layer_type)

            max_seq_len_cached = getattr(rotary_embedding, f"{layer_type}_max_seq_len_cached")
            inv_freq = getattr(rotary_embedding, f"{layer_type}_inv_freq")
            self.assertEqual(int(max_seq_len_cached), 32)
            self.assertFalse(hasattr(rotary_embedding, "None_max_seq_len_cached"))
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(inv_freq, original_inv_freq)

            short_position_ids = torch.arange(4).unsqueeze(0).to(torch_device)
            short_input = torch.zeros(1, 4, 2, 8).to(torch_device)
            rotary_embedding(short_input, short_position_ids, layer_type=layer_type)

            max_seq_len_cached = getattr(rotary_embedding, f"{layer_type}_max_seq_len_cached")
            inv_freq = getattr(rotary_embedding, f"{layer_type}_inv_freq")
            self.assertEqual(int(max_seq_len_cached), config.max_position_embeddings)
            self.assertFalse(hasattr(rotary_embedding, "None_max_seq_len_cached"))
            torch.testing.assert_close(inv_freq, original_inv_freq)

    def test_yarn_rope_numerically(self):
        # fmt: off
        EXPECTED_INV_FREQ = torch.tensor(
            [
                1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
                4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
                1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 1.0000e-01, 8.3479e-02,
                6.9590e-02, 5.7925e-02, 4.8136e-02, 3.9931e-02, 3.3061e-02, 2.7315e-02,
                2.2515e-02, 1.8512e-02, 1.5177e-02, 1.2403e-02, 1.0101e-02, 8.1924e-03,
                6.6143e-03, 5.3120e-03, 4.2400e-03, 3.3599e-03, 2.6396e-03, 2.0520e-03,
                1.5746e-03, 1.1882e-03, 8.7713e-04, 6.2810e-04, 4.3007e-04, 2.7384e-04,
                2.3714e-04, 2.0535e-04, 1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04,
                1.0000e-04, 8.6596e-05, 7.4989e-05, 6.4938e-05, 5.6234e-05, 4.8697e-05,
                4.2170e-05, 3.6517e-05, 3.1623e-05, 2.7384e-05, 2.3714e-05, 2.0535e-05,
                1.7783e-05, 1.5399e-05, 1.3335e-05, 1.1548e-05
            ], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        config = LlamaConfig()
        self.assertEqual(config.rope_parameters, {"rope_type": "default", "rope_theta": 10000.0})
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = rope_fn(config=config, device=torch_device)

        # Check 1: according to the paper, if `attention_factor` is not specified, then it has a specific default --
        # `0.1 * math.log(factor) + 1.0`
        rope_fn = ROPE_INIT_FUNCTIONS["yarn"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": factor}
            _, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 0.1 * math.log(factor) + 1.0)

            config.rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": factor,
                "attention_factor": 0.5,
            }
            _, attention_scale = rope_fn(config=config, device=torch_device, seq_len=1)
            self.assertEqual(attention_scale, 0.5)

        # Check 2: based on `beta_fast` and `beta_slow`, the frequencies will be scaled between 1 and `factor`.
        # Increasing `beta_fast` will make RoPE more interpolative (apply scaling), and the other way around.
        # `beta_slow` behaves the opposite way. Remember: `beta_fast` > `beta_slow`
        # (note: adds a margin to the test for numerical stability)
        factor = 10.0
        margin = 1e-8
        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        is_bounded_by_factor = [
            ((default_inv_freq[idx] / factor) - margin) <= yarn_inv_freq_value <= (default_inv_freq[idx] + margin)
            for idx, yarn_inv_freq_value in enumerate(inv_freq)
        ]
        self.assertTrue(all(is_bounded_by_factor))

        # super high beta_fast = interpolation (i.e. scaling) in all but the first inverse frequency. The last ~20
        # values (empirically checked for `beta_fast` = 1000) should be very small to linear scaling
        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 1000,
            "beta_slow": 1,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        is_interpolating = [
            yarn_inv_freq_value < (default_inv_freq[idx] + margin) for idx, yarn_inv_freq_value in enumerate(inv_freq)
        ]
        self.assertFalse(is_interpolating[0])
        self.assertTrue(all(is_interpolating[1:]))
        torch.testing.assert_close(inv_freq[-20:], default_inv_freq[-20:] / factor)

        # Check 3: numerical snapshot to avoid regressions
        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    @parameterized.expand([True, False])
    def test_yarn_rope_numerically_nested(self, same_rope_per_layer: bool):
        # fmt: off
        # Gemma3: head_dim=256, rope_theta=10000.0, factor=10.0, beta_fast=32, beta_slow=1
        EXPECTED_INV_FREQ = torch.tensor(
            [
                1.0000e+00, 9.3057e-01, 8.6596e-01, 8.0584e-01, 7.4989e-01, 6.9783e-01,
                6.4938e-01, 6.0430e-01, 5.6234e-01, 5.2330e-01, 4.8697e-01, 4.5316e-01,
                4.2170e-01, 3.9242e-01, 3.6517e-01, 3.3982e-01, 3.1623e-01, 2.9427e-01,
                2.7384e-01, 2.5483e-01, 2.3714e-01, 2.2067e-01, 2.0535e-01, 1.9110e-01,
                1.7783e-01, 1.6548e-01, 1.5399e-01, 1.4330e-01, 1.3335e-01, 1.2409e-01,
                1.1548e-01, 1.0746e-01, 1.0000e-01, 9.3057e-02, 8.6596e-02, 8.0584e-02,
                7.4989e-02, 6.9783e-02, 6.4938e-02, 6.0430e-02, 5.6234e-02, 5.2330e-02,
                4.8697e-02, 4.5316e-02, 4.2170e-02, 3.9242e-02, 3.6517e-02, 3.3982e-02,
                3.1623e-02, 2.9427e-02, 2.7384e-02, 2.5483e-02, 2.3714e-02, 2.2067e-02,
                2.0535e-02, 1.9110e-02, 1.7783e-02, 1.6548e-02, 1.5399e-02, 1.4330e-02,
                1.3335e-02, 1.2409e-02, 1.1548e-02, 1.0746e-02, 1.0000e-02, 9.3057e-03,
                8.6596e-03, 8.0584e-03, 7.4989e-03, 6.9783e-03, 6.4938e-03, 6.0430e-03,
                5.6234e-03, 5.2330e-03, 4.8697e-03, 4.5316e-03, 4.2170e-03, 3.9242e-03,
                3.6517e-03, 3.3982e-03, 3.1623e-03, 2.9427e-03, 2.7384e-03, 2.5483e-03,
                2.3714e-03, 2.2067e-03, 2.0535e-03, 1.9110e-03, 1.7783e-03, 1.6548e-03,
                1.5399e-03, 1.4067e-03, 1.2845e-03, 1.1726e-03, 1.0699e-03, 9.7592e-04,
                8.8980e-04, 8.1093e-04, 7.3872e-04, 6.7263e-04, 6.1216e-04, 5.5684e-04,
                5.0625e-04, 4.6001e-04, 4.1774e-04, 3.7912e-04, 3.4386e-04, 3.1166e-04,
                2.8228e-04, 2.5547e-04, 2.3103e-04, 2.0875e-04, 1.8845e-04, 1.6996e-04,
                1.5313e-04, 1.3782e-04, 1.2389e-04, 1.1124e-04, 9.9743e-05, 8.9308e-05,
                7.9841e-05, 7.1258e-05, 6.3483e-05, 5.6443e-05, 5.0075e-05, 4.4319e-05,
                3.9121e-05, 3.4431e-05
            ], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        default_config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0},
            is_nested=True,
            same_rope_per_layer=True,
        )
        self.assertEqual(default_config.hidden_size, 2304)
        self.assertEqual(default_config.num_attention_heads, 8)
        expected_defaults = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        for layer_type in set(default_config.layer_types):
            self.assertFalse(hasattr(default_config.rope_parameters[layer_type], "partial_rotary_factor"))
            self.assertEqual(default_config.rope_parameters[layer_type], expected_defaults[layer_type])

        default_inv_freq, _ = Gemma3RotaryEmbedding.compute_default_rope_parameters(
            config=default_config, layer_type="full_attention", device=torch_device
        )
        layer_types = default_config.layer_types if same_rope_per_layer else ["full_attention"]

        # Check 1: according to the paper, if `attention_factor` is not specified, then it has a specific default --
        # `0.1 * math.log(factor) + 1.0`
        rope_fn = ROPE_INIT_FUNCTIONS["yarn"]
        for factor in (2.0, 10.0, 20.0):
            config = self.get_config_with_rope_parameters(
                rope_params={"rope_type": "yarn", "rope_theta": 10000.0, "factor": factor},
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            config_attn_factor = self.get_config_with_rope_parameters(
                rope_params={
                    "rope_type": "yarn",
                    "rope_theta": 10000.0,
                    "factor": factor,
                    "attention_factor": 0.5,
                },
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                _, attention_scale = rope_fn(config=config, layer_type=layer_type, device=torch_device)
                self.assertEqual(attention_scale, 0.1 * math.log(factor) + 1.0)

                _, attention_scale = rope_fn(
                    config=config_attn_factor, layer_type=layer_type, device=torch_device, seq_len=1
                )
                self.assertEqual(attention_scale, 0.5)

        # Check 2: based on `beta_fast` and `beta_slow`, the frequencies will be scaled between 1 and `factor`.
        # Increasing `beta_fast` will make RoPE more interpolative (apply scaling), and the other way around.
        # `beta_slow` behaves the opposite way. Remember: `beta_fast` > `beta_slow`
        # (note: adds a margin to the test for numerical stability)
        factor = 10.0
        margin = 1e-8
        rope_params = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            is_bounded_by_factor = [
                ((default_inv_freq[idx] / factor) - margin) <= yarn_inv_freq_value <= (default_inv_freq[idx] + margin)
                for idx, yarn_inv_freq_value in enumerate(inv_freq)
            ]
            self.assertTrue(all(is_bounded_by_factor))

        # super high beta_fast = interpolation (i.e. scaling) in all but the first inverse frequency.
        # Note: we set original_max_position_embeddings=8 to ensure the yarn ramp covers most frequencies
        # (Gemma3's large max_position_embeddings=131072 would otherwise make most frequencies extrapolated).
        rope_params = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 1000,
            "beta_slow": 1,
            "original_max_position_embeddings": 8,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            is_interpolating = [
                yarn_inv_freq_value < (default_inv_freq[idx] + margin)
                for idx, yarn_inv_freq_value in enumerate(inv_freq)
            ]
            self.assertFalse(is_interpolating[0])
            self.assertTrue(all(is_interpolating[1:]))

        # Check 3: numerical snapshot to avoid regressions
        rope_params = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": factor,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    def test_longrope_rope_numerically(self):
        # input sanity checks: if these change, the output will also change
        config = LlamaConfig()
        self.assertEqual(config.rope_parameters, {"rope_type": "default", "rope_theta": 10000.0})
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        # longrope applies scaling on EACH inv frequency, `short_factor` or `long_factor`, depending on the seq_len
        dim = config.hidden_size // config.num_attention_heads
        short_factor = [2.0] * (dim // 2)  # scaling applied when seq_len <= max_position_embeddings
        long_factor = torch.ones(dim // 2).cumsum(0).tolist()  # scaling applied when seq_len > max_position_embeddings

        rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = rope_fn(config=config, device=torch_device)

        # Check 1: according to the paper, if `attention_factor` is not specified, then it has a specific default --
        # `math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))`
        rope_fn = ROPE_INIT_FUNCTIONS["longrope"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
            }
            _, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(
                attention_scale, math.sqrt(1 + math.log(factor) / math.log(config.max_position_embeddings))
            )

            config.rope_parameters = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
                "attention_factor": 0.5,
            }
            _, attention_scale = rope_fn(config=config, device=torch_device, seq_len=1)
            self.assertEqual(attention_scale, 0.5)

            config.rope_parameters = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
            }
            self.assertEqual(config.rope_parameters.get("attention_factor"), None)
            # Verify that "TypeError: '<' not supported between instances of 'NoneType' and 'int'" is not raised.
            config.standardize_rope_params()
            config.validate_rope()

        # Check 2: seq_len == 0 -> short factor is applied to the default frequencies
        config.rope_parameters = {
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "factor": 1.0,
            "short_factor": short_factor,
            "long_factor": long_factor,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device, seq_len=0)
        torch.testing.assert_close(inv_freq, default_inv_freq / torch.tensor(short_factor).to(torch_device))

        # Check 3: seq_len > max_position_embeddings -> long factor is applied to the default frequencies
        inv_freq, _ = rope_fn(config=config, device=torch_device, seq_len=config.max_position_embeddings + 1)
        torch.testing.assert_close(inv_freq, default_inv_freq / torch.tensor(long_factor).to(torch_device))

    @parameterized.expand([True, False])
    def test_longrope_rope_numerically_nested(self, same_rope_per_layer: bool):
        # input sanity checks: if these change, the output will also change
        default_config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0},
            is_nested=True,
            same_rope_per_layer=True,
        )
        self.assertEqual(default_config.hidden_size, 2304)
        self.assertEqual(default_config.num_attention_heads, 8)
        expected_defaults = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        for layer_type in set(default_config.layer_types):
            self.assertFalse(hasattr(default_config.rope_parameters[layer_type], "partial_rotary_factor"))
            self.assertEqual(default_config.rope_parameters[layer_type], expected_defaults[layer_type])

        # longrope applies scaling on EACH inv frequency, `short_factor` or `long_factor`, depending on the seq_len
        # Use head_dim (256) not hidden_size // num_attention_heads (288) for Gemma3
        dim = getattr(default_config, "head_dim", default_config.hidden_size // default_config.num_attention_heads)
        short_factor = [2.0] * (dim // 2)  # scaling applied when seq_len <= max_position_embeddings
        long_factor = torch.ones(dim // 2).cumsum(0).tolist()  # scaling applied when seq_len > max_position_embeddings
        layer_types = default_config.layer_types if same_rope_per_layer else ["full_attention"]

        default_inv_freq, _ = Gemma3RotaryEmbedding.compute_default_rope_parameters(
            config=default_config, layer_type="full_attention", device=torch_device
        )

        # Check 1: according to the paper, if `attention_factor` is not specified, then it has a specific default --
        # `math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))`
        rope_fn = ROPE_INIT_FUNCTIONS["longrope"]
        for factor in (2.0, 10.0, 20.0):
            rope_params = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
            }
            config = self.get_config_with_rope_parameters(
                rope_params=rope_params,
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                _, attention_scale = rope_fn(config=config, layer_type=layer_type, device=torch_device)
                self.assertEqual(
                    attention_scale, math.sqrt(1 + math.log(factor) / math.log(config.max_position_embeddings))
                )

            rope_params = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
                "attention_factor": 0.5,
            }
            config = self.get_config_with_rope_parameters(
                rope_params=rope_params,
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                _, attention_scale = rope_fn(config=config, layer_type=layer_type, device=torch_device, seq_len=1)
                self.assertEqual(attention_scale, 0.5)

            rope_params = {
                "rope_type": "longrope",
                "rope_theta": 10000.0,
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
            }
            config = self.get_config_with_rope_parameters(
                rope_params=rope_params,
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                self.assertEqual(config.rope_parameters[layer_type].get("attention_factor"), None)
                # Verify that "TypeError: '<' not supported between instances of 'NoneType' and 'int'" is not raised.
                config.standardize_rope_params()
                config.validate_rope()

        # Check 2: seq_len == 0 -> short factor is applied to the default frequencies
        rope_params = {
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "factor": 1.0,
            "short_factor": short_factor,
            "long_factor": long_factor,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device, seq_len=0)
            torch.testing.assert_close(inv_freq, default_inv_freq / torch.tensor(short_factor).to(torch_device))

            # Check 3: seq_len > max_position_embeddings -> long factor is applied to the default frequencies
            inv_freq, _ = rope_fn(
                config=config, layer_type=layer_type, device=torch_device, seq_len=config.max_position_embeddings + 1
            )
            torch.testing.assert_close(inv_freq, default_inv_freq / torch.tensor(long_factor).to(torch_device))

    def test_llama3_rope_numerically(self):
        # fmt: off
        EXPECTED_INV_FREQ = torch.tensor(
            [
                1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
                4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
                1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 1.0000e-01, 8.6596e-02,
                7.4989e-02, 6.4938e-02, 5.6234e-02, 4.8697e-02, 4.2170e-02, 3.6517e-02,
                3.1623e-02, 2.7384e-02, 2.3714e-02, 2.0535e-02, 1.7783e-02, 1.5399e-02,
                1.3335e-02, 1.0730e-02, 7.7785e-03, 5.6009e-03, 3.9991e-03, 2.8248e-03,
                1.9675e-03, 1.3449e-03, 8.9549e-04, 5.7363e-04, 3.4539e-04, 2.7384e-04,
                2.3714e-04, 2.0535e-04, 1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04,
                1.0000e-04, 8.6596e-05, 7.4989e-05, 6.4938e-05, 5.6234e-05, 4.8697e-05,
                4.2170e-05, 3.6517e-05, 3.1623e-05, 2.7384e-05, 2.3714e-05, 2.0535e-05,
                1.7783e-05, 1.5399e-05, 1.3335e-05, 1.1548e-05
            ], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        config = LlamaConfig()
        self.assertEqual(config.rope_parameters, {"rope_type": "default", "rope_theta": 10000.0})
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = rope_fn(config=config, device=torch_device)

        # Check 1: `attention_factor` is always 1
        rope_fn = ROPE_INIT_FUNCTIONS["llama3"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_parameters = {
                "rope_type": "llama3",
                "rope_theta": 10000.0,
                "factor": factor,
                "original_max_position_embeddings": 2048,
                "low_freq_factor": 1,
                "high_freq_factor": 4,
            }
            _, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 1.0)

        # Check 2: based on `low_freq_factor` and `high_freq_factor`, the frequencies will be scaled between 1 and
        # `factor` (similar to yarn). Low frequencies get scaled by `factor`, high frequencies see no change, medium
        # frequencies are scaled by a value in between. Changing `low_freq_factor` and `high_freq_factor` changes what
        # is considered low, medium, and high frequencies.
        factor = 10.0
        config.rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        is_bounded_by_factor = [
            (default_inv_freq[idx] / factor) <= llama3_inv_freq_value <= default_inv_freq[idx]
            for idx, llama3_inv_freq_value in enumerate(inv_freq)
        ]
        self.assertTrue(all(is_bounded_by_factor))

        # if we change `high_freq_factor` to a very high value, none is considered high-frequency -> ALL values will be
        # scaled
        config.rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 1000,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        is_scaled = [yarn_inv_freq_value < default_inv_freq[idx] for idx, yarn_inv_freq_value in enumerate(inv_freq)]
        self.assertTrue(all(is_scaled))

        # Check 3: numerical snapshot to avoid regressions
        config.rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    @parameterized.expand([True, False])
    def test_llama3_rope_numerically_nested(self, same_rope_per_layer: bool):
        # fmt: off
        EXPECTED_INV_FREQ = torch.tensor(
            [1.0000e+00, 9.3057e-01, 8.6596e-01, 8.0584e-01, 7.4989e-01, 6.9783e-01,
            6.4938e-01, 6.0430e-01, 5.6234e-01, 5.2330e-01, 4.8697e-01, 4.5316e-01,
            4.2170e-01, 3.9242e-01, 3.6517e-01, 3.3982e-01, 3.1623e-01, 2.9427e-01,
            2.7384e-01, 2.5483e-01, 2.3714e-01, 2.2067e-01, 2.0535e-01, 1.9110e-01,
            1.7783e-01, 1.6548e-01, 1.5399e-01, 1.4330e-01, 1.3335e-01, 1.2409e-01,
            1.1548e-01, 1.0746e-01, 1.0000e-01, 9.3057e-02, 8.6596e-02, 8.0584e-02,
            7.4989e-02, 6.9783e-02, 6.4938e-02, 6.0430e-02, 5.6234e-02, 5.2330e-02,
            4.8697e-02, 4.5316e-02, 4.2170e-02, 3.9242e-02, 3.6517e-02, 3.3982e-02,
            3.1623e-02, 2.9427e-02, 2.7384e-02, 2.5483e-02, 2.3714e-02, 2.2067e-02,
            2.0535e-02, 1.9110e-02, 1.7783e-02, 1.6548e-02, 1.5399e-02, 1.4330e-02,
            1.3335e-02, 1.2409e-02, 1.0730e-02, 9.1428e-03, 7.7785e-03, 6.6067e-03,
            5.6009e-03, 4.7383e-03, 3.9991e-03, 3.3661e-03, 2.8248e-03, 2.3623e-03,
            1.9675e-03, 1.6312e-03, 1.3449e-03, 1.1017e-03, 8.9549e-04, 7.2098e-04,
            5.7363e-04, 4.4956e-04, 3.4539e-04, 2.9427e-04, 2.7384e-04, 2.5483e-04,
            2.3714e-04, 2.2067e-04, 2.0535e-04, 1.9110e-04, 1.7783e-04, 1.6548e-04,
            1.5399e-04, 1.4330e-04, 1.3335e-04, 1.2409e-04, 1.1548e-04, 1.0746e-04,
            1.0000e-04, 9.3057e-05, 8.6596e-05, 8.0584e-05, 7.4989e-05, 6.9783e-05,
            6.4938e-05, 6.0430e-05, 5.6234e-05, 5.2330e-05, 4.8697e-05, 4.5316e-05,
            4.2170e-05, 3.9242e-05, 3.6517e-05, 3.3982e-05, 3.1623e-05, 2.9427e-05,
            2.7384e-05, 2.5483e-05, 2.3714e-05, 2.2067e-05, 2.0535e-05, 1.9110e-05,
            1.7783e-05, 1.6548e-05, 1.5399e-05, 1.4330e-05, 1.3335e-05, 1.2409e-05,
            1.1548e-05, 1.0746e-05], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        default_config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0},
            is_nested=True,
            same_rope_per_layer=True,
        )
        self.assertEqual(default_config.hidden_size, 2304)
        self.assertEqual(default_config.num_attention_heads, 8)
        expected_defaults = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        for layer_type in set(default_config.layer_types):
            self.assertFalse(hasattr(default_config.rope_parameters[layer_type], "partial_rotary_factor"))
            self.assertEqual(default_config.rope_parameters[layer_type], expected_defaults[layer_type])

        rope_fn = Gemma3RotaryEmbedding.compute_default_rope_parameters
        default_inv_freq, _ = rope_fn(config=default_config, layer_type="full_attention", device=torch_device)
        layer_types = default_config.layer_types if same_rope_per_layer else ["full_attention"]

        # Check 1: `attention_factor` is always 1
        rope_fn = ROPE_INIT_FUNCTIONS["llama3"]
        for factor in (2.0, 10.0, 20.0):
            for layer_type in layer_types:
                rope_params = {
                    "rope_type": "llama3",
                    "rope_theta": 10000.0,
                    "factor": factor,
                    "original_max_position_embeddings": 2048,
                    "low_freq_factor": 1,
                    "high_freq_factor": 4,
                }
                config = self.get_config_with_rope_parameters(
                    rope_params=rope_params,
                    is_nested=True,
                    same_rope_per_layer=same_rope_per_layer,
                )
                _, attention_scale = rope_fn(config=config, layer_type=layer_type, device=torch_device)
                self.assertEqual(attention_scale, 1.0)

        # Check 2: based on `low_freq_factor` and `high_freq_factor`, the frequencies will be scaled between 1 and
        # `factor` (similar to yarn). Low frequencies get scaled by `factor`, high frequencies see no change, medium
        # frequencies are scaled by a value in between. Changing `low_freq_factor` and `high_freq_factor` changes what
        # is considered low, medium, and high frequencies.
        factor = 10.0
        rope_params = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            is_bounded_by_factor = [
                (default_inv_freq[idx] / factor) <= llama3_inv_freq_value <= default_inv_freq[idx]
                for idx, llama3_inv_freq_value in enumerate(inv_freq)
            ]
            self.assertTrue(all(is_bounded_by_factor))

        # if we change `high_freq_factor` to a very high value, none is considered high-frequency -> ALL values will be
        # scaled
        rope_params = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 1000,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            is_scaled = [
                yarn_inv_freq_value < default_inv_freq[idx] for idx, yarn_inv_freq_value in enumerate(inv_freq)
            ]
            self.assertTrue(all(is_scaled))

        # Check 3: numerical snapshot to avoid regressions
        rope_params = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    def test_proportional_rope_numerically(self):
        # fmt: off
        EXPECTED_INV_FREQ = torch.tensor(
            [
                1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
                4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
                1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00
            ], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        config = LlamaConfig()
        self.assertEqual(config.rope_parameters, {"rope_type": "default", "rope_theta": 10000.0})
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        head_dim = config.hidden_size // config.num_attention_heads  # 128

        rope_fn = ROPE_INIT_FUNCTIONS["proportional"]
        default_rope_fn = LlamaRotaryEmbedding.compute_default_rope_parameters

        # Check 1: `attention_factor` is always 1.0, regardless of parameters
        for partial_rotary_factor in (1.0, 0.5, 0.25):
            config.rope_parameters = {
                "rope_type": "proportional",
                "rope_theta": 10000.0,
                "partial_rotary_factor": partial_rotary_factor,
            }
            _, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 1.0)

        # Check 2: output shape is always head_dim // 2, regardless of partial_rotary_factor
        for partial_rotary_factor in (1.0, 0.5, 0.25):
            config.rope_parameters = {
                "rope_type": "proportional",
                "rope_theta": 10000.0,
                "partial_rotary_factor": partial_rotary_factor,
            }
            inv_freq, _ = rope_fn(config=config, device=torch_device)
            self.assertEqual(inv_freq.shape[0], head_dim // 2)

        # Check 3: zero-padding behavior — when partial_rotary_factor < 1.0, the last (head_dim // 2 - rope_angles)
        # entries must be exactly zero, and the first rope_angles entries must be non-zero
        for partial_rotary_factor, expected_rope_angles in ((0.5, 32), (0.25, 16)):
            config.rope_parameters = {
                "rope_type": "proportional",
                "rope_theta": 10000.0,
                "partial_rotary_factor": partial_rotary_factor,
            }
            inv_freq, _ = rope_fn(config=config, device=torch_device)

            # First rope_angles entries should be non-zero (rotated frequencies)
            self.assertTrue(torch.all(inv_freq[:expected_rope_angles] != 0))
            # Remaining entries should be exactly zero (NoPE angles)
            expected_nope_angles = head_dim // 2 - expected_rope_angles
            torch.testing.assert_close(
                inv_freq[expected_rope_angles:],
                torch.zeros(expected_nope_angles, device=torch_device),
            )

        # When partial_rotary_factor = 1.0, no entries should be zero
        config.rope_parameters = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        self.assertTrue(torch.all(inv_freq != 0))

        # Check 4: factor scaling equivalences with default and linear RoPE
        # 4a: With partial_rotary_factor=1.0 and factor=1.0, proportional RoPE == default RoPE
        config.rope_parameters = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "factor": 1.0,
        }
        inv_freq_prop, _ = rope_fn(config=config, device=torch_device)
        config.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        default_inv_freq, _ = default_rope_fn(config=config, device=torch_device)
        torch.testing.assert_close(inv_freq_prop, default_inv_freq)

        # 4b: With partial_rotary_factor=1.0 and factor=2.0, proportional RoPE == linear RoPE
        linear_rope_fn = ROPE_INIT_FUNCTIONS["linear"]
        for factor in (2.0, 10.0):
            config.rope_parameters = {
                "rope_type": "proportional",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "factor": factor,
            }
            inv_freq_prop, _ = rope_fn(config=config, device=torch_device)
            config.rope_parameters = {"rope_type": "linear", "rope_theta": 10000.0, "factor": factor}
            inv_freq_linear, _ = linear_rope_fn(config=config, device=torch_device)
            torch.testing.assert_close(inv_freq_prop, inv_freq_linear)

        # 4c: With partial_rotary_factor=0.5 and factor=2.0, the non-zero portion should be the rotated subspace
        # frequencies divided by factor
        config.rope_parameters = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "factor": 2.0,
        }
        inv_freq_scaled, _ = rope_fn(config=config, device=torch_device)
        config.rope_parameters = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "factor": 1.0,
        }
        inv_freq_unscaled, _ = rope_fn(config=config, device=torch_device)
        torch.testing.assert_close(inv_freq_scaled, inv_freq_unscaled / 2.0)

        # Check 5: numerical snapshot to avoid regressions (partial_rotary_factor=0.25, factor=1.0)
        config.rope_parameters = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    @parameterized.expand([True, False])
    def test_proportional_rope_numerically_nested(self, same_rope_per_layer: bool):
        # fmt: off
        EXPECTED_INV_FREQ = torch.tensor(
            [
                1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
                4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
                1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00
            ], device=torch_device
        )
        # fmt: on

        # input sanity checks: if these change, the output will also change
        default_config = self.get_config_with_rope_parameters(
            rope_params={"rope_type": "default", "rope_theta": 10000.0},
            is_nested=True,
            same_rope_per_layer=True,
        )
        self.assertEqual(default_config.hidden_size, 2304)
        self.assertEqual(default_config.num_attention_heads, 8)
        expected_defaults = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        for layer_type in set(default_config.layer_types):
            self.assertFalse(hasattr(default_config.rope_parameters[layer_type], "partial_rotary_factor"))
            self.assertEqual(default_config.rope_parameters[layer_type], expected_defaults[layer_type])

        rope_fn = ROPE_INIT_FUNCTIONS["proportional"]
        default_inv_freq, _ = rope_fn(config=default_config, layer_type="full_attention", device=torch_device)
        layer_types = default_config.layer_types if same_rope_per_layer else ["full_attention"]

        # Check 1: `attention_factor` is always 1.0, regardless of parameters
        for partial_rotary_factor in (1.0, 0.5, 0.25):
            rope_params = {
                "rope_type": "proportional",
                "rope_theta": 10000.0,
                "partial_rotary_factor": partial_rotary_factor,
            }
            config = self.get_config_with_rope_parameters(
                rope_params=rope_params,
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                _, attention_scale = rope_fn(config=config, layer_type=layer_type, device=torch_device)
                self.assertEqual(attention_scale, 1.0)

        # Check 2: output shape is always head_dim // 2, regardless of partial_rotary_factor
        for partial_rotary_factor in (1.0, 0.5, 0.25):
            rope_params = {
                "rope_type": "proportional",
                "rope_theta": 10000.0,
                "partial_rotary_factor": partial_rotary_factor,
            }
            config = self.get_config_with_rope_parameters(
                rope_params=rope_params,
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
                self.assertEqual(inv_freq.shape[0], default_config.head_dim // 2)

        # Check 3: zero-padding behavior — when partial_rotary_factor < 1.0, the last (head_dim // 2 - rope_angles)
        # entries must be exactly zero, and the first rope_angles entries must be non-zero
        for partial_rotary_factor, expected_rope_angles in ((0.5, 32), (0.25, 16)):
            rope_params = {
                "rope_type": "proportional",
                "rope_theta": 10000.0,
                "partial_rotary_factor": partial_rotary_factor,
            }
            config = self.get_config_with_rope_parameters(
                rope_params=rope_params,
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)

                # First rope_angles entries should be non-zero (rotated frequencies)
                self.assertTrue(torch.all(inv_freq[:expected_rope_angles] != 0))
                # Remaining entries should be exactly zero (NoPE angles)
                expected_nope_angles = default_config.head_dim // 2 - expected_rope_angles
                torch.testing.assert_close(
                    inv_freq[expected_rope_angles:],
                    torch.zeros(expected_nope_angles, device=torch_device),
                )

        # When partial_rotary_factor = 1.0, no entries should be zero
        rope_params = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            self.assertTrue(torch.all(inv_freq != 0))

        # Check 4: factor scaling equivalences with default and linear RoPE
        # 4a: With partial_rotary_factor=1.0 and factor=1.0, proportional RoPE == default RoPE
        rope_params = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "factor": 1.0,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq_prop, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            torch.testing.assert_close(inv_freq_prop, default_inv_freq)

        # 4b: With partial_rotary_factor=1.0 and factor=2.0, proportional RoPE == linear RoPE
        linear_rope_fn = ROPE_INIT_FUNCTIONS["linear"]
        for factor in (2.0, 10.0):
            rope_params = {
                "rope_type": "proportional",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "factor": factor,
            }
            config_prop = self.get_config_with_rope_parameters(
                rope_params=rope_params,
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            config_linear = self.get_config_with_rope_parameters(
                rope_params={"rope_type": "linear", "rope_theta": 10000.0, "factor": factor},
                is_nested=True,
                same_rope_per_layer=same_rope_per_layer,
            )
            for layer_type in layer_types:
                inv_freq_prop, _ = rope_fn(config=config_prop, layer_type=layer_type, device=torch_device)
                inv_freq_linear, _ = linear_rope_fn(config=config_linear, layer_type=layer_type, device=torch_device)
                torch.testing.assert_close(inv_freq_prop, inv_freq_linear)

        # 4c: With partial_rotary_factor=0.5 and factor=2.0, the non-zero portion should be the rotated subspace
        # frequencies divided by factor
        rope_params = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "factor": 2.0,
        }
        config_scaled = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )

        rope_params = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "factor": 1.0,
        }
        config_unscaled = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq_scaled, _ = rope_fn(config=config_scaled, layer_type=layer_type, device=torch_device)
            inv_freq_unscaled, _ = rope_fn(config=config_unscaled, layer_type=layer_type, device=torch_device)
            torch.testing.assert_close(inv_freq_scaled, inv_freq_unscaled / 2.0)

        # Check 5: numerical snapshot to avoid regressions (partial_rotary_factor=0.25, factor=1.0)
        rope_params = {
            "rope_type": "proportional",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        }
        config = self.get_config_with_rope_parameters(
            rope_params=rope_params,
            is_nested=True,
            same_rope_per_layer=same_rope_per_layer,
        )
        for layer_type in layer_types:
            inv_freq, _ = rope_fn(config=config, layer_type=layer_type, device=torch_device)
            torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)
