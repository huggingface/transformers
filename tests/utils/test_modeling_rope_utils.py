# coding=utf-8
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


import math
import unittest

from transformers import LlamaConfig
from transformers.testing_utils import is_torch_available, require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import ROPE_INIT_FUNCTIONS
    from transformers.modeling_rope_utils import rope_config_validation


@require_torch
class RopeTest(unittest.TestCase):
    def test_rope_validation(self):
        config = LlamaConfig()
        all_rope_types = ROPE_INIT_FUNCTIONS.keys()

        # The base config is always valid (default RoPE)
        rope_config_validation(config)

        # If we explicitly set the other RoPE types, then validation should fail
        for rope_type in all_rope_types:
            if rope_type != "default":
                config.rope_scaling = {"rope_type": rope_type}
                with self.assertRaises(KeyError):
                    rope_config_validation(config)

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
            if rope_type == "default":
                continue  # checked above
            for param, valid_rope_types in valid_param_mapping.items():
                # Set `param` with a dummy value -- we want to test the dict key
                config.rope_scaling = {"rope_type": rope_type, param: True}
                if rope_type in valid_rope_types:
                    continue
                else:
                    with self.assertRaises(KeyError):
                        rope_config_validation(config)

    def test_default_rope_function_bc(self):
        config = LlamaConfig()
        device = torch_device

        rope_kwargs = {
            "rope_type": "default",
            "dim": config.hidden_size // config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
        }

        rope_fn = ROPE_INIT_FUNCTIONS["default"]
        config_freqs = rope_fn(config=config, device=device)[0]
        kwargs_freqs = rope_fn(**rope_kwargs, device=device)[0]
        torch.testing.assert_close(config_freqs, kwargs_freqs)

    def test_linear_rope_function_bc(self):
        config = LlamaConfig()
        config.rope_scaling = {"rope_type": "linear", "factor": 10.0}
        device = torch_device

        rope_kwargs = {
            "rope_type": "linear",
            "dim": config.hidden_size // config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
            "factor": 10.0,
        }

        rope_fn = ROPE_INIT_FUNCTIONS["linear"]
        config_freqs = rope_fn(config=config, device=device)[0]
        kwargs_freqs = rope_fn(**rope_kwargs, device=device)[0]
        torch.testing.assert_close(config_freqs, kwargs_freqs)

    def test_dynamic_rope_function_bc(self):
        config = LlamaConfig()
        config.rope_scaling = {"rope_type": "dynamic", "factor": 10.0}
        device = torch_device

        rope_kwargs = {
            "rope_type": "dynamic",
            "dim": config.hidden_size // config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
            "factor": 10.0,
        }

        rope_fn = ROPE_INIT_FUNCTIONS["dynamic"]
        config_freqs = rope_fn(config=config, device=device)[0]
        kwargs_freqs = rope_fn(**rope_kwargs, device=device)[0]
        torch.testing.assert_close(config_freqs, kwargs_freqs)

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
        self.assertEqual(config.rope_scaling, None)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        rope_fn = ROPE_INIT_FUNCTIONS["default"]
        inv_freq, attention_scale = rope_fn(config=config, device=torch_device)

        self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for default RoPE
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    def test_linear_rope_numerically(self):
        # This is a linear scaling strategy, the **frequencies** are scaled linearly with respect to the default
        # frequencies (= the inverse frequencies are scaled **inversely**)
        config = LlamaConfig()
        default_rope_fn = ROPE_INIT_FUNCTIONS["default"]
        default_inv_freq, _ = default_rope_fn(config=config, device=torch_device)

        rope_fn = ROPE_INIT_FUNCTIONS["linear"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_scaling = {"rope_type": "linear", "factor": factor}
            inv_freq, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for linear RoPE
            torch.testing.assert_close(inv_freq, default_inv_freq / factor)

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
        self.assertEqual(config.rope_scaling, None)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        rope_fn = ROPE_INIT_FUNCTIONS["default"]
        default_inv_freq, _ = rope_fn(config=config, device=torch_device)

        # Check 1: this is a dynamic scaling strategy, it will not scale unless we provide `seq_len` larger than the
        # model's original training sequence length
        rope_fn = ROPE_INIT_FUNCTIONS["dynamic"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_scaling = {"rope_type": "dynamic", "factor": factor}
            inv_freq, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 1.0)  # attention scale is always 1 for dynamic RoPE
            torch.testing.assert_close(inv_freq, default_inv_freq)

            inv_freq, _ = rope_fn(config=config, device=torch_device, seq_len=1)
            torch.testing.assert_close(inv_freq, default_inv_freq)

        # Check 2: if we provide `seq_len` larger than the model's original training sequence length, the frequencies
        # will scale up (i.e., the inverse frequencies will scale down).
        factor = 10.0
        config.rope_scaling = {"rope_type": "dynamic", "factor": factor}
        inv_freq, _ = rope_fn(config=config, device=torch_device, seq_len=16384)
        with self.assertRaises(AssertionError):  # It is NOT a linear factor
            torch.testing.assert_close(inv_freq, default_inv_freq / factor)
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

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
        self.assertEqual(config.rope_scaling, None)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        rope_fn = ROPE_INIT_FUNCTIONS["default"]
        default_inv_freq, _ = rope_fn(config=config, device=torch_device)

        # Check 1: according to the paper, if `attention_factor` is not specified, then it has a specific default --
        # `0.1 * math.log(factor) + 1.0`
        rope_fn = ROPE_INIT_FUNCTIONS["yarn"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_scaling = {"rope_type": "yarn", "factor": factor}
            _, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 0.1 * math.log(factor) + 1.0)

            config.rope_scaling = {"rope_type": "yarn", "factor": factor, "attention_factor": 0.5}
            _, attention_scale = rope_fn(config=config, device=torch_device, seq_len=1)
            self.assertEqual(attention_scale, 0.5)

        # Check 2: based on `beta_fast` and `beta_slow`, the frequencies will be scaled between 1 and `factor`.
        # Increasing `beta_fast` will make RoPE more interpolative (apply scaling), and the other way around.
        # `beta_slow` behaves the opposite way. Remember: `beta_fast` > `beta_slow`
        # (note: adds a margin to the test for numerical stability)
        factor = 10.0
        margin = 1e-8
        config.rope_scaling = {"rope_type": "yarn", "factor": factor, "beta_fast": 32, "beta_slow": 1}
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        is_bounded_by_factor = [
            ((default_inv_freq[idx] / factor) - margin) <= yarn_inv_freq_value <= (default_inv_freq[idx] + margin)
            for idx, yarn_inv_freq_value in enumerate(inv_freq)
        ]
        self.assertTrue(all(is_bounded_by_factor))

        # super high beta_fast = interpolation (i.e. scaling) in all but the first inverse frequency. The last ~20
        # values (empirically checked for `beta_fast` = 1000) should be very small to linear scaling
        config.rope_scaling = {"rope_type": "yarn", "factor": factor, "beta_fast": 1000, "beta_slow": 1}
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        is_interpolating = [
            yarn_inv_freq_value < (default_inv_freq[idx] + margin) for idx, yarn_inv_freq_value in enumerate(inv_freq)
        ]
        self.assertFalse(is_interpolating[0])
        self.assertTrue(all(is_interpolating[1:]))
        torch.testing.assert_close(inv_freq[-20:], default_inv_freq[-20:] / factor)

        # Check 3: numerical snapshot to avoid regressions
        config.rope_scaling = {"rope_type": "yarn", "factor": factor, "beta_fast": 32, "beta_slow": 1}
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)

    def test_longrope_rope_numerically(self):
        # input sanity checks: if these change, the output will also change
        config = LlamaConfig()
        self.assertEqual(config.rope_scaling, None)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        # longrope applies scaling on EACH inv frequency, `short_factor` or `long_factor`, depending on `factor`
        dim = config.hidden_size // config.num_attention_heads
        short_factor = [2.0] * (dim // 2)  # scaling applied when factor == 1.0
        long_factor = torch.ones(dim // 2).cumsum(0).tolist()  # scaling applied when factor > 1.0

        rope_fn = ROPE_INIT_FUNCTIONS["default"]
        default_inv_freq, _ = rope_fn(config=config, device=torch_device)

        # Check 1: according to the paper, if `attention_factor` is not specified, then it has a specific default --
        # `math.sqrt(1 + math.log(factor) / math.log(max_position_embeddings))`
        rope_fn = ROPE_INIT_FUNCTIONS["longrope"]
        max_position_embeddings = config.max_position_embeddings
        for factor in (2.0, 10.0, 20.0):
            config.rope_scaling = {
                "rope_type": "longrope",
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
            }
            _, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, math.sqrt(1 + math.log(factor) / math.log(max_position_embeddings)))

            config.rope_scaling = {
                "rope_type": "longrope",
                "factor": factor,
                "short_factor": short_factor,
                "long_factor": long_factor,
                "attention_factor": 0.5,
            }
            _, attention_scale = rope_fn(config=config, device=torch_device, seq_len=1)
            self.assertEqual(attention_scale, 0.5)

        # Check 2: Factor == 1.0 -> short factor is applied to the default frequencies
        factor = 1.0
        config.rope_scaling = {
            "rope_type": "longrope",
            "factor": factor,
            "short_factor": short_factor,
            "long_factor": long_factor,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        torch.testing.assert_close(inv_freq, default_inv_freq / torch.tensor(short_factor).to(torch_device))

        # Check 3: Factor > 1.0 -> long factor is applied to the default frequencies
        factor = 10.0
        config.rope_scaling = {
            "rope_type": "longrope",
            "factor": factor,
            "short_factor": short_factor,
            "long_factor": long_factor,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
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
        self.assertEqual(config.rope_scaling, None)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertFalse(hasattr(config, "partial_rotary_factor"))

        rope_fn = ROPE_INIT_FUNCTIONS["default"]
        default_inv_freq, _ = rope_fn(config=config, device=torch_device)

        # Check 1: `attention_factor` is always 1
        rope_fn = ROPE_INIT_FUNCTIONS["llama3"]
        for factor in (2.0, 10.0, 20.0):
            config.rope_scaling = {
                "rope_type": "llama3",
                "factor": factor,
                "original_max_position_embeddings": 2048,
                "low_freq_factor": 1,
                "high_freq_factor": 4,
            }
            _, attention_scale = rope_fn(config=config, device=torch_device)
            self.assertEqual(attention_scale, 1.0)

        # Check 2: based on `low_freq_factor` and `high_freq_factor`, the frequencies will be scaled between 1 and
        # `factor` (similar to yarn). Low frequencies get scaled by `factor`, high frequences see no change, medium
        # frequencies are scaled by a value in between. Changing `low_freq_factor` and `high_freq_factor` changes what
        # is considered low, medium, and high frequencies.
        factor = 10.0
        config.rope_scaling = {
            "rope_type": "llama3",
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
        config.rope_scaling = config.rope_scaling = {
            "rope_type": "llama3",
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 1000,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        is_scaled = [yarn_inv_freq_value < default_inv_freq[idx] for idx, yarn_inv_freq_value in enumerate(inv_freq)]
        self.assertTrue(all(is_scaled))

        # Check 3: numerical snapshot to avoid regressions
        config.rope_scaling = {
            "rope_type": "llama3",
            "factor": factor,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
        }
        inv_freq, _ = rope_fn(config=config, device=torch_device)
        torch.testing.assert_close(inv_freq, EXPECTED_INV_FREQ)
