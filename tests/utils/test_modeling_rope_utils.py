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


# TODO(joao): numerical checks for the different RoPE fns
