# Copyright 2026 Poolside and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Laguna model."""

import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import (
        LagunaConfig,
        LagunaModel,
    )


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class LagunaModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = LagunaModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.vocab_size = 64
        self.head_dim = 8
        self.sliding_window = 32
        self.shared_expert_intermediate_size = 16
        self.mlp_layer_types = ["dense", "sparse"]
        self.layer_types = ["full_attention", "sliding_attention"]


@require_torch
class LagunaModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = LagunaModelTester
    model_split_percents = [0.5, 0.8, 0.9]

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip(
        "RoPE-scaling-from-config test doesn't match Laguna's nested per-layer-type rope_parameters (same as e.g. Gemma3)."
    )
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def test_model_rope_scaling_frequencies(self):
        """
        Tests the frequency properties of the different RoPE scaling types on the model RoPE layer.
        Copied from Gemma3 to adapt to per layer rope configs.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.layer_types = ["full_attention", "sliding_attention"]

        # Retrieves the RoPE layer class from the base model class. Uses `.named_modules()` to avoid hardcoding the
        # named location of the RoPE layer class.
        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",  # most common case
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        rope_params = {"rope_type": "default", "rope_theta": 10_000.0}
        config.rope_parameters = {"full_attention": rope_params, "sliding_attention": rope_params}
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, layer_type="sliding_attention")
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        rope_params = {"rope_type": "linear", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"full_attention": rope_params, "sliding_attention": rope_params}
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        rope_params = {"rope_type": "dynamic", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"full_attention": rope_params, "sliding_attention": rope_params}
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue(
            (ntk_scaling_rope.sliding_attention_inv_freq <= original_rope.sliding_attention_inv_freq).all()
        )

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        rope_params = {"rope_type": "yarn", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"full_attention": rope_params, "sliding_attention": rope_params}
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)

    def test_apply_router_weight_on_input_not_supported(self):
        """
        `moe_apply_router_weight_on_input=True` is not supported yet so we explicitly check that it
        raises and error on config construction time
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cfg_kwargs = config.to_dict()
        cfg_kwargs["moe_apply_router_weight_on_input"] = True
        with self.assertRaises(NotImplementedError):
            LagunaConfig(**cfg_kwargs)


@require_torch
class LagunaIntegrationTest(unittest.TestCase):
    """Slow integration tests — need a public Hub checkpoint.

    TODO: replace the placeholder id once the Laguna model card is published.
    """

    @slow
    @unittest.skip("public Laguna checkpoint not yet published")
    def test_logits_and_generation(self):
        pass
