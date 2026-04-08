# Copyright 2025 Sarvam AI and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SarvamMoe model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        SarvamMoeForCausalLM,
        SarvamMoeModel,
    )
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class SarvamMoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = SarvamMoeModel

    def __init__(
        self,
        parent,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        num_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self.n_group = n_group
        self.topk_group = topk_group
        self.first_k_dense_replace = first_k_dense_replace
        self.num_shared_experts = num_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor


@require_torch
class SarvamMoeModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = SarvamMoeModelTester

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_experts = 4
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = SarvamMoeForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape[-1], config.num_experts)
        self.assertIsNotNone(result.aux_loss)

        # Make sure adding padding tokens doesn't change the loss
        pad_length = input_ids.shape[1] * 4
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)
