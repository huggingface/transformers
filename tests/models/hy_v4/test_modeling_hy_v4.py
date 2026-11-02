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
"""Testing suite for the PyTorch HYV4 model."""

import unittest

import torch

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    from transformers import HYV4ForCausalLM, HYV4Model


class HYV4ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HYV4Model
        causal_lm_class = HYV4ForCausalLM

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        moe_intermediate_size=16,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        num_hidden_layers=2,
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "shared"],
        index_topk=8,
        index_head_dim=8,
        index_n_heads=4,
        hc_mult=4,
    ):
        super().__init__(parent=parent, num_hidden_layers=num_hidden_layers)
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mlp_layer_types = mlp_layer_types
        self.indexer_types = indexer_types
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.hc_mult = hc_mult


@require_torch
class HYV4ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HYV4ModelTester
    # HYV4 routes each token to a subset of experts, so not every expert receives a gradient.
    test_all_params_have_gradient = False
    model_split_percents = [0.5, 0.8, 0.9]

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        """Override to account for the difference in MHC and the final state shapes"""
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (output_length - prompt_length))

        # When `output_hidden_states=True`, each iteration of generate appends the hidden states corresponding to the
        # new token(s)
        # NOTE: `StaticCache` may have different lengths on different layers, if this test starts failing add more
        # elaborate checks
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length

            # We have raw MHC shapes until the final one which is collapsed
            mhc_shape = (batch_size, model_input_length, config.hc_mult, config.hidden_size)
            final_shape = (batch_size, model_input_length, config.hidden_size)
            expected_shapes = [mhc_shape] * (len(iter_hidden_states) - 1)
            expected_shapes.append(final_shape)

            # check hidden size
            self.assertListEqual(
                [state.shape for state in iter_hidden_states],
                expected_shapes,
            )

    def test_hidden_states_output(self):
        """Override to account for the difference in MHC and the final state shapes"""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            text_config = model.config.get_text_config()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), text_config.num_hidden_layers + 1)

            batch_size, seq_len = inputs_dict["input_ids"].shape

            # Raw MHC shapes
            for layer_hidden_states in hidden_states[:-1]:
                self.assertEqual(
                    layer_hidden_states.shape,
                    (
                        batch_size,
                        seq_len,
                        text_config.hc_mult,
                        text_config.hidden_size,
                    ),
                )

            # Final output is standard again
            self.assertEqual(
                hidden_states[-1].shape,
                (
                    batch_size,
                    seq_len,
                    text_config.hidden_size,
                ),
            )

    @unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(
        "mHC hidden states are wrongfully cropped, needs adjustments in `_split_model_outputs` in generation"
    )
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass
