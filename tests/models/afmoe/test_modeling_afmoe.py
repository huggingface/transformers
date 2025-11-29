# Copyright 2025 Arcee AI and the HuggingFace Inc. team. All rights reserved.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    from transformers import AfmoeForCausalLM, AfmoeModel

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class AfmoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = AfmoeModel

    def __init__(
        self,
        parent,
        batch_size=4,
        seq_length=128,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_dense_layers=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=16384,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        rope_theta=10000.0,
        rope_parameters=None,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=2,
        route_norm=True,
        route_scale=1.0,
        global_attn_every_n_layers=2,
        sliding_window=128,
        attention_dropout=0.0,
    ):
        super().__init__(
            parent=parent,
            batch_size=batch_size,
            seq_length=seq_length,
            is_training=is_training,
            use_input_mask=use_input_mask,
            use_token_type_ids=use_token_type_ids,
            use_labels=use_labels,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
        )
        self.use_cache = use_cache
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.moe_intermediate_size = moe_intermediate_size
        self.num_dense_layers = num_dense_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout


@require_torch
class AfmoeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = AfmoeModelTester
    all_model_classes = (AfmoeModel, AfmoeForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": AfmoeModel, "text-generation": AfmoeForCausalLM} if is_torch_available() else {}
    )

    @unittest.skip("Afmoe applies key/query norm which doesn't work with packing")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Afmoe  applies key/query norm which doesn't work with packing")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Afmoe  applies key/query norm which doesn't work with packing")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip("Afmoe has moe, output can be different")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    # TODO: Add integration tests once we have a checkpoint on the Hub
