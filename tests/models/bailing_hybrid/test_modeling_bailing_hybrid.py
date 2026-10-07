# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Ling 3.0 / Bailing MoE V3 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import BailingHybridForCausalLM, BailingHybridModel, DynamicCache


class BailingHybridModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = BailingHybridModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.attention_probs_dropout_prob = 0.0
        self.hidden_act = "silu"
        self.num_hidden_layers = 2
        self.layer_types = ["linear_attention", "full_attention"]
        self.first_k_dense_replace = 1
        self.short_conv_kernel_size = 2
        self.head_dim = 16
        self.q_lora_rank = None
        self.kv_lora_rank = 16
        self.qk_nope_head_dim = 32
        self.qk_rope_head_dim = 16
        self.v_head_dim = 32
        self.moe_intermediate_size = 16
        self.num_local_experts = 4
        self.n_shared_experts = 1
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.gated_attention_proj_granularity_type = "head_wise"
        self.num_nextn_predict_layers = 0


@require_torch
class BailingHybridModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = BailingHybridModelTester
    model_tester: BailingHybridModelTester
    has_attentions = False

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, config.num_attention_heads * config.linear_head_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.num_attention_heads, config.linear_head_dim, config.linear_head_dim)

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("MLA uses different query/key and value head dimensions.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_hybrid_layer_pattern(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = BailingHybridModel(config)
        self.assertEqual(model.config.layer_types, ["linear_attention", "full_attention"])
        self.assertEqual(model.layers[0].self_attn.__class__.__name__, "BailingHybridKimiDeltaAttention")
        self.assertEqual(model.layers[1].self_attn.__class__.__name__, "BailingHybridAttention")

    def test_cached_decode_matches_full_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = BailingHybridForCausalLM(config).to(torch_device).eval()
        input_ids = ids_tensor((1, 8), config.vocab_size).to(torch_device)

        with torch.no_grad():
            full_logits = model(input_ids=input_ids).logits
            cache = DynamicCache(config=config)
            cached_logits = []
            for token in input_ids.split(1, dim=1):
                cached_logits.append(model(input_ids=token, past_key_values=cache, use_cache=True).logits)

        torch.testing.assert_close(torch.cat(cached_logits, dim=1), full_logits, rtol=1e-4, atol=1e-4)
