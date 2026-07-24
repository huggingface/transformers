# Copyright 2026 SK Telecom and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch A.X-K2 model."""

import unittest

from transformers import Cache, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AXK2ForCausalLM,
        AXK2Model,
    )


class AXK2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = AXK2Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        num_experts_per_tok=2,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=32,
        index_n_heads=2,
        index_head_dim=64,
        index_topk=8,
        gated_norm_rank=4,
    ):
        super().__init__(parent=parent)
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.gated_norm_rank = gated_norm_rank
        self.mlp_layer_types = ["dense", "sparse"]


@require_torch
class AXK2ModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = AXK2ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = AXK2ForCausalLM if is_torch_available() else None

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as A.X-K2 has the MLA cache format (same as DeepSeek-V3.2)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    @unittest.skip("Can be fixed by #47438, currently does not properly considers cases where topk > prefill")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip("Fundamentally incompatible with indexer as there is no boundary between sequences")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Fundamentally incompatible with indexer as there is no boundary between sequences")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass


@slow
@require_torch_accelerator
class AXK2IntegrationTest(unittest.TestCase):
    # A.X-K2's released checkpoint is a 20B+ MoE that does not fit a CI GPU, so these tests run on a tiny
    # *randomized* checkpoint (same layout as `tiny-axk1`) that still exercises the full stack: fused
    # attention output gate, gated RMSNorm, and the SGA (sparse-gated) indexer. Regenerate it
    # byte-identically (and upload) with:
    #
    #     torch.manual_seed(0)
    #     config = AXK2Config(
    #         vocab_size=163840, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
    #         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=16,
    #         num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=16, qk_nope_head_dim=8,
    #         qk_rope_head_dim=8, v_head_dim=8, index_n_heads=2, index_head_dim=16, index_topk=8,
    #         gated_norm_rank=4, max_position_embeddings=4096,
    #         mlp_layer_types=["dense"] + ["sparse"] * 3,
    #     )
    #     AXK2ForCausalLM(config).to(torch.bfloat16).push_to_hub("hf-internal-testing/tiny-axk2")
    #
    # The logits expectations below were recorded from exactly this seeded model (bf16, eager, A100).
    model_id = "hf-internal-testing/tiny-axk2"

    def test_generation(self):
        # Weights are randomly initialized so the decoded text is arbitrary; this just exercises the full
        # greedy generation loop end to end and checks the output shape.
        model = AXK2ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=torch_device)
        generated_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        self.assertEqual(generated_ids.shape, (1, input_ids.shape[1] + 20))

    def test_model_logits_batched(self):
        model = AXK2ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        # Last-3x3 logits slice, left-padded (batch 0) and unpadded (batch 1) rows.
        EXPECTED_LOGITS_LEFT_PADDED = Expectations(
            {
                ("cuda", 8): [[0.2441, 0.1201, 0.2129], [0.0659, 0.0635, 0.0525], [-0.019, -0.1104, 0.0454]],
            }
        )
        expected_left_padded = torch.tensor(EXPECTED_LOGITS_LEFT_PADDED.get_expectation(), device=torch_device)

        EXPECTED_LOGITS_UNPADDED = Expectations(
            {
                ("cuda", 8): [[0.1152, -0.1245, -0.4258], [-0.1797, -0.3145, -0.3223], [-0.0684, 0.3418, -0.027]],
            }
        )
        expected_unpadded = torch.tensor(EXPECTED_LOGITS_UNPADDED.get_expectation(), device=torch_device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()
        torch.testing.assert_close(logits[0, -3:, -3:], expected_left_padded, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, -3:, -3:], expected_unpadded, atol=1e-3, rtol=1e-3)
