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

from transformers import AutoModelForCausalLM, AutoTokenizer, Cache, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AXK2Model


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

    @unittest.skip("Mask is built per layer no matter what but FA backend needs no mask")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@slow
@require_torch_accelerator
class AXK1IntegrationTest(unittest.TestCase):
    model_id = "hf-internal-testing/tiny-axk2"

    def setup(self):
        cleanup(torch_device, gc_collect=False)

    def tearDown(self):
        cleanup(torch_device, gc_collect=False)

    def test_model_logits_batched(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")

        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(model.device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        # Last-3x3 logits slice, left-padded (batch 0) and unpadded (batch 1) rows.
        EXPECTED_LOGITS_LEFT_PADDED = Expectations(
            {("cuda", (8, 6)): [[-1.9062, -3.9688, 2.8438], [-3.5625, -1.6562, 4.2500], [-1.6172, -2.7812, 2.6094]]}
        )
        expected_left_padded = torch.tensor(EXPECTED_LOGITS_LEFT_PADDED.get_expectation(), device=model.device)
        EXPECTED_LOGITS_UNPADDED = Expectations(
            {("cuda", (8, 6)): [[0.6211, -0.4336, 1.8906], [-3.4219, -1.9219, 2.7188], [-2.0156, -1.5547, -1.3906]]}
        )
        expected_unpadded = torch.tensor(EXPECTED_LOGITS_UNPADDED.get_expectation(), device=model.device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()
        torch.testing.assert_close(logits[0, -3:, -3:], expected_left_padded, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, -3:, -3:], expected_unpadded, atol=1e-3, rtol=1e-3)

    def test_model_generation(self):
        expected_texts = Expectations(
            {
                ("cuda", (8, 6)): 'Tell me about the french revolution. 세상은됨에 Philipp{asày 값에서 쪽은Pkgày속성amentals년여 focalaure 달간を実{acknowledgements 사건과-OctCTPコロ passengers Dice GD workloads 울진 Fibonacci announcesdest denote 이야기도 scrap',
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()

        tokenizer = AutoTokenizer.from_pretrained("skt/A.X-K1")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", dtype="auto", experts_implementation="eager"
        )
        input_text = ["Tell me about the french revolution."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=32, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)
