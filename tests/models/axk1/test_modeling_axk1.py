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
"""Testing suite for the PyTorch A.X-K1 model."""

import unittest

from transformers import AutoTokenizer, is_torch_available
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

    torch.set_float32_matmul_precision("highest")

    from transformers import (
        AXK1ForCausalLM,
        AXK1Model,
        Cache,
    )


class AXK1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = AXK1Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_intermediate_size=16,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
    ):
        super().__init__(parent=parent)
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_intermediate_size = moe_intermediate_size
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim


@require_torch
class AXK1ModelTest(CausalLMModelTest, unittest.TestCase):
    # Routed experts that receive no token in a step get no gradient.
    test_all_params_have_gradient = False
    model_tester_class = AXK1ModelTester

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as A.X-K1 has the MLA cache format (keys carry the rope+nope dims)."""
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

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims (MLA qk/v dims differ)")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@slow
class AXK1IntegrationTest(unittest.TestCase):
    # The released checkpoint is a ~1T-parameter MoE: these tests need a large multi-GPU node
    # (e.g. 16x80GB with `tp_plan="auto"` or `device_map="auto"`).
    def tearDown(self):
        cleanup(torch_device, gc_collect=False)

    def _generate(self, cache_implementation=None):
        tokenizer = AutoTokenizer.from_pretrained("skt/A.X-K1")
        model = AXK1ForCausalLM.from_pretrained("skt/A.X-K1", dtype=torch.bfloat16, device_map="auto")

        messages = [{"role": "user", "content": "대한민국의 수도는?"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        generate_kwargs = {"max_new_tokens": 32, "do_sample": False}
        if cache_implementation is not None:
            generate_kwargs["cache_implementation"] = cache_implementation
        generated_ids = model.generate(**inputs, **generate_kwargs)
        return tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    @require_torch_accelerator
    def test_dynamic_cache(self):
        expected_texts = Expectations(
            {
                ("cuda", 8): "대한민국의 수도는 **서울특별시**입니다.",
            }
        )
        EXPECTED_TEXT_COMPLETION = expected_texts.get_expectation()

        dynamic_text = self._generate()
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text.strip())

    @require_torch_accelerator
    def test_static_cache(self):
        expected_texts = Expectations(
            {
                ("cuda", 8): "대한민국의 수도는 **서울특별시**입니다.",
            }
        )
        EXPECTED_TEXT_COMPLETION = expected_texts.get_expectation()

        static_text = self._generate(cache_implementation="static")
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text.strip())

    @require_torch_accelerator
    def test_model_logits_batched(self):
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = AXK1ForCausalLM.from_pretrained("skt/A.X-K1", dtype=torch.bfloat16, device_map="auto")

        EXPECTED_LOGITS_LEFT_PADDED = Expectations(
            {
                ("cuda", 8): [[236.0, 235.0, 236.0], [234.0, 234.0, 235.0], [372.0, 372.0, 372.0]],
            }
        )
        expected_left_padded = torch.tensor(EXPECTED_LOGITS_LEFT_PADDED.get_expectation(), device=torch_device)

        EXPECTED_LOGITS_UNPADDED = Expectations(
            {
                ("cuda", 8): [[368.0, 368.0, 368.0], [370.0, 372.0, 372.0], [372.0, 372.0, 372.0]],
            }
        )
        expected_unpadded = torch.tensor(EXPECTED_LOGITS_UNPADDED.get_expectation(), device=torch_device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()
        torch.testing.assert_close(logits[0, -3:, -3:], expected_left_padded, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, -3:, -3:], expected_unpadded, atol=1e-3, rtol=1e-3)
