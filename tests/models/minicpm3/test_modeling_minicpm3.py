# Copyright 2025 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniCPM3 model."""

import unittest

from transformers import Cache, is_torch_available
from transformers.testing_utils import require_torch, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, MiniCPM3ForCausalLM, MiniCPM3Model


class MiniCPM3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniCPM3Model

    def __init__(
        self,
        parent,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
    ):
        super().__init__(parent=parent)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim


@require_torch
class MiniCPM3ModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = MiniCPM3ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = MiniCPM3ForCausalLM if is_torch_available() else None

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as MiniCPM3 has a special MLA cache format inherited from DeepSeek-V2."""
        self.assertIsInstance(past_key_values, Cache)

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

    def test_tp_plan_matches_params(self):
        """Need to overwrite as the plan contains keys that are valid but depend on some configs flags and cannot
        be valid all at the same time"""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        if config.q_lora_rank is not None:
            config.base_model_tp_plan.pop("layers.*.self_attn.q_proj")
        super().test_tp_plan_matches_params()
        config.base_model_tp_plan.update({"layers.*.self_attn.q_proj": "colwise"})


@require_torch
class MiniCPM3IntegrationTest(unittest.TestCase):
    @slow
    def test_minicpm3_4b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = MiniCPM3ForCausalLM.from_pretrained("openbmb/MiniCPM3-4B", device_map="auto")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        self.assertEqual(out.shape[-1], model.config.vocab_size)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    @slow
    def test_minicpm3_4b_generation(self):
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM3-4B", use_fast=False)
        model = MiniCPM3ForCausalLM.from_pretrained("openbmb/MiniCPM3-4B", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertTrue(text.startswith(prompt))
        self.assertGreater(len(text), len(prompt))
