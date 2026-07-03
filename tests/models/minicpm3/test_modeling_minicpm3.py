# Copyright 2026 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
from transformers.testing_utils import Expectations, require_torch, require_torch_accelerator, slow, torch_device

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

    @unittest.skip(
        reason="MiniCPM3 uses MLA so the query/key and value head dims differ, which flash can't dispatch on"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@slow
@require_torch
class MiniCPM3IntegrationTest(unittest.TestCase):
    model_id = "openbmb/MiniCPM3-4B"

    @require_torch_accelerator
    def test_minicpm3_4b_logits(self):
        input_ids = torch.tensor([[1, 306, 4658, 278, 6593, 310, 2834, 338]], device=torch_device)
        model = MiniCPM3ForCausalLM.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        with torch.no_grad():
            logits = model(input_ids).logits.float()

        # Slice of the last-token logits. Reference values come from an A100 (bf16) run; the
        # maintainer can adjust per-hardware entries as needed (see `Expectations`).
        expected_slices = Expectations(
            {
                ("cuda", 8): [0.765625, 3.640625, -0.189453125, -0.8359375, -0.8359375],
                ("cuda", (8, 6)): [0.7344, 3.6562, -0.1060, -0.8633, -0.8633],
            }
        )  # fmt: skip
        expected = expected_slices.get_expectation()
        torch.testing.assert_close(
            logits[0, -1, :5].cpu(),
            torch.tensor(expected),
            atol=1e-3,
            rtol=1e-3,
        )

    @require_torch_accelerator
    def test_minicpm3_4b_generation(self):
        expected_texts = Expectations(
            {
                ("cuda", 8): "My favourite condiment is \n[A]. ketchup \n[B]. mustard \n[C]. mayonnaise \n[D]. must",
            }
        )  # fmt: skip
        expected_text = expected_texts.get_expectation()

        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
        model = MiniCPM3ForCausalLM.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(input_ids, max_new_tokens=32, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(text, expected_text)
