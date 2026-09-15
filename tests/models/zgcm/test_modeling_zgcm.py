# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the PyTorch ZGCM model."""

import tempfile
import unittest

from transformers import AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import ZgcmConfig, ZgcmForCausalLM, ZgcmModel


class ZgcmModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = ZgcmModel

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            hidden_act="silu",
            num_key_value_heads=1,
            # Exercise window eviction separately from the common generation shape checks.
            sliding_window=128,
            window_attn_skip_freq=2,
            attention_probs_dropout_prob=0.0,
        )


@require_torch
class ZgcmModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = ZgcmModelTester


@require_torch
class ZgcmAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = ZgcmConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            sliding_window=4,
            window_attn_skip_freq=2,
            pad_token_id=0,
            eos_token_id=1,
        )
        self.model = ZgcmForCausalLM(self.config).eval()
        self.tokens = torch.randint(2, 64, (2, 9))

    @torch.no_grad()
    def test_cache_beyond_sliding_window(self):
        for backend in ("eager", "sdpa"):
            for prefix_length in (3, 4, 6):
                with self.subTest(backend=backend, prefix_length=prefix_length):
                    self.model.set_attn_implementation(backend)
                    full = self.model(self.tokens, use_cache=False).logits
                    cache = self.model(self.tokens[:, :prefix_length], use_cache=True).past_key_values
                    for i in range(prefix_length, self.tokens.shape[1]):
                        step = self.model(self.tokens[:, i : i + 1], past_key_values=cache, use_cache=True)
                        cache = step.past_key_values
                        torch.testing.assert_close(step.logits[:, 0], full[:, i], atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_generation_beyond_sliding_window(self):
        # Like Gemma3, exercise both layer types with a prompt longer than the window.
        self.assertGreater(self.tokens.shape[1], self.config.sliding_window)
        self.assertEqual(self.config.layer_types, ["sliding_attention", "full_attention"])
        mask = torch.ones_like(self.tokens)
        mask[0, :2] = 0
        tokens = self.tokens.masked_fill(mask == 0, self.config.pad_token_id)
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                self.model.set_attn_implementation(backend)
                kwargs = {"attention_mask": mask, "max_new_tokens": 5, "do_sample": False, "eos_token_id": None}
                expected = self.model.generate(tokens, use_cache=False, **kwargs)
                # Repeat generation to catch cache state leaking between calls.
                for _ in range(2):
                    actual = self.model.generate(tokens, use_cache=True, **kwargs)
                    self.assertEqual(actual.shape[1], tokens.shape[1] + 5)
                    torch.testing.assert_close(actual, expected)

    @torch.no_grad()
    def test_attention_backends_and_padding(self):
        mask = torch.ones_like(self.tokens)
        mask[0, :2] = 0
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        self.model.set_attn_implementation("eager")
        eager = self.model(self.tokens, attention_mask=mask, position_ids=positions, use_cache=False).logits
        self.model.set_attn_implementation("sdpa")
        sdpa = self.model(self.tokens, attention_mask=mask, position_ids=positions, use_cache=False).logits
        torch.testing.assert_close(eager[mask.bool()], sdpa[mask.bool()], atol=1e-6, rtol=1e-5)

    def test_save_load_auto_and_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            loaded = AutoModelForCausalLM.from_pretrained(directory)
            self.assertIsInstance(loaded, ZgcmForCausalLM)
            torch.testing.assert_close(self.model(self.tokens).logits, loaded(self.tokens).logits)
        self.model.train()
        self.model.gradient_checkpointing_enable()
        loss = self.model(self.tokens, labels=self.tokens, use_cache=False).loss
        loss.backward()
        self.assertIsNotNone(self.model.model.layers[0].self_attn.g_proj.weight.grad)

    def test_configuration(self):
        self.assertEqual(self.model.config._attn_implementation, "sdpa")
        self.assertEqual(self.config.layer_types, ["sliding_attention", "full_attention"])
        self.assertEqual(self.config.attention_gate_layers, [True, False])
        with self.assertRaises(ValueError):
            ZgcmConfig(layer_types=[])
        with self.assertRaises(ValueError):
            ZgcmConfig(attention_gate_layers=[])
