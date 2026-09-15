"""Focused regression tests for ZGCM's mixed attention and gated checkpoint layout."""

import tempfile
import unittest

import torch

from transformers import AutoModelForCausalLM, ZgcmConfig, ZgcmForCausalLM


class ZgcmModelTest(unittest.TestCase):
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
        full = self.model(self.tokens, use_cache=False).logits
        prefix = self.model(self.tokens[:, :6], use_cache=True)
        for i in range(6, 9):
            step = self.model(self.tokens[:, i : i + 1], past_key_values=prefix.past_key_values, use_cache=True)
            torch.testing.assert_close(step.logits[:, 0], full[:, i], atol=1e-6, rtol=1e-5)

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
