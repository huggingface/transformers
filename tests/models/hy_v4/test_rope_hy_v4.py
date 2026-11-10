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

import unittest

import torch

from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config
from transformers.models.hy_v4.modeling_hy_v4 import HYV4ForCausalLM, HYV4RotaryEmbedding, apply_rotary_pos_emb


class HYV4RotaryTest(unittest.TestCase):
    def test_checkpoint_coordinates_match_complex_rotation(self):
        torch.manual_seed(123)
        config = HYV4Config(qk_rope_head_dim=8, rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
        positions = torch.tensor([[0, 1, 7, 2048, 32767], [3, 4, 9, 4096, 65535]])
        rotary = HYV4RotaryEmbedding(config)
        cos, sin = rotary(torch.zeros(2, 5, 8), positions)
        # Independent adjacent-pair complex arithmetic in checkpoint coordinates.
        angles = positions[..., None].float() * (10000.0 ** (-torch.arange(0, 8, 2).float() / 8))
        phase = torch.polar(torch.ones_like(angles), angles)
        for axis in (1, 2):
            with self.subTest(head_axis=axis):
                shape = (2, 3, 5, 8) if axis == 1 else (2, 5, 3, 8)
                key_shape = (2, 1, 5, 8) if axis == 1 else (2, 5, 1, 8)
                inputs = [torch.randn(s, requires_grad=True) for s in (shape, key_shape)]
                refs = [x.detach().clone().requires_grad_() for x in inputs]
                expected = [
                    torch.view_as_real(
                        torch.view_as_complex(x.reshape(*x.shape[:-1], 4, 2)) * phase.unsqueeze(axis)
                    ).flatten(-2)
                    for x in refs
                ]
                actual = apply_rotary_pos_emb(*inputs, cos, sin, unsqueeze_dim=axis)
                upstream = [torch.randn_like(x) for x in actual]
                for output, reference in zip(actual, expected):
                    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
                sum((x * w).sum() for x, w in zip(actual, upstream)).backward()
                sum((x * w).sum() for x, w in zip(expected, upstream)).backward()
                for x, ref in zip(inputs, refs):
                    torch.testing.assert_close(x.grad, ref.grad, atol=2e-6, rtol=2e-6)

    def test_decode_positions_match_prefill(self):
        config = HYV4Config(qk_rope_head_dim=8)
        rotary = HYV4RotaryEmbedding(config)
        x = torch.randn(1, 2, 7, 8)
        positions = torch.arange(7).unsqueeze(0)
        cos, sin = rotary(x, positions)
        prefill, _ = apply_rotary_pos_emb(x, x, cos, sin)
        for start in (1, 4, 6):
            token = x[:, :, start : start + 1]
            cos, sin = rotary(token, positions[:, start : start + 1])
            decode, _ = apply_rotary_pos_emb(token, token, cos, sin)
            torch.testing.assert_close(decode, prefill[:, :, start : start + 1], atol=0, rtol=0)

    def test_model_cached_decode_matches_full_sequence(self):
        torch.manual_seed(42)
        config = HYV4Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            index_head_dim=16,
            index_n_heads=4,
            index_topk=4,
            hc_mult=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "eager"
        model = HYV4ForCausalLM(config).eval()
        tokens = torch.tensor([[1, 5, 7, 8, 9, 10, 11]])
        with torch.no_grad():
            expected = model(tokens, use_cache=False).logits
            prefill = model(tokens[:, :4], use_cache=True)
            actual = [prefill.logits]
            cache = prefill.past_key_values
            for position in range(4, tokens.shape[1]):
                output = model(tokens[:, position : position + 1], past_key_values=cache, use_cache=True)
                actual.append(output.logits)
                cache = output.past_key_values
        torch.testing.assert_close(torch.cat(actual, dim=1), expected, atol=1e-5, rtol=1e-4)
