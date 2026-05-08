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

import math
import tempfile
import unittest

import torch

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.integrations.mup import MuReadout, build_mup_param_groups, coord_check


def _max_ratio(records_by_width: dict) -> float:
    """Across modules, max over widths of (mean|act|(width) / mean|act|(min width))."""
    widths = sorted(records_by_width.keys())
    base = widths[0]
    layers = set(records_by_width[base].keys())
    for w in widths[1:]:
        layers &= set(records_by_width[w].keys())
    worst = 0.0
    for layer in layers:
        base_val = records_by_width[base][layer][-1]
        if base_val == 0:
            continue
        for w in widths[1:]:
            val = records_by_width[w][layer][-1]
            ratio = max(val / base_val, base_val / val)
            worst = max(worst, ratio)
    return worst


def _make_llama(width: int, mup: bool, base_width: int = 64, vocab: int = 64):
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=width,
        intermediate_size=2 * width,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
        mup=mup,
        mup_base_width=base_width if mup else None,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


class MupLlamaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_readout_swapped_when_mup(self):
        m_sp = _make_llama(width=64, mup=False)
        m_mu = _make_llama(width=128, mup=True, base_width=64)
        self.assertNotIsInstance(m_sp.lm_head, MuReadout)
        self.assertIsInstance(m_mu.lm_head, MuReadout)
        self.assertAlmostEqual(m_mu.lm_head.width_mult, 2.0)

    def test_attention_scale_uses_inverse_d_head(self):
        m = _make_llama(width=64, mup=True, base_width=64)
        attn = m.model.layers[0].self_attn
        self.assertAlmostEqual(attn.scaling, 1.0 / attn.head_dim)

    def test_apply_mup_init_rescales_hidden_weights(self):
        torch.manual_seed(0)
        base = _make_llama(width=64, mup=True, base_width=64)
        torch.manual_seed(0)
        wide = _make_llama(width=256, mup=True, base_width=64)
        std_base = base.model.layers[0].mlp.gate_proj.weight.detach().float().std().item()
        std_wide = wide.model.layers[0].mlp.gate_proj.weight.detach().float().std().item()
        self.assertAlmostEqual(std_wide, std_base / math.sqrt(4.0), places=2)

    def test_param_groups_apply_mu_adam_rule(self):
        # width_mult = 2: matrix-like (hidden Linear weights) get lr/m, vector-like (readout weight,
        # embeddings, biases, LayerNorm) keep lr. Mirrors `mup.MuAdam`.
        m = _make_llama(width=128, mup=True, base_width=64)
        groups = build_mup_param_groups(m, lr=1e-3)
        matrix_group, vector_group = groups
        self.assertAlmostEqual(matrix_group["lr"], 5e-4)
        self.assertAlmostEqual(vector_group["lr"], 1e-3)
        q_proj_w = m.model.layers[0].self_attn.q_proj.weight
        self.assertTrue(any(p is q_proj_w for p in matrix_group["params"]))
        # The (tied) embedding/lm_head weight is vector-like under μP.
        self.assertTrue(any(p is m.model.embed_tokens.weight for p in vector_group["params"]))

    def test_coord_check_mup_flatter_than_sp(self):
        # Empirical correctness: across widths, μP keeps activation magnitudes much closer to
        # width-invariant than SP after a few optimizer steps.
        torch.manual_seed(0)
        widths = [32, 64, 128]
        input_ids = torch.randint(0, 32, (2, 16))
        batch = {"input_ids": input_ids, "labels": input_ids.clone()}

        def factory_mup(w):
            return _make_llama(width=w, mup=True, base_width=widths[0], vocab=32)

        def factory_sp(w):
            return _make_llama(width=w, mup=False, vocab=32)

        rec_mup = coord_check(factory_mup, widths=widths, batch=batch, n_steps=3, lr=1e-3)
        rec_sp = coord_check(factory_sp, widths=widths, batch=batch, n_steps=3, lr=1e-3)

        ratio_mup = _max_ratio(rec_mup)
        ratio_sp = _max_ratio(rec_sp)
        self.assertLess(ratio_mup, ratio_sp)

    def test_save_load_round_trip(self):
        m = _make_llama(width=128, mup=True, base_width=64)
        m.eval()
        input_ids = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            expected = m(input_ids=input_ids).logits

        with tempfile.TemporaryDirectory() as tmpdir:
            m.save_pretrained(tmpdir)
            loaded = LlamaForCausalLM.from_pretrained(tmpdir, attn_implementation="eager")

        self.assertIsInstance(loaded.lm_head, MuReadout)
        self.assertAlmostEqual(loaded.lm_head.width_mult, 2.0)
        attn = loaded.model.layers[0].self_attn
        self.assertAlmostEqual(attn.scaling, 1.0 / attn.head_dim)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
