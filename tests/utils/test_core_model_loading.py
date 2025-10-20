# Copyright 2019 HuggingFace Inc.
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

from transformers.core_model_loading import build_glob_alt, match_glob


class TestWeightGlobMatching(unittest.TestCase):
    def setUp(self):
        self.weight_globs_digits = [
            "model.layers.*.mlp.gate_up_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "embed_tokens.weight",
        ]
        self.alt_digits, self.map_digits = build_glob_alt(self.weight_globs_digits, digits_only=True)

        self.weight_globs_any = [
            "model.layers.*.mlp.gate_up_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "embed_tokens.weight",
        ]
        self.alt_any, self.map_any = build_glob_alt(self.weight_globs_any, digits_only=False)

    def test_exact_match(self):
        self.assertEqual(match_glob("embed_tokens.weight", self.alt_digits, self.map_digits), "embed_tokens.weight")

    def test_digits_only_star_accepts_digits(self):
        self.assertEqual(
            match_glob("model.layers.0.mlp.gate_up_proj.weight", self.alt_digits, self.map_digits),
            "model.layers.*.mlp.gate_up_proj.weight",
        )
        self.assertEqual(
            match_glob("model.layers.12.self_attn.q_proj.weight", self.alt_digits, self.map_digits),
            "model.layers.*.self_attn.q_proj.weight",
        )

    def test_digits_only_star_rejects_nondigits(self):
        # 'a' is not digits, so it should not match with digits_only=True
        self.assertIsNone(match_glob("model.layers.a.mlp.gate_up_proj.weight", self.alt_digits, self.map_digits))

    def test_anychar_star_accepts_nondigits(self):
        self.assertEqual(
            match_glob("model.layers.a.mlp.gate_up_proj.weight", self.alt_any, self.map_any),
            "model.layers.*.mlp.gate_up_proj.weight",
        )
        self.assertEqual(
            match_glob("model.layers.00x.mlp.gate_up_proj.weight", self.alt_any, self.map_any),
            "model.layers.*.mlp.gate_up_proj.weight",
        )

    def test_no_match(self):
        self.assertIsNone(match_glob("model.layers.0.mlp.up_proj.weight", self.alt_digits, self.map_digits))

    def test_leftmost_alternative_wins_for_overlapping_patterns(self):
        # Overlapping patterns: both could match; ensure leftmost wins
        globs = [
            "model.layers.*.mlp.*.weight",  # broader (first)
            "model.layers.0.mlp.gate_up_proj.weight",  # more specific (second)
        ]
        alt, mapping = build_glob_alt(globs, digits_only=False)

        # Both branches match; Python's regex picks the leftmost alternative → index 0
        self.assertEqual(
            match_glob("model.layers.0.mlp.gate_up_proj.weight", alt, mapping), "model.layers.*.mlp.*.weight"
        )

    def test_multiple_patterns_same_prefix(self):
        globs = [
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight",
        ]
        alt, mapping = build_glob_alt(globs, digits_only=True)

        self.assertEqual(
            match_glob("model.layers.3.self_attn.q_proj.weight", alt, mapping),
            "model.layers.*.self_attn.q_proj.weight",
        )
        self.assertEqual(
            match_glob("model.layers.3.self_attn.k_proj.weight", alt, mapping),
            "model.layers.*.self_attn.k_proj.weight",
        )
        self.assertEqual(
            match_glob("model.layers.3.self_attn.v_proj.weight", alt, mapping),
            "model.layers.*.self_attn.v_proj.weight",
        )

    def test_anchor_full_match_only(self):
        # Make sure partial strings don't match—anchors ^...$ are in each branch
        self.assertIsNone(match_glob("foo.model.layers.0.mlp.gate_up_proj.weight.bar", self.alt_any, self.map_any))

    def test_large_batch_performance_smoke(self):
        # Not a perf benchmark, but ensures building and matching a larger alternation is OK
        globs = [f"model.layers.*.mlp.block{i}.weight" for i in range(200)]
        alt, mapping = build_glob_alt(globs, digits_only=True)
        key = "model.layers.123.mlp.block57.weight"
        self.assertEqual(match_glob(key, alt, mapping), "model.layers.*.mlp.block57.weight")


if __name__ == "__main__":
    unittest.main()
