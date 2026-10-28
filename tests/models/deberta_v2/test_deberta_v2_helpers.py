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

"""Lightweight helper tests for DeBERTa-v2 (no pipeline/datasets deps)."""

import importlib
import unittest
import warnings

from transformers.testing_utils import require_torch


@require_torch
class DebertaV2JitScriptRemovalTest(unittest.TestCase):
    """Regression for https://github.com/huggingface/transformers/issues/50069."""

    def test_import_does_not_emit_jit_script_deprecation(self):
        import torch
        import transformers.models.deberta_v2.modeling_deberta_v2 as mod

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            importlib.reload(mod)

        for name in (
            "make_log_bucket_position",
            "c2p_dynamic_expand",
            "p2c_dynamic_expand",
            "pos_dynamic_expand",
            "scaled_size_sqrt",
            "build_rpos",
        ):
            fn = getattr(mod, name)
            self.assertFalse(
                isinstance(fn, torch.jit.ScriptFunction),
                f"{name} should be a plain Python function, not torch.jit.script",
            )

    def test_helpers_match_torchscript_when_scriptable(self):
        import torch
        from transformers.models.deberta_v2.modeling_deberta_v2 import (
            build_relative_position,
            build_rpos,
            c2p_dynamic_expand,
            make_log_bucket_position,
            p2c_dynamic_expand,
            pos_dynamic_expand,
            scaled_size_sqrt,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            j_make = torch.jit.script(make_log_bucket_position)
            j_c2p = torch.jit.script(c2p_dynamic_expand)
            j_p2c = torch.jit.script(p2c_dynamic_expand)
            j_pos = torch.jit.script(pos_dynamic_expand)
            j_scaled = torch.jit.script(scaled_size_sqrt)
            j_rpos = torch.jit.script(build_rpos)

        torch.manual_seed(0)
        relative_pos = torch.randint(-128, 128, (4, 16, 16))
        self.assertTrue(
            torch.equal(make_log_bucket_position(relative_pos, 32, 512), j_make(relative_pos, 32, 512))
        )

        B, H, Q, K, D = 2, 3, 5, 7, 8
        query = torch.randn(B, H, Q, D)
        key = torch.randn(B, H, K, D)
        relative = torch.randint(0, 10, (B, H, Q, K))
        c2p_pos = torch.randn(1, 1, Q, K)
        self.assertTrue(torch.equal(c2p_dynamic_expand(c2p_pos, query, relative), j_c2p(c2p_pos, query, relative)))
        p2c_pos = torch.randn(1, 1, 1, K)
        self.assertTrue(torch.equal(p2c_dynamic_expand(p2c_pos, query, key), j_p2c(p2c_pos, query, key)))
        p2c_att = torch.randn(B, H, Q, K)
        pos_index = torch.randint(0, Q, (B, H, Q, 1))
        self.assertTrue(torch.equal(pos_dynamic_expand(pos_index, p2c_att, key), j_pos(pos_index, p2c_att, key)))
        self.assertTrue(torch.equal(scaled_size_sqrt(query, 1), j_scaled(query, 1)))

        key_same = torch.randn(B, H, Q, D)
        rel = build_relative_position(query, key_same, bucket_size=32, max_position=512)
        self.assertTrue(torch.equal(build_rpos(query, key_same, rel, 32, 512), j_rpos(query, key_same, rel, 32, 512)))
        rel2 = build_relative_position(query, key, bucket_size=32, max_position=512)
        self.assertTrue(torch.equal(build_rpos(query, key, rel2, 32, 512), j_rpos(query, key, rel2, 32, 512)))
