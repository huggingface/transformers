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
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_poster():
    """`utils/` is not an importable package, so load the script by path."""
    path = REPO_ROOT / "utils" / "post_mlinter_review.py"
    spec = importlib.util.spec_from_file_location("post_mlinter_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


poster = _load_poster()

FINDINGS = [
    {
        "path": "src/transformers/models/acme/modeling_acme.py",
        "line": 10,
        "rule": "TRF023",
        "message": "`nn.Linear` is built with the hardcoded dimension(s) 768.",
    },
    {
        "path": "src/transformers/models/acme/modeling_acme.py",
        "line": 16,
        "rule": "TRF026",
        "message": "bare `assert` in a model file.",
    },
]
RULES = {
    "TRF023": {
        "description": "Layer dimensions must come from the config.",
        "why_bad": "A hardcoded width pins the module to one checkpoint size.",
        "diff": "-nn.Linear(768, 768)\n+nn.Linear(config.hidden_size, config.hidden_size)",
    },
    "TRF026": {"description": "No bare assert.", "why_bad": "`python -O` strips asserts.", "diff": ""},
}


class CommentableLinesTest(unittest.TestCase):
    def test_counts_added_and_context_lines_only(self):
        patch = "@@ -1,3 +1,6 @@\n context1\n+added2\n+added3\n-deleted\n context4\n"
        self.assertEqual(poster.commentable_lines(patch), {1, 2, 3, 4})

    def test_restarts_at_each_hunk_header(self):
        patch = "@@ -1,1 +1,1 @@\n c1\n@@ -20,2 +30,3 @@\n c30\n+a31\n"
        self.assertEqual(poster.commentable_lines(patch), {1, 30, 31})

    def test_handles_single_line_hunk_header_without_count(self):
        self.assertEqual(poster.commentable_lines("@@ -1 +7 @@\n+added7\n"), {7})

    def test_handles_missing_and_malformed_patches(self):
        self.assertEqual(poster.commentable_lines(None), set())
        self.assertEqual(poster.commentable_lines(""), set())
        self.assertEqual(poster.commentable_lines("@@ garbage @@\n+line\n"), set())


class RenderingTest(unittest.TestCase):
    def test_review_body_carries_marker_and_rule_table(self):
        body = poster.review_body(FINDINGS, RULES, inline_count=2, skipped=[], version="0.1.2")
        self.assertIn(poster.MARKER, body)
        self.assertIn("TRF023", body)
        self.assertIn("TRF026", body)
        self.assertIn("2 item(s)", body)
        self.assertNotIn("point at lines outside this diff", body)

    def test_review_body_lists_findings_it_could_not_anchor(self):
        body = poster.review_body(FINDINGS, RULES, inline_count=1, skipped=FINDINGS[1:], version="0.1.2")
        self.assertIn("1 item(s) point at lines outside this diff", body)
        self.assertIn("modeling_acme.py:16", body)

    def test_review_body_escapes_pipes_in_descriptions(self):
        rules = {"TRF023": {"description": "a | b", "why_bad": "", "diff": ""}}
        body = poster.review_body(FINDINGS[:1], rules, inline_count=0, skipped=[], version="0.1.2")
        self.assertIn("a \\| b", body)

    def test_comment_body_includes_rationale_and_the_escape_hatch(self):
        body = poster.comment_body(FINDINGS[0], RULES)
        self.assertIn("**TRF023**", body)
        self.assertIn("pins the module to one checkpoint size", body)
        self.assertIn("```diff", body)
        self.assertIn("# trf-ignore: TRF023", body)

    def test_comment_body_without_a_rule_spec_is_still_valid(self):
        body = poster.comment_body({"rule": "TRF999", "message": "something"}, {})
        self.assertIn("**TRF999**", body)
        self.assertNotIn("<details>", body)


if __name__ == "__main__":
    unittest.main()
