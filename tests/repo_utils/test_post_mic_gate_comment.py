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

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import post_mic_gate_comment as mod  # noqa: E402


BLOCKED = {
    "fleet": "prototypes",
    "variant": "bfloat16",
    "contract": "GPT-J",
    "target": "gpt-j-6b-mic",
    "status": "fail",
    "base_status": "pass",
    "verdict": "blocked",
    "reason": "green on the base (pass), red with this change",
    "failure": "next_token_logits: max delta 0.0156 over tolerance",
}
WARNING = {**BLOCKED, "contract": "Llama", "target": "llama-mic", "verdict": "warning"}
CODEGEN = {**BLOCKED, "contract": "CodeGen2", "target": "codegen2-16B_P-mic"}
OBSERVATION = {
    "selections": [
        {},
        {"GPT-J": "models/gptj (src/transformers/models/gptj/modeling_gptj.py)"},
        {"CodeGen2": "models/codegen (src/transformers/models/codegen/modeling_codegen.py)"},
    ]
}
SHA = "65271b5e2af765151535545bb6c86483c9d30beb"


class BlockedCasesTest(unittest.TestCase):
    def test_keeps_only_blocked_cases(self):
        cases = mod.blocked_cases({"decisions": [BLOCKED, WARNING]})
        self.assertEqual(cases, [("GPT-J", "gpt-j-6b-mic", "bfloat16", BLOCKED["failure"])])

    def test_drops_names_that_are_not_plain(self):
        hostile = [
            {**BLOCKED, "contract": "GPT-J](https://evil.example)"},
            {**BLOCKED, "target": "@huggingface/maintainers"},
            {**BLOCKED, "variant": None},
            "not a dict",
        ]
        self.assertEqual(mod.blocked_cases({"decisions": hostile}), [])

    def test_error_text_cannot_leave_its_code_span(self):
        cases = mod.blocked_cases({"decisions": [{**BLOCKED, "failure": "a `b` | c\n\n# heading " + "x" * 500}]})
        error = cases[0][3]
        self.assertNotIn("`", error)
        self.assertNotIn("|", error)
        self.assertNotIn("\n", error)
        self.assertLessEqual(len(error), mod.MAX_ERROR + 1)


class ModelNamesTest(unittest.TestCase):
    def test_uses_the_selecting_model_directory(self):
        cases = mod.blocked_cases({"decisions": [BLOCKED, CODEGEN, {**BLOCKED, "variant": "default"}]})
        self.assertEqual(mod.model_names(cases, OBSERVATION), ["gptj", "codegen"])

    def test_falls_back_to_the_contract_name(self):
        cases = mod.blocked_cases({"decisions": [BLOCKED]})
        self.assertEqual(mod.model_names(cases, None), ["GPT-J"])
        self.assertEqual(mod.model_names(cases, {"selections": [{"GPT-J": "models/../x"}]}), ["GPT-J"])


class RenderTest(unittest.TestCase):
    def render(self, enforced=True):
        cases = mod.blocked_cases({"decisions": [BLOCKED, CODEGEN]})
        return mod.render(
            cases,
            pr=51298,
            sha=SHA,
            enforced=enforced,
            run_id="45802675898",
            attempt="2",
            job_url="https://github.com/o/r/actions/runs/1/job/2",
            models=mod.model_names(cases, OBSERVATION),
        )

    def test_comment_has_errors_links_and_record_mic(self):
        body = self.render()
        self.assertTrue(body.startswith(mod.MARKER))
        self.assertIn("2 blocked case(s)", body)
        self.assertIn(
            "| GPT-J | gpt-j-6b-mic | bfloat16 | `next_token_logits: max delta 0.0156 over tolerance` |", body
        )
        self.assertIn(
            "[contracts run](https://transformers-ci.huggingface.cool/d/pytest-observability-by-run/"
            "pytest-observability-run?var-run_id=45802675898%3A2&var-pr=51298&var-gh_run_id=45802675898)",
            body,
        )
        self.assertIn("[job log](https://github.com/o/r/actions/runs/1/job/2)", body)
        self.assertIn('make record-mic MODEL="gptj codegen" PR=51298', body)
        self.assertIn("/mic?scope=prs&group=pr&q=51298", body)
        self.assertIn("?var-pr=51298", body)
        self.assertIn("record-mic: gptj codegen", body)
        self.assertIn('make run-mic MODEL="gptj codegen"', body)
        self.assertIn("can't merge", body)

    def test_shadow_mode_does_not_claim_the_pr_is_blocked_from_merging(self):
        self.assertNotIn("can't merge", self.render(enforced=False))


RECORD_REPLY = {
    "user": {"login": "github-actions[bot]", "type": "Bot"},
    "html_url": "https://github.com/huggingface/transformers/pull/51302#issuecomment-1",
    "body": "\n".join(
        [
            "**record-mic** (asked by @tarekziade): published for this PR only, from PR CI run 45802675898. "
            "They apply from the next contracts run.",
            "",
            "### record-mic for #51302: ",
            "",
            "| contract | target | variant | platform | verified | changed outputs |",
            "|---|---|---|---|---|---|",
            "| CodeGen2 | codegen2-16B_P-mic | bfloat16 | cpu-x86_64-bf16 | ✅ | `next_token_logits` max Δ 0.0117 |",
            "| GPT-J | gpt-j-6b-mic | bfloat16 | cpu-x86_64-bf16 | ✅ | `next_token_logits` max Δ 0.0156 |",
        ]
    ),
}


class RecordRepliesTest(unittest.TestCase):
    def test_reads_requester_and_contracts_from_the_record_mic_reply(self):
        self.assertEqual(
            mod.record_replies([RECORD_REPLY]), [("tarekziade", ["CodeGen2", "GPT-J"], RECORD_REPLY["html_url"])]
        )

    def test_ignores_the_same_text_from_a_user(self):
        spoof = {**RECORD_REPLY, "user": {"login": "someone", "type": "User"}}
        self.assertEqual(mod.record_replies([spoof]), [])

    def test_ignores_failed_requests(self):
        failed = {**RECORD_REPLY, "body": "**record-mic** (asked by @tarekziade): failed: no blocked case"}
        self.assertEqual(mod.record_replies([failed]), [])

    def test_resolved_comment_credits_without_pinging(self):
        body = mod.render_resolved(SHA, mod.record_replies([RECORD_REPLY]))
        self.assertIn("no blocked case", body)
        self.assertIn("- CodeGen2, GPT-J: requested by [tarekziade](https://github.com/tarekziade)", body)
        self.assertNotIn("@tarekziade", body)

    def test_resolved_comment_without_record_mic(self):
        self.assertNotIn("record-mic", mod.render_resolved(SHA))


class MainTest(unittest.TestCase):
    def run_main(self, gate, existing=None, open_pr=True, others=()):
        calls = []

        def get(url, token=None):
            if "/pulls?" in url:
                return [{"number": 51298, "head": {"sha": SHA if open_pr else "0" * 40}}]
            if "/jobs?" in url:
                return {"jobs": [{"name": "pr-ci / fast-integration / transformers", "html_url": "https://job"}]}
            if "/comments?" in url:
                return ([existing] if existing else []) + list(others)
            raise AssertionError(url)

        def request(url, token=None, method="GET", payload=None):
            calls.append((method, url, payload["body"]))
            return {"html_url": "https://comment"}

        env = {
            "GITHUB_TOKEN": "t",
            "GITHUB_REPOSITORY": "huggingface/transformers",
            "RUN_ID": "1",
            "HEAD_SHA": SHA,
            "HEAD_OWNER": "someone",
            "HEAD_BRANCH": "branch",
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "contracts-gate.json").write_text(json.dumps(gate), encoding="utf-8")
            Path(tmp, "contracts-observation.json").write_text(json.dumps(OBSERVATION), encoding="utf-8")
            with (
                patch.dict(os.environ, env),
                patch.object(sys, "argv", ["x", tmp]),
                patch.object(mod, "get_github_json", get),
                patch.object(mod, "github_request", request),
            ):
                self.assertEqual(mod.main(), 0)
        return calls

    def test_posts_a_new_comment_for_blocked_cases(self):
        calls = self.run_main({"enforced": True, "decisions": [BLOCKED]})
        self.assertEqual(len(calls), 1)
        method, url, body = calls[0]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/issues/51298/comments"))
        self.assertIn("record-mic: gptj", body)

    def test_updates_the_existing_comment(self):
        existing = {"user": {"type": "Bot"}, "body": mod.MARKER + "\nold", "url": "https://api/c/9", "html_url": "h"}
        calls = self.run_main({"decisions": [BLOCKED]}, existing=existing)
        self.assertEqual([(m, u) for m, u, _ in calls], [("PATCH", "https://api/c/9")])

    def test_marks_the_comment_resolved_when_nothing_is_blocked(self):
        existing = {"user": {"type": "Bot"}, "body": mod.MARKER + "\nold", "url": "https://api/c/9", "html_url": "h"}
        calls = self.run_main({"decisions": [WARNING]}, existing=existing)
        self.assertEqual(len(calls), 1)
        self.assertIn("no blocked case", calls[0][2])

    def test_resolved_comment_names_the_record_mic_request(self):
        existing = {"user": {"type": "Bot"}, "body": mod.MARKER + "\nold", "url": "https://api/c/9", "html_url": "h"}
        calls = self.run_main({"decisions": []}, existing=existing, others=[RECORD_REPLY])
        self.assertIn("requested by [tarekziade]", calls[0][2])

    def test_posts_nothing_without_blocked_cases_or_comment(self):
        self.assertEqual(self.run_main({"decisions": [WARNING]}), [])

    def test_ignores_a_comment_marker_from_a_user(self):
        spoof = {"user": {"type": "User"}, "body": mod.MARKER, "url": "https://api/c/1", "html_url": "h"}
        calls = self.run_main({"decisions": [BLOCKED]}, existing=spoof)
        self.assertEqual(calls[0][0], "POST")

    def test_skips_a_run_whose_commit_is_no_longer_the_pr_head(self):
        self.assertEqual(self.run_main({"decisions": [BLOCKED]}, open_pr=False), [])


if __name__ == "__main__":
    unittest.main()
