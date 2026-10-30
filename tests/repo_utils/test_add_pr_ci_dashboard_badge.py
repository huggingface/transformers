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
import unittest
from unittest.mock import Mock, mock_open, patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import add_pr_ci_dashboard_badge as badge_mod  # noqa: E402
from add_pr_ci_dashboard_badge import (  # noqa: E402
    BADGE_END,
    BADGE_START,
    inject_ci_badge,
    render_ci_badge,
    replace_marked_block,
)


class RenderBadgeTest(unittest.TestCase):
    def test_static_badge_links_to_the_requested_pr(self):
        for pr in (123, 999):
            url = f"https://dash.example/d/x?var-pr={pr}"
            badge = render_ci_badge(url)
            self.assertEqual(
                badge,
                f"{BADGE_START}\n[![🤗 Transformers CI]({badge_mod.BADGE_URL})]({url})\n{BADGE_END}",
            )
            self.assertNotIn("event=", badge)
            self.assertNotIn("/badge/pr", badge)

    def test_static_badge_replaces_old_cpu_gpu_header_without_changing_description(self):
        old = (
            f"{BADGE_START}\n[![CPU CI](https://old/badge/pr?pr=123&event=pr-ci)](https://dash) "
            f"[![GPU run-slow](https://old/badge/pr?pr=123&event=run-slow)](https://dash)\n{BADGE_END}"
        )
        description = "## What changed\n\nA description with a [link](https://example.com)."
        badge = render_ci_badge("https://dash.example/d/x?var-pr=123")
        updated = inject_ci_badge(f"{old}\n\n{description}", badge)
        self.assertEqual(updated, f"{badge}\n\n{description}")
        self.assertEqual(inject_ci_badge(updated, badge), updated)


class BadgeUpdateTest(unittest.TestCase):
    def run_main(self, body, event=None, state="open"):
        if event is None:
            event = {"pull_request": {"number": 123}}
        pr = {"number": 123, "html_url": "https://gh/pr/123", "body": body, "state": state}
        request = Mock(return_value=pr)
        with (
            patch.dict(
                os.environ,
                {"GITHUB_TOKEN": "test", "GITHUB_REPOSITORY": "test/repo", "GITHUB_EVENT_PATH": "event.json"},
            ),
            patch("builtins.open", mock_open(read_data=json.dumps(event))),
            patch.multiple(
                badge_mod,
                github_request=request,
            ),
        ):
            badge_mod.main()
        return request

    def test_opened_and_reopened_prs_use_current_description_without_waiting_for_ci(self):
        for action in ("opened", "reopened"):
            with self.subTest(action=action):
                event = {
                    "action": action,
                    "pull_request": {"number": 123, "body": "Outdated description", "head": {"sha": "old"}},
                }
                request = self.run_main("Edited description", event=event)
                self.assertEqual(request.call_count, 2)
                self.assertEqual(request.call_args_list[0].args, ("https://api.github.com/repos/test/repo/pulls/123",))
                self.assertEqual(request.call_args_list[0].kwargs, {"token": "test"})
                self.assertEqual(request.call_args.kwargs["method"], "PATCH")
                body = request.call_args.kwargs["payload"]["body"]
                self.assertIn("?var-pr=123", body)
                self.assertTrue(body.endswith("Edited description"))
                self.assertNotIn("Outdated description", body)

    def test_open_event_does_not_rewrite_an_existing_badge(self):
        badge = render_ci_badge(f"{badge_mod.DASHBOARD_URL}?var-pr=123")
        request = self.run_main(f"{badge}\n\nDescription", event={"pull_request": {"number": 123}})
        request.assert_called_once_with("https://api.github.com/repos/test/repo/pulls/123", token="test")

    def test_open_event_skips_a_pr_closed_while_the_job_was_queued(self):
        request = self.run_main("Description", event={"pull_request": {"number": 123}}, state="closed")
        request.assert_called_once_with("https://api.github.com/repos/test/repo/pulls/123", token="test")

    def test_updates_header_when_badge_is_missing(self):
        request = self.run_main("Description")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.kwargs["method"], "PATCH")
        body = request.call_args.kwargs["payload"]["body"]
        self.assertIn("![🤗 Transformers CI]", body)
        self.assertIn("?var-pr=123", body)
        self.assertTrue(body.endswith("Description"))


class ReplaceMarkedBlockTest(unittest.TestCase):
    def test_returns_none_when_markers_absent(self):
        self.assertIsNone(replace_marked_block("no markers here", "<!--s-->", "<!--e-->", "X"))
        self.assertIsNone(replace_marked_block(None, "<!--s-->", "<!--e-->", "X"))

    def test_replaces_existing_block(self):
        body = "before\n<!--s-->\nold\ncontent\n<!--e-->\nafter"
        out = replace_marked_block(body, "<!--s-->", "<!--e-->", "<!--s-->\nnew\n<!--e-->")
        self.assertEqual(out, "before\n<!--s-->\nnew\n<!--e-->\nafter")

    def test_replaces_all_marked_regions(self):
        body = "<!--s-->a<!--e--> mid <!--s-->b<!--e-->"
        out = replace_marked_block(body, "<!--s-->", "<!--e-->", "REPL")
        # non-greedy match replaces each marked region
        self.assertEqual(out, "REPL mid REPL")


class InjectBadgeTest(unittest.TestCase):
    def test_prepends_when_no_existing_block(self):
        badge = f"{BADGE_START}\nbadge\n{BADGE_END}"
        out = inject_ci_badge("Hello world", badge)
        self.assertEqual(out, f"{badge}\n\nHello world")

    def test_handles_none_body(self):
        badge = f"{BADGE_START}\nbadge\n{BADGE_END}"
        out = inject_ci_badge(None, badge)
        self.assertEqual(out, badge)

    def test_replaces_existing_block(self):
        old = f"{BADGE_START}\nold\n{BADGE_END}\n\nHello"
        badge = f"{BADGE_START}\nnew\n{BADGE_END}"
        out = inject_ci_badge(old, badge)
        self.assertEqual(out, f"{badge}\n\nHello")
        self.assertNotIn("old", out)


if __name__ == "__main__":
    unittest.main()
