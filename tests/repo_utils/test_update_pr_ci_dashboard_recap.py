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

import update_pr_ci_dashboard_recap as recap_mod  # noqa: E402
from update_pr_ci_dashboard_recap import (  # noqa: E402
    BADGE_END,
    BADGE_START,
    RECAP_END,
    RECAP_START,
    delete_old_dashboard_comments,
    delete_recap_comments,
    find_open_pr_for_sha,
    github_paginate,
    inject_ci_badge,
    remove_marked_block,
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
                f"{BADGE_START}\n[![🤗 Transformers CI]({recap_mod.BADGE_URL})]({url})\n{BADGE_END}",
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
    def run_main(self, body):
        request = Mock()
        event = {"workflow_run": {"id": 1, "event": "pull_request", "head_sha": "abc", "html_url": "https://gh/run/1"}}
        pr = {"number": 123, "html_url": "https://gh/pr/123", "body": body}
        with (
            patch.dict(
                os.environ,
                {"GITHUB_TOKEN": "test", "GITHUB_REPOSITORY": "test/repo", "GITHUB_EVENT_PATH": "event.json"},
            ),
            patch("builtins.open", mock_open(read_data=json.dumps(event))),
            patch.multiple(
                recap_mod,
                find_open_pr_for_sha=Mock(return_value=pr),
                delete_old_dashboard_comments=Mock(),
                delete_recap_comments=Mock(),
                github_request=request,
            ),
        ):
            recap_mod.main()
        return request

    def test_updates_header_when_badge_is_missing(self):
        request = self.run_main("Description")
        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["method"], "PATCH")
        body = request.call_args.kwargs["payload"]["body"]
        self.assertIn("![🤗 Transformers CI]", body)
        self.assertIn("?var-pr=123", body)
        self.assertTrue(body.endswith("Description"))

    def test_does_not_rewrite_an_unchanged_header(self):
        badge = render_ci_badge(f"{recap_mod.DASHBOARD_URL}?var-pr=123")
        request = self.run_main(f"{badge}\n\nDescription")
        request.assert_not_called()


class ReplaceMarkedBlockTest(unittest.TestCase):
    def test_returns_none_when_markers_absent(self):
        self.assertIsNone(replace_marked_block("no markers here", "<!--s-->", "<!--e-->", "X"))
        self.assertIsNone(replace_marked_block(None, "<!--s-->", "<!--e-->", "X"))

    def test_replaces_existing_block(self):
        body = "before\n<!--s-->\nold\ncontent\n<!--e-->\nafter"
        out = replace_marked_block(body, "<!--s-->", "<!--e-->", "<!--s-->\nnew\n<!--e-->")
        self.assertEqual(out, "before\n<!--s-->\nnew\n<!--e-->\nafter")

    def test_only_replaces_first_block_region(self):
        body = "<!--s-->a<!--e--> mid <!--s-->b<!--e-->"
        out = replace_marked_block(body, "<!--s-->", "<!--e-->", "REPL")
        # non-greedy match replaces each marked region
        self.assertEqual(out, "REPL mid REPL")


class RemoveMarkedBlockTest(unittest.TestCase):
    def test_keeps_body_when_no_existing_block(self):
        out = remove_marked_block("Hello world", RECAP_START, RECAP_END)
        self.assertEqual(out, "Hello world")

    def test_handles_none_body(self):
        out = remove_marked_block(None, RECAP_START, RECAP_END)
        self.assertEqual(out, "")

    def test_removes_existing_block(self):
        old = f"text\n{RECAP_START}\nold\n{RECAP_END}\ntail"
        out = remove_marked_block(old, RECAP_START, RECAP_END)
        self.assertEqual(out, "text\n\ntail")
        self.assertNotIn("old", out)


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

    def test_badge_and_recap_compose(self):
        badge = f"{BADGE_START}\nbadge\n{BADGE_END}"
        recap = f"{RECAP_START}\nrecap\n{RECAP_END}"
        body = inject_ci_badge("Original PR text", badge)
        body = f"{body}\n\n{recap}"
        body = remove_marked_block(body, RECAP_START, RECAP_END)
        self.assertTrue(body.startswith(badge))
        self.assertIn("Original PR text", body)
        self.assertNotIn(RECAP_START, body)


class GithubPaginateTest(unittest.TestCase):
    def test_stops_on_short_page(self):
        page = [{"i": n} for n in range(100)]
        with patch.object(recap_mod, "github_request", side_effect=[page, [{"i": 100}]]) as mocked:
            items = github_paginate("/repos/x/y/pulls", token="t")
        self.assertEqual(len(items), 101)
        self.assertEqual(mocked.call_count, 2)

    def test_stops_on_empty_first_page(self):
        with patch.object(recap_mod, "github_request", return_value=[]) as mocked:
            items = github_paginate("/repos/x/y/pulls", token="t")
        self.assertEqual(items, [])
        self.assertEqual(mocked.call_count, 1)

    def test_extracts_keyed_payload(self):
        with patch.object(recap_mod, "github_request", return_value={"jobs": [{"name": "a"}]}):
            items = github_paginate("/repos/x/y/actions/runs/1/jobs", token="t", key="jobs")
        self.assertEqual(items, [{"name": "a"}])

    def test_builds_query_separator_when_path_has_query(self):
        with patch.object(recap_mod, "github_request", return_value=[]) as mocked:
            github_paginate("/repos/x/y/pulls?state=open", token="t")
        url = mocked.call_args.args[0]
        self.assertIn("?state=open&per_page=100&page=1", url)


class FindOpenPrTest(unittest.TestCase):
    def test_matches_head_sha(self):
        prs = [
            {"number": 1, "head": {"sha": "aaa"}},
            {"number": 2, "head": {"sha": "bbb"}},
        ]
        with patch.object(recap_mod, "github_paginate", return_value=prs):
            self.assertEqual(find_open_pr_for_sha("x/y", "t", "bbb")["number"], 2)

    def test_returns_none_when_no_match(self):
        with patch.object(recap_mod, "github_paginate", return_value=[{"number": 1, "head": {"sha": "aaa"}}]):
            self.assertIsNone(find_open_pr_for_sha("x/y", "t", "zzz"))


class DeleteOldCommentsTest(unittest.TestCase):
    def test_deletes_only_dashboard_comments(self):
        marker = recap_mod.OLD_DASHBOARD_COMMENT_MARKERS[0]
        comments = [
            {"id": 1, "body": f"hello {marker} there"},
            {"id": 2, "body": "an unrelated comment"},
            {"id": 3, "body": None},
        ]
        with (
            patch.object(recap_mod, "github_paginate", return_value=comments),
            patch.object(recap_mod, "github_request") as request_mock,
        ):
            delete_old_dashboard_comments("x/y", "t", 5)
        self.assertEqual(request_mock.call_count, 1)
        deleted_url = request_mock.call_args.args[0]
        self.assertIn("/comments/1", deleted_url)
        self.assertEqual(request_mock.call_args.kwargs["method"], "DELETE")


class DeleteRecapCommentsTest(unittest.TestCase):
    def test_deletes_recap_comments_only(self):
        comments = [
            {"id": 1, "body": "an unrelated comment"},
            {"id": 2, "body": f"{RECAP_START}\nold\n{RECAP_END}"},
        ]
        with (
            patch.object(recap_mod, "github_paginate", return_value=comments),
            patch.object(recap_mod, "github_request") as request_mock,
        ):
            delete_recap_comments("x/y", "t", 5)
        self.assertEqual(request_mock.call_count, 1)
        self.assertIn("/comments/2", request_mock.call_args.args[0])
        self.assertEqual(request_mock.call_args.kwargs["method"], "DELETE")

    def test_deletes_multiple_recap_comments(self):
        comments = [
            {"id": 1, "body": f"{RECAP_START}\nold 1\n{RECAP_END}"},
            {"id": 2, "body": f"{RECAP_START}\nold 2\n{RECAP_END}"},
        ]
        with (
            patch.object(recap_mod, "github_paginate", return_value=comments),
            patch.object(recap_mod, "github_request") as request_mock,
        ):
            delete_recap_comments("x/y", "t", 5)
        self.assertEqual(request_mock.call_count, 2)
        self.assertEqual([c.kwargs["method"] for c in request_mock.call_args_list], ["DELETE", "DELETE"])

    def test_no_op_when_no_recap_comments(self):
        with (
            patch.object(recap_mod, "github_paginate", return_value=[{"id": 1, "body": "unrelated"}]),
            patch.object(recap_mod, "github_request") as request_mock,
        ):
            delete_recap_comments("x/y", "t", 5)
        request_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
