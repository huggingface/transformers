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

import os
import sys
import unittest
from unittest.mock import patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import update_pr_ci_dashboard_recap as recap_mod  # noqa: E402
from update_pr_ci_dashboard_recap import (  # noqa: E402
    BADGE_END,
    BADGE_START,
    RECAP_END,
    RECAP_START,
    delete_old_dashboard_comments,
    find_open_pr_for_sha,
    first_value,
    format_duration,
    format_number,
    get_ci_recap,
    get_latest_run_id,
    get_metric_value,
    github_paginate,
    inject_ci_badge,
    prometheus_string,
    quality_job_failed,
    recreate_ci_recap_comment,
    remove_marked_block,
    render_ci_badge,
    render_ci_recap,
    replace_marked_block,
)


class PrometheusStringTest(unittest.TestCase):
    def test_casts_non_string_values(self):
        self.assertEqual(prometheus_string(1234), "1234")

    def test_escapes_backslash_and_quote(self):
        self.assertEqual(prometheus_string('a"b'), 'a\\"b')
        self.assertEqual(prometheus_string("a\\b"), "a\\\\b")
        # backslash is escaped before the quote so an escaped quote is not double-escaped
        self.assertEqual(prometheus_string('a\\"b'), 'a\\\\\\"b')


class FirstValueTest(unittest.TestCase):
    def test_returns_none_on_empty(self):
        self.assertIsNone(first_value([]))
        self.assertIsNone(first_value(None))

    def test_parses_float_from_prometheus_result(self):
        result = [{"value": [1700000000, "42.5"]}]
        self.assertEqual(first_value(result), 42.5)

    def test_returns_none_on_malformed_entry(self):
        self.assertIsNone(first_value([{"value": [1700000000]}]))  # missing index 1
        self.assertIsNone(first_value([{"value": [1700000000, "not-a-number"]}]))
        self.assertIsNone(first_value([{}]))  # missing "value" key


class FormatNumberTest(unittest.TestCase):
    def test_none(self):
        self.assertEqual(format_number(None), "n/a")

    def test_integer_value_gets_thousands_separator(self):
        self.assertEqual(format_number(1234.0), "1,234")
        self.assertEqual(format_number(0.0), "0")

    def test_non_integer_value_keeps_two_decimals(self):
        self.assertEqual(format_number(1234.5), "1,234.50")
        self.assertEqual(format_number(3.14159), "3.14")


class FormatDurationTest(unittest.TestCase):
    def test_none(self):
        self.assertEqual(format_duration(None), "n/a")

    def test_seconds_only(self):
        self.assertEqual(format_duration(0), "0s")
        self.assertEqual(format_duration(45), "45s")

    def test_minutes_and_seconds(self):
        self.assertEqual(format_duration(61), "1m 1s")
        self.assertEqual(format_duration(599), "9m 59s")

    def test_hours_and_minutes(self):
        self.assertEqual(format_duration(3661), "1h 1m")
        self.assertEqual(format_duration(7200), "2h 0m")

    def test_rounds_before_formatting(self):
        self.assertEqual(format_duration(44.4), "44s")
        self.assertEqual(format_duration(59.6), "1m 0s")


class RenderBadgeTest(unittest.TestCase):
    def test_render_ci_badge(self):
        badge = render_ci_badge(123, "https://dash.example/d/x?var-pr=123")
        lines = badge.split("\n")
        self.assertEqual(lines[0], BADGE_START)
        self.assertEqual(lines[-1], BADGE_END)
        self.assertIn(f"{recap_mod.BADGE_URL}?pr=123", badge)
        self.assertIn("https://dash.example/d/x?var-pr=123", badge)


class RenderRecapTest(unittest.TestCase):
    def _recap(self, **overrides):
        base = {
            "metrics_available": True,
            "latest_run_id": "run-1",
            "current_run_url": "https://gh/run/1",
            "current_run_conclusion": "success",
            "job_count": 10.0,
            "total_tests": 1234.0,
            "failed_tests": 0.0,
            "duration_seconds": 3661.0,
        }
        base.update(overrides)
        return base

    def test_metrics_available_renders_summary_line(self):
        out = render_ci_recap("https://dash", self._recap(), workflow_run={}, quality_failed=False)
        self.assertTrue(out.startswith(RECAP_START))
        self.assertTrue(out.endswith(RECAP_END))
        self.assertIn("**Result:** `success`", out)
        self.assertIn("**Jobs:** 10", out)
        self.assertIn("**Tests:** 1,234", out)
        self.assertIn("**Failures:** 0", out)
        self.assertIn("**Duration:** 1h 1m", out)
        self.assertIn("run-1", out)

    def test_unknown_conclusion_falls_back(self):
        out = render_ci_recap(
            "https://dash", self._recap(current_run_conclusion=None), workflow_run={}, quality_failed=False
        )
        self.assertIn("**Result:** `unknown`", out)

    def test_metrics_unavailable_uses_workflow_run(self):
        workflow_run = {"id": 999, "html_url": "https://gh/run/999", "conclusion": "failure"}
        out = render_ci_recap(
            "https://dash", {"metrics_available": False}, workflow_run=workflow_run, quality_failed=False
        )
        self.assertIn("Grafana metrics are not available yet.", out)
        self.assertIn("999", out)
        self.assertIn("`failure`", out)

    def test_quality_failed_appends_warning(self):
        out = render_ci_recap("https://dash", self._recap(), workflow_run={}, quality_failed=True)
        self.assertIn("Code quality check failed", out)


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


class GetMetricValueTest(unittest.TestCase):
    def test_returns_primary_value(self):
        with patch.object(recap_mod, "query_prometheus", return_value=[{"value": [0, "7"]}]):
            self.assertEqual(get_metric_value("q"), 7.0)

    def test_falls_back_when_primary_is_none(self):
        results = {"primary": [], "fallback": [{"value": [0, "3"]}]}

        def fake(query):
            return results["primary"] if query == "primary" else results["fallback"]

        with patch.object(recap_mod, "query_prometheus", side_effect=fake):
            self.assertEqual(get_metric_value("primary", fallback_query="fallback"), 3.0)

    def test_fallback_on_zero(self):
        def fake(query):
            return [{"value": [0, "0"]}] if query == "primary" else [{"value": [0, "5"]}]

        with patch.object(recap_mod, "query_prometheus", side_effect=fake):
            # without fallback_on_zero, the zero primary value is kept
            self.assertEqual(get_metric_value("primary", fallback_query="fallback"), 0.0)
            # with fallback_on_zero, the fallback query is consulted
            self.assertEqual(get_metric_value("primary", fallback_query="fallback", fallback_on_zero=True), 5.0)


class GetLatestRunIdTest(unittest.TestCase):
    def test_returns_none_when_empty(self):
        with patch.object(recap_mod, "query_prometheus", return_value=[]):
            self.assertIsNone(get_latest_run_id(42))

    def test_returns_run_id_from_metric(self):
        with patch.object(recap_mod, "query_prometheus", return_value=[{"metric": {"run_id": "abc"}}]):
            self.assertEqual(get_latest_run_id(42), "abc")


class GetCiRecapTest(unittest.TestCase):
    def test_no_metrics_when_no_latest_run(self):
        with patch.object(recap_mod, "get_latest_run_id", return_value=None):
            out = get_ci_recap(42, "https://gh/run", "success")
        self.assertEqual(out, {"metrics_available": False, "latest_run_id": None})

    def test_collects_all_metrics(self):
        with (
            patch.object(recap_mod, "get_latest_run_id", return_value="run-7"),
            patch.object(recap_mod, "get_metric_value", return_value=12.0),
        ):
            out = get_ci_recap(42, "https://gh/run", "success")
        self.assertTrue(out["metrics_available"])
        self.assertEqual(out["latest_run_id"], "run-7")
        self.assertEqual(out["current_run_url"], "https://gh/run")
        self.assertEqual(out["current_run_conclusion"], "success")
        for key in ("duration_seconds", "failed_tests", "job_count", "total_tests"):
            self.assertEqual(out[key], 12.0)


class GithubPaginateTest(unittest.TestCase):
    def test_stops_on_short_page(self):
        page = [{"i": n} for n in range(100)]
        with patch.object(recap_mod, "request_json", side_effect=[page, [{"i": 100}]]) as mocked:
            items = github_paginate("/repos/x/y/pulls", token="t")
        self.assertEqual(len(items), 101)
        self.assertEqual(mocked.call_count, 2)

    def test_stops_on_empty_first_page(self):
        with patch.object(recap_mod, "request_json", return_value=[]) as mocked:
            items = github_paginate("/repos/x/y/pulls", token="t")
        self.assertEqual(items, [])
        self.assertEqual(mocked.call_count, 1)

    def test_extracts_keyed_payload(self):
        with patch.object(recap_mod, "request_json", return_value={"jobs": [{"name": "a"}]}):
            items = github_paginate("/repos/x/y/actions/runs/1/jobs", token="t", key="jobs")
        self.assertEqual(items, [{"name": "a"}])

    def test_builds_query_separator_when_path_has_query(self):
        with patch.object(recap_mod, "request_json", return_value=[]) as mocked:
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


class QualityJobFailedTest(unittest.TestCase):
    def test_true_when_quality_job_failed(self):
        jobs = [{"name": "Check code quality", "conclusion": "failure"}]
        with patch.object(recap_mod, "github_paginate", return_value=jobs):
            self.assertTrue(quality_job_failed("x/y", "t", 1))

    def test_false_when_quality_job_succeeded(self):
        jobs = [{"name": "Check code quality", "conclusion": "success"}]
        with patch.object(recap_mod, "github_paginate", return_value=jobs):
            self.assertFalse(quality_job_failed("x/y", "t", 1))

    def test_false_when_quality_job_absent(self):
        jobs = [{"name": "Run tests", "conclusion": "failure"}]
        with patch.object(recap_mod, "github_paginate", return_value=jobs):
            self.assertFalse(quality_job_failed("x/y", "t", 1))


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
            patch.object(recap_mod, "request_json") as request_mock,
        ):
            delete_old_dashboard_comments("x/y", "t", 5)
        self.assertEqual(request_mock.call_count, 1)
        deleted_url = request_mock.call_args.args[0]
        self.assertIn("/comments/1", deleted_url)
        self.assertEqual(request_mock.call_args.kwargs["method"], "DELETE")


class RecreateCiRecapCommentTest(unittest.TestCase):
    def test_deletes_existing_recap_comment_then_creates_new_one(self):
        comments = [
            {"id": 1, "body": "an unrelated comment"},
            {"id": 2, "body": f"{RECAP_START}\nold\n{RECAP_END}"},
        ]
        recap = f"{RECAP_START}\nnew\n{RECAP_END}"
        with (
            patch.object(recap_mod, "github_paginate", return_value=comments),
            patch.object(recap_mod, "request_json") as request_mock,
        ):
            recreate_ci_recap_comment("x/y", "t", 5, recap)

        self.assertEqual(request_mock.call_count, 2)
        delete_call, post_call = request_mock.call_args_list
        self.assertIn("/comments/2", delete_call.args[0])
        self.assertEqual(delete_call.kwargs["method"], "DELETE")
        self.assertIn("/issues/5/comments", post_call.args[0])
        self.assertEqual(post_call.kwargs["method"], "POST")
        self.assertEqual(post_call.kwargs["payload"], {"body": recap})

    def test_deletes_all_existing_recap_comments_before_creating_new_one(self):
        comments = [
            {"id": 1, "body": f"{RECAP_START}\nold 1\n{RECAP_END}"},
            {"id": 2, "body": f"{RECAP_START}\nold 2\n{RECAP_END}"},
        ]
        recap = f"{RECAP_START}\nnew\n{RECAP_END}"
        with (
            patch.object(recap_mod, "github_paginate", return_value=comments),
            patch.object(recap_mod, "request_json") as request_mock,
        ):
            recreate_ci_recap_comment("x/y", "t", 5, recap)

        self.assertEqual(request_mock.call_count, 3)
        self.assertEqual([call.kwargs["method"] for call in request_mock.call_args_list], ["DELETE", "DELETE", "POST"])

    def test_creates_recap_comment_when_missing(self):
        recap = f"{RECAP_START}\nnew\n{RECAP_END}"
        with (
            patch.object(recap_mod, "github_paginate", return_value=[{"id": 1, "body": "unrelated"}]),
            patch.object(recap_mod, "request_json") as request_mock,
        ):
            recreate_ci_recap_comment("x/y", "t", 5, recap)

        request_mock.assert_called_once()
        self.assertIn("/issues/5/comments", request_mock.call_args.args[0])
        self.assertEqual(request_mock.call_args.kwargs["method"], "POST")
        self.assertEqual(request_mock.call_args.kwargs["payload"], {"body": recap})


if __name__ == "__main__":
    unittest.main()
