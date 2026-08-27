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


# utils/ is not an installable package; add it to the path so CI utility modules
# can be imported without modifying the installed transformers package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))

from get_previous_daily_ci import get_daily_ci_runs  # noqa: E402


_WORKFLOW_ID = 90575235
_CURRENT_RUN_ID = 33038227895

# Simulates get_github_json("/runs/{GITHUB_RUN_ID}") — same workflow as the one being queried.
_CURRENT_RUN_RESPONSE = {"id": _CURRENT_RUN_ID, "workflow_id": _WORKFLOW_ID}

# Simulates a current run belonging to a *different* workflow (e.g. AMD CI querying Nvidia).
_DIFFERENT_WORKFLOW_RUN_RESPONSE = {"id": 11111111111, "workflow_id": 99999}

_STALE_RESPONSE = {
    "total_count": 209,
    "workflow_runs": [
        {
            "id": 30781855254,
            "status": "completed",
            "conclusion": "failure",
            "created_at": "2026-08-03T03:27:23Z",
            "event": "schedule",
        },
        {
            "id": 30730647697,
            "status": "completed",
            "conclusion": "failure",
            "created_at": "2026-08-02T03:27:19Z",
            "event": "schedule",
        },
    ],
}

_FRESH_RESPONSE = {
    "total_count": 413,
    "workflow_runs": [
        {
            "id": _CURRENT_RUN_ID,
            "status": "completed",
            "conclusion": "failure",
            "created_at": "2026-08-27T04:03:26Z",
            "event": "schedule",
        },
        {
            "id": 32923823246,
            "status": "completed",
            "conclusion": "failure",
            "created_at": "2026-08-26T02:44:02Z",
            "event": "schedule",
        },
    ],
}


class GetDailyCiRunsRetryTest(unittest.TestCase):
    """Unit tests for the stale-cache retry logic in get_daily_ci_runs.

    get_github_json is called once up-front to fetch the current run's workflow_id
    (for stale-check eligibility), then once per schedule-query attempt.  Tests use
    side_effect lists ordered as: [current-run lookup, attempt-1, attempt-2, ...].
    """

    def test_fresh_on_first_attempt_no_retry(self):
        """When the first response contains the current run, return immediately without retrying."""
        with (
            patch("get_previous_daily_ci.get_github_json", side_effect=[_CURRENT_RUN_RESPONSE, _FRESH_RESPONSE]),
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
            patch.dict("os.environ", {"GITHUB_RUN_ID": str(_CURRENT_RUN_ID), "GITHUB_EVENT_NAME": "schedule"}),
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID)

        mock_sleep.assert_not_called()
        self.assertEqual(runs[0]["id"], _CURRENT_RUN_ID)

    def test_stale_then_fresh_retries_once(self):
        """A stale first response triggers one retry that returns fresh data."""
        with (
            patch(
                "get_previous_daily_ci.get_github_json",
                side_effect=[_CURRENT_RUN_RESPONSE, _STALE_RESPONSE, _FRESH_RESPONSE],
            ),
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
            patch.dict("os.environ", {"GITHUB_RUN_ID": str(_CURRENT_RUN_ID), "GITHUB_EVENT_NAME": "schedule"}),
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID)

        mock_sleep.assert_called_once_with(30)
        self.assertEqual(runs[0]["id"], _CURRENT_RUN_ID)

    def test_all_stale_exhausts_max_attempts(self):
        """When all attempts return stale data, proceed after max_attempts with the last result."""
        with (
            patch(
                "get_previous_daily_ci.get_github_json",
                side_effect=[_CURRENT_RUN_RESPONSE] + [_STALE_RESPONSE] * 5,
            ) as mock_api,
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
            patch.dict("os.environ", {"GITHUB_RUN_ID": str(_CURRENT_RUN_ID), "GITHUB_EVENT_NAME": "schedule"}),
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID)

        # 1 current-run lookup + 5 schedule queries (max_attempts=5)
        self.assertEqual(mock_api.call_count, 6)
        # sleep between attempts 1-2, 2-3, 3-4, 4-5 — not after the last attempt
        self.assertEqual(mock_sleep.call_count, 4)
        mock_sleep.assert_called_with(30)
        # Stale data is returned as a graceful fallback
        self.assertEqual(runs[0]["id"], 30781855254)

    def test_different_workflow_skips_stale_check(self):
        """When workflow_id differs from the current run's, stale check is skipped entirely."""
        with (
            patch(
                "get_previous_daily_ci.get_github_json",
                side_effect=[_DIFFERENT_WORKFLOW_RUN_RESPONSE, _STALE_RESPONSE],
            ) as mock_api,
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
            patch.dict("os.environ", {"GITHUB_RUN_ID": "11111111111", "GITHUB_EVENT_NAME": "schedule"}),
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID)

        # 1 current-run lookup + 1 schedule query (max_attempts=1, no retries)
        self.assertEqual(mock_api.call_count, 2)
        mock_sleep.assert_not_called()
        # Returns whatever the single attempt gave (stale in this case)
        self.assertEqual(runs[0]["id"], 30781855254)

    def test_non_schedule_event_skips_stale_check(self):
        """When the triggering event is not 'schedule' (e.g. push), stale check is skipped."""
        with (
            patch(
                "get_previous_daily_ci.get_github_json",
                side_effect=[_CURRENT_RUN_RESPONSE, _STALE_RESPONSE],
            ) as mock_api,
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
            patch.dict("os.environ", {"GITHUB_RUN_ID": str(_CURRENT_RUN_ID), "GITHUB_EVENT_NAME": "push"}),
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID)

        # 1 current-run lookup + 1 schedule query (max_attempts=1, no retries)
        self.assertEqual(mock_api.call_count, 2)
        mock_sleep.assert_not_called()
        self.assertEqual(runs[0]["id"], 30781855254)

    def test_no_github_run_id_skips_stale_check(self):
        """With GITHUB_RUN_ID unset, the current-run lookup is skipped and no retry is done."""
        with (
            patch("get_previous_daily_ci.get_github_json", side_effect=[_STALE_RESPONSE]) as mock_api,
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
            patch.dict("os.environ", {}, clear=False),
        ):
            os.environ.pop("GITHUB_RUN_ID", None)
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID)

        # No current-run lookup, 1 schedule query, no retries
        mock_api.assert_called_once()
        mock_sleep.assert_not_called()
        self.assertEqual(runs[0]["id"], 30781855254)

    def test_empty_schedule_falls_back_to_workflow_run(self):
        """When event=schedule returns no runs, fall back to event=workflow_run without retrying."""
        fallback_response = {
            "total_count": 5,
            "workflow_runs": [
                {
                    "id": _CURRENT_RUN_ID,
                    "status": "completed",
                    "conclusion": "failure",
                    "created_at": "2026-08-27T04:03:26Z",
                    "event": "workflow_run",
                },
            ],
        }
        with (
            patch(
                "get_previous_daily_ci.get_github_json",
                side_effect=[_CURRENT_RUN_RESPONSE, {"total_count": 0, "workflow_runs": []}, fallback_response],
            ),
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
            patch.dict("os.environ", {"GITHUB_RUN_ID": str(_CURRENT_RUN_ID)}),
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID)

        mock_sleep.assert_not_called()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["event"], "workflow_run")
