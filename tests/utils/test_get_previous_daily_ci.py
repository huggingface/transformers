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
            "id": 33038227895,
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

_CURRENT_RUN_ID = 33038227895
_WORKFLOW_ID = 90575235


class GetDailyCiRunsRetryTest(unittest.TestCase):
    """Unit tests for the stale-cache retry logic in get_daily_ci_runs."""

    def test_fresh_on_first_attempt_no_retry(self):
        """When the first response contains current_run_id, return immediately without retrying."""
        with (
            patch("get_previous_daily_ci.get_github_json", return_value=_FRESH_RESPONSE) as mock_api,
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID, current_run_id=_CURRENT_RUN_ID)

        mock_api.assert_called_once()
        mock_sleep.assert_not_called()
        self.assertEqual(runs[0]["id"], _CURRENT_RUN_ID)

    def test_stale_then_fresh_retries_once(self):
        """A stale first response triggers one retry that returns fresh data."""
        with (
            patch("get_previous_daily_ci.get_github_json", side_effect=[_STALE_RESPONSE, _FRESH_RESPONSE]) as mock_api,
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID, current_run_id=_CURRENT_RUN_ID)

        self.assertEqual(mock_api.call_count, 2)
        mock_sleep.assert_called_once_with(30)
        self.assertEqual(runs[0]["id"], _CURRENT_RUN_ID)

    def test_all_stale_exhausts_max_attempts(self):
        """When all attempts return stale data, proceed after max_attempts with the last result."""
        with (
            patch("get_previous_daily_ci.get_github_json", return_value=_STALE_RESPONSE) as mock_api,
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID, current_run_id=_CURRENT_RUN_ID)

        # max_attempts=5 → 5 API calls, 4 sleeps (between consecutive attempts, not after the last)
        self.assertEqual(mock_api.call_count, 5)
        self.assertEqual(mock_sleep.call_count, 4)
        mock_sleep.assert_called_with(30)
        # Stale data is returned as a graceful fallback
        self.assertEqual(runs[0]["id"], 30781855254)

    def test_no_current_run_id_skips_stale_check(self):
        """With current_run_id=0 (GITHUB_RUN_ID unset) the stale check is skipped entirely."""
        with (
            patch("get_previous_daily_ci.get_github_json", return_value=_STALE_RESPONSE) as mock_api,
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
            patch.dict("os.environ", {}, clear=False),
        ):
            os.environ.pop("GITHUB_RUN_ID", None)
            # Don't pass current_run_id → defaults to int(os.environ.get("GITHUB_RUN_ID", 0)) = 0
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID)

        mock_api.assert_called_once()
        mock_sleep.assert_not_called()
        self.assertEqual(runs[0]["id"], 30781855254)

    def test_empty_schedule_falls_back_to_workflow_run(self):
        """When event=schedule returns no runs, fall back to event=workflow_run without retrying."""
        fallback_response = {
            "total_count": 5,
            "workflow_runs": [
                {
                    "id": 33038227895,
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
                side_effect=[{"total_count": 0, "workflow_runs": []}, fallback_response],
            ),
            patch("get_previous_daily_ci.time.sleep") as mock_sleep,
        ):
            runs = get_daily_ci_runs(token="tok", workflow_id=_WORKFLOW_ID, current_run_id=_CURRENT_RUN_ID)

        mock_sleep.assert_not_called()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["event"], "workflow_run")
