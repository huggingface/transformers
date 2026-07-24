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

import github_utils as gh  # noqa: E402
from github_utils import (  # noqa: E402
    build_github_headers,
    get_github_json,
    github_request,
)


class Headers(dict):
    """A dict with a ``.get(key, default)`` like the header mappings GitHub responses expose."""

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _response(status, headers=None, body=""):
    """Build the ``(status, headers, body)`` tuple that :func:`github_utils._request` returns."""
    return status, Headers(headers or {}), body


class BuildGithubHeadersTest(unittest.TestCase):
    def test_adds_authorization_when_token_present(self):
        headers = build_github_headers("secret-token")
        self.assertEqual(headers["Authorization"], "Bearer secret-token")
        self.assertEqual(headers["Accept"], "application/vnd.github+json")

    def test_omits_authorization_without_token(self):
        for token in (None, ""):
            with self.subTest(token=token):
                self.assertNotIn("Authorization", build_github_headers(token))


class IsExpiredOrBadTokenTest(unittest.TestCase):
    def test_non_401_is_never_a_token_problem(self):
        for status in (200, 403, 404, 429, 500):
            with self.subTest(status=status):
                self.assertIsNone(gh._is_expired_or_bad_token(status, "Bad credentials"))

    def test_401_is_always_a_rejected_token_even_without_a_body(self):
        self.assertEqual(gh._is_expired_or_bad_token(401, ""), "rejected")
        self.assertEqual(gh._is_expired_or_bad_token(401, None), "rejected")

    def test_401_body_sharpens_the_reason(self):
        self.assertEqual(gh._is_expired_or_bad_token(401, "Bad credentials"), "bad credentials")
        # GitHub wording is not guaranteed; matching is case-insensitive and substring based.
        self.assertEqual(gh._is_expired_or_bad_token(401, "This token has EXPIRED"), "expired")


class RateLimitWaitTest(unittest.TestCase):
    def test_non_rate_limit_status_returns_none(self):
        self.assertIsNone(gh._rate_limit_wait(200, Headers({}), "", 0))

    def test_permission_403_is_not_a_rate_limit(self):
        # A 403 with no rate-limit signal is a genuine permission error and must not be retried.
        wait = gh._rate_limit_wait(403, Headers({}), "Resource not accessible by integration", 0)
        self.assertIsNone(wait)

    def test_429_is_always_a_rate_limit(self):
        self.assertIsNotNone(gh._rate_limit_wait(429, Headers({}), "", 0))

    def test_secondary_rate_limit_detected_from_body(self):
        wait = gh._rate_limit_wait(403, Headers({}), "You have exceeded a secondary rate limit", 0)
        self.assertIsNotNone(wait)

    def test_retry_after_header_is_honored_and_clamped(self):
        # Retry-After below the floor is clamped up to 30s; above the ceiling down to 300s.
        self.assertEqual(gh._rate_limit_wait(429, Headers({"Retry-After": "5"}), "", 0), 30)
        self.assertEqual(gh._rate_limit_wait(429, Headers({"Retry-After": "9999"}), "", 0), 300)

    def test_primary_limit_uses_reset_epoch(self):
        with patch.object(gh.time, "time", return_value=1_000):
            wait = gh._rate_limit_wait(
                403, Headers({"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1120"}), "", 0
            )
        self.assertEqual(wait, 120)

    def test_secondary_limit_without_hints_grows_per_attempt(self):
        first = gh._rate_limit_wait(429, Headers({}), "", 0)
        later = gh._rate_limit_wait(429, Headers({}), "", 2)
        self.assertLess(first, later)


class GithubRequestTest(unittest.TestCase):
    def setUp(self):
        # Never actually sleep while exercising the retry loop.
        sleep_patcher = patch.object(gh.time, "sleep", return_value=None)
        self.addCleanup(sleep_patcher.stop)
        sleep_patcher.start()

    def _patch_request(self, side_effect):
        patcher = patch.object(gh, "_request", side_effect=side_effect)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_returns_parsed_json_on_200(self):
        self._patch_request([_response(200, body='{"ok": true}')])
        self.assertEqual(github_request("https://api.github.com/x", token="t"), {"ok": True})

    def test_get_github_json_is_a_get_shortcut(self):
        mock = self._patch_request([_response(200, body='{"n": 1}')])
        self.assertEqual(get_github_json("https://api.github.com/x", token="t"), {"n": 1})
        self.assertEqual(mock.call_args.kwargs["method"], "GET")

    def test_empty_body_returns_none(self):
        # A 204 (e.g. a DELETE) has no body to parse.
        self._patch_request([_response(204, body="")])
        self.assertIsNone(github_request("https://api.github.com/x", token="t", method="DELETE"))

    def test_rejected_token_401_fails_hard_without_retry(self):
        mock = self._patch_request([_response(401, body="Bad credentials")])
        with self.assertRaises(RuntimeError) as ctx:
            github_request("https://api.github.com/x", token="expired")
        self.assertIn("bad credentials", str(ctx.exception).lower())
        # Fail hard: the transport is hit exactly once, never retried.
        self.assertEqual(mock.call_count, 1)

    def test_401_never_falls_back_to_anonymous(self):
        # The whole point: a rejected token must not trigger a token-less retry (which would only
        # trip the anonymous rate limit into a cascade of 403s). Every call keeps the auth header.
        mock = self._patch_request([_response(401, body="Bad credentials")])
        with self.assertRaises(RuntimeError):
            github_request("https://api.github.com/x", token="expired")
        for call in mock.call_args_list:
            headers = call.args[1]
            self.assertEqual(headers["Authorization"], "Bearer expired")

    def test_404_fails_hard_without_retry(self):
        mock = self._patch_request([_response(404, body="Not Found")])
        with self.assertRaises(RuntimeError):
            github_request("https://api.github.com/x", token="t")
        self.assertEqual(mock.call_count, 1)

    def test_5xx_fails_hard_without_retry(self):
        # Only rate limiting is retried; a server error is raised immediately.
        mock = self._patch_request([_response(503, body="boom")])
        with self.assertRaises(RuntimeError):
            github_request("https://api.github.com/x", token="t")
        self.assertEqual(mock.call_count, 1)

    def test_network_error_fails_hard(self):
        self._patch_request(gh.urllib.error.URLError("connection reset"))
        with self.assertRaises(RuntimeError) as ctx:
            github_request("https://api.github.com/x", token="t")
        self.assertIn("connection reset", str(ctx.exception))

    def test_rate_limit_is_retried_then_succeeds(self):
        mock = self._patch_request(
            [
                _response(403, body="You have exceeded a secondary rate limit"),
                _response(200, body='{"ok": true}'),
            ]
        )
        self.assertEqual(github_request("https://api.github.com/x", token="t"), {"ok": True})
        self.assertEqual(mock.call_count, 2)

    def test_rate_limit_exhausts_retries_and_raises(self):
        # Always rate limited: loop up to max_retries, then fail loudly.
        mock = self._patch_request([_response(429, body="rate limit") for _ in range(5)])
        with self.assertRaises(RuntimeError) as ctx:
            github_request("https://api.github.com/x", token="t", max_retries=3)
        self.assertIn("still rate limited", str(ctx.exception))
        self.assertEqual(mock.call_count, 3)

    def test_post_sends_json_payload_and_parses_response(self):
        mock = self._patch_request([_response(201, body='{"id": 5}')])
        result = github_request("https://api.github.com/x", token="t", method="POST", payload={"body": "hi"})
        self.assertEqual(result, {"id": 5})
        headers = mock.call_args.args[1]
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(mock.call_args.kwargs["method"], "POST")
        self.assertEqual(mock.call_args.kwargs["data"], b'{"body": "hi"}')


if __name__ == "__main__":
    unittest.main()
