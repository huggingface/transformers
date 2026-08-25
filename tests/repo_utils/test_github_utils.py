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

import http.client
import logging
import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
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


class LogTokenStatusTest(unittest.TestCase):
    def setUp(self):
        # Reset the once-per-process guard before every test.
        gh._token_status_logged = False

    def _patch_request(self, side_effect):
        patcher = patch.object(gh, "_request", side_effect=side_effect)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_ci_without_token_raises(self):
        with patch.dict(os.environ, {"CI": "true"}):
            with self.assertRaises(RuntimeError) as ctx:
                gh._log_token_status(token=None)
        self.assertIn("no github token", str(ctx.exception).lower())

    def test_no_ci_without_token_does_not_raise(self):
        env = {k: v for k, v in os.environ.items() if k != "CI"}
        with patch.dict(os.environ, env, clear=True):
            self._patch_request(
                [_response(200, body='{"resources": {"core": {"limit": 60, "remaining": 59, "reset": 9999999999}}}')]
            )
            gh._log_token_status(token=None)  # must not raise

    def test_token_rejected_401_raises(self):
        self._patch_request([_response(401, body="Bad credentials")])
        with self.assertRaises(RuntimeError) as ctx:
            gh._log_token_status(token="bad-token")
        self.assertIn("rejected", str(ctx.exception).lower())

    def test_401_with_token_raises_with_refresh_message(self):
        self._patch_request([_response(401, body="Bad credentials")])
        with self.assertRaises(RuntimeError) as ctx:
            gh._log_token_status(token="bad-token")
        self.assertIn("refresh the token", str(ctx.exception).lower())

    def test_401_without_token_raises_with_unexpected_message(self):
        # Must unset CI so the no-token-in-CI guard doesn't fire before the /rate_limit call.
        env = {k: v for k, v in os.environ.items() if k != "CI"}
        with patch.dict(os.environ, env, clear=True):
            self._patch_request([_response(401, body="")])
            with self.assertRaises(RuntimeError) as ctx:
                gh._log_token_status(token=None)
        self.assertIn("unexpected", str(ctx.exception).lower())

    def test_remaining_zero_raises(self):
        self._patch_request(
            [_response(200, body='{"resources": {"core": {"limit": 5000, "remaining": 0, "reset": 9999999999}}}')]
        )
        with self.assertRaises(RuntimeError) as ctx:
            gh._log_token_status(token="t")
        self.assertIn("exhausted", str(ctx.exception).lower())

    def test_remaining_nonzero_does_not_raise(self):
        self._patch_request(
            [_response(200, body='{"resources": {"core": {"limit": 5000, "remaining": 4999, "reset": 9999999999}}}')]
        )
        gh._log_token_status(token="t")  # must not raise

    def test_called_only_once(self):
        mock = self._patch_request(
            [_response(200, body='{"resources": {"core": {"limit": 5000, "remaining": 4999, "reset": 9999999999}}}')]
        )
        gh._log_token_status(token="t")
        gh._log_token_status(token="t")
        self.assertEqual(mock.call_count, 1)

    def test_network_error_does_not_raise(self):
        # A connectivity failure is logged but must not abort the caller.
        self._patch_request(gh.urllib.error.URLError("timeout"))
        gh._log_token_status(token="t")  # must not raise


class GithubRequestTest(unittest.TestCase):
    def setUp(self):
        # Never actually sleep while exercising the retry loop.
        sleep_patcher = patch.object(gh.time, "sleep", return_value=None)
        self.addCleanup(sleep_patcher.stop)
        sleep_patcher.start()

        # Bypass the pre-flight token check — it is tested separately in LogTokenStatusTest
        # and would otherwise consume mock responses intended for the actual API call.
        token_check_patcher = patch.object(gh, "_log_token_status", return_value=None)
        self.addCleanup(token_check_patcher.stop)
        token_check_patcher.start()
        gh._token_status_logged = False

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
        self.assertIn("401", str(ctx.exception))
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

    def test_connection_error_is_wrapped_as_url_error(self):
        # ConnectionError (and subclasses) are OSError, not urllib.error.URLError — _request must
        # normalize them so callers see a single consistent exception type.
        with patch("urllib.request.urlopen", side_effect=ConnectionResetError("reset")):
            with self.assertRaises(gh.urllib.error.URLError):
                gh._request("https://api.github.com/x", {})

    def test_remote_disconnected_is_wrapped_as_url_error(self):
        # The concrete error seen in CI: RemoteDisconnected is a ConnectionResetError subclass.
        exc = http.client.RemoteDisconnected("Remote end closed connection without response")
        with patch("urllib.request.urlopen", side_effect=exc):
            with self.assertRaises(gh.urllib.error.URLError):
                gh._request("https://api.github.com/x", {})

    def test_network_error_retries_then_fails(self):
        # All attempts raise URLError → exhausts max_retries → raises RuntimeError.
        mock = self._patch_request(gh.urllib.error.URLError("connection reset"))
        with self.assertRaises(RuntimeError) as ctx:
            github_request("https://api.github.com/x", token="t", max_retries=3)
        self.assertIn("connection reset", str(ctx.exception))
        self.assertEqual(mock.call_count, 3)

    def test_connection_error_retries_then_succeeds(self):
        # _request normalizes ConnectionError subclasses (e.g. RemoteDisconnected) into URLError
        # before they reach github_request. The mock raises URLError to simulate that — transient
        # error on attempt 1, success on attempt 2.
        mock = self._patch_request(
            [
                gh.urllib.error.URLError("connection reset"),
                _response(200, body='{"ok": true}'),
            ]
        )
        self.assertEqual(github_request("https://api.github.com/x", token="t"), {"ok": True})
        self.assertEqual(mock.call_count, 2)

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


def test_github_diagnostics_logger_uses_stream_handler_not_stdout():
    assert any(
        isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is not sys.stdout
        for handler in gh.logger.handlers
    )


def test_github_request_diagnostics_are_logged_not_printed_to_stdout():
    stdout = StringIO()
    with (
        patch.object(gh, "_log_token_status", return_value=None),
        patch.object(gh, "_request", return_value=_response(200, {"X-RateLimit-Limit": "5000"}, '{"ok": true}')),
        redirect_stdout(stdout),
        unittest.TestCase().assertLogs(gh.logger, level="INFO") as logs,
    ):
        assert github_request("https://api.github.com/x", token="t") == {"ok": True}

    assert stdout.getvalue() == ""
    output = "\n".join(logs.output)
    assert "[initial] GET https://api.github.com/x" in output
    assert "GitHub rate-limit" in output


if __name__ == "__main__":
    unittest.main()
