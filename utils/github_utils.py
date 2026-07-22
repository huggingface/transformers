# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Shared helper for talking to the GitHub REST API from CI utilities.

Every GitHub API call in ``utils/`` should go through :func:`github_request` (or the
:func:`get_github_json` GET shortcut) so that rate limiting, transient errors, and rejected tokens
are handled identically everywhere. Two hard rules the whole repo relies on live here:

  * **No anonymous fallback.** When a token is supplied it is *always* kept. A rejected token is a
    reason to stop, never a reason to retry without auth: the anonymous 60-request/hour limit is
    exhausted within a few pages of a large run and turns into a cascade of 403s that looks like a
    rate-limit problem but is really an expired credential.
  * **Fail hard.** A rejected token (401) or any other non-retryable status raises ``RuntimeError``
    instead of returning an error payload, so callers never index into ``{"message": ...}`` by
    mistake.

This module is intentionally **standard-library only** (``urllib``, not ``requests``): it is
imported by GitHub Actions steps that run a bare Python with no third-party packages installed.
"""

import json
import time
import urllib.error
import urllib.request


def build_github_headers(token=None):
    """Build request headers for the GitHub REST API, adding the auth header only when a token is set."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "huggingface-transformers-ci",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _rate_limit_wait(status, response_headers, body, attempt):
    """Return how many seconds to wait before retrying a rate-limited GitHub response, or ``None``.

    Distinguishes the two GitHub rate limits, which look different on the wire:

      * primary limit: ``X-RateLimit-Remaining: 0`` plus an ``X-RateLimit-Reset`` epoch;
      * secondary limit: a 403/429 that does *not* touch the primary quota (``X-RateLimit-Remaining``
        may still be non-zero) and often ships no ``Retry-After`` header, only a body message like
        "You have exceeded a secondary rate limit". This is the one that breaks daily CI reporting
        when it walks the ~24 pages of a large run's jobs, so it must be detected by body too.

    see https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
    """
    if status not in (403, 429):
        return None

    retry_after = response_headers.get("Retry-After")
    remaining = response_headers.get("X-RateLimit-Remaining")
    reset = response_headers.get("X-RateLimit-Reset")
    body = (body or "").lower()
    # A 429 is always "too many requests"; a 403 only counts as a rate limit if something says so
    # (a 403 without any rate-limit signal is a genuine permission error and must not be retried).
    is_rate_limited = status == 429 or (
        retry_after is not None
        or remaining == "0"
        or "rate limit" in body
        or "secondary rate" in body
        or "abuse" in body
    )
    if not is_rate_limited:
        return None

    if retry_after is not None:
        wait = int(retry_after)
    elif remaining == "0" and reset is not None:
        wait = max(0, int(reset) - int(time.time()))
    else:
        # Secondary limit without hints: GitHub asks to wait ~1 min; grow it per attempt.
        wait = 60 * (attempt + 1)
    # Clamp so a far-off primary reset can't stall CI, but always wait long enough for a secondary
    # limit (which is measured in tens of seconds) to actually clear.
    return min(max(wait, 30), 300)


def _is_expired_or_bad_token(status, body):
    """Return why a 401 rejected the token (``"expired"`` / ``"bad credentials"`` / ``"rejected"``), else ``None``.

    A 401 (Unauthorized) is fundamentally a statement about the credential, not the resource: if a
    token was sent and GitHub still answered 401, the token itself was refused. GitHub spells this
    out in the body (``"Bad credentials"``; expired credentials also mention ``"expired"``), but any
    401 received *while sending a token* is by definition a rejected token, so the body is only used
    to sharpen the error message, never to decide.
    """
    if status != 401:
        return None
    body = (body or "").lower()
    if "expired" in body:
        return "expired"
    return "bad credentials" if "bad credentials" in body else "rejected"


def _request(url, headers, method="GET", data=None):
    """Perform a single HTTP request and return ``(status, headers, body_text)``.

    Non-2xx responses come back through :class:`urllib.error.HTTPError`, which is itself a response
    object, so error bodies (rate-limit / auth messages) are returned like any other body instead of
    raising. Only genuine network-level failures raise (``urllib.error.URLError``), for the caller to
    treat as transient.
    """
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.headers, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read().decode("utf-8", errors="replace")


def github_request(url, token=None, method="GET", payload=None, max_retries=8):
    """Call a GitHub REST API URL and return the parsed JSON (or ``None`` for an empty body, e.g. a 204).

    Hardened against the failure modes that silently broke daily CI reporting (callers indexed into
    the response with e.g. ``result["jobs"]`` and raised a bare ``KeyError`` when GitHub returned an
    error payload instead of data).

    **Rate limiting is the only thing that is ever retried.** Both the primary limit
    (``X-RateLimit-Remaining: 0`` + ``X-RateLimit-Reset``) and the secondary limit are waited out
    (``Retry-After`` / reset epoch when present, otherwise ~1 min, growing per attempt) up to
    ``max_retries`` times, and the token is always kept -- retrying without it would only lower the
    limit. Everything else fails hard immediately with ``RuntimeError`` (no retry):

      * a 401 with a token: the token is bad/revoked/expired. It is *never* retried and *never* falls
        back to an unauthenticated request, because the anonymous 60-request/hour limit is exhausted
        almost at once while crawling a large run and every subsequent call comes back 403 -- an
        expired credential masquerading as a rate limit.
      * 5xx server errors, network failures, 404s, permission 403s, and any other non-2xx status:
        raised at once so callers fail loudly instead of indexing into an error payload or masking a
        real outage behind silent retries.
    """
    headers = build_github_headers(token)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    for attempt in range(max_retries):
        try:
            status, response_headers, body = _request(url, headers, method=method, data=data)
        except urllib.error.URLError as error:
            # Network-level failure (DNS, connection reset, timeout): not a rate limit, so fail hard.
            raise RuntimeError(f"GitHub API request to {method} {url} failed: {error}") from error

        wait = _rate_limit_wait(status, response_headers, body, attempt)
        if wait is not None:
            print(
                f"GitHub API rate limited on {method} {url} (status {status}); waiting {wait}s before "
                f"retry {attempt + 1}/{max_retries}"
            )
            time.sleep(wait)
            continue

        # A rejected token (401) must not be retried and must not fall back to anonymous requests
        # (that only trips the anonymous rate limit into a cascade of 403s). Fail hard so the
        # operator refreshes the token instead of chasing phantom rate limits.
        token_state = _is_expired_or_bad_token(status, body)
        if token_state is not None:
            raise RuntimeError(
                f"GitHub rejected the token on {method} {url} with 401 ({token_state}). Not retrying "
                "unauthenticated (that only trips the anonymous rate limit and returns 403s). Refresh "
                "the token and rerun."
            )

        if 200 <= status < 300:
            return json.loads(body) if body else None

        # Anything else (5xx, 404, permission 403, ...) is non-retryable: fail hard.
        raise RuntimeError(f"Could not complete {method} {url}: status {status}: {body[:300]}")

    raise RuntimeError(f"GitHub API still rate limited on {method} {url} after {max_retries} attempt(s)")


def get_github_json(url, token=None, max_retries=8):
    """GET a GitHub REST API URL and return the parsed JSON. See :func:`github_request` for semantics."""
    return github_request(url, token=token, method="GET", max_retries=max_retries)
