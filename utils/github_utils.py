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
import logging
import os
import time
import urllib.error
import urllib.request


logger = logging.getLogger(__name__)
if not logger.handlers:
    # StreamHandler defaults to stderr, keeping diagnostics visible in CI logs without polluting stdout
    # that some workflow steps capture as machine-readable output.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


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


def _log_rate_limit_headers(response_headers, prefix=""):
    """Log all GitHub rate-limit response headers for diagnostics."""
    limit = response_headers.get("X-RateLimit-Limit", "n/a")
    used = response_headers.get("X-RateLimit-Used", "n/a")
    remaining = response_headers.get("X-RateLimit-Remaining", "n/a")
    reset_ts = response_headers.get("X-RateLimit-Reset")
    resource = response_headers.get("X-RateLimit-Resource", "n/a")
    retry_after = response_headers.get("Retry-After")

    if reset_ts is not None:
        secs_until = int(reset_ts) - int(time.time())
        reset_str = f"{reset_ts} (resets in {secs_until}s, i.e. {secs_until // 60}m {secs_until % 60}s)"
    else:
        reset_str = "n/a"

    parts = [
        f"limit={limit}",
        f"used={used}",
        f"remaining={remaining}",
        f"reset={reset_str}",
        f"resource={resource}",
    ]
    if retry_after is not None:
        parts.append(f"Retry-After={retry_after}s")

    tag = f"[{prefix}] " if prefix else ""
    logger.info("%sGitHub rate-limit: %s", tag, ", ".join(parts))


_token_status_logged = False


def _log_token_status(token=None):
    """Call GET /rate_limit once to confirm token validity and log quota.

    ``/rate_limit`` does not consume quota, so it is safe to call as a pure diagnostic.
    Prints whether the token is accepted (authenticated, limit=5000+) or rejected (401).

    Raises ``RuntimeError`` immediately if:
      * running in CI (``CI`` env var) with no token — anonymous 60 req/hr is never enough;
      * a token was provided but rejected (HTTP 401) — all subsequent calls will fail the same way;
      * the quota is already exhausted (remaining=0) — no point starting any API calls.
    """
    global _token_status_logged
    if _token_status_logged:
        return
    _token_status_logged = True

    if not token and os.environ.get("CI"):
        raise RuntimeError(
            "[token check] No GitHub token provided in a CI environment. "
            "Set the GITHUB_TOKEN env var — anonymous requests are limited to 60/hr and will "
            "exhaust immediately on any multi-page crawl."
        )

    try:
        status, response_headers, body = _request("https://api.github.com/rate_limit", build_github_headers(token))
    except urllib.error.URLError as e:
        logger.info("[token check] Could not reach GitHub API: %s", e)
        return

    if status == 401:
        if token:
            raise RuntimeError(
                "[token check] GitHub rejected the token (HTTP 401) — it is invalid, expired, or revoked. "
                "Refresh the token and rerun."
            )
        else:
            raise RuntimeError(
                "[token check] GitHub rejected an anonymous /rate_limit request (HTTP 401) — unexpected, "
                "as anonymous access should always be allowed. Check for a GitHub outage or network proxy issue."
            )

    if status != 200:
        logger.info("[token check] Unexpected status %s from /rate_limit: %r", status, body[:200])
        return

    # Parse the JSON body for richer quota info (headers only carry a subset).
    try:
        data = json.loads(body)
        core = data.get("resources", {}).get("core", data.get("rate", {}))
        remaining = str(core.get("remaining", "n/a"))
        limit = str(core.get("limit", "n/a"))
        reset_ts = core.get("reset")
    except Exception:
        remaining = response_headers.get("X-RateLimit-Remaining", "n/a")
        limit = response_headers.get("X-RateLimit-Limit", "n/a")
        reset_ts = response_headers.get("X-RateLimit-Reset")
        reset_ts = int(reset_ts) if reset_ts is not None else None

    if reset_ts is not None:
        secs_until = int(reset_ts) - int(time.time())
        reset_str = f"{reset_ts} (resets in {secs_until}s, i.e. {secs_until // 60}m {secs_until % 60}s)"
    else:
        reset_str = "n/a"

    auth_status = "AUTHENTICATED" if token else "UNAUTHENTICATED"
    logger.info("[token check] %s — limit=%s, remaining=%s, reset=%s", auth_status, limit, remaining, reset_str)

    if remaining == "0":
        msg = f"[token check] Rate-limit quota EXHAUSTED (remaining=0). Resets at {reset_str}."
        raise RuntimeError(msg)


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
    except ConnectionError as error:
        # http.client.RemoteDisconnected (and siblings like ConnectionResetError) are OSError
        # subclasses that urllib does not always wrap in URLError — re-raise so callers see a
        # consistent urllib.error.URLError and can decide whether to retry.
        raise urllib.error.URLError(reason=error) from error


def github_request(url, token=None, method="GET", payload=None, max_retries=8):
    """Call a GitHub REST API URL and return the parsed JSON (or ``None`` for an empty body, e.g. a 204).

    Hardened against the failure modes that silently broke daily CI reporting (callers indexed into
    the response with e.g. ``result["jobs"]`` and raised a bare ``KeyError`` when GitHub returned an
    error payload instead of data).

    Two categories of failures are retried up to ``max_retries`` times:

      * **Rate limiting** — both the primary limit (``X-RateLimit-Remaining: 0`` +
        ``X-RateLimit-Reset``) and the secondary limit are waited out (``Retry-After`` / reset epoch
        when present, otherwise ~1 min, growing per attempt). The token is always kept — retrying
        without it would only lower the limit.
      * **Transient network errors** (``urllib.error.URLError``, including
        ``http.client.RemoteDisconnected`` and other ``ConnectionError`` subclasses) — retried with
        exponential backoff (1 s, 2 s, 4 s … capped at 60 s).

    Everything else fails hard immediately with ``RuntimeError`` (no retry):

      * a 401 with a token: the token is bad/revoked/expired. It is *never* retried and *never* falls
        back to an unauthenticated request, because the anonymous 60-request/hour limit is exhausted
        almost at once while crawling a large run and every subsequent call comes back 403 -- an
        expired credential masquerading as a rate limit.
      * 5xx server errors, 404s, permission 403s, and any other non-2xx HTTP status:
        raised at once so callers fail loudly instead of indexing into an error payload or masking a
        real outage behind silent retries.
    """
    # Check token validity and quota once before entering the retry loop. Raises immediately if
    # running in CI without a token, the token is rejected, or the quota is already exhausted.
    _log_token_status(token)

    headers = build_github_headers(token)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    for attempt in range(max_retries):
        label = "initial" if attempt == 0 else f"retry {attempt}/{max_retries - 1}"

        try:
            status, response_headers, body = _request(url, headers, method=method, data=data)
        except urllib.error.URLError as error:
            # Transient network-level failure (DNS, connection reset, RemoteDisconnected, timeout).
            # Retry with exponential backoff rather than failing immediately, because these are
            # almost always transient (the runner saw RemoteDisconnected mid-TLS handshake).
            if attempt < max_retries - 1:
                wait = min(2**attempt, 60)
                logger.warning("[%s] Network error on %s %s (%s) — retrying in %ss", label, method, url, error, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"GitHub API request to {method} {url} failed after {max_retries} attempts: {error}"
            ) from error

        logger.info("[%s] %s %s → HTTP %s", label, method, url, status)
        # Rate-limit headers are absent on 401 (auth rejected before rate-limit machinery runs).
        if status != 401:
            _log_rate_limit_headers(response_headers, prefix=label)

        wait = _rate_limit_wait(status, response_headers, body, attempt)
        if wait is not None:
            next_label = f"retry {attempt + 1}/{max_retries - 1}"
            logger.info("[%s] Rate limited (HTTP %s) — waiting %ss before %s", label, status, wait, next_label)
            time.sleep(wait)
            continue

        if 200 <= status < 300:
            return json.loads(body) if body else None

        # Anything else (5xx, 404, permission 403, ...) is non-retryable: fail hard.
        raise RuntimeError(f"Could not complete {method} {url}: status {status}: {body[:300]}")

    raise RuntimeError(f"GitHub API still rate limited on {method} {url} after {max_retries} attempt(s)")


def get_github_json(url, token=None, max_retries=8):
    """GET a GitHub REST API URL and return the parsed JSON. See :func:`github_request` for semantics."""
    return github_request(url, token=token, method="GET", max_retries=max_retries)
