import argparse
import json
import logging
import math
import os
import time
import traceback
import zipfile
from collections import Counter

import requests


logger = logging.getLogger(__name__)


def _rate_limit_wait(response, attempt):
    """Return how many seconds to wait before retrying a rate-limited GitHub response, or ``None``.

    Distinguishes the two GitHub rate limits, which look different on the wire:

      * primary limit: ``X-RateLimit-Remaining: 0`` plus an ``X-RateLimit-Reset`` epoch;
      * secondary limit: a 403/429 that does *not* touch the primary quota (``X-RateLimit-Remaining``
        may still be non-zero) and often ships no ``Retry-After`` header, only a body message like
        "You have exceeded a secondary rate limit". This is the one that breaks daily CI reporting
        when it walks the ~24 pages of a large run's jobs, so it must be detected by body too.

    see https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
    """
    if response.status_code not in (403, 429):
        return None

    retry_after = response.headers.get("Retry-After")
    remaining = response.headers.get("X-RateLimit-Remaining")
    reset = response.headers.get("X-RateLimit-Reset")
    body = (response.text or "").lower()
    # A 429 is always "too many requests"; a 403 only counts as a rate limit if something says so
    # (a 403 without any rate-limit signal is a genuine permission error and must not be retried).
    is_rate_limited = response.status_code == 429 or (
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


def get_github_json(url, token=None, max_retries=8):
    """GET a GitHub REST API URL and return the parsed JSON.

    Hardened against the failure modes that silently broke daily CI reporting (the reports indexed
    into the response with e.g. ``result["jobs"]`` / ``workflow_run["created_at"]`` and raised a
    bare ``KeyError`` when GitHub returned an error payload instead of data):

      * primary *and* secondary rate limiting: retried with a backoff (``Retry-After`` /
        ``X-RateLimit-Reset`` when present, otherwise ~1 min for secondary limits), never parsed
        as data. Retrying without the token would only lower the limit, so the token is kept.
      * transient 5xx errors: retried with exponential backoff.

    Only a genuine 401/404 with a token falls back to an unauthenticated retry; a 403 is treated as
    a rate limit (above) or a non-retryable error, never as a reason to drop the token. Raises
    ``RuntimeError`` if no usable response is obtained, so callers fail loudly instead of indexing
    into an error payload.
    """
    headers = None
    if token:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    response = None
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        status = response.status_code

        wait = _rate_limit_wait(response, attempt)
        if wait is not None:
            print(
                f"GitHub API rate limited on {url} (status {status}); waiting {wait}s before "
                f"retry {attempt + 1}/{max_retries}"
            )
            time.sleep(wait)
            continue

        # Genuine auth/not-found with a token: retry once unauthenticated (previous behaviour).
        if headers is not None and status in (401, 404):
            response = requests.get(url)
            status = response.status_code

        if status >= 500:
            wait = min(2**attempt, 60)
            print(f"GitHub API server error {status} on {url}; retrying in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue

        if status == 200:
            return response.json()

        # Any other (non-retryable) status: stop and fail loudly below.
        break

    last_status = response.status_code if response is not None else "no response"
    raise RuntimeError(f"Could not fetch {url}: last status {last_status} after {max_retries} attempt(s)")


def _get_paginated_items(url, key, token=None):
    """Return all items found under ``key`` across the paginated pages of a GitHub API endpoint.

    ``url`` must already request ``per_page=50``. A missing ``key`` in a page raises ``KeyError``,
    but only after :func:`get_github_json` has already retried transient/rate-limit errors, so this
    only fires on a genuinely unexpected payload.
    """
    result = get_github_json(url, token=token)
    items = list(result[key])
    total_count = result.get("total_count", len(items))
    pages_to_iterate_over = math.ceil((total_count - 50) / 50)

    for i in range(pages_to_iterate_over):
        # Space out requests: a large run has ~20+ pages of jobs, and hammering them back-to-back is
        # what trips GitHub's secondary rate limit in the first place.
        time.sleep(3)
        result = get_github_json(url + f"&page={i + 2}", token=token)
        items.extend(result[key])

    return items


def get_jobs(workflow_run_id, token=None):
    """Extract jobs in a GitHub Actions workflow run"""

    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}/jobs?per_page=50"
    try:
        return _get_paginated_items(url, "jobs", token=token)
    except Exception:
        print(f"Unknown error, could not fetch jobs:\n{traceback.format_exc()}")

    return []


def get_job_links(workflow_run_id, token=None):
    """Extract job names and their job links in a GitHub Actions workflow run"""

    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}/jobs?per_page=50"
    try:
        jobs = _get_paginated_items(url, "jobs", token=token)
        return {job["name"]: job["html_url"] for job in jobs}
    except Exception:
        print(f"Unknown error, could not fetch links:\n{traceback.format_exc()}")

    return {}


def get_artifacts_links(workflow_run_id, token=None):
    """Get all artifact links from a workflow run"""

    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}/artifacts?per_page=50"
    try:
        artifacts = _get_paginated_items(url, "artifacts", token=token)
        return {artifact["name"]: artifact["archive_download_url"] for artifact in artifacts}
    except Exception:
        print(f"Unknown error, could not fetch links:\n{traceback.format_exc()}")

    return {}


def download_artifact(artifact_name, artifact_url, output_dir, token):
    """Download a GitHub Action artifact from a URL.

    The URL is of the form `https://api.github.com/repos/huggingface/transformers/actions/artifacts/{ARTIFACT_ID}/zip`,
    but it can't be used to download directly. We need to get a redirect URL first.
    See https://docs.github.com/en/rest/actions/artifacts#download-an-artifact
    """
    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    result = requests.get(artifact_url, headers=headers, allow_redirects=False)
    download_url = result.headers["Location"]
    response = requests.get(download_url, allow_redirects=True)
    file_path = os.path.join(output_dir, f"{artifact_name}.zip")
    with open(file_path, "wb") as fp:
        fp.write(response.content)


def get_errors_from_single_artifact(artifact_zip_path, job_links=None):
    """Extract errors from a downloaded artifact (in .zip format)"""
    errors = []
    failed_tests = []
    job_name = None

    with zipfile.ZipFile(artifact_zip_path) as z:
        for filename in z.namelist():
            if not os.path.isdir(filename):
                # read the file
                if filename in ["failures_line.txt", "summary_short.txt", "job_name.txt"]:
                    with z.open(filename) as f:
                        for line in f:
                            line = line.decode("UTF-8").strip()
                            if filename == "failures_line.txt":
                                try:
                                    # `error_line` is the place where `error` occurs
                                    error_line = line[: line.index(": ")]
                                    error = line[line.index(": ") + len(": ") :]
                                    errors.append([error_line, error])
                                except Exception:
                                    # skip un-related lines that don't match the expected format
                                    logger.debug(f"Skipping unrelated line: {line}")
                            elif filename == "summary_short.txt" and line.startswith("FAILED "):
                                # `test` is the test method that failed
                                test = line[len("FAILED ") :]
                                failed_tests.append(test)
                            elif filename == "job_name.txt":
                                job_name = line

    if len(errors) != len(failed_tests):
        raise ValueError(
            f"`errors` and `failed_tests` should have the same number of elements. Got {len(errors)} for `errors` "
            f"and {len(failed_tests)} for `failed_tests` instead. The test reports in {artifact_zip_path} have some"
            " problem."
        )

    job_link = None
    if job_name and job_links:
        job_link = job_links.get(job_name, None)

    # A list with elements of the form (line of error, error, failed test)
    result = [x + [y] + [job_link] for x, y in zip(errors, failed_tests)]

    return result


def get_all_errors(artifact_dir, job_links=None):
    """Extract errors from all artifact files"""

    errors = []

    paths = [os.path.join(artifact_dir, p) for p in os.listdir(artifact_dir) if p.endswith(".zip")]
    for p in paths:
        errors.extend(get_errors_from_single_artifact(p, job_links=job_links))

    return errors


def reduce_by_error(logs, error_filter=None):
    """count each error"""

    counter = Counter()
    counter.update([x[1] for x in logs])
    counts = counter.most_common()
    r = {}
    for error, count in counts:
        if error_filter is None or error not in error_filter:
            r[error] = {"count": count, "failed_tests": [(x[2], x[0]) for x in logs if x[1] == error]}

    r = dict(sorted(r.items(), key=lambda item: item[1]["count"], reverse=True))
    return r


def get_model(test):
    """Get the model name from a test method"""
    test = test.split("::")[0]
    if test.startswith("tests/models/"):
        test = test.split("/")[2]
    else:
        test = None

    return test


def reduce_by_model(logs, error_filter=None):
    """count each error per model"""

    logs = [(x[0], x[1], get_model(x[2])) for x in logs]
    logs = [x for x in logs if x[2] is not None]
    tests = {x[2] for x in logs}

    r = {}
    for test in tests:
        counter = Counter()
        # count by errors in `test`
        counter.update([x[1] for x in logs if x[2] == test])
        counts = counter.most_common()
        error_counts = {error: count for error, count in counts if (error_filter is None or error not in error_filter)}
        n_errors = sum(error_counts.values())
        if n_errors > 0:
            r[test] = {"count": n_errors, "errors": error_counts}

    r = dict(sorted(r.items(), key=lambda item: item[1]["count"], reverse=True))
    return r


def make_github_table(reduced_by_error):
    header = "| no. | error | status |"
    sep = "|-:|:-|:-|"
    lines = [header, sep]
    for error in reduced_by_error:
        count = reduced_by_error[error]["count"]
        line = f"| {count} | {error[:100]} |  |"
        lines.append(line)

    return "\n".join(lines)


def make_github_table_per_model(reduced_by_model):
    header = "| model | no. of errors | major error | count |"
    sep = "|-:|-:|-:|-:|"
    lines = [header, sep]
    for model in reduced_by_model:
        count = reduced_by_model[model]["count"]
        error, _count = list(reduced_by_model[model]["errors"].items())[0]
        line = f"| {model} | {count} | {error[:60]} | {_count} |"
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--workflow_run_id", type=str, required=True, help="A GitHub Actions workflow run id.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store the downloaded artifacts and other result files.",
    )
    parser.add_argument("--token", default=None, type=str, help="A token that has actions:read permission.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _job_links = get_job_links(args.workflow_run_id, token=args.token)
    job_links = {}
    # To deal with `workflow_call` event, where a job name is the combination of the job names in the caller and callee.
    # For example, `PyTorch 1.11 / Model tests (models/albert, single-gpu)`.
    if _job_links:
        for k, v in _job_links.items():
            # This is how GitHub actions combine job names.
            if " / " in k:
                index = k.find(" / ")
                k = k[index + len(" / ") :]
            job_links[k] = v
    with open(os.path.join(args.output_dir, "job_links.json"), "w", encoding="UTF-8") as fp:
        json.dump(job_links, fp, ensure_ascii=False, indent=4)

    artifacts = get_artifacts_links(args.workflow_run_id, token=args.token)
    with open(os.path.join(args.output_dir, "artifacts.json"), "w", encoding="UTF-8") as fp:
        json.dump(artifacts, fp, ensure_ascii=False, indent=4)

    for idx, (name, url) in enumerate(artifacts.items()):
        download_artifact(name, url, args.output_dir, args.token)
        # Be gentle to GitHub
        time.sleep(1)

    errors = get_all_errors(args.output_dir, job_links=job_links)

    # `e[1]` is the error
    counter = Counter()
    counter.update([e[1] for e in errors])

    # print the top 30 most common test errors
    most_common = counter.most_common(30)
    for item in most_common:
        print(item)

    with open(os.path.join(args.output_dir, "errors.json"), "w", encoding="UTF-8") as fp:
        json.dump(errors, fp, ensure_ascii=False, indent=4)

    reduced_by_error = reduce_by_error(errors)
    reduced_by_model = reduce_by_model(errors)

    s1 = make_github_table(reduced_by_error)
    s2 = make_github_table_per_model(reduced_by_model)

    with open(os.path.join(args.output_dir, "reduced_by_error.txt"), "w", encoding="UTF-8") as fp:
        fp.write(s1)
    with open(os.path.join(args.output_dir, "reduced_by_model.txt"), "w", encoding="UTF-8") as fp:
        fp.write(s2)
