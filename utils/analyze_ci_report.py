"""
Generate CI failure report files from a GitHub Actions workflow run (e.g. the vLLM and SGLang
integration workflows).

Fetches job logs via the GitHub REST API (no gh CLI dependency — works inside
Docker containers that don't have gh installed). Imports log-parsing functions
from analyze_gh_wf_runs.py, which must be in the same directory.

Usage:
  python utils/analyze_ci_report.py --run-id <RUN_ID> --workflow <WORKFLOW_FILE>
                                    [--token <TOKEN>] [--output-dir <DIR>]
                                    [--repo <OWNER/REPO>] [--plugin vllm]

  --token   GitHub token with actions:read scope.
            Defaults to GITHUB_TOKEN env var. Without a token, only public repos
            at reduced rate limits are accessible (unauthenticated: 60 req/hr).

Output files written to --output-dir (default: current directory):
  failing_tests_YYYY_MM_DD.json               all failing tests per job, no trace
  failing_tests_with_trace_YYYY_MM_DD.json    all failing tests per job, with trace
  new_failing_tests_YYYY_MM_DD.json           only new failures vs prev run, no trace
  new_failing_tests_with_trace_YYYY_MM_DD.json  only new failures vs prev run, with trace
  failing_tests_categories_YYYY_MM_DD.json    failing tests grouped by root-cause error
  job_results.json                            run metadata + per-job status, test counts and failures

Failures are grouped by job, keyed by the real job name (matrix shards are kept
as separate jobs). Each job entry looks like:
  "Test vLLM transformers backend (shard 2 / 4)": {
      "job_url": "https://github.com/<repo>/actions/runs/<run_id>/job/<job_id>",
      "num_failing": 3,
      "tests": [...]   # test IDs, or {test, short_error, trace} in *_with_trace files
  }

Known gaps (not handled yet — we will monitor the real CI runs and improve if
any of these show up):
  - A failed job without any parsed pytest failure is silently left out of the
    reports: e.g. a failing setup step (pip/uv install), OOM kill (exit 137), job
    timeout, or the non-pytest example scripts in "Test vLLM integration". The
    report then shows 0 failures for a red CI run. This already happened on the
    main run https://github.com/huggingface/transformers/actions/runs/41959891568
    (2026-11-05): all 13 jobs failed in "Install dependencies" (torch==2.14.0+cpu
    could not be resolved). It also hits jobs killed before pytest prints its
    summary (see the first gap of analyze_gh_wf_runs.py). Possible fix: list such
    jobs with their job URL and the last "##[error]" line.
    The same applies to the *previous* (baseline) run: if its jobs failed this
    way, it has 0 parsed failures and every current failure is reported as new
    (e.g. run 42196234595 vs baseline 42154858517). Possible fix: warn (or skip
    the comparison) when the baseline has failed jobs but no parsed failures.
  - Log-parsing gaps of analyze_gh_wf_runs.py (see its module docstring) apply here.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime


# ---------------------------------------------------------------------------
# Import parsing functions from analyze_gh_wf_runs (same directory as this file)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from analyze_gh_wf_runs import (  # noqa: E402
    PLUGINS,
    TracePlugin,
    apply_plugin,
    classify_by_scanning_log,
    extract_failures_with_errors,
    extract_test_counts,
)


REPO = "huggingface/transformers"


# ---------------------------------------------------------------------------
# GitHub REST API helpers (urllib only, no gh CLI)
# ---------------------------------------------------------------------------


def _api_get(url, token=None):
    """GET a GitHub API URL and return parsed JSON, or None on error."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} fetching {url}: {e.reason}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"  URL error fetching {url}: {e.reason}", file=sys.stderr)
        return None


def fetch_run_jobs(run_id, repo=REPO, token=None):
    """Return all jobs for a workflow run, handling pagination."""
    jobs = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}"
        data = _api_get(url, token=token)
        if data is None:
            break
        batch = data.get("jobs", [])
        jobs.extend(batch)
        if len(jobs) >= data.get("total_count", len(jobs)) or not batch:
            break
        page += 1
    return jobs


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirect handler that refuses to follow redirects so we can read Location ourselves."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_job_log(job_id, repo=REPO, token=None):
    """
    Return raw log text for a job.

    GitHub responds with a 302 redirect to a signed Azure Blob Storage URL.
    We must NOT forward the Authorization header to the blob URL — Azure
    rejects any request that carries a Bearer token it doesn't own (401).
    Fix: intercept the redirect, read the Location URL, then fetch the blob
    without any auth headers.
    """
    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Step 1: ask GitHub for the signed blob URL (expect 302, do NOT follow it)
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener.open(req, timeout=30) as resp:
            # Unexpected 200 — read directly
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308) and "Location" in e.headers:
            blob_url = e.headers["Location"]
        else:
            print(f"  HTTP {e.code} fetching log for job {job_id}: {e.reason}", file=sys.stderr)
            return ""
    except urllib.error.URLError as e:
        print(f"  URL error fetching log for job {job_id}: {e.reason}", file=sys.stderr)
        return ""

    # Step 2: fetch the signed blob URL without any auth headers. Retry: the blob storage can fail temporarily
    # (e.g. "HTTP Error 503: Egress is over the account limit").
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(blob_url, timeout=120) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  Error fetching log content for job {job_id} (attempt {attempt}/3): {e}", file=sys.stderr)
            if attempt < 3:
                time.sleep(10)
    return ""


def get_previous_run(current_run_id, workflow, repo=REPO, token=None):
    """
    Return the most recent *completed* workflow run on the main branch that is
    not the current run. Returns the run dict or None if not found.
    """
    url = (
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs"
        f"?branch=main&status=completed&per_page=10"
    )
    data = _api_get(url, token=token)
    if data is None:
        return None
    for run in data.get("workflow_runs", []):
        if str(run["id"]) != str(current_run_id):
            return run
    return None


# ---------------------------------------------------------------------------
# Failure collection
# ---------------------------------------------------------------------------


def collect_run_failures(run_id, repo=REPO, token=None, plugin=None, with_job_results=False):
    """
    Fetch all job logs for a workflow run and parse pytest failures.

    Only fetches logs for jobs that failed, plus (with `with_job_results`) the successful
    ones, to get their test counts.

    Returns:
        (jobs_meta, grouped_failures, job_results)
        - jobs_meta: list of raw job dicts from the API
        - grouped_failures: dict  job_name ->
              list of {"test": str, "short_error": str, "trace": str}
          Tests are deduplicated within each job.
        - job_results: dict  job_name -> {"status", "job_link", "passed", "failed", "errors", "skipped"}
          (empty without `with_job_results`). "status" is "passed", "failed", or "no_summary" for a
          failed job without pytest's final line (killed, setup failure, or log unavailable), which
          then has an "error" (its first "##[error]" line) instead of the test counts. Successful jobs
          without pytest's final line (e.g. setup jobs) are left out.
    """
    if plugin is None:
        plugin = TracePlugin()

    jobs = fetch_run_jobs(run_id, repo=repo, token=token)
    print(f"  Run {run_id}: {len(jobs)} jobs found", file=sys.stderr)

    grouped = defaultdict(list)
    job_results = {}

    for job in jobs:
        conclusion = job.get("conclusion") or "unknown"
        # Only fetch logs for jobs that actually ran and failed (and, with `with_job_results`, succeeded).
        # Skip skipped (never ran) and unknown (still running — e.g. the report job itself querying its own run).
        if conclusion != "failure" and not (with_job_results and conclusion == "success"):
            continue

        job_name = job["name"]
        print(f"  [{conclusion}] {job_name}", file=sys.stderr)

        log = fetch_job_log(job["id"], repo=repo, token=token)

        if with_job_results:
            counts = extract_test_counts(log) if log else None
            if counts is not None:
                status = "passed" if conclusion == "success" else "failed"
                job_results[job_name] = {"status": status, "job_link": job.get("html_url"), **counts}
            elif conclusion == "failure":
                error = next((line for line in log.splitlines() if "##[error]" in line), "") if log else ""
                job_results[job_name] = {
                    "status": "no_summary",
                    "job_link": job.get("html_url"),
                    "error": error.split("##[error]", 1)[1].strip() if error else "(log unavailable)",
                }

        if conclusion != "failure" or not log:
            continue

        failures = extract_failures_with_errors(log)
        if not failures:
            continue

        test_ids = [t for t, _ in failures]
        trace_map = classify_by_scanning_log(log, test_ids)
        short_error_map = dict(failures)

        seen_tests = set()
        for test_id in test_ids:
            if test_id in seen_tests:
                continue
            seen_tests.add(test_id)

            base_trace = trace_map.get(test_id, "(not found)")
            enhanced_trace = apply_plugin(plugin, log, test_id, base_trace)
            grouped[job_name].append(
                {
                    "test": test_id,
                    "short_error": short_error_map.get(test_id, ""),
                    "trace": enhanced_trace,
                }
            )

    return jobs, dict(grouped), job_results


# ---------------------------------------------------------------------------
# Categorisation of failures by root-cause error
# ---------------------------------------------------------------------------

# TODO: We could improve this to have a better grouping. The category is the full
# (lightly normalized) error message, so failures with the same cause but different
# numbers end up in separate categories, e.g. two CUDA OOMs:
#   "OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. ..."
#   "OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. ..."
# Seen on run 42196234595: 10 identical vLLM KV-cache errors ("Available memory on
# node 0 (X/111.33 GiB) ... less than requested memory for kv (Y/102.42 GiB)") ended
# up in 10 categories.
# Also, for tests without vLLM worker output the category comes from the pytest summary
# line, i.e. the *last* exception of a chain, which may embed input data. Seen on run
# 42207123286: 3 FireRedASR2 failures ended up in 2 categories, because the summary error
# is "ValueError: Failed to apply FireRedASR2Processor on data={... array([0.64, ...". The
# real cause is the first exception of the chain in the trace:
# "ModuleNotFoundError: No module named 'kaldi_native_fbank'".
# Possible approach: fixed labels for well-known cases (OOM, gated repo, timeout,
# network) matched on the root-cause line only; prefer the first exception of a chain
# (the first "E   SomeError: ..." line in the trace) over the summary line; strip data
# reprs (e.g. "data={...}", "array([...])") and replace numbers with a placeholder in the
# remaining messages; keep one original message as an example.

# A top-level exception line, e.g. "ValueError: No common block size for 16."
# or "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered ...".
_EXC_LINE_RE = re.compile(r"^((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt))(?::\s*(.*))?$")
# Short error from the pytest summary line: "<exc type>: <message>".
_SHORT_ERROR_RE = re.compile(r"^((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*):\s*(.*)$")
# pydantic's actual message, e.g. "  Value error, Model Runner V2 requires Triton. [type=value_error, ...".
_PYDANTIC_VALUE_ERROR_RE = re.compile(r"Value error, (.*?)(?: \[type=.*)?$", re.MULTILINE)
# c10 log prefix: "[rank0]:[F825 10:33:04.650907418 cpu_wna16.cpp:308] " -> "[cpu_wna16.cpp:308] "
_C10_PREFIX_RE = re.compile(r"(?:\[rank\d+\]:)?\[[IWEF]\d{3,4} [\d:.]+ ([^\]]+)\]")

# Header a trace plugin prepends to the pytest trace, e.g. "[vllm plugin] Subprocess ERROR lines for ...".
_PLUGIN_HEADER_RE = re.compile(r"\[\w+ plugin\]")

_MAX_ROOT_CAUSE_LEN = 300


def root_cause_error(item):
    """
    Return a normalized "<ExcType>: <message>" string describing why a test failed.

    - vLLM engine failures (EngineDeadError, "Engine core initialization failed")
      only wrap the real error raised in a worker subprocess. When the vLLM plugin
      prepended the subprocess ERROR lines, use the first exception found there.
    - Otherwise use the short error from the pytest summary line (first line only).
    - pydantic ValidationError: use the actual "Value error, ..." message instead
      of the generic "1 validation error for VllmConfig".
    - Run-specific noise (memory addresses, c10 log timestamps) is removed so the
      same failure gets the same category across runs.
    """
    trace, short_error = item["trace"], item["short_error"]
    exc_type, message = None, ""

    if _PLUGIN_HEADER_RE.match(trace):
        subprocess_lines = trace.split("\n--- pytest trace ---\n", 1)[0]
        for line in subprocess_lines.splitlines():
            m = _EXC_LINE_RE.match(line.rstrip())
            if m:
                exc_type, message = m.group(1), m.group(2) or ""
                break

    if exc_type is None and short_error:
        first_line = short_error.splitlines()[0].strip()
        m = _SHORT_ERROR_RE.match(first_line)
        if m:
            exc_type, message = m.group(1), m.group(2)
        else:
            exc_type, message = "", first_line

    if exc_type is None:
        return "Unknown"

    exc_type = exc_type.rsplit(".", 1)[-1]
    if exc_type == "ValidationError":
        m = _PYDANTIC_VALUE_ERROR_RE.search(trace)
        if m:
            message = m.group(1)

    message = _C10_PREFIX_RE.sub(r"[\1]", message)
    message = re.sub(r"\[rank\d+\]:", "", message)
    message = re.sub(r"0x[0-9a-fA-F]+", "0x...", message).strip()
    if len(message) > _MAX_ROOT_CAUSE_LEN:
        message = message[:_MAX_ROOT_CAUSE_LEN] + "..."

    if not exc_type:
        return message or "Unknown"
    return f"{exc_type}: {message}" if message else exc_type


def group_by_category(grouped_failures, job_urls):
    """
    Group failing tests by their root-cause error (see root_cause_error).

    Input:  grouped_failures  (job_name -> list of {test, short_error, trace})
    Output: dict  root_cause -> {"count": int, "jobs": {job_name -> {job_url, num_failing, tests}}}
            sorted by descending count
    """
    by_cause = defaultdict(lambda: defaultdict(list))
    for job_name, items in grouped_failures.items():
        for item in items:
            by_cause[root_cause_error(item)][job_name].append(item)

    def count(cause_jobs):
        return sum(len(items) for items in cause_jobs.values())

    return {
        cause: {
            "count": count(cause_jobs),
            "jobs": _format_jobs(cause_jobs, job_urls, with_trace=False),
        }
        for cause, cause_jobs in sorted(by_cause.items(), key=lambda x: (-count(x[1]), x[0]))
    }


# ---------------------------------------------------------------------------
# New-failure computation
# ---------------------------------------------------------------------------


def compute_new_failures(current_grouped, previous_grouped):
    """
    Return a grouped_failures dict containing only tests that are *new*:
    failing in the current run but not present in the previous run's failures.
    """
    prev_tests = {item["test"] for items in previous_grouped.values() for item in items}
    new_grouped = {}
    for job_name, items in current_grouped.items():
        new_items = [item for item in items if item["test"] not in prev_tests]
        if new_items:
            new_grouped[job_name] = new_items
    return new_grouped


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def _format_jobs(grouped, job_urls, with_trace):
    """
    Convert grouped_failures into the report's "jobs" mapping, sorted by job name:
      job_name -> {"job_url": str, "num_failing": int, "tests": [...]}
    With with_trace=False, "tests" holds only test IDs.
    """
    return {
        job_name: {
            "job_url": job_urls.get(job_name),
            "num_failing": len(items),
            "tests": items if with_trace else [item["test"] for item in items],
        }
        for job_name, items in sorted(grouped.items())
    }


def write_job_results(run, previous_run, job_results, current_grouped, previous_grouped, output_dir):
    """
    Write job_results.json: the run's metadata and, per job, its status, test counts and failures
    (like `model_results.json` of the daily CI). Used by `ci_report_notify.py`.
    """
    new_grouped = compute_new_failures(current_grouped, previous_grouped)
    jobs = {}
    for job_name, result in sorted(job_results.items()):
        jobs[job_name] = {
            **result,
            "new_failures": len(new_grouped.get(job_name, [])),
            "failures": [
                {"line": item["test"], "trace": item["short_error"]} for item in current_grouped.get(job_name, [])
            ],
        }

    path = os.path.join(output_dir, "job_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run": {
                    "id": str(run["id"]),
                    "url": run["html_url"],
                    "number": run["run_number"],
                    "workflow": run["name"],
                    "event": run["event"],
                    "branch": run["head_branch"],
                    "head_sha": run["head_sha"],
                    "created_at": run["created_at"],
                },
                "previous_run": {"id": str(previous_run["id"]), "url": previous_run["html_url"]}
                if previous_run
                else None,
                "jobs": jobs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {path}", file=sys.stderr)


def write_reports(run_id, run_url, current_grouped, job_urls, previous_run, previous_grouped, output_dir):
    """Write the five report files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y_%m_%d")
    date_iso = date_str.replace("_", "-")

    total_failing = sum(len(v) for v in current_grouped.values())
    prev_run_id = str(previous_run["id"]) if previous_run else None
    prev_run_url = previous_run.get("html_url") if previous_run else None

    # ------------------------------------------------------------------
    # 1. failing_tests_YYYY_MM_DD.json — all failures, no trace
    # ------------------------------------------------------------------
    path = os.path.join(output_dir, f"failing_tests_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": str(run_id),
                "run_url": run_url,
                "date": date_iso,
                "total_failing": total_failing,
                "jobs": _format_jobs(current_grouped, job_urls, with_trace=False),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 2. failing_tests_with_trace_YYYY_MM_DD.json — all failures, with trace
    # ------------------------------------------------------------------
    path = os.path.join(output_dir, f"failing_tests_with_trace_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": str(run_id),
                "run_url": run_url,
                "date": date_iso,
                "total_failing": total_failing,
                "jobs": _format_jobs(current_grouped, job_urls, with_trace=True),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 3 & 4. new_failing_tests — tests absent from the previous run's failures
    # ------------------------------------------------------------------
    new_grouped = compute_new_failures(current_grouped, previous_grouped)
    total_new = sum(len(v) for v in new_grouped.values())
    new_note = "Tests that fail in the current run but were not failing in the most recent completed main-branch run."

    path = os.path.join(output_dir, f"new_failing_tests_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": str(run_id),
                "run_url": run_url,
                "previous_run_id": prev_run_id,
                "previous_run_url": prev_run_url,
                "date": date_iso,
                "total_new_failing": total_new,
                "note": new_note,
                "jobs": _format_jobs(new_grouped, job_urls, with_trace=False),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {path}", file=sys.stderr)

    path = os.path.join(output_dir, f"new_failing_tests_with_trace_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": str(run_id),
                "run_url": run_url,
                "previous_run_id": prev_run_id,
                "previous_run_url": prev_run_url,
                "date": date_iso,
                "total_new_failing": total_new,
                "note": new_note,
                "jobs": _format_jobs(new_grouped, job_urls, with_trace=True),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 5. failing_tests_categories_YYYY_MM_DD.json — error category breakdown
    # ------------------------------------------------------------------
    categories = group_by_category(current_grouped, job_urls)
    path = os.path.join(output_dir, f"failing_tests_categories_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": str(run_id),
                "run_url": run_url,
                "date": date_iso,
                "total_failing": total_failing,
                "categories": categories,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate CI failure report files from a GitHub Actions workflow run."
    )
    parser.add_argument("--run-id", required=True, help="GitHub Actions workflow run ID")
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub token with actions:read scope (default: GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write report files (default: current directory)",
    )
    parser.add_argument(
        "--repo",
        default=REPO,
        help=f"GitHub repo in owner/name format (default: {REPO})",
    )
    parser.add_argument(
        "--workflow",
        required=True,
        help="Workflow file name (e.g. vllm-ci-caller.yml), used to find the previous run on main",
    )
    parser.add_argument(
        "--plugin",
        default=None,
        choices=list(PLUGINS),
        help="Trace-enhancement plugin (e.g. vllm)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "WARNING: No GitHub token provided. Using unauthenticated requests "
            "(60 req/hr rate limit, public repos only).",
            file=sys.stderr,
        )

    plugin = PLUGINS[args.plugin]() if args.plugin else TracePlugin()
    run_url = f"https://github.com/{args.repo}/actions/runs/{args.run_id}"

    # ------------------------------------------------------------------
    # Collect failures for the current run
    # ------------------------------------------------------------------
    print(f"\n=== Current run: {args.run_id} ===", file=sys.stderr)
    current_jobs, current_grouped, job_results = collect_run_failures(
        args.run_id, repo=args.repo, token=token, plugin=plugin, with_job_results=True
    )
    job_urls = {job["name"]: job.get("html_url") for job in current_jobs}
    total = sum(len(v) for v in current_grouped.values())
    print(f"  Total unique failures: {total}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Find and collect failures for the previous completed run on main
    # ------------------------------------------------------------------
    print("\n=== Finding previous completed run on main branch ===", file=sys.stderr)
    previous_run = get_previous_run(args.run_id, repo=args.repo, workflow=args.workflow, token=token)
    previous_grouped = {}
    if previous_run:
        prev_id = str(previous_run["id"])
        print(
            f"  Found: run {prev_id}  {previous_run.get('html_url', '')}",
            file=sys.stderr,
        )
        print(f"\n=== Previous run: {prev_id} ===", file=sys.stderr)
        _, previous_grouped, _ = collect_run_failures(prev_id, repo=args.repo, token=token)
        prev_total = sum(len(v) for v in previous_grouped.values())
        print(f"  Total previous failures: {prev_total}", file=sys.stderr)
    else:
        print(
            "  No previous run found — new_failing_tests will equal all failing tests.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Write the five report files
    # ------------------------------------------------------------------
    print(f"\n=== Writing reports to: {args.output_dir} ===", file=sys.stderr)
    write_reports(
        args.run_id,
        run_url,
        current_grouped,
        job_urls,
        previous_run,
        previous_grouped,
        args.output_dir,
    )
    run = _api_get(f"https://api.github.com/repos/{args.repo}/actions/runs/{args.run_id}", token=token)
    if run is not None:
        write_job_results(run, previous_run, job_results, current_grouped, previous_grouped, args.output_dir)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
