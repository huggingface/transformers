"""
Generate vLLM CI failure report files from a GitHub Actions workflow run.

Fetches job logs via the GitHub REST API (no gh CLI dependency — works inside
Docker containers that don't have gh installed). Imports log-parsing functions
from analyze_gh_wf_runs.py, which must be in the same directory.

Usage:
  python analyze_vllm_ci.py --run-id <RUN_ID> [--token <TOKEN>]
                             [--output-dir <DIR>] [--repo <OWNER/REPO>]
                             [--workflow <WORKFLOW_FILE>] [--plugin vllm]

  --token   GitHub token with actions:read scope.
            Defaults to GITHUB_TOKEN env var. Without a token, only public repos
            at reduced rate limits are accessible (unauthenticated: 60 req/hr).

Output files written to --output-dir (default: current directory):
  failing_tests_YYYY_MM_DD.json               all failing tests per job, no trace
  failing_tests_with_trace_YYYY_MM_DD.json    all failing tests per job, with trace
  new_failing_tests_YYYY_MM_DD.json           only new failures vs prev run, no trace
  new_failing_tests_with_trace_YYYY_MM_DD.json  only new failures vs prev run, with trace
  failing_tests_categories_YYYY_MM_DD.json    failure counts per error category

Failures are grouped by job, keyed by the real job name (matrix shards are kept
as separate jobs). Each job entry looks like:
  "Test vLLM transformers backend (shard 2 / 4)": {
      "job_url": "https://github.com/<repo>/actions/runs/<run_id>/job/<job_id>",
      "num_failing": 3,
      "tests": [...]   # test IDs, or {test, short_error, trace} in *_with_trace files
  }
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

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
)

REPO = "huggingface/transformers"
WORKFLOW_FILE = "vllm-ci-caller.yml"


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

    # Step 2: fetch the signed blob URL without any auth headers
    try:
        with urllib.request.urlopen(blob_url, timeout=120) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Error fetching log content for job {job_id}: {e}", file=sys.stderr)
        return ""


def get_previous_run(current_run_id, repo=REPO, workflow=WORKFLOW_FILE, token=None):
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

def collect_run_failures(run_id, repo=REPO, token=None, plugin=None):
    """
    Fetch all job logs for a workflow run and parse pytest failures.

    Only fetches logs for jobs that did *not* succeed (skips successful jobs to
    avoid downloading large logs unnecessarily).

    Returns:
        (jobs_meta, grouped_failures)
        - jobs_meta: list of raw job dicts from the API
        - grouped_failures: dict  job_name ->
              list of {"test": str, "short_error": str, "trace": str}
          Tests are deduplicated within each job.
    """
    if plugin is None:
        plugin = TracePlugin()

    jobs = fetch_run_jobs(run_id, repo=repo, token=token)
    print(f"  Run {run_id}: {len(jobs)} jobs found", file=sys.stderr)

    grouped = defaultdict(list)

    for job in jobs:
        conclusion = job.get("conclusion") or "unknown"
        # Only fetch logs for jobs that actually ran and failed.
        # Skip success (no failures), skipped (never ran), and unknown (still
        # running — e.g. the report job itself querying its own run).
        if conclusion != "failure":
            continue

        job_name = job["name"]
        print(f"  [{conclusion}] {job_name}", file=sys.stderr)

        log = fetch_job_log(job["id"], repo=repo, token=token)
        if not log:
            continue

        failures = extract_failures_with_errors(log)
        if not failures:
            continue

        test_ids = [t for t, _ in failures]
        trace_map = classify_by_scanning_log(log, test_ids)
        short_error_map = {t: e for t, e in failures}

        seen_tests = set()
        for test_id in test_ids:
            if test_id in seen_tests:
                continue
            seen_tests.add(test_id)

            base_trace = trace_map.get(test_id, "(not found)")
            enhanced_trace = apply_plugin(plugin, log, test_id, base_trace)
            grouped[job_name].append({
                "test": test_id,
                "short_error": short_error_map.get(test_id, ""),
                "trace": enhanced_trace,
            })

    return jobs, dict(grouped)


# ---------------------------------------------------------------------------
# Categorisation of failures by error type
# ---------------------------------------------------------------------------

def group_by_category(grouped_failures):
    """
    Categorise all failing tests by error type.

    Input:  grouped_failures  (job_name -> list of {test, short_error, trace})
    Output: dict  category -> {"count": int, "tests": [test_id, ...]}
            sorted by descending count
    """
    categories = defaultdict(list)

    for items in grouped_failures.values():
        for item in items:
            text = item["short_error"] + "\n" + item["trace"]

            if "AmbiguousGlobalPerLayerAttributeError" in text:
                cat = "AmbiguousGlobalPerLayerAttributeError"
            elif "GatedRepoError" in text or "gated repo" in text.lower() or "restricted" in text:
                cat = "GatedRepoError (403 - no Hub access)"
            elif "ImportError" in text or "ModuleNotFoundError" in text:
                cat = "ImportError / missing module"
            elif "SKIPPED" in text:
                cat = "Skipped (not a real failure)"
            elif "Available memory" in text or "memory utilization" in text:
                cat = "OOM / memory utilization"
            elif "exit code 137" in text or "Killed" in text:
                cat = "OOM kill (exit 137)"
            elif "NotImplementedError" in text:
                cat = "NotImplementedError"
            elif "assert" in text.lower() or "AssertionError" in text:
                cat = "AssertionError"
            elif "timeout" in text.lower() or "Timeout" in text:
                cat = "Timeout"
            elif "ConnectionError" in text or "HTTPError" in text:
                cat = "Network / HTTP error"
            elif "FileNotFoundError" in text or "No such file" in text:
                cat = "FileNotFoundError"
            elif "RuntimeError" in text:
                cat = "RuntimeError"
            elif "ValueError" in text:
                cat = "ValueError"
            else:
                cat = "Other / unknown"

            categories[cat].append(item["test"])

    return {
        cat: {"count": len(tests), "tests": tests}
        for cat, tests in sorted(categories.items(), key=lambda x: -len(x[1]))
    }


# ---------------------------------------------------------------------------
# New-failure computation
# ---------------------------------------------------------------------------

def compute_new_failures(current_grouped, previous_grouped):
    """
    Return a grouped_failures dict containing only tests that are *new*:
    failing in the current run but not present in the previous run's failures.
    """
    prev_tests = {
        item["test"]
        for items in previous_grouped.values()
        for item in items
    }
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


def write_reports(run_id, run_url, current_grouped, job_urls, previous_run, previous_grouped, output_dir):
    """Write the five report files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y_%m_%d")
    date_iso = date_str.replace("_", "-")

    total_failing = sum(len(v) for v in current_grouped.values())
    prev_run_id = str(previous_run["id"]) if previous_run else None
    prev_run_url = previous_run.get("html_url") if previous_run else None

    # ------------------------------------------------------------------
    # 1. failing_tests_YYYY_MM_DD.json — all failures, no trace
    # ------------------------------------------------------------------
    path = os.path.join(output_dir, f"failing_tests_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": str(run_id),
            "run_url": run_url,
            "date": date_iso,
            "total_failing": total_failing,
            "jobs": _format_jobs(current_grouped, job_urls, with_trace=False),
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 2. failing_tests_with_trace_YYYY_MM_DD.json — all failures, with trace
    # ------------------------------------------------------------------
    path = os.path.join(output_dir, f"failing_tests_with_trace_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": str(run_id),
            "run_url": run_url,
            "date": date_iso,
            "total_failing": total_failing,
            "jobs": _format_jobs(current_grouped, job_urls, with_trace=True),
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 3 & 4. new_failing_tests — tests absent from the previous run's failures
    # ------------------------------------------------------------------
    new_grouped = compute_new_failures(current_grouped, previous_grouped)
    total_new = sum(len(v) for v in new_grouped.values())
    new_note = (
        "Tests that fail in the current run but were not failing "
        "in the most recent completed main-branch run."
    )

    path = os.path.join(output_dir, f"new_failing_tests_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": str(run_id),
            "run_url": run_url,
            "previous_run_id": prev_run_id,
            "previous_run_url": prev_run_url,
            "date": date_iso,
            "total_new_failing": total_new,
            "note": new_note,
            "jobs": _format_jobs(new_grouped, job_urls, with_trace=False),
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}", file=sys.stderr)

    path = os.path.join(output_dir, f"new_failing_tests_with_trace_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": str(run_id),
            "run_url": run_url,
            "previous_run_id": prev_run_id,
            "previous_run_url": prev_run_url,
            "date": date_iso,
            "total_new_failing": total_new,
            "note": new_note,
            "jobs": _format_jobs(new_grouped, job_urls, with_trace=True),
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 5. failing_tests_categories_YYYY_MM_DD.json — error category breakdown
    # ------------------------------------------------------------------
    categories = group_by_category(current_grouped)
    path = os.path.join(output_dir, f"failing_tests_categories_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": str(run_id),
            "run_url": run_url,
            "date": date_iso,
            "total_failing": total_failing,
            "categories": categories,
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate vLLM CI failure report files from a GitHub Actions workflow run."
    )
    parser.add_argument("--run-id", required=True, help="GitHub Actions workflow run ID")
    parser.add_argument(
        "--token", default=None,
        help="GitHub token with actions:read scope (default: GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory to write report files (default: current directory)",
    )
    parser.add_argument(
        "--repo", default=REPO,
        help=f"GitHub repo in owner/name format (default: {REPO})",
    )
    parser.add_argument(
        "--workflow", default=WORKFLOW_FILE,
        help=f"Workflow file name used to find the previous run (default: {WORKFLOW_FILE})",
    )
    parser.add_argument(
        "--plugin", default=None, choices=list(PLUGINS),
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
    current_jobs, current_grouped = collect_run_failures(
        args.run_id, repo=args.repo, token=token, plugin=plugin
    )
    job_urls = {job["name"]: job.get("html_url") for job in current_jobs}
    total = sum(len(v) for v in current_grouped.values())
    print(f"  Total unique failures: {total}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Find and collect failures for the previous completed run on main
    # ------------------------------------------------------------------
    print(f"\n=== Finding previous completed run on main branch ===", file=sys.stderr)
    previous_run = get_previous_run(
        args.run_id, repo=args.repo, workflow=args.workflow, token=token
    )
    previous_grouped = {}
    if previous_run:
        prev_id = str(previous_run["id"])
        print(
            f"  Found: run {prev_id}  {previous_run.get('html_url', '')}",
            file=sys.stderr,
        )
        print(f"\n=== Previous run: {prev_id} ===", file=sys.stderr)
        _, previous_grouped = collect_run_failures(prev_id, repo=args.repo, token=token)
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
        args.run_id, run_url,
        current_grouped, job_urls, previous_run, previous_grouped,
        args.output_dir,
    )
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
