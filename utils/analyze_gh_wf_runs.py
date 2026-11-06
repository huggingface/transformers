"""
Fetch and parse raw logs from GitHub Actions workflow or job runs.

Low-level utilities for:
- Fetching job metadata and raw logs via the gh CLI
- Parsing pytest failure output (summary section + failure blocks)
- Extracting per-test error traces

Import the parsing functions from other scripts, or run directly for interactive debugging.

Usage:
  python utils/analyze_gh_wf_runs.py <url-or-id> [--list | --traces] [--plugin vllm] [--file <log.txt>]

  url-or-id can be:
    - Workflow run URL:  https://github.com/huggingface/transformers/actions/runs/31430051419
    - Job URL:          https://github.com/huggingface/transformers/actions/runs/31430051419/job/93590992881
    - Bare run ID:      31430051419

Known gaps (not handled yet — we will monitor the real CI runs and improve if
any of these show up):
  - Only the first "short test summary info" section is parsed, so a job that
    runs pytest more than once only reports the first run's failures.
  - Node ids containing spaces (e.g. a parametrize id with a space) are not matched.
  - Failure blocks are matched by name (the "____ name ____" header has no file
    path), so two tests with the same name in different files of one job both
    get the last block; if no block matches, the fuzzy fallback may return the
    first lines of another parametrization's block.
  - VLLMPlugin only recognizes "(Worker pid=N)" / "(EngineCore pid=N)" ERROR
    lines; other tags such as "(EngineCore_DP0 pid=N)" or "(APIServer pid=N)"
    are not picked up. Possible fix: accept any process name in the
    "(<name> pid=N) ERROR" prefix instead of listing the names (in both the
    match and the prefix-strip regexes).
"""

import json
import re
import subprocess
import sys


REPO = "huggingface/transformers"


# ---------------------------------------------------------------------------
# GitHub API via gh CLI
# ---------------------------------------------------------------------------


def gh_api(path, extra_args=None):
    """Call the gh CLI and return stdout, or None on error."""
    # Strip leading slash — Windows Git Bash rewrites /foo as a filesystem path
    path = path.lstrip("/")
    cmd = ["gh", "api", path]
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"ERROR: gh api {path}\n{result.stderr}", file=sys.stderr)
        return None
    return result.stdout


def parse_url(url):
    """
    Parse a GitHub Actions URL or bare run ID.
    Returns ('run', run_id) or ('job', job_id).
    """
    m = re.search(r"/runs/(\d+)/job/(\d+)", url)
    if m:
        return "job", m.group(2)
    m = re.search(r"/runs/(\d+)", url)
    if m:
        return "run", m.group(1)
    if url.isdigit():
        return "run", url
    raise ValueError(f"Cannot parse URL or ID: {url!r}")


def get_run_jobs(run_id):
    """Return all jobs for a workflow run (paginated)."""
    data = gh_api(f"repos/{REPO}/actions/runs/{run_id}/jobs", ["--paginate"])
    if data is None:
        return []
    return json.loads(data)["jobs"]


def get_job_info(job_id):
    """Return job metadata dict."""
    data = gh_api(f"repos/{REPO}/actions/jobs/{job_id}")
    if data is None:
        return None
    return json.loads(data)


def get_job_log(job_id):
    """Return raw log text for a job via gh CLI."""
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def _strip_timestamp(line):
    """Remove GitHub Actions log timestamp prefix (e.g. '2026-08-07T14:16:35.123Z ')."""
    return re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ", "", line)


# A failing entry of pytest's short test summary: "FAILED <nodeid> - <message>" or "ERROR <nodeid> - <message>"
# (ERROR = error in setup/teardown/collection). The node id must look like a test path, so that log lines such as
# "ERROR 08-25 10:31:30 [file.py:12] ..." are not mistaken for test errors.
_SUMMARY_ENTRY_RE = re.compile(r"(FAILED|ERROR)\s+(\S+\.py(?:::\S+)?)(?:\s+-\s+(.*))?$")


def extract_failures_with_errors(log):
    """
    Parse the 'short test summary info' section of a pytest log.
    Returns list of (test_nodeid, short_error_text) tuples, for both FAILED and ERROR entries.

    Handles multi-line error messages — continuation lines between entries
    are joined and included in the error text, e.g.:

        FAILED tests/foo.py::test_bar[ExaoneForCausalLM] - OSError: You are trying to access a gated repo.
        Make sure to have access to it at https://huggingface.co/LGAI-EXAONE/...
        403 Client Error. ...
        ERROR tests/foo.py::TestBaz::test_qux - ValueError: ...

    Falls back to scanning bare FAILED/ERROR lines anywhere in the log if the summary
    section is not present (e.g. truncated log).
    """
    lines = [_strip_timestamp(l) for l in log.splitlines()]

    summary_start = None
    for i, line in enumerate(lines):
        if "short test summary info" in line:
            summary_start = i + 1
            break

    if summary_start is None:
        # Fallback: parse FAILED/ERROR lines anywhere in log (single-line errors only)
        results = []
        for line in lines:
            m = _SUMMARY_ENTRY_RE.match(line.strip())
            if m and m.group(3) is not None:
                results.append((m.group(2), m.group(3).strip()))
        return results

    results = []
    current_test = None
    current_error_lines = []

    for line in lines[summary_start:]:
        stripped = line.strip()

        # End of summary section (e.g. "=== 44 failed, 17 passed ... ===")
        if re.match(r"=+", stripped):
            break

        m = _SUMMARY_ENTRY_RE.match(stripped)
        if m:
            if current_test is not None:
                results.append((current_test, "\n".join(current_error_lines).strip()))
            current_test = m.group(2)
            current_error_lines = [(m.group(3) or "").strip()]
        elif current_test is not None:
            # Continuation line of a multi-line error (may be empty)
            current_error_lines.append(stripped)

    if current_test is not None:
        results.append((current_test, "\n".join(current_error_lines).strip()))

    return results


def extract_failures(log):
    """Return list of failed test node IDs."""
    return [test_id for test_id, _ in extract_failures_with_errors(log)]


# Prefix of failure-block headers for errors outside the test call, e.g. "ERROR at setup of TestFoo.test_bar".
_BLOCK_ERROR_PREFIX_RE = re.compile(r"^ERROR (?:at (?:setup|teardown) of|collecting) ")


def extract_all_failure_blocks(log):
    """
    Split a pytest log into {test_name: [lines]} by parsing failure blocks.

    Pytest structure:
      ________ test_name ________            (or "TestClass.test_name" for class-based tests,
      ...traceback...                         "ERROR at setup of ..." for setup/teardown errors)
      E   ActualError: message
      ________ next test ________

    Keys are the header names without the "ERROR at setup of " / "ERROR at teardown of " prefix.
    """
    lines = [_strip_timestamp(l) for l in log.splitlines()]

    blocks = {}
    current_name = None
    current_lines = []

    for line in lines:
        m = re.match(r"_+\s+(.+?)\s+_+$", line)
        if m:
            if current_name:
                blocks[current_name] = current_lines
            current_name = _BLOCK_ERROR_PREFIX_RE.sub("", m.group(1))
            current_lines = []
        elif "short test summary info" in line or "warnings summary" in line:
            if current_name:
                blocks[current_name] = current_lines
            break
        elif current_name:
            current_lines.append(line)

    return blocks


def classify_by_scanning_log(log, failures):
    """
    Match each failed test to its failure block and return error lines.
    Returns dict: test_nodeid -> error_string
    """
    blocks = extract_all_failure_blocks(log)
    results = {}

    for test_nodeid in failures:
        test_name = test_nodeid.split("::")[-1]  # e.g. test_foo[param-bar]
        # Class-based tests: node id "file.py::TestFoo::test_bar" has the header "TestFoo.test_bar"
        qualified_name = ".".join(test_nodeid.split("::")[1:])
        if qualified_name in blocks:
            results[test_nodeid] = "\n".join(blocks[qualified_name])
        elif test_name in blocks:
            results[test_nodeid] = "\n".join(blocks[test_name])
        else:
            # Fuzzy: find a block key that starts with the same function name
            func_name = re.sub(r"\[.*\]$", "", test_name)
            matches = [k for k in blocks if k.startswith(func_name)]
            if matches:
                results[test_nodeid] = "\n".join(blocks[matches[0]][:5])
            else:
                results[test_nodeid] = "(block not found)"

    return results


# ---------------------------------------------------------------------------
# Plugin system
# ---------------------------------------------------------------------------


class TracePlugin:
    """
    Base plugin. enhance_trace() receives the log text, test node id, and the
    base trace extracted by classify_by_scanning_log(), and returns a
    (possibly augmented) trace string. Default implementation is a no-op.
    """

    name = "default"

    def enhance_trace(self, log: str, test_nodeid: str, base_trace: str) -> str:
        return base_trace


class VLLMPlugin(TracePlugin):
    """
    vLLM-aware plugin.

    vLLM spawns worker/engine-core subprocesses that print their own ERROR
    lines to stdout *before* pytest's failure block, e.g.:

        2026-08-12T14:23:03Z  tests/…::test_foo[DeepSeekMTPModel] Fork a new process to run a test 0
        ...
        (Worker pid=1669) ERROR … ImportError: cannot import name …

    When the base trace says "See root cause above", this plugin finds the
    "Fork a new process" anchor for the test and collects all subsequent
    (Worker|EngineCore) ERROR lines up to the next test or summary section.
    Those lines are prepended to the base trace so the real root cause is
    visible.
    """

    name = "vllm"

    _WORKER_ERROR_RE = re.compile(r"\((?:Worker|EngineCore) pid=\d+\)\s+ERROR")
    _TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ")
    _FAILED_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z FAILED\b")

    def _find_range(self, lines: list, stripped: list, test_nodeid: str):
        """
        Return (start, end) line indices bounding this test's subprocess output.
        Tries two strategies:
          1. "Fork a new process to run a test 0" anchor
          2. First timestamped line mentioning test_nodeid → nearest FAILED line
        Returns None if neither anchor is found.
        """
        # Strategy 1: fork anchor
        fork_anchor = f"{test_nodeid} Fork a new process to run a test 0"
        for i, s in enumerate(stripped):
            if fork_anchor in s:
                end = len(lines)
                for j, s2 in enumerate(stripped[i + 1 :], i + 1):
                    if "Fork a new process to run a test 0" in s2 or re.match(r"(PASSED|FAILED|={5,})", s2.strip()):
                        end = j
                        break
                return i + 1, end

        # Strategy 2: first line where pytest starts reporting this test (it begins with the node id, unlike
        # e.g. a "Running N items in this shard: <all node ids>" listing) → nearest FAILED line.
        # If that line already contains "FAILED", there is no live log block in between: nothing to extract.
        start = None
        for i, raw in enumerate(lines):
            if not self._TIMESTAMP_RE.match(raw):
                continue
            stripped_raw = re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ", "", raw)
            if stripped_raw == test_nodeid or stripped_raw.startswith(test_nodeid + " "):
                if "FAILED" in stripped_raw:
                    return None
                start = i
                break
        if start is None:
            return None
        end = len(lines)
        for j, raw in enumerate(lines[start + 1 :], start + 1):
            if self._FAILED_LINE_RE.match(raw):
                end = j + 1
                break
        return start, end

    def enhance_trace(self, log: str, test_nodeid: str, base_trace: str) -> str:
        lines = log.splitlines()
        stripped = [re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ", "", l) for l in lines]

        result = self._find_range(lines, stripped, test_nodeid)
        if result is None:
            return base_trace
        start, end = result

        error_lines = []
        last_proc_prefix = None
        for s in stripped[start:end]:
            if self._WORKER_ERROR_RE.search(s):
                m_prefix = re.match(r"(\((?:Worker|EngineCore) pid=\d+\))", s)
                proc_prefix = m_prefix.group(1) if m_prefix else None
                if error_lines and proc_prefix != last_proc_prefix:
                    error_lines.append("")
                last_proc_prefix = proc_prefix
                # Strip the "(Worker pid=XXXX) ERROR HH:MM:SS [mod:line] " prefix for readability
                clean = re.sub(r"^\((?:Worker|EngineCore) pid=\d+\)\s+ERROR[\d: .-]*\[[\w/.]+:\d+\] ?", "", s)
                error_lines.append(clean if clean.strip() else s)

        if not error_lines:
            return base_trace

        header = f"[vllm plugin] Subprocess ERROR lines for {test_nodeid}:\n"
        return header + "\n".join(error_lines) + "\n\n--- pytest trace ---\n" + base_trace


PLUGINS: dict[str, type[TracePlugin]] = {
    VLLMPlugin.name: VLLMPlugin,
}


def apply_plugin(plugin: TracePlugin, log: str, test_nodeid: str, base_trace: str) -> str:
    return plugin.enhance_trace(log, test_nodeid, base_trace)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and parse GitHub Actions job logs for pytest failures.")
    parser.add_argument("url", nargs="?", help="Workflow run URL, job URL, or bare run ID")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print (test_nodeid, short_error) pairs and save findings_list.json",
    )
    parser.add_argument(
        "--traces",
        action="store_true",
        help="Print (test_nodeid, short_error, full_trace) and save findings_traces.json",
    )
    parser.add_argument(
        "--plugin",
        default=None,
        choices=list(PLUGINS),
        help="Trace-enhancement plugin (e.g. vllm)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Read log from a local file instead of fetching from GitHub",
    )
    args = parser.parse_args()

    if not args.url and not args.file:
        parser.error("Provide a workflow run URL/ID or --file <log.txt>")

    plugin: TracePlugin = PLUGINS[args.plugin]() if args.plugin else TracePlugin()

    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            job_logs = {"(local file)": f.read()}
        print(f"Loaded local file: {args.file}", file=sys.stderr)
    else:
        kind, id_ = parse_url(args.url)
        if kind == "job":
            job = get_job_info(id_)
            jobs = [job] if job else []
        else:
            jobs = get_run_jobs(id_)

        print(f"Found {len(jobs)} jobs", file=sys.stderr)
        job_logs = {}
        for job in jobs:
            print(f"Fetching log: {job['name']} ...", file=sys.stderr)
            job_logs[job["name"]] = get_job_log(job["id"])

    # Deduplicated (test_nodeid, short_error) pairs across all jobs
    seen = set()
    failures_with_errors = []
    for log in job_logs.values():
        for test_nodeid, error in extract_failures_with_errors(log):
            if test_nodeid not in seen:
                seen.add(test_nodeid)
                failures_with_errors.append((test_nodeid, error))

    print(f"Unique failures: {len(failures_with_errors)}", file=sys.stderr)

    if args.list:
        for test_nodeid, error in failures_with_errors:
            print(f"{test_nodeid}\n  {error}\n")
        data = [[t, e] for t, e in failures_with_errors]
        with open("findings_list.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} entries to findings_list.json", file=sys.stderr)
        return

    # Build full traces (needed for --traces)
    nodeid_to_log = {}
    for log in job_logs.values():
        for test_nodeid, _ in extract_failures_with_errors(log):
            if test_nodeid not in nodeid_to_log:
                nodeid_to_log[test_nodeid] = log

    failures_with_traces = []
    for log in job_logs.values():
        job_nodeids = [t for t, _ in extract_failures_with_errors(log) if nodeid_to_log.get(t) is log]
        error_map = classify_by_scanning_log(log, job_nodeids)
        short_error_map = dict(extract_failures_with_errors(log))
        for test_nodeid in job_nodeids:
            base_trace = error_map.get(test_nodeid, "(not found)")
            enhanced_trace = apply_plugin(plugin, log, test_nodeid, base_trace)
            failures_with_traces.append(
                (
                    test_nodeid,
                    short_error_map.get(test_nodeid, ""),
                    enhanced_trace,
                )
            )

    if args.traces:
        for test_nodeid, short_error, full_trace in failures_with_traces:
            print(f"{test_nodeid}")
            print(f"  short: {short_error}")
            print("  trace:\n    " + "\n    ".join(full_trace.splitlines()))
            print()
        data = [[t, s, tr] for t, s, tr in failures_with_traces]
        with open("findings_traces.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} entries to findings_traces.json", file=sys.stderr)
        return

    # Default: print summary
    print(f"\nTotal unique failures: {len(failures_with_errors)}")
    print("Pass --list or --traces for detailed output.")


if __name__ == "__main__":
    main()
