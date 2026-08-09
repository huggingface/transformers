"""
Memory tracker pytest plugin.

Measures RSS memory at the start and end of every test — no background thread,
no sampling gaps. Works for fast sub-second tests and with pytest-xdist (-n N).

USAGE
-----
# Activate via flag:
    pytest -p memory_tracker_plugin --track-memory -n 8 tests/test_xxx.py

# Activate via env var (useful in CI):
    MEMORY_TRACK=1 pytest -n 8 tests/test_xxx.py

# Optional env-var tuning:
    MEMORY_LOG_DIR=memory_logs   # where to write per-worker .jsonl files (default: memory_logs)

ANALYZING RESULTS
-----------------
    python memory_tracker_plugin.py memory_logs/
    python memory_tracker_plugin.py memory_logs/ --top 30

HOW IT WORKS
------------
- RSS is read synchronously right before and right after each test body (~0.1 ms overhead).
- No sampling thread: every test is measured exactly, regardless of duration.
- Each xdist worker writes memory_logs/worker_<id>.jsonl (line-buffered, safe mid-run).

FIELDS PER RECORD
-----------------
  start_mb  RSS when the test started
  end_mb    RSS when the test finished
  delta_mb  end - start: positive = memory retained after test (cumulative OOM risk)
  high_mb   running high-water mark of RSS seen by this worker up to this test

FINDING OOM CULPRITS
--------------------
- Sort by delta_mb  -> tests that retain memory; stacking deltas across a worker cause OOM.
- Sort by high_mb   -> absolute RSS at that point in the worker session; shows when limit is hit.
- Per-worker cumulative growth shows if one worker runs away while others are fine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# RSS reading (psutil preferred, /proc fallback for Linux without psutil)
# ---------------------------------------------------------------------------

def _get_rss_bytes() -> int:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        pass
    # Fallback: read /proc/self/status (Linux only)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except Exception:
        pass
    return 0


def _mb(n_bytes: int) -> float:
    return round(n_bytes / 1024**2, 1)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class MemoryTrackerPlugin:

    def __init__(self, log_dir: str) -> None:
        self._log_dir = log_dir
        self._worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        self._log_file = None
        self._high_water_bytes = 0  # running peak RSS seen by this worker

    def _get_log_file(self):
        """Open lazily so the file is created in the worker process, not the xdist controller."""
        if self._log_file is None:
            Path(self._log_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(self._log_dir, f"worker_{self._worker_id}.jsonl")
            self._log_file = open(path, "a", buffering=1)  # line-buffered: survives hard kills
        return self._log_file

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        """Wraps the test body only (not setup/teardown)."""
        start = _get_rss_bytes()
        try:
            yield  # ← test runs here
        finally:
            # Always runs — even when the test raises (failed/errored tests).
            end = _get_rss_bytes()

            if end > self._high_water_bytes:
                self._high_water_bytes = end

            # Stash on the item; pytest_runtest_makereport reads it below.
            item._mem_start_mb = _mb(start)
            item._mem_end_mb = _mb(end)
            item._mem_delta_mb = _mb(end - start)
            item._mem_high_mb = _mb(self._high_water_bytes)

            record = {
                "nodeid": item.nodeid,
                "worker": self._worker_id,
                "start_mb": item._mem_start_mb,
                "end_mb": item._mem_end_mb,
                "delta_mb": item._mem_delta_mb,
                "high_mb": item._mem_high_mb,
            }
            self._get_log_file().write(json.dumps(record) + "\n")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Attach memory data to the report via user_properties.

        user_properties is a List[Tuple[str, Any]] that pytest and pytest-xdist
        always serialize when sending reports from worker to controller — unlike
        arbitrary custom attributes which are version-dependent.
        """
        outcome = yield
        if call.when == "call":
            report = outcome.get_result()
            delta_mb = getattr(item, "_mem_delta_mb", None)
            if delta_mb is not None:
                report.user_properties.append(("memory_delta_mb", delta_mb))

    def pytest_sessionfinish(self, session, exitstatus):
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def pytest_terminal_summary(self, terminalreporter):
        """Print aggregated memory summary after pytest's own pass/fail/skip line."""
        import glob

        files = sorted(glob.glob(os.path.join(self._log_dir, "worker_*.jsonl")))
        if not files:
            return

        records = []
        for path in files:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        if not records:
            return

        # Only the controller process has a real terminalreporter; workers have a
        # no-op one. Guard so we don't print 8 copies of the summary.
        if not hasattr(terminalreporter, "write_sep"):
            return

        top_n = 30
        by_delta = sorted(records, key=lambda r: r["delta_mb"], reverse=True)

        terminalreporter.write_sep("=", f"memory usage — top {top_n} tests by retained RSS after test")
        terminalreporter.write_line(
            f"  {'Delta MB':>9}  {'End MB':>9}  {'Worker':>6}  Test"
        )
        terminalreporter.write_line(
            f"  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*60}"
        )
        for r in by_delta[:top_n]:
            sign = "+" if r["delta_mb"] >= 0 else ""
            terminalreporter.write_line(
                f"  {sign}{r['delta_mb']:>8.1f}  {r['end_mb']:>9.1f}  {r['worker']:>6}  {r['nodeid']}"
            )

        # Per-worker totals
        from collections import defaultdict
        by_worker = defaultdict(list)
        for r in records:
            by_worker[r["worker"]].append(r)

        terminalreporter.write_sep("-", "per-worker RSS summary")
        terminalreporter.write_line(
            f"  {'Worker':>6}  {'Tests':>6}  {'Peak RSS MB':>12}  {'Net growth MB':>14}"
        )
        for worker, recs in sorted(by_worker.items()):
            peak = max(r["high_mb"] for r in recs)
            growth = recs[-1]["end_mb"] - recs[0]["start_mb"]
            sign = "+" if growth >= 0 else ""
            terminalreporter.write_line(
                f"  {worker:>6}  {len(recs):>6}  {peak:>12.1f}  {sign}{growth:>13.1f}"
            )


# ---------------------------------------------------------------------------
# Pytest hooks (picked up when loaded via -p memory_tracker_plugin)
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    group = parser.getgroup("memory-tracker", "Memory usage tracking per test")
    group.addoption(
        "--memory-log-dir",
        default=None,
        metavar="DIR",
        help="Directory for per-worker .jsonl log files (default: memory_logs, or MEMORY_LOG_DIR env var).",
    )


def pytest_configure(config):
    log_dir = _safe_getoption(config, "--memory-log-dir") or os.environ.get("MEMORY_LOG_DIR", "memory_logs")

    # Avoid double-registration when xdist workers re-import this module
    if not config.pluginmanager.has_plugin("memory_tracker_plugin_instance"):
        config.pluginmanager.register(MemoryTrackerPlugin(log_dir=log_dir), "memory_tracker_plugin_instance")


def _safe_getoption(config, name):
    try:
        return config.getoption(name)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Stand-alone summary script:  python memory_tracker_plugin.py memory_logs/
# ---------------------------------------------------------------------------

def _summarize(log_dir: str, top_n: int = 20) -> None:
    import glob
    from collections import defaultdict

    files = sorted(glob.glob(os.path.join(log_dir, "worker_*.jsonl")))
    if not files:
        print(f"No worker_*.jsonl files found in {log_dir!r}")
        return

    records: list[dict] = []
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not records:
        print("No records found.")
        return

    print(f"\nTotal tests tracked: {len(records)}")

    # ---- Top N by memory retained after test (primary OOM signal for fast tests) ----
    by_delta = sorted(records, key=lambda r: r["delta_mb"], reverse=True)
    print(f"\nTop {top_n} by memory retained after test  (delta_mb = end - start):")
    print(f"  {'Delta MB':>9}  {'Start MB':>9}  {'End MB':>9}  Test")
    print(f"  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*60}")
    for r in by_delta[:top_n]:
        print(f"  {r['delta_mb']:>+9.1f}  {r['start_mb']:>9.1f}  {r['end_mb']:>9.1f}  {r['nodeid']}")

    # ---- Top N by absolute RSS at test end (shows when worker approached OOM) ----
    by_high = sorted(records, key=lambda r: r["high_mb"], reverse=True)
    print(f"\nTop {top_n} by worker RSS high-water mark at that point in session:")
    print(f"  {'High MB':>9}  {'End MB':>9}  {'Worker':>8}  Test")
    print(f"  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*60}")
    for r in by_high[:top_n]:
        print(f"  {r['high_mb']:>9.1f}  {r['end_mb']:>9.1f}  {r['worker']:>8}  {r['nodeid']}")

    # ---- Per-worker summary ----
    worker_records: dict[str, list] = defaultdict(list)
    for r in records:
        worker_records[r["worker"]].append(r)

    print(f"\nPer-worker summary:")
    print(f"  {'Worker':>8}  {'Tests':>6}  {'Peak RSS MB':>12}  {'Total Growth MB':>16}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*16}")
    for worker, recs in sorted(worker_records.items()):
        peak = max(r["high_mb"] for r in recs)
        growth = recs[-1]["end_mb"] - recs[0]["start_mb"]
        print(f"  {worker:>8}  {len(recs):>6}  {peak:>12.1f}  {growth:>+16.1f}")

    # ---- Cumulative growth timeline per worker (detect which test crosses a threshold) ----
    threshold_mb = 100  # warn if any single test adds more than this
    big_jumps = [r for r in records if r["delta_mb"] >= threshold_mb]
    if big_jumps:
        big_jumps.sort(key=lambda r: r["delta_mb"], reverse=True)
        print(f"\nTests with delta_mb >= {threshold_mb} MB (large single-test allocations):")
        for r in big_jumps:
            print(f"  {r['delta_mb']:>+9.1f} MB  [{r['worker']}]  {r['nodeid']}")

    print()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Summarize memory_tracker_plugin JSONL logs.")
    ap.add_argument("log_dir", help="Directory containing worker_*.jsonl files")
    ap.add_argument("--top", type=int, default=20, help="Number of top entries to show (default: 20)")
    args = ap.parse_args()

    _summarize(args.log_dir, top_n=args.top)
