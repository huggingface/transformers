import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _base_test_name(nodeid: str) -> str:
    # Strip parameters like [param=..] from the last component
    name = nodeid.split("::")[-1]
    return re.sub(r"\[.*\]$", "", name)


def _class_name(nodeid: str) -> str | None:
    parts = nodeid.split("::")
    # nodeid can be: file::Class::test or file::test
    if len(parts) >= 3:
        return parts[-2]
    return None


def _file_path(nodeid: str) -> str:
    return nodeid.split("::")[0]


def _modeling_key(file_path: str) -> str | None:
    # Extract "xxx" from test_modeling_xxx.py
    m = re.search(r"test_modeling_([A-Za-z0-9_]+)\.py$", file_path)
    if m:
        return m.group(1)
    return None


def summarize(report_path: str):
    p = Path(report_path)
    if not p.exists():
        raise FileNotFoundError(f"Report file not found: {p.resolve()}")

    data = json.loads(p.read_text())
    tests = data.get("tests", [])

    # Overall counts
    outcomes = Counter(t.get("outcome", "unknown") for t in tests)

    # Filter failures (pytest-json-report uses "failed" and may have "error")
    failed = [t for t in tests if t.get("outcome") in ("failed", "error")]

    # 1) Failures per test file
    failures_per_file = Counter(_file_path(t.get("nodeid", "")) for t in failed)

    # 2) Failures per class (if any; otherwise "NO_CLASS")
    failures_per_class = Counter((_class_name(t.get("nodeid", "")) or "NO_CLASS") for t in failed)

    # 3) Failures per base test name (function), aggregating parametrized cases
    failures_per_testname = Counter(_base_test_name(t.get("nodeid", "")) for t in failed)

    # 4) Failures per test_modeling_xxx (derived from filename)
    failures_per_modeling_key = Counter()
    for t in failed:
        key = _modeling_key(_file_path(t.get("nodeid", "")))
        if key:
            failures_per_modeling_key[key] += 1

    return {
        "outcomes": outcomes,
        "failures_per_file": failures_per_file,
        "failures_per_class": failures_per_class,
        "failures_per_testname": failures_per_testname,
        "failures_per_modeling_key": failures_per_modeling_key,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize pytest JSON report failures")
    parser.add_argument(
        "--report", default="report.json", help="Path to pytest JSON report file (default: report.json)"
    )
    args = parser.parse_args()

    try:
        summary = summarize(args.report)
    except FileNotFoundError as e:
        print(str(e))
        return

    outcomes = summary["outcomes"]
    print("=== Overall ===")
    total = sum(outcomes.values())
    print(f"Total tests: {total}")
    for k in sorted(outcomes):
        print(f"{k:>10}: {outcomes[k]}")

    def _print_counter(title, counter: Counter, label=""):
        print(f"\n=== {title} ===")
        if not counter:
            print("None")
            return
        for key, cnt in sorted(counter.items(), key=lambda x: (x[1], x[0])):
            if label:
                print(f"{cnt:4d}  {label}{key}")
            else:
                print(f"{cnt:4d}  {key}")

    _print_counter("Failures per test class", summary["failures_per_class"], label="class ")
    _print_counter("Failures per test_modeling_xxx", summary["failures_per_modeling_key"], label="model ")
    _print_counter("Failures per test file", summary["failures_per_file"])
    _print_counter("Failures per test name (base)", summary["failures_per_testname"])


if __name__ == "__main__":
    main()
