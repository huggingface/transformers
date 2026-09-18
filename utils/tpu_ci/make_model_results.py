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
"""
Turn the `--make-reports` directories left behind by `utils/get_test_reports.py` into the
`model_results.json` artifact the daily model CI uploads.

The daily CI builds that file from GitHub Actions artifacts, which only exist for runners the CI can
reach. This script builds the same file from a local run instead, so a device whose wheels are not
in the shared CI images (currently TPU) can feed the same dashboard as the NVIDIA and AMD columns.

Usage:
    python3 utils/tpu_ci/make_model_results.py reports/
    python3 utils/tpu_ci/make_model_results.py reports/ --upload
    python3 utils/tpu_ci/make_model_results.py --self-check

Requires `slack_sdk`, which `utils/notification_service` imports at module level.
"""

import argparse
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# `notification_service` sits in `utils/`, one level up. Everything Slack- and GitHub-bound in it is
# under `if __name__ == "__main__":`, so importing it only brings in the report parsing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notification_service import handle_stacktraces, handle_test_results, pop_default  # noqa: E402


REPORT_DIR_SUFFIX = "_test_reports"

# Failure categories of the `model_results.json` schema, as built in `notification_service`.
TEST_CATEGORIES = [
    "PyTorch",
    "Tokenizers",
    "Pipelines",
    "Trainer",
    "ONNX",
    "Auto",
    "Quantization",
    "Unclassified",
]

# Which category a failing test id falls into, first match wins. Same rules as `notification_service`.
FAILURE_CATEGORY_RULES = [
    ("tests/quantization", "Quantization"),
    ("test_modeling", "PyTorch"),
    ("test_tokenization", "Tokenizers"),
    ("test_pipelines", "Pipelines"),
    ("test_trainer", "Trainer"),
    ("onnx", "ONNX"),
    ("auto", "Auto"),
]

# `--machine-type` as the CI spells it, mapped to the key the dashboard reads. The "gpu" in the CI
# spelling only ever shows up in report directory names, so devices reuse it rather than fork the
# naming on both sides.
MACHINE_TYPE_TO_GPU = {"single-gpu": "single", "multi-gpu": "multi"}

DEFAULT_REPO_ID = "hf-gcp-tpu-internal/transformers_daily_ci"
RESULTS_FOLDER = "ci_results_run_models_gpu"


def parse_report_dir_name(name: str) -> tuple[str, str] | None:
    """
    Split a `<machine type>_<suite>_<matrix name>_test_reports` directory name into its machine type
    and its matrix name (`models_bert`), which is the key the results are stored under. Returns
    `None` for anything that is not a report directory of a known machine type.
    """
    if not name.endswith(REPORT_DIR_SUFFIX):
        return None
    parts = name[: -len(REPORT_DIR_SUFFIX)].split("_")
    if len(parts) < 3 or parts[0] not in MACHINE_TYPE_TO_GPU:
        return None
    return parts[0], "_".join(parts[2:])


def read_report_dir(path: Path) -> dict[str, str]:
    """Read a report directory into a `{file stem: contents}` dict."""
    return {file.stem: file.read_text() for file in path.iterdir() if file.is_file()}


def categorize_failure(line: str) -> str:
    for pattern, category in FAILURE_CATEGORY_RULES:
        if re.search(pattern, line):
            return category
    return "Unclassified"


def new_entry() -> dict:
    return {
        "failed": {category: {"unclassified": 0, "single": 0, "multi": 0} for category in TEST_CATEGORIES},
        "errors": 0,
        "success": 0,
        "skipped": 0,
        "time_spent": [],
        "error": False,
        "failures": {},
        # GitHub-run specific, and the dashboard tolerates them being empty.
        "job_link": {},
        "captured_info": {},
    }


def build_model_results(reports_dir: Path) -> dict[str, dict]:
    """Aggregate every report directory under `reports_dir` into the `model_results.json` schema."""
    results: dict[str, dict] = {}

    for path in sorted(reports_dir.iterdir()):
        if not path.is_dir():
            continue
        parsed = parse_report_dir_name(path.name)
        if parsed is None:
            print(f"Skipping {path.name}: not a report directory of a known machine type")
            continue
        machine_type, matrix_name = parsed
        gpu = MACHINE_TYPE_TO_GPU[machine_type]

        artifact = read_report_dir(path)
        entry = results.setdefault(matrix_name, new_entry())

        if "summary_short" not in artifact:
            # The process was killed (CPU OOM for instance) or the run was interrupted.
            entry["error"] = True
        if "stats" not in artifact:
            continue

        _, errors, success, skipped, time_spent = handle_test_results(artifact["stats"])
        entry["success"] += success
        entry["errors"] += errors
        entry["skipped"] += skipped
        entry["time_spent"].append(float(time_spent[:-1]))

        stacktraces = handle_stacktraces(artifact["failures_line"])

        for line in artifact["summary_short"].split("\n"):
            if not line.startswith("FAILED "):
                continue
            # `run_test_using_subprocess` reports the same failure twice; the extra entry has no
            # stacktrace of its own and would shift every `stacktraces.pop` below.
            if " - Failed: (subprocess)" in line:
                continue
            line = line[len("FAILED ") :].split()[0].replace("\n", "")

            entry["failures"].setdefault(gpu, []).append(
                {"line": line, "trace": pop_default(stacktraces, 0, "Cannot retrieve error message.")}
            )
            entry["failed"][categorize_failure(line)][gpu] += 1

    return results


def render_summary(results: dict[str, dict]) -> str:
    """Render the run as markdown. Rebuilt from the reports every time, never appended to."""
    lines = ["| Model | Passed | Failed | Skipped | Errors | Time (s) |", "|---|---|---|---|---|---|"]
    totals = dict.fromkeys(["success", "failed", "skipped", "errors"], 0)
    for name, entry in sorted(results.items()):
        failed = sum(count for category in entry["failed"].values() for count in category.values())
        for key, value in (("success", entry["success"]), ("failed", failed)):
            totals[key] += value
        totals["skipped"] += entry["skipped"]
        totals["errors"] += entry["errors"]
        time_spent = sum(entry["time_spent"])
        flag = " ⚠️ incomplete" if entry["error"] else ""
        lines.append(
            f"| {name}{flag} | {entry['success']} | {failed} | {entry['skipped']} | {entry['errors']} | {time_spent:.0f} |"
        )
    lines.append(
        f"| **Total** | **{totals['success']}** | **{totals['failed']}** | "
        f"**{totals['skipped']}** | **{totals['errors']}** | |"
    )

    # A pass rate bought with skips is not a pass rate, so report the skips next to it.
    attempted = totals["success"] + totals["failed"]
    rate = f"{100 * totals['success'] / attempted:.1f}%" if attempted else "n/a"
    lines.append(f"\nPass rate over attempted tests: {rate} ({totals['skipped']} skipped, not counted).\n")

    for name, entry in sorted(results.items()):
        failures = [failure for gpu_failures in entry["failures"].values() for failure in gpu_failures]
        if not failures:
            continue
        lines.append(f"\n### {name}\n")
        lines += [f"- `{failure['line']}`\n  {failure['trace']}" for failure in failures]

    return "\n".join(lines) + "\n"


def upload(results_path: Path, repo_id: str, date: str) -> None:
    from huggingface_hub import HfApi

    path_in_repo = f"{date}/{RESULTS_FOLDER}/{results_path.name}"
    HfApi().upload_file(
        path_or_fileobj=str(results_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to {repo_id}/{path_in_repo}")


def self_check() -> None:
    """Parse a synthetic report directory, so the schema cannot silently rot."""
    # Note the plural in "2 errors": `handle_test_results` matches on "errors", so pytest's singular
    # "1 error" is not counted. That is how the NVIDIA and AMD columns are already built.
    stats = "==== 2 failed, 3 passed, 1 skipped, 1 warning, 2 errors in 12.34s ====\n"
    summary_short = (
        "FAILED tests/models/bert/test_modeling_bert.py::BertModelTest::test_a - AssertionError\n"
        "FAILED tests/models/bert/test_modeling_bert.py::BertModelTest::test_b - Failed: (subprocess)\n"
        "FAILED tests/models/bert/test_tokenization_bert.py::BertTokenizationTest::test_c - ValueError\n"
    )
    failures_line = (
        "=== FAILURES ===\n"
        "tests/models/bert/test_modeling_bert.py:42: AssertionError: nope\n"
        "tests/models/bert/test_tokenization_bert.py:99: ValueError: nope either\n"
        "\n"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_dir = Path(tmp_dir) / "multi-gpu_models_models_bert_test_reports"
        report_dir.mkdir()
        (report_dir / "stats.txt").write_text(stats)
        (report_dir / "summary_short.txt").write_text(summary_short)
        (report_dir / "failures_line.txt").write_text(failures_line)
        (Path(tmp_dir) / "not_a_report_dir").mkdir()

        results = build_model_results(Path(tmp_dir))

    assert list(results) == ["models_bert"], results
    entry = results["models_bert"]
    assert entry["success"] == 3, entry
    assert entry["skipped"] == 1, entry
    assert entry["errors"] == 2, entry
    assert entry["time_spent"] == [12.34], entry
    assert entry["error"] is False, entry
    assert entry["failed"]["PyTorch"] == {"unclassified": 0, "single": 0, "multi": 1}, entry
    assert entry["failed"]["Tokenizers"] == {"unclassified": 0, "single": 0, "multi": 1}, entry
    assert entry["failed"]["Unclassified"] == {"unclassified": 0, "single": 0, "multi": 0}, entry
    # The `- Failed: (subprocess)` line is dropped, so the two kept failures keep their own traces.
    assert [failure["line"] for failure in entry["failures"]["multi"]] == [
        "tests/models/bert/test_modeling_bert.py::BertModelTest::test_a",
        "tests/models/bert/test_tokenization_bert.py::BertTokenizationTest::test_c",
    ], entry
    assert entry["failures"]["multi"][0]["trace"] == "(line 42)  AssertionError: nope", entry
    assert entry["failures"]["multi"][1]["trace"] == "(line 99)  ValueError: nope either", entry

    summary = render_summary(results)
    assert "| models_bert | 3 | 2 | 1 | 2 | 12 |" in summary, summary
    assert "Pass rate over attempted tests: 60.0% (1 skipped, not counted)." in summary, summary
    print("Self-check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports_dir", nargs="?", type=Path, help="directory holding the `*_test_reports` dirs")
    parser.add_argument("--output", type=Path, default=Path("model_results.json"), help="where to write the results")
    parser.add_argument("--summary", type=Path, default=None, help="also write a markdown summary there")
    parser.add_argument("--upload", action="store_true", help="also upload the results to the dataset repo")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="dataset repo to upload to")
    parser.add_argument("--date", default=None, help="date folder to upload under, defaults to today (UTC)")
    parser.add_argument("--self-check", action="store_true", help="check the parsing against a synthetic report dir")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if args.reports_dir is None:
        parser.error("reports_dir is required unless --self-check is passed")

    results = build_model_results(args.reports_dir)
    if not results:
        raise SystemExit(f"No report directories found under {args.reports_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="UTF-8") as fp:
        json.dump(results, fp, indent=4, ensure_ascii=False)
    print(f"Wrote {len(results)} entries to {args.output}")

    if args.summary:
        args.summary.write_text(render_summary(results))
        print(f"Wrote the summary to {args.summary}")

    if args.upload:
        upload(args.output, args.repo_id, args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main()
