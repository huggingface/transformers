#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Inc. team.
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
Format a new_failures_with_bad_commit.json artifact into GitHub markdown files
for use in PR CI comments (self-comment-ci.yml and self-comment-ci-amd.yml).

Failures are bucketed into three categories:
- new          : bad_commit is not None, no "DIFFERENT error message"  (✅ ⟹ ❌)
- error_changed: bad_commit is not None, "DIFFERENT error message"     (❌ ⟹ ❌)
- same_error   : bad_commit is None, failure_at_base_commit non-empty  (❌ ⟹ ❌)
Flaky / bisect-failed entries (bad_commit is None, failure_at_base_commit empty) are discarded.

The red gate (header file written) fires when new OR error_changed failures exist.
Same-error-only failures produce a green status but are still shown in the details.

Outputs:
  --header-out  one-line status text written only when the red gate fires
  --details-out full <details> markdown block written whenever any failures exist
"""

import argparse
import json
import sys
from pathlib import Path


def bucket_failures(data):
    """
    Bucket failures from new_failures_with_bad_commit.json into three categories.
    Returns dict with keys 'new', 'error_changed', 'same_error',
    each a list of (model, failure) tuples.
    """
    buckets = {"new": [], "error_changed": [], "same_error": []}
    for model, model_result in data.items():
        for failures in model_result.values():
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                bad_commit = failure.get("bad_commit")
                status = failure.get("status", "")
                failure_at_base = failure.get("failure_at_base_commit", "")
                if bad_commit is not None:
                    if "DIFFERENT error message" in status:
                        buckets["error_changed"].append((model, failure))
                    else:
                        buckets["new"].append((model, failure))
                elif failure_at_base and "SAME error message" in status:
                    buckets["same_error"].append((model, failure))
    return buckets


def format_entries_by_model(entries, test_suffix_fn=None, model_indent=4, test_indent=8):
    """
    Group (model, failure) entries by model and format as indented markdown lines.
    model_indent / test_indent: number of &nbsp; to prepend to model / test lines.
    Returns a list of strings (lines), with a blank line after each model group.
    """
    nbsp = "&nbsp;"
    model_prefix = nbsp * model_indent
    test_prefix = nbsp * test_indent
    model_groups = {}
    model_order = []
    for model, failure in entries:
        if model not in model_groups:
            model_groups[model] = []
            model_order.append(model)
        model_groups[model].append(failure)
    lines = []
    for model in model_order:
        failures = model_groups[model]
        job_link = failures[0].get("job_link", "")
        lines.append(f"{model_prefix}- [{model}]({job_link}):")
        for failure in failures:
            test_name = failure.get("test", "")
            suffix = (" " + test_suffix_fn(failure)) if test_suffix_fn else ""
            lines.append(f"{test_prefix}`{test_name}`{suffix}")
        lines.append("")
    return lines


def format_report(data):
    """
    Format failure data into a GitHub markdown <details> block.
    Surfaces pre-existing failures (still failing on the PR) alongside new regressions.
    Returns (formatted_text, new_count, error_changed_count, same_error_count).
    formatted_text is an empty string if no failures of any kind.
    """
    buckets = bucket_failures(data)
    new_failures = buckets["new"]
    error_changed = buckets["error_changed"]
    same_error = buckets["same_error"]
    new_count = len(new_failures)
    error_changed_count = len(error_changed)
    same_error_count = len(same_error)
    preexisting_count = error_changed_count + same_error_count

    if new_count == 0 and preexisting_count == 0:
        return "", 0, 0, 0

    summary_parts = []
    if new_count > 0:
        summary_parts.append(f"{new_count} new")
    if preexisting_count > 0:
        summary_parts.append(f"{preexisting_count} pre-existing")
    summary = ", ".join(summary_parts)

    lines = ["<details>", f"<summary>Failures ({summary})</summary>", ""]

    # New failures section
    if new_failures:
        lines.append("&nbsp;&nbsp;&nbsp;&nbsp;**New failures:**")
        lines.append("")
        lines.extend(
            format_entries_by_model(
                new_failures,
                test_suffix_fn=lambda f: "(✅ ⟹ ❌)",
                model_indent=4,
                test_indent=8,
            )
        )

    # Separator between new and pre-existing sections
    if new_failures and preexisting_count > 0:
        lines.append("---")
        lines.append("")

    # Pre-existing section
    if preexisting_count > 0:
        lines.append(
            f"&nbsp;&nbsp;&nbsp;&nbsp;**Pre-existing failures (still failing) \u2014 {preexisting_count} total:**"
        )
        lines.append("")

        if error_changed:
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Error changed ({error_changed_count}):**")
            lines.append("")
            lines.extend(
                format_entries_by_model(
                    error_changed,
                    test_suffix_fn=lambda f: "(❌ ⟹ ❌)",
                    model_indent=8,
                    test_indent=12,
                )
            )

        if same_error:
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Same error ({same_error_count}):**")
            lines.append("")
            if same_error_count <= 10:
                lines.extend(format_entries_by_model(same_error, model_indent=8, test_indent=12))
            else:
                # Collapse with dl/dd trick; content indented relative to the dl/dd nesting
                se_lines = format_entries_by_model(same_error, model_indent=4, test_indent=8)
                lines.append(f"<dl><dd><dl><dd><details><summary>show all {same_error_count}</summary>")
                lines.append("")
                lines.append("&nbsp;")
                lines.extend(se_lines)
                lines.append("</details></dd></dl></dd></dl>")
                lines.append("")

    lines.append("</details>")
    return "\n".join(lines), new_count, error_changed_count, same_error_count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-path", required=True, help="Path to new_failures_with_bad_commit.json")
    parser.add_argument("--url-path", help="Path to new_failures_with_bad_commit_url.txt")
    parser.add_argument("--header-out", required=True, help="Output path for the one-line status header")
    parser.add_argument("--details-out", required=True, help="Output path for the <details> markdown block")
    parser.add_argument(
        "--kind",
        choices=["model", "quantization"],
        default="model",
        help="Type of CI job, used to label the header text (default: model)",
    )
    args = parser.parse_args()

    url = None
    if args.url_path:
        url_path = Path(args.url_path)
        if url_path.exists():
            url = url_path.read_text(encoding="utf-8").strip()

    report = {}
    report_path = Path(args.report_path)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    formatted, new_count, error_changed_count, _ = format_report(report)

    quant = args.kind == "quantization"

    if formatted:
        # Red gate: fires when new OR error_changed failures exist
        if new_count > 0 or error_changed_count > 0:
            if new_count > 0:
                label = "new failing quant tests" if quant else "new failing tests"
                url_text = f"[{new_count} {label}]({url})" if url else f"{new_count} {label}"
            else:
                label = "quant tests with changed errors" if quant else "tests with changed errors"
                url_text = f"[{error_changed_count} {label}]({url})" if url else f"{error_changed_count} {label}"
            Path(args.header_out).write_text(f"❌ {url_text} 😭", encoding="utf-8")
        Path(args.details_out).write_text(formatted, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
