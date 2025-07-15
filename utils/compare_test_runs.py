# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import re


def normalize_test_line(line):
    line = line.strip()

    # Normalize SKIPPED/XFAIL/etc with path:line and reason
    match = re.match(r"^(SKIPPED|XFAIL|XPASS|EXPECTEDFAIL)\s+\[?\d*\]?\s*(\S+:\d+)", line)
    if match:
        status, location = match.groups()
        return f"{status} {location}"

    # Normalize ERROR/FAILED lines with optional message
    if line.startswith("ERROR") or line.startswith("FAILED"):
        return re.split(r"\s+-\s+", line)[0].strip()

    return line


def parse_summary_file(file_path):
    test_set = set()
    with open(file_path, "r", encoding="utf-8") as f:
        in_summary = False
        for line in f:
            if line.strip().startswith("==="):
                in_summary = not in_summary
                continue
            if in_summary:
                stripped = line.strip()
                if stripped:
                    normalized = normalize_test_line(stripped)
                    test_set.add(normalized)
    return test_set


def compare_job_sets(job_set1, job_set2):
    all_job_names = sorted(set(job_set1) | set(job_set2))
    report_lines = []

    for job_name in all_job_names:
        file1 = job_set1.get(job_name)
        file2 = job_set2.get(job_name)

        tests1 = parse_summary_file(file1) if file1 else set()
        tests2 = parse_summary_file(file2) if file2 else set()

        added = tests2 - tests1
        removed = tests1 - tests2

        if added or removed:
            report_lines.append(f"=== Diff for job: {job_name} ===")
            if removed:
                report_lines.append("--- Absent in current run:")
                for test in sorted(removed):
                    report_lines.append(f"    - {test}")
            if added:
                report_lines.append("+++ Appeared in current run:")
                for test in sorted(added):
                    report_lines.append(f"    + {test}")
            report_lines.append("")  # blank line

    return "\n".join(report_lines) if report_lines else "No differences found."


# Example usage:
# job_set_1 = {
#     "albert": "prev/multi-gpu_run_models_gpu_models/albert_test_reports/summary_short.txt",
#     "bloom": "prev/multi-gpu_run_models_gpu_models/bloom_test_reports/summary_short.txt",
# }

# job_set_2 = {
#     "albert": "curr/multi-gpu_run_models_gpu_models/albert_test_reports/summary_short.txt",
#     "bloom": "curr/multi-gpu_run_models_gpu_models/bloom_test_reports/summary_short.txt",
# }

# report = compare_job_sets(job_set_1, job_set_2)
# print(report)
