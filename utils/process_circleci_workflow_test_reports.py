# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import requests


def parse_failure_lines(text: str) -> list[dict]:
    """Extract failed test entries with basic metadata."""
    failures = []
    if not text:
        return failures

    for raw_line in text.splitlines():
        if not raw_line.startswith("FAILED "):
            continue
        entry = raw_line[len("FAILED ") :].strip()
        test_id, _, reason = entry.partition(" - ")
        test_id = test_id.strip()
        reason = reason.strip()
        base_file = test_id.split("::")[0]
        model = None
        if base_file.startswith("tests/models/"):
            parts = base_file.split("/")
            if len(parts) >= 3:
                model = parts[2]
        failures.append({"test": test_id, "reason": reason or "Unknown reason", "base_file": base_file, "model": model})

    return failures


def parse_failures_long(text: str) -> list[str]:
    """Split the full stack trace report into separate stack traces."""
    if not text:
        return []

    stacktraces = []
    current_chunk = None
    for line in text.splitlines():
        if line.startswith("="):
            continue
        if re.match(r"_+\s.*\s_+$", line):
            if current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    stacktraces.append(chunk_text)
            current_chunk = []
            continue
        if current_chunk is not None:
            current_chunk.append(line)
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            stacktraces.append(chunk_text)

    return stacktraces


def update_reason_map(reason_map: dict, entry: dict) -> None:
    """Aggregate failure data per reason."""
    reason = entry["reason"]
    data = reason_map.setdefault(
        reason, {"count": 0, "models": set(), "tests": set(), "stacktrace": None}
    )
    data["count"] += 1
    if entry["model"]:
        data["models"].add(entry["model"])
    data["tests"].add(entry["test"])
    if data["stacktrace"] is None and entry.get("stacktrace"):
        data["stacktrace"] = entry["stacktrace"]


def serialize_reason_map(reason_map: dict) -> list[dict]:
    """Prepare reason map for JSON serialization."""
    serialized = []
    for reason, data in reason_map.items():
        serialized.append(
            {
                "reason": reason,
                "failures": data["count"],
                "models": sorted(data["models"]),
                "tests": sorted(data["tests"]),
                "stacktrace": data["stacktrace"] or "",
            }
        )
    serialized.sort(key=lambda x: x["failures"], reverse=True)
    return serialized


def serialize_counter(counter: Counter) -> list[dict]:
    items = [{"file": file_path, "failures": count} for file_path, count in counter.items()]
    items.sort(key=lambda x: x["failures"])
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_id", type=str, required=True)
    args = parser.parse_args()
    workflow_id = args.workflow_id

    r = requests.get(
        f"https://circleci.com/api/v2/workflow/{workflow_id}/job",
        headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")},
    )
    jobs = r.json()["items"]

    os.makedirs("outputs", exist_ok=True)

    global_failure_counts: Counter[str] = Counter()
    global_reason_map: dict[str, dict] = {}

    # for each job, download artifacts
    for job in jobs:
        project_slug = job["project_slug"]
        if job["name"].startswith(("tests_", "examples_", "pipelines_")):
            url = f"https://circleci.com/api/v2/project/{project_slug}/{job['job_number']}/artifacts"
            r = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
            job_artifacts = r.json()["items"]

            os.makedirs(job["name"], exist_ok=True)
            os.makedirs(f"outputs/{job['name']}", exist_ok=True)

            node_reports: dict[int, dict[str, str]] = defaultdict(dict)
            for artifact in job_artifacts:
                if not artifact["path"].startswith("reports/"):
                    continue
                node_index = artifact["node_index"]
                url = artifact["url"]
                if artifact["path"].endswith("/summary_short.txt"):
                    resp = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    node_reports[node_index]["summary_short"] = resp.text
                elif artifact["path"].endswith("/failures_line.txt"):
                    resp = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    node_reports[node_index]["failures_line"] = resp.text
                elif artifact["path"].endswith("/failures_long.txt"):
                    resp = requests.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    node_reports[node_index]["failures_long"] = resp.text

            job_failure_counts: Counter[str] = Counter()
            job_reason_map: dict[str, dict] = {}

            for node_index, reports in node_reports.items():
                failure_lines = reports.get("failures_line") or reports.get("summary_short", "")
                failures = parse_failure_lines(failure_lines)
                stacktraces = parse_failures_long(reports.get("failures_long", ""))
                for idx, failure in enumerate(failures):
                    if idx < len(stacktraces):
                        failure["stacktrace"] = stacktraces[idx]
                    else:
                        failure["stacktrace"] = None
                    job_failure_counts[failure["base_file"]] += 1
                    global_failure_counts[failure["base_file"]] += 1
                    update_reason_map(job_reason_map, failure)
                    update_reason_map(global_reason_map, failure)

            if job_failure_counts:
                print(f"Failure counts for job {job['name']}:")
                for item in serialize_counter(job_failure_counts):
                    print(f"{item['file']} : {item['failures']} failures")
            else:
                print(f"No failures detected for job {job['name']}.")

            job_output = {
                "failures_per_file": serialize_counter(job_failure_counts),
                "failure_reasons": serialize_reason_map(job_reason_map),
            }

            with open(f"outputs/{job['name']}/test_summary.json", "w") as fp:
                json.dump(job_output, fp, indent=4)

    if global_failure_counts:
        print("Aggregated failure counts across all processed jobs:")
        for item in serialize_counter(global_failure_counts):
            print(f"{item['file']} : {item['failures']} failures")
    else:
        print("No failures detected across all processed jobs.")

    aggregated_output = {
        "failures_per_file": serialize_counter(global_failure_counts),
        "failure_reasons": serialize_reason_map(global_reason_map),
    }

    with open("outputs/test_summary.json", "w") as fp:
        json.dump(aggregated_output, fp, indent=4)
