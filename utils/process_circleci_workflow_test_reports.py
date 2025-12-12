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
import argparse
import json
import os
import re
from collections import Counter

import httpx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_id", type=str, required=True)
    args = parser.parse_args()

    r = httpx.get(
        f"https://circleci.com/api/v2/workflow/{args.workflow_id}/job",
        headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")},
    )
    jobs = r.json()["items"]

    os.makedirs("outputs", exist_ok=True)
    workflow_summary = {}
    failure_entries = []

    for job in jobs:
        if job["name"].startswith(("tests_", "examples_", "pipelines_")):
            url = f"https://circleci.com/api/v2/project/{job['project_slug']}/{job['job_number']}/artifacts"
            r = httpx.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
            job_artifacts = r.json()["items"]

            os.makedirs(f"outputs/{job['name']}", exist_ok=True)

            job_test_summaries = {}
            job_failure_lines = {}

            for artifact in job_artifacts:
                url = artifact["url"]
                if artifact["path"].endswith("/summary_short.txt"):
                    r = httpx.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    job_test_summaries[artifact["node_index"]] = r.text
                elif artifact["path"].endswith("/failures_line.txt"):
                    r = httpx.get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    job_failure_lines[artifact["node_index"]] = r.text

            summary = {}
            for node_index, node_test_summary in job_test_summaries.items():
                for line in node_test_summary.splitlines():
                    if line.startswith("PASSED "):
                        summary[line[7:]] = "passed"
                    elif line.startswith("FAILED "):
                        summary[line[7:].split()[0]] = "failed"

            summary = dict(sorted(summary.items(), key=lambda x: (x[1], x[0])))
            workflow_summary[job["name"]] = summary

            with open(f"outputs/{job['name']}/test_summary.json", "w") as fp:
                json.dump(summary, fp, indent=4)

            # Collect failure details
            for node_index, summary_text in job_test_summaries.items():
                failure_lines_list = [
                    l.strip()
                    for l in job_failure_lines.get(node_index, "").splitlines()
                    if l.strip() and not l.strip().startswith(("=", "_", "short test summary")) and ": " in l
                ]

                failure_idx = 0
                for line in summary_text.splitlines():
                    if line.startswith("FAILED ") and " - Failed: (subprocess)" not in line:
                        test_name, _, short_error = line[7:].strip().partition(" - ")
                        test_name = test_name.strip()
                        parts = test_name.split("::", 1)[0].split("/")
                        model_name = parts[2] if len(parts) >= 3 and test_name.startswith("tests/models/") else None
                        full_error = (
                            failure_lines_list[failure_idx] if failure_idx < len(failure_lines_list) else short_error
                        )

                        failure_entries.append(
                            {
                                "job_name": job["name"],
                                "test_name": test_name,
                                "short_error": short_error,
                                "error": full_error,
                                "model_name": model_name,
                            }
                        )
                        failure_idx += 1

    # Build workflow summary
    new_workflow_summary = {}
    for job_name, job_summary in workflow_summary.items():
        for test, status in job_summary.items():
            new_workflow_summary.setdefault(test, {})[job_name] = status

    new_workflow_summary = {
        test: dict(sorted(result.items())) for test, result in sorted(new_workflow_summary.items())
    }

    with open("outputs/test_summary.json", "w") as fp:
        json.dump(new_workflow_summary, fp, indent=4)

    # Aggregate failures by test and model
    by_test, by_model = {}, {}

    for entry in failure_entries:
        # Normalize test name
        normalized = entry["test_name"].split("[", 1)[0]
        parts = normalized.split("::")
        normalized = "::".join(parts[:-1] + [re.sub(r"_\d{2,}.*$", "", parts[-1])])

        by_test.setdefault(normalized, {"count": 0, "errors": Counter(), "jobs": set(), "variants": set()})
        by_test[normalized]["count"] += 1
        by_test[normalized]["errors"][entry["error"]] += 1
        by_test[normalized]["jobs"].add(entry["job_name"])
        by_test[normalized]["variants"].add(entry["test_name"])

        if entry["model_name"]:
            by_model.setdefault(entry["model_name"], {"count": 0, "errors": Counter(), "tests": set()})
            by_model[entry["model_name"]]["count"] += 1
            by_model[entry["model_name"]]["errors"][entry["error"]] += 1
            by_model[entry["model_name"]]["tests"].add(entry["test_name"])

    # Convert Counter and sets to dicts/lists for JSON serialization
    for info in by_test.values():
        info["errors"] = dict(info["errors"].most_common())
        info["jobs"] = sorted(info["jobs"])
        info["variants"] = sorted(info["variants"])
    for info in by_model.values():
        info["errors"] = dict(info["errors"].most_common())
        info["tests"] = sorted(info["tests"])

    with open("outputs/failure_summary.json", "w") as fp:
        json.dump({"failures": failure_entries, "by_test": by_test, "by_model": by_model}, fp, indent=4)
