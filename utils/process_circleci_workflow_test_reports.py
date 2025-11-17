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
from collections import Counter
from typing import Callable

import requests


def _extract_failed_tests(summary_short: str) -> list[tuple[str, str]]:
    """
    Return a list of tuples (<test node id>, <error message from short summary>).
    """
    failed_tests = []
    for line in summary_short.splitlines():
        if not line.startswith("FAILED "):
            continue
        # Skip subprocess failures created by `run_test_using_subprocess`
        if " - Failed: (subprocess)" in line:
            continue
        failure_line = line[len("FAILED ") :].strip()
        test_node, sep, error_message = failure_line.partition(" - ")
        failed_tests.append((test_node.strip(), error_message.strip()))

    return failed_tests


def _extract_failure_lines(failures_line: str | None) -> list[str]:
    if not failures_line:
        return []

    failure_lines = []
    for raw_line in failures_line.splitlines():
        raw_line = raw_line.strip()
        if (
            not raw_line
            or raw_line.startswith("=")
            or raw_line.startswith("_")
            or raw_line.lower().startswith("short test summary")
        ):
            continue
        if ": " not in raw_line:
            continue
        failure_lines.append(raw_line)

    return failure_lines


def _derive_model_name(test_node_id: str) -> str | None:
    """
    Given a pytest node id (e.g. tests/models/bart/test_modeling_bart.py::BartModelTest::test_forward),
    extract the model name when it lives under `tests/models`.
    """
    file_path = test_node_id.split("::", maxsplit=1)[0]
    if file_path.startswith("tests/models/"):
        parts = file_path.split("/")
        if len(parts) >= 3:
            return parts[2]
    return None


def _aggregate_failures(failure_entries: list[dict]) -> tuple[dict, dict]:
    by_test: dict[str, dict] = {}
    by_model: dict[str, dict] = {}

    for entry in failure_entries:
        test_name = entry["test_name"]
        model_name = entry["model_name"]
        error_message = entry["error"]

        test_info = by_test.setdefault(test_name, {"count": 0, "errors": Counter(), "jobs": set()})
        test_info["count"] += 1
        test_info["errors"][error_message] += 1
        test_info["jobs"].add(entry["job_name"])

        if model_name:
            model_info = by_model.setdefault(model_name, {"count": 0, "errors": Counter(), "tests": set()})
            model_info["count"] += 1
            model_info["errors"][error_message] += 1
            model_info["tests"].add(test_name)

    # Convert counters and sets to serializable forms
    def _prepare(entries: dict, include_tests: bool = False):
        prepared = {}
        for key, value in entries.items():
            prepared[key] = {
                "count": value["count"],
                "errors": dict(value["errors"].most_common()),
            }
            if include_tests:
                prepared[key]["tests"] = sorted(value["tests"])
            else:
                prepared[key]["jobs"] = sorted(value["jobs"])
        return prepared

    return _prepare(by_test), _prepare(by_model, include_tests=True)


def _format_error_messages(errors: dict[str, int]) -> str:
    return "; ".join(f"{count}× {msg}" for msg, count in errors.items()) or "N/A"


def _format_markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return "No data\n"

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    table_lines = [header_line, separator]
    table_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table_lines) + "\n"


def _get_pr_details_from_env() -> tuple[str, str, str] | None:
    """
    Returns (owner, repo, pr_number) if we can infer them from the environment.

    CircleCI does not always expose `CIRCLE_PULL_REQUEST`, so the collection job exports `PR_NUMBER`
    beforehand via `utils/extract_pr_number_from_circleci.py`. We try every known source before giving up.
    """
    pr_url_candidates = [
        os.environ.get("CIRCLE_PULL_REQUEST"),
        os.environ.get("GITHUB_PULL_REQUEST_URL"),
    ]
    for pr_url in pr_url_candidates:
        if not pr_url:
            continue
        match = re.match(
            r"https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)", pr_url
        )
        if match:
            return match.group("owner"), match.group("repo"), match.group("number")

    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("PR_NUMBER")
    if not pr_number:
        github_ref = os.environ.get("GITHUB_REF", "")
        match = re.search(r"refs/pull/(\d+)/", github_ref)
        if match:
            pr_number = match.group(1)
    if repo and pr_number:
        owner, repo_name = repo.split("/", 1)
        return owner, repo_name, pr_number
    return None


def _get_github_token() -> str | None:
    for env_var in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_ACCESS_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            return token
    return None


def _post_failure_summary_comment(markdown_text: str, request_post: Callable = requests.post) -> bool:
    pr_details = _get_pr_details_from_env()
    token = _get_github_token()
    if not pr_details or not token:
        return False
    owner, repo, pr_number = pr_details
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    response = request_post(url, headers=headers, json={"body": markdown_text})
    if not (200 <= getattr(response, "status_code", 0) < 300):
        print(f"Failed to post PR comment: {getattr(response, 'status_code', 'unknown')} {getattr(response, 'text', '')}")
        return False
    return True


def process_circleci_workflow(
    workflow_id: str,
    output_dir: str = "outputs",
    request_get: Callable = requests.get,
    request_post: Callable = requests.post,
):
    response = request_get(
        f"https://circleci.com/api/v2/workflow/{workflow_id}/job",
        headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")},
    )
    jobs = response.json()["items"]

    os.makedirs(output_dir, exist_ok=True)

    workflow_summary = {}
    failure_entries: list[dict] = []
    # for each job, download artifacts
    for job in jobs:
        project_slug = job["project_slug"]
        if job["name"].startswith(("tests_", "examples_", "pipelines_")):
            url = f"https://circleci.com/api/v2/project/{project_slug}/{job['job_number']}/artifacts"
            r = request_get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
            job_artifacts = r.json()["items"]

            job_output_dir = os.path.join(output_dir, job["name"])
            os.makedirs(job_output_dir, exist_ok=True)

            job_test_summaries = {}
            job_failure_lines = {}
            for artifact in job_artifacts:
                if artifact["path"].startswith("reports/") and artifact["path"].endswith("/summary_short.txt"):
                    node_index = artifact["node_index"]
                    artifact_url = artifact["url"]
                    r = request_get(artifact_url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    test_summary = r.text
                    job_test_summaries[node_index] = test_summary
                elif artifact["path"].startswith("reports/") and artifact["path"].endswith("/failures_line.txt"):
                    node_index = artifact["node_index"]
                    artifact_url = artifact["url"]
                    r = request_get(artifact_url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    job_failure_lines[node_index] = r.text

            summary = {}
            for node_index, node_test_summary in job_test_summaries.items():
                for line in node_test_summary.splitlines():
                    if line.startswith("PASSED "):
                        test = line[len("PASSED ") :]
                        summary[test] = "passed"
                    elif line.startswith("FAILED "):
                        test = line[len("FAILED ") :].split()[0]
                        summary[test] = "failed"
            # failed before passed
            summary = dict(sorted(summary.items(), key=lambda x: (x[1], x[0])))
            workflow_summary[job["name"]] = summary

            # collected version
            with open(os.path.join(job_output_dir, "test_summary.json"), "w") as fp:
                json.dump(summary, fp, indent=4)

            # Collect failure details per node for this job
            for node_index, summary_short in job_test_summaries.items():
                failed_tests = _extract_failed_tests(summary_short)
                failure_lines = _extract_failure_lines(job_failure_lines.get(node_index))
                for idx, (test_name, short_error) in enumerate(failed_tests):
                    full_error = failure_lines[idx] if idx < len(failure_lines) else short_error
                    failure_entries.append(
                        {
                            "job_name": job["name"],
                            "node_index": node_index,
                            "test_name": test_name,
                            "short_error": short_error,
                            "error": full_error,
                            "model_name": _derive_model_name(test_name),
                        }
                    )

    new_workflow_summary = {}
    for job_name, job_summary in workflow_summary.items():
        for test, status in job_summary.items():
            if test not in new_workflow_summary:
                new_workflow_summary[test] = {}
            new_workflow_summary[test][job_name] = status

    for test, result in new_workflow_summary.items():
        new_workflow_summary[test] = dict(sorted(result.items()))
    new_workflow_summary = dict(sorted(new_workflow_summary.items()))

    with open(os.path.join(output_dir, "test_summary.json"), "w") as fp:
        json.dump(new_workflow_summary, fp, indent=4)

    failures_by_test, failures_by_model = _aggregate_failures(failure_entries)
    failure_summary = {
        "failures": failure_entries,
        "by_test": failures_by_test,
        "by_model": failures_by_model,
    }

    with open(os.path.join(output_dir, "failure_summary.json"), "w") as fp:
        json.dump(failure_summary, fp, indent=4)

    markdown_buffer = ["# Failure summary\n"]
    if failure_entries:
        markdown_buffer.append("## By test\n")
        test_rows = []
        for test_name, info in sorted(failures_by_test.items(), key=lambda x: x[1]["count"], reverse=True):
            test_rows.append(
                [
                    test_name,
                    str(info["count"]),
                    _format_error_messages(info["errors"]),
                ]
            )
        markdown_buffer.append(_format_markdown_table(test_rows, ["Test", "Failures", "Full error(s)"]))

        markdown_buffer.append("## By model\n")
        model_rows = []
        for model_name, info in sorted(failures_by_model.items(), key=lambda x: x[1]["count"], reverse=True):
            model_rows.append(
                [
                    model_name,
                    str(info["count"]),
                    _format_error_messages(info["errors"]),
                ]
            )
        markdown_buffer.append(_format_markdown_table(model_rows, ["Model", "Failures", "Full error(s)"]))
    else:
        markdown_buffer.append("No failures were reported.\n")

    markdown_text = "\n".join(markdown_buffer)
    with open(os.path.join(output_dir, "failure_summary.md"), "w") as fp:
        fp.write(markdown_text)

    _post_failure_summary_comment(markdown_text, request_post=request_post)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_id", type=str, required=True)
    args = parser.parse_args()
    workflow_id = args.workflow_id
    process_circleci_workflow(workflow_id)


if __name__ == "__main__":
    main()
