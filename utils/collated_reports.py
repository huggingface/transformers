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

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GPU_NAMES = ["mi300", "mi325", "mi355", "h100", "a10"]


def simplify_gpu_name(gpu_name: str, simplified_names: list[str]) -> str:
    matches = []
    for simplified_name in simplified_names:
        if simplified_name in gpu_name:
            matches.append(simplified_name)
    if len(matches) == 1:
        return matches[0]
    return gpu_name


def parse_short_summary_line(line: str) -> tuple[str | None, int]:
    if line.startswith("PASSED"):
        return "passed", 1
    if line.startswith("FAILED"):
        return "failed", 1
    if line.startswith("SKIPPED"):
        line = line.split("[", maxsplit=1)[1]
        line = line.split("]", maxsplit=1)[0]
        return "skipped", int(line)
    if line.startswith("ERROR"):
        return "error", 1
    return None, 0


def validate_path(p: str) -> Path:
    # Validate path and apply glob pattern if provided
    path = Path(p)
    assert path.is_dir(), f"Path {path} is not a directory"
    return path


def get_gpu_name(gpu_name: str | None) -> str:
    # Get GPU name if available
    if gpu_name is None:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name()
        except Exception as e:
            print(f"Failed to get GPU name with {e}")
            gpu_name = "unknown"
    else:
        gpu_name = gpu_name.replace(" ", "_").lower()
        gpu_name = simplify_gpu_name(gpu_name, DEFAULT_GPU_NAMES)

    return gpu_name


def get_commit_hash(commit_hash: str | None) -> str:
    # Get commit hash if available
    if commit_hash is None:
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception as e:
            print(f"Failed to get commit hash with {e}")
            commit_hash = "unknown"

    return commit_hash[:7]


@dataclass
class Args:
    path: Path
    machine_type: str
    gpu_name: str
    commit_hash: str
    job: str | None
    report_repo_id: str | None


def get_arguments(args: argparse.Namespace) -> Args:
    path = validate_path(args.path)
    machine_type = args.machine_type
    gpu_name = get_gpu_name(args.gpu_name)
    commit_hash = get_commit_hash(args.commit_hash)
    job = args.job
    report_repo_id = args.report_repo_id
    return Args(path, machine_type, gpu_name, commit_hash, job, report_repo_id)


def upload_collated_report(job: str, report_repo_id: str, filename: str):
    # Alternatively we can check for the existence of the collated_reports file and upload in notification_service.py
    import os

    from get_previous_daily_ci import get_last_daily_ci_run
    from huggingface_hub import HfApi

    api = HfApi()

    # if it is not a scheduled run, upload the reports to a subfolder under `report_repo_folder`
    report_repo_subfolder = ""
    if os.getenv("GITHUB_EVENT_NAME") != "schedule":
        report_repo_subfolder = f"{os.getenv('GITHUB_RUN_NUMBER')}-{os.getenv('GITHUB_RUN_ID')}"
        report_repo_subfolder = f"runs/{report_repo_subfolder}"

    workflow_run = get_last_daily_ci_run(
        token=os.environ["ACCESS_REPO_INFO_TOKEN"], workflow_run_id=os.getenv("GITHUB_RUN_ID")
    )
    workflow_run_created_time = workflow_run["created_at"]
    report_repo_folder = workflow_run_created_time.split("T")[0]

    if report_repo_subfolder:
        report_repo_folder = f"{report_repo_folder}/{report_repo_subfolder}"

    api.upload_file(
        path_or_fileobj=f"{filename}",
        path_in_repo=f"{report_repo_folder}/ci_results_{job}/{filename}",
        repo_id=report_repo_id,
        repo_type="dataset",
        token=os.getenv("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post process models test reports.")
    parser.add_argument("--path", "-p", help="Path to the reports folder")
    parser.add_argument(
        "--machine-type", "-m", help="Process single or multi GPU results", choices=["single-gpu", "multi-gpu"]
    )
    parser.add_argument("--gpu-name", "-g", help="GPU name", default=None)
    parser.add_argument("--commit-hash", "-c", help="Commit hash", default=None)
    parser.add_argument("--job", "-j", help="Optional job name required for uploading reports", default=None)
    parser.add_argument(
        "--report-repo-id", "-r", help="Optional report repository ID required for uploading reports", default=None
    )
    args = get_arguments(parser.parse_args())

    # Initialize accumulators for collated report
    total_status_count = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "error": 0,
        None: 0,
    }
    collated_report_buffer = []

    path = args.path
    machine_type = args.machine_type
    gpu_name = args.gpu_name
    commit_hash = args.commit_hash
    job = args.job
    report_repo_id = args.report_repo_id

    # Loop through model directories and create collated reports
    for model_dir in sorted(path.iterdir()):
        if not model_dir.name.startswith(machine_type):
            continue

        # Create a new entry for the model
        model_name = model_dir.name.split("models_")[-1].removesuffix("_test_reports")
        report = {"model": model_name, "results": []}
        results = []

        # Read short summary
        with open(model_dir / "summary_short.txt", "r") as f:
            short_summary_lines = f.readlines()

        # Parse short summary
        for line in short_summary_lines[1:]:
            status, count = parse_short_summary_line(line)
            total_status_count[status] += count
            if status:
                result = {
                    "status": status,
                    "test": line.split(status.upper(), maxsplit=1)[1].strip(),
                    "count": count,
                }
                results.append(result)

        # Add short summaries to report
        report["results"] = results

        collated_report_buffer.append(report)

    filename = f"collated_reports_{machine_type}_{commit_hash}.json"
    # Write collated report
    with open(filename, "w") as f:
        json.dump(
            {
                "gpu_name": gpu_name,
                "machine_type": machine_type,
                "commit_hash": commit_hash,
                "total_status_count": total_status_count,
                "results": collated_report_buffer,
            },
            f,
            indent=2,
        )

    # Upload collated report
    if job and report_repo_id:
        upload_collated_report(job, report_repo_id, filename)
