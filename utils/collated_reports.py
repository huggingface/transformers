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
from pathlib import Path


DEFAULT_GPU_NAMES = ["mi300", "mi355", "h100", "a10"]


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


def get_path(s: str) -> Path:
    # validate path
    path = Path(s)
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


def get_arguments(args: argparse.Namespace) -> tuple[Path, str, str]:
    path = get_path(args.path)
    gpu_name = get_gpu_name(args.gpu_name)
    commit_hash = get_commit_hash(args.commit_hash)
    return path, gpu_name, commit_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post process models test reports.")
    parser.add_argument("--path", "-p", help="Path to the reports folder")
    parser.add_argument("--gpu-name", "-g", help="GPU name", default=None)
    parser.add_argument("--commit-hash", "-c", help="Commit hash", default=None)

    path, gpu_name, commit_hash = get_arguments(parser.parse_args())

    # Initialize accumulators for collated report
    total_status_count = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "error": 0,
        None: 0,
    }
    collated_report_buffer = []

    for model_dir in sorted(path.iterdir()):
        # Create a new entry for the model
        model_name = model_dir.name.removesuffix("_test_reports")
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

    # Write collated report
    with open(f"collated_reports_{commit_hash}.json", "w") as f:
        json.dump(
            {
                "gpu_name": gpu_name,
                "commit_hash": commit_hash,
                "total_status_count": total_status_count,
                "results": collated_report_buffer,
            },
            f,
            indent=2,
        )
