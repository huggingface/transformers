#!/usr/bin/env python
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
Format extras smoke test results for Slack notification.

This script reads failure reports from a JSON file and outputs environment
variables for GitHub Actions to use in Slack notifications.
"""

import argparse
import json
import os
import sys


def format_slack_message(failures_file, workflow_url, output_file=None):
    """
    Format extras smoke test results into Slack message components.

    Args:
        failures_file: Path to JSON file containing failure reports
        workflow_url: URL to the GitHub Actions workflow run
        output_file: Optional path to output file (defaults to GITHUB_ENV)

    Returns:
        Dictionary with title, message, and workflow_url
    """
    # Read failures
    with open(failures_file) as f:
        failures = json.load(f)

    if not failures:
        # Success case
        title = "Extras Smoke Test - All tests passed"
        message = "All extras installed successfully across all Python versions."
    else:
        # Failure case - group by Python version
        failures_by_python = {}
        for failure in failures:
            py_ver = failure.get("python_version", "unknown")
            extra = failure.get("extra", "unknown")

            if py_ver not in failures_by_python:
                failures_by_python[py_ver] = []
            failures_by_python[py_ver].append(extra)

        title = f"Extras Smoke Test Failed - {len(failures)} failure(s)"

        # Build failure details
        details = []
        for py_ver in sorted(failures_by_python.keys()):
            extras = failures_by_python[py_ver]
            extras_list = "\n".join([f"• `{extra}`" for extra in sorted(extras)])
            details.append(f"*Python {py_ver}*\n{extras_list}")

        message = "\n\n".join(details)

    # Determine output destination
    if output_file is None:
        output_file = os.environ.get("GITHUB_ENV")
        if not output_file:
            print("Error: GITHUB_ENV not set and no output file specified", file=sys.stderr)
            sys.exit(1)

    # Write environment variables
    with open(output_file, "a") as f:
        f.write(f"SLACK_TITLE={title}\n")
        f.write(f"SLACK_WORKFLOW_URL={workflow_url}\n")
        # Use heredoc for multiline message
        f.write("SLACK_MESSAGE<<EOF\n")
        f.write(f"{message}\n")
        f.write("EOF\n")

    return {"title": title, "message": message, "workflow_url": workflow_url}


def main():
    parser = argparse.ArgumentParser(description="Format extras smoke test results for Slack")
    parser.add_argument(
        "--failures",
        required=True,
        help="Path to JSON file containing failure reports",
    )
    parser.add_argument(
        "--workflow-url",
        required=True,
        help="URL to the GitHub Actions workflow run",
    )
    parser.add_argument(
        "--output",
        help="Output file path (defaults to GITHUB_ENV)",
    )

    args = parser.parse_args()

    result = format_slack_message(args.failures, args.workflow_url, args.output)
    print(f"Formatted Slack message: {result['title']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
