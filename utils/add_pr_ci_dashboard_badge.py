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

"""Inject a Grafana CI badge into the PR description."""

import json
import os
import re
import sys

# GitHub REST calls go through the shared helper (rate-limit handling, no anonymous fallback, fail
# hard on a rejected token); see github_utils.py for details.
from github_utils import github_request


GITHUB_API_URL = "https://api.github.com"
DASHBOARD_URL = "https://transformers-ci.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch"
BADGE_URL = "https://img.shields.io/badge/%F0%9F%A4%97%20Transformers%20CI-333333?style=flat"
BADGE_START = "<!-- ci-dashboard-badge:start -->"
BADGE_END = "<!-- ci-dashboard-badge:end -->"


def replace_marked_block(body, start_marker, end_marker, replacement):
    """Replace all Markdown regions delimited by marker comments, if any exist."""
    existing_body = body or ""
    pattern = re.compile(f"{re.escape(start_marker)}[\\s\\S]*?{re.escape(end_marker)}")
    if pattern.search(existing_body):
        return pattern.sub(replacement, existing_body)
    return None


def render_ci_badge(dashboard_url):
    """Render a static dashboard link, independent of CI status and metrics."""
    return f"{BADGE_START}\n[![🤗 Transformers CI]({BADGE_URL})]({dashboard_url})\n{BADGE_END}"


def inject_ci_badge(body, badge):
    """Insert or replace the CI dashboard badge block in a PR body."""
    replaced = replace_marked_block(body, BADGE_START, BADGE_END, badge)
    if replaced is not None:
        return replaced
    existing_body = body or ""
    return f"{badge}\n\n{existing_body.lstrip()}".rstrip()


def main():
    """Add the CI dashboard badge when a PR opens or is reopened."""
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    event_path = os.environ["GITHUB_EVENT_PATH"]

    with open(event_path, encoding="utf-8") as event_file:
        event = json.load(event_file)

    # The description may have changed while the job was queued. Fetch it by PR number
    # instead of reusing the event body or searching for a potentially outdated head SHA.
    pr_number = event["pull_request"]["number"]
    pr = github_request(f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr_number}", token=token)
    if pr["state"] != "open":
        print(f"PR #{pr_number} is no longer open, skipping")
        return

    print(f"Matched PR #{pr['number']}: {pr['html_url']}")
    dashboard_url = f"{DASHBOARD_URL}?var-pr={pr['number']}"
    updated_body = inject_ci_badge(pr.get("body"), render_ci_badge(dashboard_url))
    if updated_body != (pr.get("body") or ""):
        github_request(
            f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr['number']}",
            token=token,
            method="PATCH",
            payload={"body": updated_body},
        )


if __name__ == "__main__":
    sys.exit(main())
