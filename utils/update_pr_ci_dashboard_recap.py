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

"""Inject a Grafana CI badge into the PR description and clean up legacy recap comments."""

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
RECAP_START = "<!-- ci-dashboard-recap:start -->"
RECAP_END = "<!-- ci-dashboard-recap:end -->"
OLD_DASHBOARD_COMMENT_MARKERS = (
    "**CI Observability Dashboard:** [View test results in Grafana]",
    "**CI Dashboard:** [View test results in Grafana]",
)


def log_workflow_run(workflow_run):
    """Print the GitHub Actions workflow_run payload fields used for debugging."""
    print("=== Triggering PR CI workflow_run info ===")
    print(f"  Run ID:           {workflow_run.get('id')}")
    print(f"  Run number:       {workflow_run.get('run_number')}")
    print(f"  Run URL:          {workflow_run.get('html_url')}")
    print(f"  Triggering event: {workflow_run.get('event')}")
    print(f"  Conclusion:       {workflow_run.get('conclusion')}")
    print(f"  Head branch:      {workflow_run.get('head_branch')}")
    print(f"  Head SHA:         {workflow_run.get('head_sha')}")
    print(f"  Actor:            {(workflow_run.get('actor') or {}).get('login')}")
    print(f"  Triggering actor: {(workflow_run.get('triggering_actor') or {}).get('login')}")
    print(f"  Created at:       {workflow_run.get('created_at')}")
    print(f"  Run started at:   {workflow_run.get('run_started_at')}")
    print(f"  Updated at:       {workflow_run.get('updated_at')}")
    print("==========================================")


def github_paginate(path, token, key=None):
    """Return all items from a paginated GitHub API endpoint."""
    page = 1
    items = []
    separator = "&" if "?" in path else "?"
    while True:
        url = f"{GITHUB_API_URL}{path}{separator}per_page=100&page={page}"
        payload = github_request(url, token=token)
        page_items = payload[key] if key is not None else payload
        if not page_items:
            break
        items.extend(page_items)
        if len(page_items) < 100:
            break
        page += 1
    return items


def replace_marked_block(body, start_marker, end_marker, replacement):
    """Replace all Markdown regions delimited by marker comments, if any exist."""
    existing_body = body or ""
    pattern = re.compile(f"{re.escape(start_marker)}[\\s\\S]*?{re.escape(end_marker)}")
    if pattern.search(existing_body):
        return pattern.sub(replacement, existing_body)
    return None


def remove_marked_block(body, start_marker, end_marker):
    """Remove all marked Markdown regions from a body and normalize blank lines."""
    updated = replace_marked_block(body, start_marker, end_marker, "")
    if updated is None:
        return body or ""
    return re.sub(r"\n{3,}", "\n\n", updated).strip()


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


def find_open_pr_for_sha(repo, token, head_sha):
    """Find the open pull request whose head commit matches a workflow run SHA."""
    prs = github_paginate(f"/repos/{repo}/pulls?state=open", token)
    return next((pr for pr in prs if pr["head"]["sha"] == head_sha), None)


def delete_old_dashboard_comments(repo, token, pr_number):
    """Delete legacy dashboard comments created before the recap marker flow."""
    comments = github_paginate(f"/repos/{repo}/issues/{pr_number}/comments", token)
    for comment in comments:
        body = comment.get("body") or ""
        if any(marker in body for marker in OLD_DASHBOARD_COMMENT_MARKERS):
            github_request(
                f"{GITHUB_API_URL}/repos/{repo}/issues/comments/{comment['id']}", token=token, method="DELETE"
            )


def delete_recap_comments(repo, token, pr_number):
    """Delete any existing CI recap comments (cleanup from before this change)."""
    comments = github_paginate(f"/repos/{repo}/issues/{pr_number}/comments", token)
    for comment in comments:
        if RECAP_START in (comment.get("body") or ""):
            github_request(
                f"{GITHUB_API_URL}/repos/{repo}/issues/comments/{comment['id']}",
                token=token,
                method="DELETE",
            )


def main():
    """Entrypoint for the workflow_run-triggered GitHub Action."""
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    event_path = os.environ["GITHUB_EVENT_PATH"]

    with open(event_path, encoding="utf-8") as event_file:
        event = json.load(event_file)

    workflow_run = event["workflow_run"]
    log_workflow_run(workflow_run)

    if workflow_run.get("event") != "pull_request":
        print(f"Workflow run event is {workflow_run.get('event')!r}, skipping")
        return

    pr = find_open_pr_for_sha(repo, token, workflow_run["head_sha"])
    if pr is None:
        print(f"No open PR found for SHA {workflow_run['head_sha']}, skipping")
        return

    print(f"Matched PR #{pr['number']}: {pr['html_url']}")
    delete_old_dashboard_comments(repo, token, pr["number"])
    delete_recap_comments(repo, token, pr["number"])

    dashboard_url = f"{DASHBOARD_URL}?var-pr={pr['number']}"
    updated_body = inject_ci_badge(pr.get("body"), render_ci_badge(dashboard_url))
    updated_body = remove_marked_block(updated_body, RECAP_START, RECAP_END)
    if updated_body != (pr.get("body") or ""):
        github_request(
            f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr['number']}",
            token=token,
            method="PATCH",
            payload={"body": updated_body},
        )


if __name__ == "__main__":
    sys.exit(main())
