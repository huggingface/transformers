#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Post the model integration contracts gate on the PR, as one comment kept up to date.

Runs in the workflow_run privileged job (pull-requests: write), after `PR CI`. Reads
`contracts-gate.json` from the `contracts` artifact of the `fast-integration / transformers`
job. Never checks out or executes PR code.

That job runs the PR's code, so the artifact is untrusted: the PR is resolved from the
workflow_run event (head repository, branch and SHA), never from the artifact, and every
field rendered from the artifact is validated or quoted.

- blocked cases: create or update the comment (errors, links, `record-mic` suggestion);
- no blocked case and a comment exists: mark it resolved;
- no blocked case and no comment: post nothing.

Usage:
    GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/repo RUN_ID=... RUN_ATTEMPT=... HEAD_SHA=... HEAD_OWNER=... HEAD_BRANCH=... \
        python utils/post_mic_gate_comment.py <artifact dir>
"""

import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

# github_utils is present because the workflow checks out the repo.
from github_utils import get_github_json, github_request


GITHUB_API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
DASHBOARD = "https://transformers-ci.huggingface.cool"
DOCS = "https://huggingface.co/docs/transformers/main/en/pr_checks#model-integration-contracts"
MARKER = "<!-- mic-gate-comment -->"
JOB_SUFFIX = "fast-integration / transformers"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}$")
RECORD_BOT = "github-actions[bot]"
RECORD_REPLY = re.compile(r"^\*\*record-mic\*\* \(asked by @([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\): published\b")
RECORD_ROW = re.compile(r"^\| ([A-Za-z0-9][A-Za-z0-9_.\-]{0,99}) \| ")
SELECTED_BY = re.compile(r"^models/([a-z0-9_]{1,64})\b")
MAX_ROWS = 20
MAX_ERROR = 200


def clean_name(value):
    """A contract, target or variant name, or None when it is not a plain name."""
    return value if isinstance(value, str) and NAME.match(value) else None


def clean_error(value):
    """One line of error text, safe inside a Markdown code span in a table cell."""
    text = " ".join(str(value or "").split())
    text = text.replace("`", "'").replace("|", "/")
    return text[:MAX_ERROR] + ("…" if len(text) > MAX_ERROR else "")


def blocked_cases(gate):
    """The blocked cases of a gate report, validated; malformed entries are dropped."""
    cases = []
    for decision in gate.get("decisions") or []:
        if not isinstance(decision, dict) or decision.get("verdict") != "blocked":
            continue
        names = [clean_name(decision.get(field)) for field in ("contract", "target", "variant")]
        if None in names:
            continue
        cases.append((*names, clean_error(decision.get("failure"))))
    return cases


def model_names(cases, observation):
    """What to name in `make run-mic` and `record-mic`: the `models/` directory that selected each
    blocked contract (both accept it), else the contract name (only `record-mic` accepts that)."""
    reasons = {}
    for selection in (observation or {}).get("selections") or []:
        if isinstance(selection, dict):
            reasons.update(selection)
    names = []
    for contract, *_ in cases:
        reason = reasons.get(contract)
        match = SELECTED_BY.match(reason) if isinstance(reason, str) else None
        name = match.group(1) if match else contract
        if name not in names:
            names.append(name)
    return names


def run_url(pr, run_id, attempt):
    """The contracts run on the dashboard: the run page, keyed like the MIC page's `run_url`."""
    query = urllib.parse.urlencode(
        {"var-run_id": f"{run_id}:{attempt}", "var-pr": str(pr), "var-gh_run_id": str(run_id)}
    )
    return f"{DASHBOARD}/d/pytest-observability-by-run/pytest-observability-run?{query}"


def render(cases, *, pr, sha, enforced, run_id, attempt, job_url, models):
    models = " ".join(models)
    query = urllib.parse.urlencode({"scope": "prs", "group": "pr", "q": str(pr)})
    links = [
        f"[contracts run]({run_url(pr, run_id, attempt)})",
        f"[MIC dashboard]({DASHBOARD}/mic?{query})",
        f"[PR CI dashboard]({DASHBOARD}/d/pytest-observability-by-pr/pytest-observability-branch?var-pr={pr})",
    ]
    if job_url:
        links.append(f"[job log]({job_url})")
    effect = "This PR can't merge until they are resolved." if enforced else "Reviewers treat them as failures."
    lines = [
        MARKER,
        f"### ❌ Model integration contracts: {len(cases)} blocked case(s)",
        "",
        f"These contracts are green on the base and red with this PR (commit {sha[:10]}). {effect}",
        "",
        "| contract | target | variant | error |",
        "|---|---|---|---|",
    ]
    lines += [f"| {c} | {t} | {v} | `{e}` |" for c, t, v, e in cases[:MAX_ROWS]]
    if len(cases) > MAX_ROWS:
        lines.append(f"| … | | | {len(cases) - MAX_ROWS} more in the job summary |")
    lines += [
        "",
        " · ".join(links),
        "",
        f'- **Regression**: fix it in this PR. Reproduce it with `make run-mic MODEL="{models}"`.',
        "- **Intended change**: push your final changes and let the contracts job finish, then someone with "
        "write access re-records the expected values for this PR only, either by commenting:",
        "",
        "  ```text",
        f"  record-mic: {models}",
        "  ```",
        "",
        "  or from any checkout of transformers, with `gh` logged in (same workflow, same checks):",
        "",
        "  ```bash",
        f'  make record-mic MODEL="{models}" PR={pr}',
        "  ```",
        "",
        "  A bot replies here with the re-recorded values; the next contracts run uses them.",
        "",
        f"[What a blocked case is and what happens on merge]({DOCS})",
    ]
    return "\n".join(lines)


def record_replies(comments):
    """`(requester, contracts, url)` of each `record-mic` publication on the PR, oldest first.

    Read from the record-mic workflow's own replies: only a privileged workflow posts as
    github-actions[bot], so unlike the artifact these are trusted.
    """
    replies = []
    for comment in comments:
        if comment["user"].get("login") != RECORD_BOT:
            continue
        lines = comment["body"].splitlines()
        match = RECORD_REPLY.match(lines[0]) if lines else None
        if not match:
            continue
        contracts = []
        for line in lines[1:]:
            row = RECORD_ROW.match(line)
            if row and row.group(1) != "contract" and row.group(1) not in contracts:
                contracts.append(row.group(1))
        replies.append((match.group(1), contracts, comment["html_url"]))
    return replies


def render_resolved(sha, replies=()):
    lines = [
        MARKER,
        "### ✅ Model integration contracts: no blocked case",
        "",
        f"No contract is blocked as of commit {sha[:10]}.",
    ]
    if replies:
        lines += ["", "Expected values re-recorded for this PR with `record-mic`:", ""]
        for login, contracts, url in replies:
            # A profile link, not an @-mention: editing this comment must not ping anyone.
            names = ", ".join(contracts) if contracts else "see the reply"
            lines.append(f"- {names}: requested by [{login}](https://github.com/{login}) ([reply]({url}))")
        lines += ["", "They apply to this PR until it merges, then to every later commit."]
    return "\n".join(lines)


def find_pr(repo, owner, branch, sha, token):
    """The open PR whose head is exactly this commit, from the event, not the artifact."""
    query = urllib.parse.urlencode({"head": f"{owner}:{branch}", "state": "open"})
    for pr in get_github_json(f"{GITHUB_API_URL}/repos/{repo}/pulls?{query}", token=token) or []:
        if pr["head"]["sha"] == sha:
            return pr["number"]
    return None


def find_job_url(repo, run_id, token):
    url = f"{GITHUB_API_URL}/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&filter=latest"
    for job in (get_github_json(url, token=token) or {}).get("jobs", []):
        if job["name"].endswith(JOB_SUFFIX):
            return job["html_url"]
    return None


def list_comments(repo, pr, token):
    comments, page = [], 1
    while True:
        url = f"{GITHUB_API_URL}/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}"
        batch = get_github_json(url, token=token) or []
        comments += batch
        if len(batch) < 100:
            return comments
        page += 1


def find_comment(comments):
    for comment in comments:
        if comment["user"]["type"] == "Bot" and comment["body"].startswith(MARKER):
            return comment
    return None


def read_json(path):
    """A JSON object from the artifact, or None when it is missing or malformed."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def main():
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    sha = os.environ["HEAD_SHA"]
    artifact = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    gate_path = artifact / "contracts-gate.json"
    if not gate_path.exists():
        print("No contracts gate report; nothing to post.")
        return 0
    gate = read_json(gate_path)
    if gate is None:
        print("Unreadable contracts gate report; nothing to post.")
        return 0

    pr = find_pr(repo, os.environ["HEAD_OWNER"], os.environ["HEAD_BRANCH"], sha, token)
    if pr is None:
        print(f"No open PR has head {sha}; the run is stale or the PR closed. Nothing to post.")
        return 0

    cases = blocked_cases(gate)
    comments = list_comments(repo, pr, token)
    existing = find_comment(comments)
    if cases:
        job_url = find_job_url(repo, os.environ["RUN_ID"], token)
        models = model_names(cases, read_json(artifact / "contracts-observation.json"))
        body = render(
            cases,
            pr=pr,
            sha=sha,
            enforced=gate.get("enforced") is True,
            run_id=os.environ["RUN_ID"],
            attempt=os.environ.get("RUN_ATTEMPT", "1"),
            job_url=job_url,
            models=models,
        )
    elif existing:
        body = render_resolved(sha, record_replies(comments))
    else:
        print(f"#{pr}: no blocked case and no comment; nothing to post.")
        return 0

    if existing:
        if existing["body"] != body:
            github_request(existing["url"], token=token, method="PATCH", payload={"body": body})
        print(f"#{pr}: updated comment {existing['html_url']}")
    else:
        comment = github_request(
            f"{GITHUB_API_URL}/repos/{repo}/issues/{pr}/comments", token=token, method="POST", payload={"body": body}
        )
        print(f"#{pr}: posted {comment['html_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
