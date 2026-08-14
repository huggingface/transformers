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
"""Post mlinter findings as an inline PR review.

Runs in the workflow_run privileged job (pull-requests: write). Reads
mlinter-findings.json and mlinter-pr-number.txt produced by the
check_code_quality job. Never checks out or executes PR code.

Usage:
    GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/repo python utils/post_mlinter_review.py
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# github_utils is present because the workflow checks out the repo.
from github_utils import get_github_json, github_request


GITHUB_API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
MAX_INLINE = 40
MAX_PER_FILE = 10

# Delimiters for the mlinter state block embedded in the PR description.
_STATE_START = "<!-- mlinter-state-start -->"
_STATE_END = "<!-- mlinter-state-end -->"


def _findings_hash(findings):
    """Short SHA-256 hash of findings content, excluding parse-error sentinels (rule=None)."""
    key = json.dumps(
        sorted([(f["path"], f["line"], f["rule"], f["message"]) for f in findings if f.get("rule")])
    ).encode()
    return hashlib.sha256(key).hexdigest()[:8]


def _read_pr_state(pr_body):
    """Extract the mlinter hash from the PR description state block. Returns hash string or None."""
    m = re.search(
        r"<!-- mlinter-state-start -->.*?hash: `([0-9a-f]{8})`.*?<!-- mlinter-state-end -->",
        pr_body or "",
        re.DOTALL,
    )
    return m.group(1) if m else None


def _update_pr_description(pr_body, findings_hash, commit_sha, review_url):
    """Add or replace the mlinter state block in the PR description."""
    block = "\n".join(
        [
            _STATE_START,
            "<details>",
            "<summary>🤖 mlinter review state</summary>",
            "",
            f"- hash: `{findings_hash}`",
            f"- commit: `{commit_sha}`",
            f"- review: {review_url}",
            "",
            "</details>",
            _STATE_END,
        ]
    )
    if _STATE_START in (pr_body or ""):
        return re.sub(
            r"<!-- mlinter-state-start -->.*?<!-- mlinter-state-end -->",
            block,
            pr_body,
            flags=re.DOTALL,
        )
    return (pr_body or "").rstrip() + "\n\n" + block


def _paginate(url, token):
    items = []
    page = 1
    while page <= 20:
        sep = "&" if "?" in url else "?"
        batch = get_github_json(f"{url}{sep}per_page=100&page={page}", token=token)
        if not isinstance(batch, list) or not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def _commentable_lines(patch):
    """Return line numbers on the head side that GitHub accepts as comment anchors."""
    lines = set()
    head = 0
    for raw in (patch or "").splitlines():
        if raw.startswith("@@"):
            try:
                head = int(raw.split("+", 1)[1].split(",")[0].split()[0])
            except (IndexError, ValueError):
                head = 0
            continue
        if raw.startswith("-"):
            continue
        if raw.startswith(("+", " ")) and head:
            lines.add(head)
            head += 1
    return lines


def _comment_body(finding, rules):
    rule = finding["rule"]
    spec = rules.get(rule, {})
    lines = [f"**{rule}** — {finding['message']}"]
    why = (spec.get("why_bad") or "").strip()
    diff = (spec.get("diff") or "").strip()
    if why or diff:
        lines += ["", "<details><summary>Why this matters</summary>", ""]
        if why:
            lines += [why, ""]
        if diff:
            lines += ["```diff", diff, "```", ""]
        lines += [f"Suppress with `# trf-ignore: {rule}` if intentional.", "", "</details>"]
    return "\n".join(lines)


def _review_body(findings, rules, skipped):
    counts = Counter(f["rule"] for f in findings)
    lines = [
        "## Model linter — first pass",
        "",
        f"`transformers-mlinter` found **{len(findings)} item(s)** in the model files this PR touches. "
        "These are structural conventions a maintainer would otherwise flag by hand.",
        "",
        "This is automated and advisory — it does not block merging.",
        "",
        "| rule | count | what it checks |",
        "| --- | --- | --- |",
    ]
    for rule, count in counts.most_common():
        description = (rules.get(rule, {}).get("description") or "").replace("|", "\\|")
        lines.append(f"| `{rule}` | {count} | {description} |")
    if skipped:
        lines += [
            "",
            f"<details><summary>{len(skipped)} item(s) outside this diff</summary>",
            "",
        ]
        for f in skipped:
            lines.append(f"- `{f['path']}:{f['line']}` **{f['rule']}** — {f['message']}")
        lines += ["", "</details>"]
    return "\n".join(lines)


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")

    pr_number_path = Path("mlinter-pr-number.txt")
    findings_path = Path("mlinter-findings.json")

    if not pr_number_path.exists() or not findings_path.exists():
        print("Artifact files missing; skipping.")
        return 0

    pr_number = pr_number_path.read_text().strip()
    if not pr_number:
        print("No PR number; skipping.")
        return 0

    payload = json.loads(findings_path.read_text())
    findings = payload.get("findings") or []
    rules = payload.get("rules") or {}

    if not findings:
        print("No findings; not posting.")
        return 0

    pulls_url = f"{GITHUB_API_URL}/repos/{repo}/pulls/{pr_number}"
    current_hash = _findings_hash(findings)

    # Read PR description once — used for both deduplication and state update.
    pr_data = get_github_json(pulls_url, token=token)
    pr_body = pr_data.get("body") or ""
    commit_sha = pr_data["head"]["sha"][:8]

    existing_hash = _read_pr_state(pr_body)
    if existing_hash == current_hash:
        print(f"Findings unchanged (hash {current_hash}); skipping.")
        return 0
    if existing_hash:
        print(f"Findings changed (was {existing_hash}, now {current_hash}); posting new review.")

    # Map each changed file to its commentable lines.
    anchors = {f["filename"]: _commentable_lines(f.get("patch")) for f in _paginate(f"{pulls_url}/files", token)}

    comments = []
    skipped = []
    per_file = defaultdict(int)
    for finding in findings:
        path, line = finding["path"], finding["line"]
        if line in anchors.get(path, set()) and len(comments) < MAX_INLINE and per_file[path] < MAX_PER_FILE:
            comments.append({"path": path, "line": line, "side": "RIGHT", "body": _comment_body(finding, rules)})
            per_file[path] += 1
        else:
            skipped.append(finding)

    body = _review_body(findings, rules, skipped)
    try:
        review = github_request(
            f"{pulls_url}/reviews",
            token=token,
            method="POST",
            payload={"event": "COMMENT", "body": body, "comments": comments},
        )
        print(f"Posted review {review['id']} with {len(comments)} inline comment(s).")
    except RuntimeError as exc:
        # Anchor rejection (e.g. force-push race): fall back to summary-only.
        print(f"Inline review failed: {exc}; posting summary only.", file=sys.stderr)
        review = github_request(
            f"{pulls_url}/reviews",
            token=token,
            method="POST",
            payload={"event": "COMMENT", "body": body},
        )
        print(f"Posted summary-only review {review['id']}.")

    # Update PR description with the new state (1 API call, replaces existing block if present).
    review_url = f"https://github.com/{repo}/pull/{pr_number}#pullrequestreview-{review['id']}"
    new_pr_body = _update_pr_description(pr_body, current_hash, commit_sha, review_url)
    github_request(pulls_url, token=token, method="PATCH", payload={"body": new_pr_body})
    print(f"Updated PR description with mlinter state (hash {current_hash}, commit {commit_sha}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
