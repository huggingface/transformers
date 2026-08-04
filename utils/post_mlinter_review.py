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

"""Post the model linter's findings on a PR as a single review with inline comments.

Reads the JSON that `collect_mlinter_findings.py` produced in the untrusted job. Never checks out or
executes PR code: it only talks to the GitHub API.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


GITHUB_API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
RULES_DOC_URL = "https://github.com/huggingface/transformers-mlinter#rules"
# Lets a re-run recognise its own previous review instead of stacking duplicates.
MARKER = "<!-- mlinter-review -->"
MAX_INLINE_COMMENTS = 40


def request_json(url, token, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "transformers-mlinter-review")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def paginate(url, token):
    items = []
    page = 1
    while page <= 20:
        separator = "&" if "?" in url else "?"
        batch = request_json(f"{url}{separator}per_page=100&page={page}", token)
        if not isinstance(batch, list) or not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def commentable_lines(patch):
    """Line numbers on the head side of a file that a review comment can anchor to.

    GitHub rejects a comment whose line is not part of the diff, so added and context lines are the
    only valid anchors. Deleted lines never appear on the head side.
    """
    lines = set()
    head_line = 0
    for raw in (patch or "").splitlines():
        if raw.startswith("@@"):
            # @@ -old,+count +new,+count @@
            try:
                head_line = int(raw.split("+", 1)[1].split(",", 1)[0].split(" ", 1)[0])
            except (IndexError, ValueError):
                head_line = 0
            continue
        if raw.startswith("-"):
            continue
        if raw.startswith("+") or raw.startswith(" "):
            if head_line:
                lines.add(head_line)
                head_line += 1
    return lines


def comment_body(finding, rules):
    rule = finding["rule"]
    spec = rules.get(rule, {})
    lines = [f"**{rule}** — {finding['message']}"]
    why = (spec.get("why_bad") or "").strip()
    diff = (spec.get("diff") or "").strip()
    if why or diff:
        lines.append("")
        lines.append("<details><summary>Why this matters</summary>")
        lines.append("")
        if why:
            lines.append(why)
            lines.append("")
        if diff:
            lines.append("```diff")
            lines.append(diff)
            lines.append("```")
            lines.append("")
        lines.append(f"If this model genuinely needs to deviate, add `# trf-ignore: {rule}` above the line.")
        lines.append("")
        lines.append("</details>")
    return "\n".join(lines)


def review_body(findings, rules, inline_count, skipped, version):
    counts = Counter(finding["rule"] for finding in findings)
    lines = [
        MARKER,
        "## Model linter — first pass",
        "",
        f"`transformers-mlinter` {version} found **{len(findings)} item(s)** in the model files this PR touches. "
        "These are the structural conventions a maintainer would otherwise flag by hand, so getting them out of "
        "the way first makes the human review shorter.",
        "",
        "This is automated and advisory. It does not block merging, and a maintainer may well tell you to ignore "
        "some of it.",
        "",
        "| rule | count | what it checks |",
        "| --- | --- | --- |",
    ]
    for rule, count in counts.most_common():
        description = (rules.get(rule, {}).get("description") or "").replace("|", "\\|")
        lines.append(f"| [`{rule}`]({RULES_DOC_URL}) | {count} | {description} |")
    lines.append("")
    if inline_count:
        lines.append(f"{inline_count} of these are attached inline below.")
    if skipped:
        lines.append("")
        lines.append(f"<details><summary>{len(skipped)} item(s) point at lines outside this diff</summary>")
        lines.append("")
        for finding in skipped:
            lines.append(f"- `{finding['path']}:{finding['line']}` — **{finding['rule']}** {finding['message']}")
        lines.append("")
        lines.append("</details>")
    lines.append("")
    lines.append(
        "Run it locally with `pip install transformers-mlinter && mlinter --changed-only`, or see one rule in "
        "detail with `mlinter --rule TRF001`. Suppress a single line with `# trf-ignore: TRFxxx`."
    )
    return "\n".join(lines)


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is not set.", file=sys.stderr)
        return 1
    repository = os.environ.get("GITHUB_REPOSITORY")
    findings_path = Path(os.environ.get("MLINTER_FINDINGS", "mlinter-findings.json"))
    pr_number = os.environ.get("PR_NUMBER", "").strip()

    if not pr_number:
        print("No PR number available; nothing to post.")
        return 0
    if not findings_path.exists():
        print(f"{findings_path} is missing; the lint job produced nothing.")
        return 0

    payload = json.loads(findings_path.read_text(encoding="utf-8"))
    findings = payload.get("findings") or []
    rules = payload.get("rules") or {}
    version = payload.get("mlinter_version", "")

    if not findings:
        print("No findings; not posting a review.")
        return 0

    pulls_url = f"{GITHUB_API_URL}/repos/{repository}/pulls/{pr_number}"

    # Already reviewed this PR once? The point is a first pass, not a comment on every push.
    for review in paginate(f"{pulls_url}/reviews", token):
        if MARKER in (review.get("body") or ""):
            print(f"Already posted review {review.get('id')}; leaving it alone.")
            return 0

    anchors = {}
    for entry in paginate(f"{pulls_url}/files", token):
        anchors[entry["filename"]] = commentable_lines(entry.get("patch"))

    comments = []
    skipped = []
    per_file = defaultdict(int)
    for finding in findings:
        path, line = finding["path"], finding["line"]
        # Cap per file so one noisy file cannot bury the rest of the review.
        if line in anchors.get(path, set()) and len(comments) < MAX_INLINE_COMMENTS and per_file[path] < 10:
            comments.append({"path": path, "line": line, "side": "RIGHT", "body": comment_body(finding, rules)})
            per_file[path] += 1
        else:
            skipped.append(finding)

    body = review_body(findings, rules, len(comments), skipped, version)
    try:
        review = request_json(
            f"{pulls_url}/reviews",
            token,
            method="POST",
            payload={"event": "COMMENT", "body": body, "comments": comments},
        )
        print(f"Posted review {review.get('id')} with {len(comments)} inline comment(s).")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(f"Review with inline comments failed ({exc.code}): {detail}", file=sys.stderr)
        # An anchor can still be rejected (force-push races, renamed files). A summary-only review keeps
        # the feedback rather than losing the whole run.
        body = review_body(findings, rules, 0, findings, version)
        review = request_json(f"{pulls_url}/reviews", token, method="POST", payload={"event": "COMMENT", "body": body})
        print(f"Posted summary-only review {review.get('id')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
