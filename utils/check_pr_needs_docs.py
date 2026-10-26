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

"""Check whether an approved PR needs docs, and leave a comment if it does.

The `docs-check.yml` workflow runs this script after a maintainer approves a PR.
The script asks Serge to read the PR and decide. It only comments
when the answer is yes, otherwise, it stays quiet, including when the PR
was already checked or something goes wrong.

The prompt is in `.ai/docs-check-rules.md`. Change
it there if you need. You don't need to edit this file.
"""

import argparse
import json
import logging
import os
import re
import sys

from reviewbot.github_client import GitHubClient
from reviewbot.llm_client import ChatCompletionClient, ToolCall
from reviewbot.tools import TOOL_SPECS, ToolEnv, run_tool


log = logging.getLogger("docs-check")

MARKER = "<!-- docs-check -->"
MAINTAINER_ASSOCIATIONS = {"MEMBER", "OWNER", "COLLABORATOR"}
BROWSE_TOOLS = {"read_file", "list_dir", "grep"}
MAX_TOOL_ROUNDS = 6
MAX_DIFF_CHARS = 120_000
MAX_BODY_CHARS = 4_000
MAX_REASON_CHARS = 1_000
PAGE_RE = re.compile(r"docs/source/en/[\w./-]+")


def has_maintainer_approval(reviews: list[dict]) -> bool:
    """Check whether a maintainer has approved the PR.

    Looks at each maintainer's most recent review. If any of them is an
    approval, returns True. Reviews that are only comments don't count, so
    approving and then leaving a comment still counts as an approval.
    """
    latest = {}
    for review in reviews:
        if review.get("author_association") in MAINTAINER_ASSOCIATIONS and review.get("state") != "COMMENTED":
            latest[review["user"]["login"]] = review["state"]
    return "APPROVED" in latest.values()


def skip_reason(files: list[dict]) -> str | None:
    """Quickly rule out PRs that clearly don't need docs, so Serge isn't asked.

    A PR is skipped if it already edits the docs, doesn't touch the library code
    in `src/transformers/`, or only changes private files.

    Returns a short reason when the PR is skipped, or None when Serge should decide.

    `files` is GitHub's list of changed files. Each one has a `filename`, a
    `status` ("added", "modified", "removed", or "renamed"), the number of
    lines added and deleted, and usually a `patch` showing the changes.
    """
    paths = [f["filename"] for f in files]
    if any(p.startswith("docs/source/en/") for p in paths):
        return "already edits docs/source/en"
    src = [p for p in paths if p.startswith("src/transformers/")]
    if not src:
        return "no changes under src/transformers (CI, tests, or tooling)"
    if all(os.path.basename(p).startswith("_") and not os.path.basename(p).startswith("__") for p in src):
        return "only changes private modules"
    return None


def format_diff(files: list[dict], budget: int = MAX_DIFF_CHARS) -> str:
    """Put the changes from every file into one block of text for Serge.

    The text is capped at `budget` characters so it doesn't overwhelm Serge.
    When a file's changes don't fit, only its name is listed. When even the
    name doesn't fit, the text ends with a note saying how many files were left
    out. Serge can still open any file itself.
    """
    parts = []
    used = 0
    for i, f in enumerate(files):
        header = f"### {f['filename']} ({f['status']}, +{f['additions']} -{f['deletions']})"
        patch = f.get("patch") or "(no patch: binary or too large)"
        if used + len(header) + len(patch) > budget:
            patch = "(patch omitted: diff budget exhausted, use read_file if needed)"
        if used + len(header) + len(patch) > budget:
            parts.append(f"(... {len(files) - i} more changed files not shown)")
            break
        used += len(header) + len(patch)
        parts.append(f"{header}\n{patch}")
    return "\n\n".join(parts)


def build_messages(rules: str, pr: dict, files: list[dict]) -> list[dict]:
    """Write what the AI is sent: the rules first, then the PR's title, description, and changes."""
    body = (pr.get("body") or "")[:MAX_BODY_CHARS]
    user = (
        f"PR #{pr['number']}: {pr['title']}\n\n"
        f"Description:\n{body or '(empty)'}\n\n"
        f"Changed files and diff:\n\n{format_diff(files)}"
    )
    return [{"role": "system", "content": rules}, {"role": "user", "content": user}]


def tool_call_dict(tc: ToolCall) -> dict:
    """Record Serge's request to read a file in the expected format.
    """
    call = {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
    if tc.thought_signature:
        call["extra_content"] = {"google": {"thought_signature": tc.thought_signature}}
    return call


def ask_llm(llm: ChatCompletionClient, messages: list[dict], tool_env: ToolEnv | None) -> str:
    """Ask Serge whether the PR needs docs, and return its answer.

    Before answering, Serge can look around the PR's code: read a file, list a
    folder, or search for text. After `MAX_TOOL_ROUNDS` rounds of looking, it's
    told to stop and answer.
    """
    tools = [spec for spec in TOOL_SPECS if spec["function"]["name"] in BROWSE_TOOLS] if tool_env else None
    for _ in range(MAX_TOOL_ROUNDS):
        chat = llm.complete(messages, tools=tools, temperature=0.0, max_tokens=4096)
        if not chat.tool_calls:
            return chat.content
        messages.append(
            {
                "role": "assistant",
                "content": chat.content or "",
                "tool_calls": [tool_call_dict(tc) for tc in chat.tool_calls],
            }
        )
        for tc in chat.tool_calls:
            try:
                args = json.loads(tc.arguments or "{}")
            except ValueError:
                args = {}
            result = (
                run_tool(tool_env, tc.name, args) if tc.name in BROWSE_TOOLS else f"error: unknown tool {tc.name!r}"
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})
    messages.append({"role": "user", "content": "Stop browsing. Reply with the JSON object now."})
    return llm.complete(messages, response_format={"type": "json_object"}, temperature=0.0, max_tokens=1024).content


def parse_verdict(content: str) -> dict | None:
    """Read Serge's answer. Returns None if the answer can't be understood.

    Only suggested pages that are plain file paths inside `docs/source/en/` are
    kept, and no more than 3 of them.
    """
    raw = content or ""
    try:
        obj = json.loads(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        try:
            obj = json.loads(raw[start : end + 1]) if 0 <= start < end else None
        except ValueError:
            obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("needs_docs"), bool):
        return None
    pages = obj.get("suggested_pages") or []
    return {
        "needs_docs": obj["needs_docs"],
        "reason": str(obj.get("reason", "")).strip(),
        "suggested_pages": [p for p in pages if isinstance(p, str) and PAGE_RE.fullmatch(p)][:3],
    }


def neutralize(text: str) -> str:
    """Clean up Serge's text so it's safe to post on the PR.

    Anyone can write a PR, and Serge might repeat what they wrote. So this
    removes links, stops @mentions from notifying anyone, turns HTML into plain
    text, and shortens the text to `MAX_REASON_CHARS` characters.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?:https?|ftp)://\S+|www\.\S+", "(link removed)", text)
    text = text.replace("@", "@​")
    return text[:MAX_REASON_CHARS]


def format_comment(verdict: dict) -> str:
    """Write the comment to post on the PR from Serge's answer.

    The comment starts with a hidden tag so the bot can find its own comment
    later and doesn't comment twice.
    """
    lines = [MARKER, "**This PR might need docs.**", "", neutralize(verdict["reason"])]
    if verdict["suggested_pages"]:
        lines += ["", "Pages that might need an update:", *[f"- `{p}`" for p in verdict["suggested_pages"]]]
    lines += [
        "",
        "<sub>Posted automatically after approval. Docs live in `docs/source/en`. "
        "Ignore this comment if the change doesn't need docs.</sub>",
    ]
    return "\n".join(lines)


def already_commented(gh: GitHubClient, owner: str, repo: str, number: int) -> bool:
    """Check whether the bot has already commented on this PR, by looking for its hidden tag."""
    page = 1
    while True:
        r = gh.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments",
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if any(MARKER in (c.get("body") or "") for c in batch):
            return True
        if len(batch) < 100:
            return False
        page += 1


def main() -> int:
    """Run the full check on one PR.

    Returns 0 whether or not it comments, so a quiet run isn't marked as failed.
    The run only fails if GitHub can't be reached or the arguments are wrong.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head-sha", help="Skip the PR unless a maintainer approved this commit on it.")
    parser.add_argument("--rules", required=True, help="Path to the rules file the LLM follows.")
    parser.add_argument(
        "--checkout", help="Folder with the PR's code for the LLM to read. Leave out to turn off file reading."
    )
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "huggingface/transformers"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the comment instead of posting it. Also skips the approval and earlier-comment checks.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    owner, repo = args.repo.split("/")
    gh = GitHubClient(os.environ["GITHUB_TOKEN"])
    pr = gh.get_pr(owner, repo, args.pr)

    # Step 1: double-check the trigger to confirm the PR is open, approved, and that the
    # commit the trigger ran on was approved on this PR. New commits pushed since then are fine.
    if pr["state"] != "open" or pr["base"]["ref"] != "main":
        log.info("PR #%d is not open against main; skipping.", args.pr)
        return 0
    reviews = gh.get_pr_reviews(owner, repo, args.pr)
    approved_shas = {
        r.get("commit_id")
        for r in reviews
        if r.get("state") == "APPROVED" and r.get("author_association") in MAINTAINER_ASSOCIATIONS
    }
    if args.head_sha and args.head_sha not in approved_shas:
        log.info("PR #%d has no maintainer approval on %s; skipping.", args.pr, args.head_sha)
        return 0
    if not args.dry_run and not has_maintainer_approval(reviews):
        log.info("PR #%d has no maintainer approval; skipping.", args.pr)
        return 0
    # Step 2: check whether the bot already commented to avoid commenting twice.
    if not args.dry_run and already_commented(gh, owner, repo, args.pr):
        log.info("PR #%d already has a docs-check comment; skipping.", args.pr)
        return 0

    # Step 3: skip obvious cases without calling Serge.
    files = gh.get_pr_files(owner, repo, args.pr)
    if reason := skip_reason(files):
        log.info("PR #%d skipped before the LLM call: %s", args.pr, reason)
        return 0

    # Step 4: ask Serge, giving it the rules, the PR, and read-only browse tools.
    with open(args.rules, encoding="utf-8") as f:
        rules = f.read()
    llm = ChatCompletionClient(
        api_base=os.environ.get("LLM_API_BASE", "https://router.huggingface.co/v1"),
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ.get("LLM_MODEL") or None,
        bill_to=os.environ.get("LLM_BILL_TO") or None,
    )
    tool_env = ToolEnv(repo_root=args.checkout) if args.checkout else None

    # Step 5: read the answer and if it fails or outputs gibberish for whatever reason, stay silent.
    try:
        verdict = parse_verdict(ask_llm(llm, build_messages(rules, pr, files), tool_env))
    except Exception:
        log.exception("LLM call failed; staying silent.")
        return 0
    log.info("PR #%d verdict: %s", args.pr, verdict)
    # Tell the workflow a verdict was reached, so it doesn't ask again for this commit.
    if verdict and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write("verdict=done\n")

    # Step 6: comment only when the PR needs docs.
    if not verdict or not verdict["needs_docs"] or not verdict["reason"]:
        return 0
    body = format_comment(verdict)
    if args.dry_run:
        print(body)
    else:
        gh.post_issue_comment(owner, repo, args.pr, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
