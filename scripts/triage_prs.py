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
"""
Close PRs that link an issue their author may not work on. See "Contribution rules" in CONTRIBUTING.md.

A PR links an issue with a closing keyword (`Fixes #123`); other mentions of issues are just context. A PR that links
no issue is standalone, and gets the `standalone` label so maintainers can check its evidence. A PR that links issues
passes if one of them is open and is labeled `contributions-welcome` or `New model`, or is labeled `triaged` and was
opened by the PR author. Otherwise the PR is closed.

PRs are re-checked whenever their description is edited, so links can be added or removed at any time: the
`standalone` label follows the description, and a PR this script closed is reopened once it links a qualifying issue.
Exempt authors are skipped, and so are PRs with the `triaged` label, which maintainers can add to override a closure.

Checks the PR in `PR_NUMBER`, or every open PR if it is unset. Nothing is changed unless `DRY_RUN=false`.
"""

import os
import re
from datetime import datetime, timezone

from github import Github
from github.GithubException import UnknownObjectException


START_DATE = datetime(2026, 10, 20, tzinfo=timezone.utc)  # Set to the day enforcement starts; earlier PRs are skipped
TRIAGED_LABEL = "triaged"
STANDALONE_LABEL = "standalone"
OPEN_LABELS = {"contributions-welcome", "New model"}
BOT_LOGIN = "github-actions[bot]"
# Hugging Face members whose membership is private and who have no access to this repo show up as CONTRIBUTOR or NONE,
# so they need an allowlist entry or public membership.
EXEMPT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
# Long-standing contributors and bots, by user ID (https://api.github.com/users/<login>), since logins can be renamed.
# This file is outside src/, tests/, docs/ and utils/, so the PR CI security gate stops forks from editing it.
ALLOWLIST = {
    49699333,  # dependabot[bot]
    283583894,  # sergereview[bot]
    265538906,  # hf-security-analysis[bot]
}
MAX_ISSUE_REFERENCES = 10
# GitHub's closing keywords: https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue
ISSUE_LINK = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?):?\s+"
    r"(?:#|huggingface/transformers#|https://github\.com/huggingface/transformers/issues/)(\d+)\b",
    re.IGNORECASE,
)
CLOSURE_MARKER = "<!-- triage-prs: closed -->"
CLOSURE_MESSAGE = (
    f"{CLOSURE_MARKER}\n\n"
    "Thanks for your PR! We're closing it because the issue it links isn't one you can work on. "
    "You can open a PR for an issue you opened once a maintainer has labeled it `triaged`, or for any issue labeled "
    f"{' or '.join(f'`{label}`' for label in sorted(OPEN_LABELS))}.\n\n"
    "Once the PR links an issue that qualifies, edit the description (for example `Fixes #123`) and we'll reopen it "
    "automatically; there's no need to open a new one. If you think this was a mistake, ping a maintainer here.\n\n"
    "See the [contribution rules](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#contribution-rules)."
)


def linked_issues(body):
    numbers = []
    for number in ISSUE_LINK.findall(body or ""):
        if int(number) not in numbers:
            numbers.append(int(number))
    return numbers[:MAX_ISSUE_REFERENCES]


def may_work_on(repo, number, user_id):
    try:
        issue = repo.get_issue(number)
    except UnknownObjectException:
        return False
    if issue.pull_request is not None or issue.state != "open":
        return False
    labels = {label.name for label in issue.labels}
    return bool(OPEN_LABELS & labels) or (TRIAGED_LABEL in labels and issue.user.id == user_id)


def closed_by_us(pr):
    closer = pr.as_issue().closed_by
    if closer is None or closer.login != BOT_LOGIN:
        return False
    return any(CLOSURE_MARKER in (comment.body or "") for comment in pr.get_issue_comments())


def triage(repo, pr, dry_run):
    if pr.created_at < START_DATE or pr.merged:
        return
    if pr.author_association in EXEMPT_ASSOCIATIONS or pr.user.id in ALLOWLIST:
        return
    labels = {label.name for label in pr.labels}
    if TRIAGED_LABEL in labels:
        return

    issues = linked_issues(pr.body)
    standalone = not issues
    passes = standalone or any(may_work_on(repo, number, pr.user.id) for number in issues)

    if pr.state != "open":
        # Only undo our own closures, and only once the PR links a qualifying issue. Dropping the link from a closed PR
        # doesn't reopen it: a PR that fixes an existing issue has to go through that issue.
        if standalone or not passes or not closed_by_us(pr):
            return
        print(f"{'Would' if dry_run else 'Will'} reopen PR #{pr.number}")
        if not dry_run:
            pr.edit(state="open")
    else:
        action = "label as standalone" if standalone else "pass" if passes else "close"
        print(f"{'Would' if dry_run else 'Will'} {action} PR #{pr.number}")
    if dry_run:
        return

    if standalone and STANDALONE_LABEL not in labels:
        pr.add_to_labels(STANDALONE_LABEL)
    elif not standalone and STANDALONE_LABEL in labels:
        pr.remove_from_labels(STANDALONE_LABEL)
    if passes:
        return
    # A PR reopened without a valid link is closed again, but only commented on once
    if not any(CLOSURE_MARKER in (comment.body or "") for comment in pr.get_issue_comments()):
        pr.create_issue_comment(CLOSURE_MESSAGE)
    pr.edit(state="closed")


def main():
    dry_run = os.environ.get("DRY_RUN", "true").lower() != "false"
    repo = Github(os.environ["GITHUB_TOKEN"]).get_repo("huggingface/transformers")
    if os.environ.get("PR_NUMBER"):
        prs = [repo.get_pull(int(os.environ["PR_NUMBER"]))]
    else:
        # Snapshot the list first, since closing PRs changes the pagination of open ones
        prs = list(repo.get_pulls(state="open"))
    for pr in prs:
        triage(repo, pr, dry_run)


if __name__ == "__main__":
    main()
