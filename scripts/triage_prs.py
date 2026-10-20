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
Close PRs that don't link an issue their author may work on. See "Contribution rules" in CONTRIBUTING.md.

A PR passes if its author is exempt, or if its description links an open issue that is labeled `contributions-welcome`,
or labeled `triaged` and opened by the PR author. Passing PRs from non-exempt authors get the `triaged` label and are
skipped on later runs, so maintainers can also add that label to override a closure.

Checks the PR in `PR_NUMBER`, or every open PR if it is unset. Nothing is changed unless `DRY_RUN=false`.
"""

import os
import re
from datetime import datetime, timezone

from github import Github
from github.GithubException import UnknownObjectException


START_DATE = datetime(2026, 10, 20, tzinfo=timezone.utc)  # Set to the day enforcement starts; earlier PRs are skipped
TRIAGED_LABEL = "triaged"
OPEN_LABEL = "contributions-welcome"
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
ISSUE_REFERENCE = re.compile(
    r"(?<![\w/])#(\d+)|huggingface/transformers#(\d+)|github\.com/huggingface/transformers/issues/(\d+)"
)
CLOSURE_MARKER = "<!-- triage-prs: closed -->"
CLOSURE_MESSAGE = (
    f"{CLOSURE_MARKER}\n\n"
    "Thanks for your PR! We're closing it because its description doesn't link an open issue that you can work on. "
    "You can open a PR for an issue you opened once a maintainer has labeled it `triaged`, or for any issue labeled "
    f"`{OPEN_LABEL}`.\n\n"
    "Once that's the case, link the issue in the description (for example `Fixes #123`) and reopen this PR; "
    "there's no need to open a new one. If you think this was a mistake, ping a maintainer here.\n\n"
    "See the [contribution rules](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#contribution-rules)."
)


def linked_issues(body):
    numbers = []
    for groups in ISSUE_REFERENCE.findall(body or ""):
        number = int(next(group for group in groups if group))
        if number not in numbers:
            numbers.append(number)
    return numbers[:MAX_ISSUE_REFERENCES]


def may_work_on(repo, number, user_id):
    try:
        issue = repo.get_issue(number)
    except UnknownObjectException:
        return False
    if issue.pull_request is not None or issue.state != "open":
        return False
    labels = {label.name for label in issue.labels}
    return OPEN_LABEL in labels or (TRIAGED_LABEL in labels and issue.user.id == user_id)


def triage(repo, pr, dry_run):
    if pr.state != "open" or pr.created_at < START_DATE:
        return
    if pr.author_association in EXEMPT_ASSOCIATIONS or pr.user.id in ALLOWLIST:
        return
    if any(label.name == TRIAGED_LABEL for label in pr.labels):
        return

    passes = any(may_work_on(repo, number, pr.user.id) for number in linked_issues(pr.body))
    print(f"{'Would' if dry_run else 'Will'} {'label' if passes else 'close'} PR #{pr.number}")
    if dry_run:
        return
    if passes:
        pr.add_to_labels(TRIAGED_LABEL)
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
