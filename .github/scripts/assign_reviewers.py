# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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

"""Request reviewers on a pull request, ranked by how much of the code they own.

Runs from `pull_request_target`, so it has a write token and reads some files at the PR's head
commit (`read_head_file`) to see models and `# Reviewers:` tags the PR adds. Nothing from the head
is ever checked out or executed: the fetched content is only parsed as text, and `is_collaborator`
gates every name before a review is requested, so a `# Reviewers:` tag in an untrusted PR cannot
be used to ping arbitrary accounts.
"""

import json
import os
from collections import Counter
from pathlib import Path

import github
from codeowners_resolver import (
    CODEOWNERS_PATH,
    owners_for_file,
    pr_author_is_in_hf,
    read_local_file,
    unknown_modality_slugs,
)
from github import Github


MAX_REVIEWERS = 2

# Set by `main`; only needed for a file this PR adds or edits, which the checked-out base either
# lacks or has in its pre-PR state.
_REPO = None
_HEAD_SHA = None
_CHANGED_PATHS = frozenset()
_FILE_CACHE = {}

def warn(message):
    # Surfaced on the workflow run itself. Without it an unassignable reviewer is a line in a step
    # log on a green job, which is how #48070 ended up with none.
    print(f"::warning::{message}")

def set_pr_context(repo, head_sha, changed_paths):
    global _REPO, _HEAD_SHA, _CHANGED_PATHS
    _REPO, _HEAD_SHA, _CHANGED_PATHS = repo, head_sha, frozenset(changed_paths)

def read_pr_file(path):
    # Contents of `path`, preferring the PR's own version for a file it touches: a model this PR
    # adds is missing from the `pull_request_target` base checkout, and a `# Reviewers:` tag it adds
    # is not yet visible there. Only ever parsed as data, never executed.
    if path in _FILE_CACHE:
        return _FILE_CACHE[path]
    text = read_head_file(path) if path in _CHANGED_PATHS else ""
    if not text:
        text = read_local_file(path) or read_head_file(path)
    _FILE_CACHE[path] = text
    return text

def read_head_file(path):
    if _REPO is None or _HEAD_SHA is None:
        return ""
    try:
        return _REPO.get_contents(path, ref=_HEAD_SHA).decoded_content.decode("utf-8")
    except github.GithubException:
        return ""  # a file in neither the base nor the head is a normal outcome

def is_collaborator(repo, login):
    # A review can only be requested from a collaborator, and GitHub rejects the WHOLE request
    # rather than the offending name -- so one codeowner who has left takes their valid co-owners
    # down with them. Check before asking.
    try:
        return repo.has_in_collaborators(login)
    except github.GithubException as e:
        warn(f"Could not check whether {login} is a collaborator: {e}")
        return False

def request_reviews(repo, pr, candidates, limit=MAX_REVIEWERS):
    # Request up to `limit` reviews, in ranked order, one name per call so a rejection cannot cancel
    # the others. Owners who cannot be asked are skipped and the next-ranked owner takes the slot.
    # Returns the logins requested.
    requested = []
    for login in candidates:
        if len(requested) == limit:
            break
        if not is_collaborator(repo, login):
            warn(f"Skipping {login}: not a collaborator of {repo.full_name} (stale codeowners entry?)")
            continue
        try:
            pr.create_review_request([login])
        except github.GithubException as e:
            warn(f"Failed to request review from {login}: {e}")
            continue
        requested.append(login)
    return requested

def main():
    script_dir = Path(__file__).parent.absolute()
    with open(script_dir / Path(CODEOWNERS_PATH).name) as f:
        codeowners_lines = f.readlines()
    for slug in unknown_modality_slugs(codeowners_lines):
        warn(f"Unknown modality {slug!r} in {CODEOWNERS_PATH}; that rule is ignored")

    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo("huggingface/transformers")
    with open(os.environ['GITHUB_EVENT_PATH']) as f:
        event = json.load(f)

    # The PR number is available in the event payload
    pr_number = event['pull_request']['number']
    pr = repo.get_pull(pr_number)
    pr_author = pr.user.login
    if pr_author_is_in_hf(pr_author, codeowners_lines):
        print(f"PR author {pr_author} is in codeowners, skipping review request.")
        return

    existing_reviews = list(pr.get_reviews())
    if existing_reviews:
        print(f"Already has reviews: {[r.user.login for r in existing_reviews]}")
        return

    users_requested, teams_requested = pr.get_review_requests()
    users_requested = list(users_requested)
    if users_requested:
        print(f"Reviewers already requested: {users_requested}")
        return

    changed_files = list(pr.get_files())
    set_pr_context(repo, pr.head.sha, [f.filename for f in changed_files])

    # Tally per person, not per spelling: a login is case-insensitive on GitHub, and the same owner
    # appearing as `@ArthurZucker` on one line and `@arthurzucker` on another would otherwise split
    # their total across two entries (and be requested twice).
    locs_per_owner = Counter()
    spelling = {}
    for file in changed_files:
        for owner in owners_for_file(file.filename, codeowners_lines, read_pr_file):
            key = owner.casefold()
            spelling.setdefault(key, owner)
            locs_per_owner[key] += file.changes

    # Assign the top 2 based on locs changed as reviewers, but skip the owner if present
    locs_per_owner.pop(pr_author.casefold(), None)
    ranked_owners = [spelling[key] for key, _ in locs_per_owner.most_common()]
    print("Top owners", [(spelling[key], locs) for key, locs in locs_per_owner.most_common(MAX_REVIEWERS)])
    requested = request_reviews(repo, pr, ranked_owners)
    if requested:
        print(f"Requested review from {requested}")
    elif ranked_owners:
        warn(f"No reviewer could be requested for #{pr_number} out of {ranked_owners}")
    else:
        warn(f"No codeowner matched the files changed in #{pr_number}")



if __name__ == "__main__":
    main()
