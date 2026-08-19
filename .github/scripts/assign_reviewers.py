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

import json
import os
import re
from collections import Counter
from pathlib import Path

import github
from github import Github


MAX_REVIEWERS = 2


def pattern_to_regex(pattern):
    if pattern.startswith("/"):
        start_anchor = True
        pattern = re.escape(pattern[1:])
    else:
        start_anchor = False
        pattern = re.escape(pattern)
    # Replace `*` with "any number of non-slash characters"
    pattern = pattern.replace(r"\*", "[^/]*")
    if start_anchor:
        pattern = r"^\/?" + pattern  # Allow an optional leading slash after the start of the string
    return pattern

def get_file_owners(file_path, codeowners_lines):
    # Process lines in reverse (last matching pattern takes precedence)
    for line in reversed(codeowners_lines):
        # Skip comments and empty lines, strip inline comments
        line = line.split('#')[0].strip()
        if not line:
            continue

        # Split into pattern and owners
        parts = line.split()
        pattern = parts[0]
        # Can be empty, e.g. for dummy files with explicitly no owner!
        owners = [owner.removeprefix("@") for owner in parts[1:]]

        # Check if file matches pattern
        file_regex = pattern_to_regex(pattern)
        if re.search(file_regex, file_path) is not None:
            return owners  # Remember, can still be empty!
    return []  # Should never happen, but just in case

def pr_author_is_in_hf(pr_author, codeowners_lines):
    # Check if the PR author is in the codeowners file
    for line in codeowners_lines:
        line = line.split('#')[0].strip()
        if not line:
            continue

        # Split into pattern and owners
        parts = line.split()
        owners = [owner.removeprefix("@") for owner in parts[1:]]

        if pr_author.casefold() in {owner.casefold() for owner in owners}:
            return True
    return False

def warn(message):
    # Surfaced on the workflow run itself. Without it an unassignable reviewer is a
    # line in a step log on a green job, which is how #48070 ended up with none.
    print(f"::warning::{message}")

def is_collaborator(repo, login):
    # A review can only be requested from a collaborator, and GitHub rejects the
    # WHOLE request rather than the offending name -- so one codeowner who has left
    # takes their valid co-owners down with them. Check before asking.
    try:
        return repo.has_in_collaborators(login)
    except github.GithubException as e:
        warn(f"Could not check whether {login} is a collaborator: {e}")
        return False

def request_reviews(repo, pr, candidates, limit=MAX_REVIEWERS):
    # Request up to `limit` reviews, in ranked order, one name per call so a
    # rejection cannot cancel the others. Owners who cannot be asked are skipped
    # and the next-ranked owner takes the slot. Returns the logins requested.
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
    with open(script_dir / "codeowners_for_review_action") as f:
        codeowners_lines = f.readlines()

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

    # Tally per person, not per spelling: a login is case-insensitive on GitHub, and the same
    # owner appearing as `@ArthurZucker` on one line and `@arthurzucker` on another would
    # otherwise split their total across two entries (and be requested twice).
    locs_per_owner = Counter()
    spelling = {}
    for file in pr.get_files():
        owners = get_file_owners(file.filename, codeowners_lines)
        for owner in owners:
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
