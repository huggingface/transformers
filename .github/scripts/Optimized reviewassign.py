# Auto reviewer assignment (production ready)

import json
import os
import fnmatch
from collections import Counter
from pathlib import Path
from github import Github, GithubException


def match_pattern(file_path, pattern):
    """Proper CODEOWNERS glob matching."""
    pattern = pattern.lstrip("/")
    return fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch("/" + file_path, pattern)


def parse_codeowners(path):
    entries = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue

        parts = line.split()
        pattern = parts[0]
        owners = [o.replace("@", "") for o in parts[1:]]
        entries.append((pattern, owners))
    return entries


def get_file_owners(file_path, entries):
    for pattern, owners in reversed(entries):
        if match_pattern(file_path, pattern):
            return owners
    return []


def get_all_pr_files(pr):
    """Handles pagination correctly."""
    files = []
    page = 0
    while True:
        batch = pr.get_files().get_page(page)
        if not batch:
            break
        files.extend(batch)
        page += 1
    return files


def main():
    token = os.environ["GITHUB_TOKEN"]
    repo_name = os.environ["GITHUB_REPOSITORY"]

    g = Github(token)
    repo = g.get_repo(repo_name)

    event = json.load(open(os.environ["GITHUB_EVENT_PATH"]))
    pr = repo.get_pull(event["pull_request"]["number"])
    author = pr.user.login

    entries = parse_codeowners("codeowners_for_review_action")

    # Skip if reviews already exist
    if list(pr.get_reviews()):
        print("PR already reviewed. Skipping.")
        return

    # Count LOC per owner
    locs = Counter()

    for file in get_all_pr_files(pr):
        owners = get_file_owners(file.filename, entries)
        for owner in owners:
            if owner != author:
                locs[owner] += file.changes

    if not locs:
        print("No owners found → assigning fallback reviewer")
        fallback = os.getenv("FALLBACK_REVIEWER")
        if fallback:
            pr.create_review_request([fallback])
        return

    reviewers = [o for o, _ in locs.most_common(2)]

    print("Requesting reviews from:", reviewers)

    try:
        pr.create_review_request(reviewers)
    except GithubException as e:
        print("GitHub API error:", e)


if __name__ == "__main__":
    main()
