import os
import github
import json
from github import Github
from fnmatch import fnmatch
from collections import Counter
from pathlib import Path

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
        owners = parts[1:]  # Can be empty, e.g. for dummy files with explicitly no owner!

        # Check if file matches pattern
        if fnmatch(file_path, pattern):
            return owners  # Remember, can still be empty!
    return []  # Should never happen, but just in case

def main():
    # TODO Remove PR owner from owners list so they don't tag themselves
    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo("huggingface/transformers")
    with open(os.environ['GITHUB_EVENT_PATH']) as f:
        event = json.load(f)
    script_dir = Path(__file__).parent.absolute()
    with open(script_dir / "codeowners_for_review_action") as f:
        codeowners_lines = f.readlines()

    # The PR number is available in the event payload
    pr_number = event['pull_request']['number']
    pr = repo.get_pull(pr_number)

    existing_reviews = list(pr.get_reviews())
    if existing_reviews:
        print(f"Already has reviews: {[r.user.login for r in existing_reviews]}")
        return

    users_requested, teams_requested = pr.get_review_requests()
    users_requested = list(users_requested)
    if users_requested:
        print(f"Reviewers already requested: {users_requested}")
        return

    locs_per_owner = Counter()
    for file in pr.get_files():
        owners = get_file_owners(file.filename, codeowners_lines)
        for owner in owners:
            locs_per_owner[owner] += file.changes

    # Assign the top 2 based on locs changed as reviewers
    top_owners = locs_per_owner.most_common(2)
    print("Top owners", top_owners)
    top_owners = [owner[0].removeprefix("@") for owner in top_owners]
    try:
        pr.create_review_request(top_owners)
    except github.GithubException as e:
        print(f"Failed to request review for {top_owners}: {e}")



if __name__ == "__main__":
    main()
