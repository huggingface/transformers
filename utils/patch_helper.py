"""
Pass in a list of PR:
`python utils/patch
"""

import argparse
from git import Repo, GitCommandError
from packaging import version

def get_merge_commit(repo, pr_number, since_tag):
    try:
        # Use git log to find the merge commit for the PR within the given tag range
        merge_commit = next(repo.iter_commits(f'v{since_tag}...HEAD', grep=f'#{pr_number}'))
        return merge_commit
    except StopIteration:
        print(f"No merge commit found for PR #{pr_number} between tags {since_tag} and {main}")
        return None
    except GitCommandError as e:
        print(f"Error finding merge commit for PR #{pr_number}: {str(e)}")
        return None

def main(pr_numbers):
    repo = Repo('..')  # Initialize the Repo object for the current directory
    merge_commits = []

    tags = {}
    for tag in repo.tags:
        try:
            # Parse and sort tags, skip invalid ones
            tag_ver = version.parse(tag.name)
            tags[tag_ver] = tag
        except Exception as e:
            print(f"Skipping invalid version tag: {tag.name}")

    last_tag = sorted(tags)[-1]
    major_minor = f"{last_tag.major}.{last_tag.minor}.0"
    # Iterate through tag ranges to find the merge commits
    for pr in pr_numbers:
        commit = get_merge_commit(repo, pr, major_minor)
        if commit:
            merge_commits.append(commit)

    # Sort commits by date
    merge_commits.sort(key=lambda commit: commit.committed_datetime)

    # Output the git cherry-pick commands
    print("Git cherry-pick commands to run:")
    for commit in merge_commits:
        print(f"git cherry-pick {commit.hexsha} #{commit.committed_datetime}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and sort merge commits for specified PRs.")
    parser.add_argument('--prs', nargs='+', type=int, help="PR numbers to find merge commits for")

    args = parser.parse_args()
    main(args.prs)
