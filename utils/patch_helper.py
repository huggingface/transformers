# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
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
This should help you prepare a patch, automatically extracting the commits to cherry-pick
in chronological order to avoid merge conflicts. An equivalent way to do this is to use
`git log --pretty=oneline HEAD...v4.41.0` and grep.

Potential TODO: automatically cherry-picks them.

Pass in a list of PR:
`python utils/patch_helper.py --prs 31108 31054 31008 31010 31004`
will produce the following:
```bash
Skipping invalid version tag: list
Skipping invalid version tag: localattn1
Git cherry-pick commands to run:
git cherry-pick 03935d300d60110bb86edb49d2315089cfb19789 #2024-05-24 11:00:59+02:00
git cherry-pick bdb9106f247fca48a71eb384be25dbbd29b065a8 #2024-05-24 19:02:55+02:00
git cherry-pick 84c4b72ee99e8e65a8a5754a5f9d6265b45cf67e #2024-05-27 10:34:14+02:00
git cherry-pick 936ab7bae5e040ec58994cb722dd587b9ab26581 #2024-05-28 11:56:05+02:00
git cherry-pick 0bef4a273825d2cfc52ddfe62ba486ee61cc116f #2024-05-29 13:33:26+01:00
```
"""

import argparse

from git import GitCommandError, Repo
from packaging import version


def get_merge_commit(repo, pr_number, since_tag):
    try:
        # Use git log to find the merge commit for the PR within the given tag range
        merge_commit = next(repo.iter_commits(f"v{since_tag}...origin/main", grep=f"#{pr_number}"))
        return merge_commit
    except StopIteration:
        print(f"No merge commit found for PR #{pr_number} between tags {since_tag} and {main}")
        return None
    except GitCommandError as e:
        print(f"Error finding merge commit for PR #{pr_number}: {str(e)}")
        return None


def main(pr_numbers):
    repo = Repo(".")  # Initialize the Repo object for the current directory
    merge_commits = []

    tags = {}
    for tag in repo.tags:
        try:
            # Parse and sort tags, skip invalid ones
            tag_ver = version.parse(tag.name)
            tags[tag_ver] = tag
        except Exception:
            print(f"Skipping invalid version tag: {tag.name}")

    last_tag = sorted(tags)[-1]
    major_minor = f"{last_tag.major}.{last_tag.minor}.0"
    # Iterate through tag ranges to find the merge commits
    for pr in pr_numbers:
        pr = pr.split("https://github.com/huggingface/transformers/pull/")[-1]
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
    parser.add_argument("--prs", nargs="+", required=False, type=str, help="PR numbers to find merge commits for")

    args = parser.parse_args()
    if args.prs is None:
        args.prs = "https://github.com/huggingface/transformers/pull/33753  https://github.com/huggingface/transformers/pull/33861  https://github.com/huggingface/transformers/pull/33906  https://github.com/huggingface/transformers/pull/33761  https://github.com/huggingface/transformers/pull/33586  https://github.com/huggingface/transformers/pull/33766  https://github.com/huggingface/transformers/pull/33958 https://github.com/huggingface/transformers/pull/33965".split()
    main(args.prs)
