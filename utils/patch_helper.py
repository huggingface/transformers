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

import json
import subprocess

import transformers


LABEL = "for patch"  # Replace with your label
REPO = "huggingface/transformers"  # Optional if already in correct repo


def get_release_branch_name():
    """Derive branch name from transformers version."""
    major, minor, *_ = transformers.__version__.split(".")
    major = int(major)
    minor = int(minor)

    if minor == 0:
        # Handle major version rollback, e.g., from 5.0 to 4.latest (if ever needed)
        major -= 1
        # You'll need logic to determine the last minor of the previous major version
        raise ValueError("Minor version is 0; need logic to find previous major version's last minor")
    # else:
    #     minor -= 1

    return f"v{major}.{minor}-release"


def checkout_branch(branch):
    """Checkout the target branch."""
    try:
        subprocess.run(["git", "checkout", branch], check=True)
        print(f"✅ Checked out branch: {branch}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to checkout branch: {branch}. Does it exist?")
        exit(1)


def get_prs_by_label(label):
    """Call gh CLI to get PRs with a specific label."""
    cmd = [
        "gh",
        "pr",
        "list",
        "--label",
        label,
        "--state",
        "all",
        "--json",
        "number,title,mergeCommit,url",
        "--limit",
        "100",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    result.check_returncode()
    prs = json.loads(result.stdout)
    for pr in prs:
        is_merged = pr.get("mergeCommit", {})
        if is_merged:
            pr["oid"] = is_merged.get("oid")
    return prs


def get_commit_timestamp(commit_sha):
    """Get UNIX timestamp of a commit using git."""
    result = subprocess.run(["git", "show", "-s", "--format=%ct", commit_sha], capture_output=True, text=True)
    result.check_returncode()
    return int(result.stdout.strip())


def cherry_pick_commit(sha):
    """Cherry-pick a given commit SHA."""
    try:
        subprocess.run(["git", "cherry-pick", sha], check=True)
        print(f"✅ Cherry-picked commit {sha}")
    except subprocess.CalledProcessError:
        print(f"⚠️ Failed to cherry-pick {sha}. Manual intervention required.")


def commit_in_history(commit_sha, base_branch="HEAD"):
    """Return True if commit is already part of base_branch history."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit_sha, base_branch],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main(verbose=False):
    branch = get_release_branch_name()
    checkout_branch(branch)
    prs = get_prs_by_label(LABEL)
    # Attach commit timestamps
    for pr in prs:
        sha = pr.get("oid")
        if sha:
            pr["timestamp"] = get_commit_timestamp(sha)
        else:
            print("\n" + "=" * 80)
            print(f"⚠️  WARNING: PR #{pr['number']} ({sha}) is NOT in main!")
            print("⚠️  A core maintainer must review this before cherry-picking.")
            print("=" * 80 + "\n")
    # Sort by commit timestamp (ascending)
    prs = [pr for pr in prs if pr.get("timestamp") is not None]
    prs.sort(key=lambda pr: pr["timestamp"])
    for pr in prs:
        sha = pr.get("oid")
        if sha:
            if commit_in_history(sha):
                if verbose:
                    print(f"🔁 PR #{pr['number']} ({pr['title']}) already in history. Skipping.")
            else:
                print(f"🚀 PR #{pr['number']} ({pr['title']}) not in history. Cherry-picking...")
                cherry_pick_commit(sha)


if __name__ == "__main__":
    main()
