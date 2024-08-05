# Copyright 2024 The HuggingFace Team. All rights reserved.
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
This script is used to get the models for which to run slow CI.

A new model added in a pull request will be included, as well as models specified in a commit message with a prefix
`[run-slow]`, `[run_slow]` or `[run slow]`. For example, the commit message `[run_slow]bert, gpt2` will give `bert` and
`gpt2`.

Usage:

```bash
python utils/pr_slow_ci_models.py.py
```
"""

import argparse
import re
from pathlib import Path
from typing import List

from git import Repo


PATH_TO_REPO = Path(__file__).parent.parent.resolve()


def get_new_python_files_between_commits(base_commit: str, commits: List[str]) -> List[str]:
    """
    Get the list of added python files between a base commit and one or several commits.

    Args:
        repo (`git.Repo`):
            A git repository (for instance the Transformers repo).
        base_commit (`str`):
            The commit reference of where to compare for the diff. This is the current commit, not the branching point!
        commits (`List[str]`):
            The list of commits with which to compare the repo at `base_commit` (so the branching point).

    Returns:
        `List[str]`: The list of python files added between a base commit and one or several commits.
    """
    code_diff = []
    for commit in commits:
        for diff_obj in commit.diff(base_commit):
            # We always add new python files
            if diff_obj.change_type == "A" and diff_obj.b_path.endswith(".py"):
                code_diff.append(diff_obj.b_path)

    return code_diff


def get_new_python_files() -> List[str]:
    """
    Return a list of python files that have been added between the current head and the main branch.

    Returns:
        `List[str]`: The list of python files added.
    """
    repo = Repo(PATH_TO_REPO)

    try:
        # For the cases where the main branch exists locally
        main = repo.refs.main
    except AttributeError:
        # On GitHub Actions runners, it doesn't have local main branch
        main = repo.remotes.origin.refs.main

    print(f"main is at {main.commit}")
    print(f"Current head is at {repo.head.commit}")

    branching_commits = repo.merge_base(main, repo.head)
    for commit in branching_commits:
        print(f"Branching commit: {commit}")
    return get_new_python_files_between_commits(repo.head.commit, branching_commits)


def get_new_model():
    new_files = get_new_python_files()
    reg = re.compile(r"src/transformers/(models/.*)/modeling_.*\.py")

    new_model = ""
    for x in new_files:
        find_new_model = reg.findall(x)
        if len(find_new_model) > 0:
            new_model = find_new_model[0]
            # It's unlikely we have 2 new modeling files in a pull request.
            break
    return new_model


def parse_commit_message(commit_message: str) -> str:
    """
    Parses the commit message to find the models specified in it to run slow CI.

    Args:
        commit_message (`str`): The commit message of the current commit.

    Returns:
        `str`: The substring in `commit_message` after `[run-slow]`, [run_slow]` or [run slow]`. If no such prefix is
         found, the empty string is returned.
    """
    if commit_message is None:
        return ""

    command_search = re.search(r"\[([^\]]*)\](.*)", commit_message)
    if command_search is None:
        return ""

    command = command_search.groups()[0]
    command = command.lower().replace("-", " ").replace("_", " ")
    run_slow = command == "run slow"
    if run_slow:
        models = command_search.groups()[1].strip()
        return models
    else:
        return ""


def get_models(commit_message: str):
    models = parse_commit_message(commit_message)
    return [f"models/{x}" for x in models.replace(",", " ").split()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit_message", type=str, default="", help="The commit message.")
    args = parser.parse_args()

    new_model = get_new_model()
    specified_models = get_models(args.commit_message)
    models = ([] if new_model == "" else [new_model]) + specified_models
    print(sorted(set(models)))
