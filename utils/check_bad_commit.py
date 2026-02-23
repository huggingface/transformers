#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import argparse
import copy
import json
import os
import re
import subprocess

import git
import requests


def create_script(target_test):
    """Create a python script to be run by `git bisect run` to determine if `target_test` passes or fails.
    If a test is not found in a commit, the script with exit code `0` (i.e. `Success`).

    Args:
        target_test (`str`): The test to check.

    Returns:
        `str`: The script to be run by `git bisect run`.
    """

    script = f"""
import os
import subprocess

_ = subprocess.run(
    ["python3", "-m", "pip", "install", "-e", "."],
    capture_output = True,
    text=True,
)

result = subprocess.run(
    ["python3", "-m", "pytest", "-v", "--flake-finder", "--flake-runs=4", "-rfEp", f"{target_test}"],
    capture_output = True,
    text=True,
)
print(result.stdout)

if f"FAILED {target_test}" in result.stdout:
    print("test failed")
    exit(1)
elif result.returncode != 0:
    if "ERROR: file or directory not found: " in result.stderr:
        print("test file or directory not found in this commit")
        # git bisect treats exit code 125 as `test not found`. But this causes it not be able to make the conclusion
        # if a test is added between the `good commit` (exclusive) and `bad commit` (inclusive) (in git bisect terminology).
        # So we return 0 here in order to allow the process being able to identify the first commit that fails the test.
        exit(0)
    elif "ERROR: not found: " in result.stderr:
        print("test not found in this commit")
        exit(0)
    else:
        print(f"pytest gets unknown error: {{result.stderr}}")
        exit(1)

print(f"pytest runs successfully.")
exit(0)
"""

    with open("target_script.py", "w") as fp:
        fp.write(script.strip())


def is_bad_commit(target_test, commit):
    repo = git.Repo(".")  # or specify path to your repo

    # Save the current HEAD reference
    original_head = repo.head.commit

    # Checkout to the commit
    repo.git.checkout(commit)

    create_script(target_test=target_test)

    result = subprocess.run(
        ["python3", "target_script.py"],
        capture_output=True,
        text=True,
    )

    # Restore to original commit
    repo.git.checkout(original_head)

    n_passed = 0
    o = re.findall(r"====.* (\d+) passed", result.stdout)
    if len(o) > 0:
        n_passed = int(o[0])

    n_failed = 0
    o = re.findall(r"====.* (\d+) failed", result.stdout)
    if len(o) > 0:
        n_failed = int(o[0])

    error_message = ""
    if n_failed > 0:
        match = re.search(r"^(FAILED .+ - .+)$", result.stdout, re.MULTILINE)
        error_message = match.group(1).strip() if match else "Cannot retrieve error message."

    return result.returncode != 0, n_failed, n_passed, error_message


def find_bad_commit(target_test, start_commit, end_commit):
    """Find (backward) the earliest commit between `start_commit` (inclusive) and `end_commit` (exclusive) at which `target_test` fails.

    Args:
        target_test (`str`): The test to check.
        start_commit (`str`): The latest commit (inclusive).
        end_commit (`str`): The earliest commit (exclusive).

    Returns:
        `dict`: A dict containing the info about the earliest commit at which `target_test` fails.
    """
    result = {
        "bad_commit": None,
        "status": None,
        "failure_at_workflow_commit": None,
        "failure_at_base_commit": None,
        "failure_at_bad_commit": None,
    }

    is_pr_ci = os.environ.get("GITHUB_EVENT_NAME") in ["issue_comment", "pull_request"]

    # For PR comment CI, we "assume" all tests at `end_commit` passed, so any failing test during a PR CI run is
    # "a new failing test", and we can perform more detailed checks with this script.
    # For "a failing tes at start_commit", we check the test against `end_commit` (run multiple times):
    #   - if all passing at end_commit: an actual new failing test at start_commit
    #   - if all failing at end_commit: get the failure message and compare it against the one from start_commit:
    #     - same failure message: not a new failing test --> don't report it
    #      - different failure message: kind of a new failing test --> need to report it
    #   - if both failing and passing at end_commit: mark it as flaky

    # check if `end_commit` fails the test
    failed_before, n_failed, n_passed, failure_at_base_commit = is_bad_commit(target_test, end_commit)
    # We only need one failure to conclude the test is flaky on the previous run with `end_commit`.
    # However, when running on CI, we need at least one failure and one pass to conclude.
    is_flaky_at_end_commit = ((not is_pr_ci) and n_failed > 0) or (is_pr_ci and n_failed > 0 and n_passed > 0)
    # `n_passed == 0` itself is not enough, as the test may not exist in the codebase at `end_commit`.
    is_failing_at_end_commit = failed_before and n_passed == 0

    if is_flaky_at_end_commit:
        result["status"] = (
            f"flaky: test both passed and failed during the check of the current run on the previous commit: {end_commit}"
        )
        return result

    elif (not is_pr_ci) and is_failing_at_end_commit:
        result["status"] = (
            f"flaky: test passed in the previous run (commit: {end_commit}) but failed (on the same commit) during the check of the current run."
        )
        return result

    # if there is no new commit (e.g. 2 different CI runs on the same commit):
    #   - failed once on `start_commit` but passed on `end_commit`, which are the same commit --> flaky (or something change externally) --> don't report
    if start_commit == end_commit:
        result["status"] = (
            f"flaky: test fails on the current CI run but passed in the previous run which is running on the same commit {end_commit}."
        )
        return result

    # Now, we are (almost) sure `target_test` is not failing at `end_commit`. (For a PR CI, it may fail at `end_commit`)
    # Check if `start_commit` fails the test.
    # **IMPORTANT** we only need one pass to conclude the test is flaky on the current run with `start_commit`!
    _, n_failed, n_passed, failure_at_workflow_commit = is_bad_commit(target_test, start_commit)
    if n_passed > 0:
        # failed on CI run, but not reproducible here --> don't report
        result["status"] = (
            f"flaky: test fails on the current CI run (commit: {start_commit}) but passes during the check."
        )
        return result

    # The test fails on `start_commit`, and
    #   - if the CI is run on PR: this block checks if the test also failed on `start_commit`.
    #   - otherwise: the test passed on `end_commit` --> an actual new failing test, this block is skipped.
    if is_pr_ci and failure_at_base_commit != "" and failure_at_workflow_commit != failure_at_base_commit:
        result["bad_commit"] = start_commit
        result["status"] = (
            f"test fails both on the current commit ({start_commit}) and the previous commit ({end_commit}), but with DIFFERENT error message!"
        )
        result["failure_at_workflow_commit"] = failure_at_workflow_commit
        result["failure_at_base_commit"] = failure_at_base_commit
        result["failure_at_bad_commit"] = failure_at_workflow_commit
        return result
    # Fail on both commits but with the same error message ==> don't include
    elif is_pr_ci and failure_at_workflow_commit == failure_at_base_commit:
        result["bad_commit"] = None
        result["status"] = (
            f"test fails both on the current commit ({start_commit}) and the previous commit ({end_commit}) with the SAME error message!"
        )
        result["failure_at_workflow_commit"] = failure_at_workflow_commit
        result["failure_at_base_commit"] = failure_at_base_commit
        result["failure_at_bad_commit"] = failure_at_workflow_commit
        return result

    # The test fails on `start_commit` but passed on `end_commit`.
    create_script(target_test=target_test)

    bash = f"""
git bisect reset
git bisect start --first-parent {start_commit} {end_commit}
git bisect run python3 target_script.py
"""

    with open("run_git_bisect.sh", "w") as fp:
        fp.write(bash.strip())

    bash_result = subprocess.run(
        ["bash", "run_git_bisect.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(bash_result.stdout)

    # This happens if running the script gives exit code < 0  or other issues
    if "error: bisect run failed" in bash_result.stderr:
        error_msg = f"Error when running git bisect:\nbash error: {bash_result.stderr}\nbash output:\n{bash_result.stdout}\nset `bad_commit` to `None`."
        print(error_msg)
        return None, "git bisect failed"

    pattern = r"(.+) is the first bad commit"
    commits = re.findall(pattern, bash_result.stdout)

    bad_commit = None
    failure_at_bad_commit = ""
    if len(commits) > 0:
        bad_commit = commits[0]
        _, _, _, failure_at_bad_commit = is_bad_commit(target_test, bad_commit)

    print(f"Between `start_commit` {start_commit} and `end_commit` {end_commit}")
    print(f"bad_commit: {bad_commit}\n")

    result["bad_commit"] = bad_commit
    result["status"] = "git bisect found the bad commit."
    result["failure_at_workflow_commit"] = failure_at_workflow_commit
    result["failure_at_base_commit"] = failure_at_base_commit
    result["failure_at_bad_commit"] = failure_at_bad_commit
    return result


def get_commit_info(commit, pr_number=None):
    """Get information for a commit via `api.github.com`."""
    if commit is None:
        return {"commit": None, "pr_number": None, "author": None, "merged_by": None}

    author = None
    merged_author = None

    # Use PR number from environment if not provided
    if pr_number is None:
        pr_number = os.environ.get("pr_number")

    # First, get commit info to check if it's a merge commit
    url = f"https://api.github.com/repos/huggingface/transformers/commits/{commit}"
    commit_info = requests.get(url).json()

    commit_to_query = commit

    # Check if this is a merge commit created by GitHub
    if commit_info.get("parents") and len(commit_info["parents"]) > 1:
        commit_message = commit_info.get("commit", {}).get("message", "")
        # Parse message like "Merge 1ac46bed... into 5a67f0a7..."
        import re

        match = re.match(r"^Merge ([a-f0-9]{40}) into ([a-f0-9]{40})", commit_message)
        if match:
            # Use the first SHA (the PR commit)
            commit_to_query = match.group(1)

    # If no PR number yet, try to discover it from the commit
    if not pr_number:
        url = f"https://api.github.com/repos/huggingface/transformers/commits/{commit_to_query}/pulls"
        pr_info_for_commit = requests.get(url).json()
        if len(pr_info_for_commit) > 0:
            pr_number = pr_info_for_commit[0]["number"]

    # If we have a PR number, get author and merged_by info
    if pr_number:
        url = f"https://api.github.com/repos/huggingface/transformers/pulls/{pr_number}"
        pr_for_commit = requests.get(url).json()
        author = pr_for_commit["user"]["login"]
        if pr_for_commit["merged_by"] is not None:
            merged_author = pr_for_commit["merged_by"]["login"]

    parent = commit_info["parents"][0]["sha"]
    if author is None:
        author = commit_info["author"]["login"]

    return {"commit": commit, "pr_number": pr_number, "author": author, "merged_by": merged_author, "parent": parent}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_commit", type=str, required=True, help="The latest commit hash to check.")
    parser.add_argument("--end_commit", type=str, required=True, help="The earliest commit hash to check.")
    parser.add_argument("--test", type=str, help="The test to check.")
    parser.add_argument("--file", type=str, help="The report file.")
    parser.add_argument("--output_file", type=str, required=True, help="The path of the output file.")
    args = parser.parse_args()

    print(f"start_commit: {args.start_commit}")
    print(f"end_commit: {args.end_commit}")

    # `get_commit_info` uses `requests.get()` to request info. via `api.github.com` without using token.
    # If there are many new failed tests in a workflow run, this script may fail at some point with `KeyError` at
    # `pr_number = pr_info_for_commit[0]["number"]` due to the rate limit.
    # Let's cache the commit info. and reuse them whenever possible.
    commit_info_cache = {}

    if len({args.test is None, args.file is None}) != 2:
        raise ValueError("Exactly one argument `test` or `file` must be specified.")

    if args.test is not None:
        commit, status = find_bad_commit(
            target_test=args.test, start_commit=args.start_commit, end_commit=args.end_commit
        )
        with open(args.output_file, "w", encoding="UTF-8") as fp:
            fp.write(f"{args.test}\n{commit}\n{status}")
    elif os.path.isfile(args.file):
        with open(args.file, "r", encoding="UTF-8") as fp:
            reports = json.load(fp)

        for model in reports:
            # We change the format of "new_failures.json" in PR #XXXXX, let's handle both formats for a few weeks.
            if "failures" in reports[model]:
                if "job_link" in reports[model]:
                    for device, device_failures in reports[model]["failures"].items():
                        if device in reports[model]["job_link"]:
                            for failure in device_failures:
                                failure["job_link"] = reports[model]["job_link"][device]
                    del reports[model]["job_link"]
                reports[model] = reports[model]["failures"]

            # TODO: make this script able to deal with both `single-gpu` and `multi-gpu` via a new argument.
            reports[model].pop("multi-gpu", None)
            failed_tests = reports[model].get("single-gpu", [])

            failed_tests_with_bad_commits = []
            for failure in failed_tests:
                test = failure["line"]
                bad_commit_info = find_bad_commit(
                    target_test=test, start_commit=args.start_commit, end_commit=args.end_commit
                )
                info = {"test": test}
                info.update(bad_commit_info)

                bad_commit = bad_commit_info["bad_commit"]

                if bad_commit in commit_info_cache:
                    commit_info = commit_info_cache[bad_commit]
                else:
                    commit_info = get_commit_info(bad_commit)
                    commit_info_cache[bad_commit] = commit_info

                commit_info_copied = copy.deepcopy(commit_info)
                commit_info_copied.pop("commit")
                commit_info_copied.update({"workflow_commit": args.start_commit, "base_commit": args.end_commit})
                info.update(commit_info_copied)
                # put failure message toward the end
                info = {k: v for k, v in info.items() if not k.startswith(("failure_at_", "job_link"))} | {
                    k: v for k, v in info.items() if k.startswith(("failure_at_", "job_link"))
                }

                failed_tests_with_bad_commits.append(info)

            # If no single-gpu test failures, remove the key
            if len(failed_tests_with_bad_commits) > 0:
                reports[model]["single-gpu"] = failed_tests_with_bad_commits
            else:
                reports[model].pop("single-gpu", None)

        # remove the models without any test failure
        reports = {k: v for k, v in reports.items() if len(v) > 0}

        with open(args.output_file, "w", encoding="UTF-8") as fp:
            json.dump(reports, fp, ensure_ascii=False, indent=4)
