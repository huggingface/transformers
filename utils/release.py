# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import os
import re

import git
import packaging.version


PATH_TO_EXAMPLES = "examples/"
REPLACE_PATTERNS = {
    "examples": (re.compile(r'^check_min_version\("[^"]+"\)\s*$', re.MULTILINE), 'check_min_version("VERSION")\n'),
    "init": (re.compile(r'^__version__\s+=\s+"([^"]+)"\s*$', re.MULTILINE), '__version__ = "VERSION"\n'),
    "setup": (re.compile(r'^(\s*)version\s*=\s*"[^"]+",', re.MULTILINE), r'\1version="VERSION",'),
    "doc": (re.compile(r'^(\s*)release\s*=\s*"[^"]+"$', re.MULTILINE), 'release = "VERSION"\n'),
}
REPLACE_FILES = {
    "init": "src/transformers/__init__.py",
    "setup": "setup.py",
}
README_FILE = "README.md"
CUSTOM_JS_FILE = "docs/source/_static/js/custom.js"
DEPLOY_SH_FILE = ".circleci/deploy.sh"


def update_version_in_file(fname, version, pattern):
    """Update the version in one file using a specific pattern."""
    with open(fname, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()
    re_pattern, replace = REPLACE_PATTERNS[pattern]
    replace = replace.replace("VERSION", version)
    code = re_pattern.sub(replace, code)
    with open(fname, "w", encoding="utf-8", newline="\n") as f:
        f.write(code)


def update_version_in_examples(version):
    """Update the version in all examples files."""
    for folder, directories, fnames in os.walk(PATH_TO_EXAMPLES):
        # Removing some of the folders with non-actively maintained examples from the walk
        if "research_projects" in directories:
            directories.remove("research_projects")
        if "legacy" in directories:
            directories.remove("legacy")
        for fname in fnames:
            if fname.endswith(".py"):
                update_version_in_file(os.path.join(folder, fname), version, pattern="examples")


def global_version_update(version, patch=False):
    """Update the version in all needed files."""
    for pattern, fname in REPLACE_FILES.items():
        update_version_in_file(fname, version, pattern)
    if not patch:
        update_version_in_examples(version)


def clean_master_ref_in_model_list():
    """Replace the links from master doc tp stable doc in the model list of the README."""
    # If the introduction or the conclusion of the list change, the prompts may need to be updated.
    _start_prompt = "🤗 Transformers currently provides the following architectures"
    _end_prompt = "1. Want to contribute a new model?"
    with open(README_FILE, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Find the start of the list.
    start_index = 0
    while not lines[start_index].startswith(_start_prompt):
        start_index += 1
    start_index += 1

    index = start_index
    # Update the lines in the model list.
    while not lines[index].startswith(_end_prompt):
        if lines[index].startswith("1."):
            lines[index] = lines[index].replace(
                "https://huggingface.co/transformers/master/model_doc",
                "https://huggingface.co/transformers/model_doc",
            )
        index += 1

    with open(README_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


def get_version():
    """Reads the current version in the __init__."""
    with open(REPLACE_FILES["init"], "r") as f:
        code = f.read()
    default_version = REPLACE_PATTERNS["init"][0].search(code).groups()[0]
    return packaging.version.parse(default_version)


def pre_release_work(patch=False):
    """Do all the necessary pre-release steps."""
    # First let's get the default version: base version if we are in dev, bump minor otherwise.
    default_version = get_version()
    if patch and default_version.is_devrelease:
        raise ValueError("Can't create a patch version from the dev branch, checkout a released version!")
    if default_version.is_devrelease:
        default_version = default_version.base_version
    elif patch:
        default_version = f"{default_version.major}.{default_version.minor}.{default_version.micro + 1}"
    else:
        default_version = f"{default_version.major}.{default_version.minor + 1}.0"

    # Now let's ask nicely if that's the right one.
    version = input(f"Which version are you releasing? [{default_version}]")
    if len(version) == 0:
        version = default_version

    print(f"Updating version to {version}.")
    global_version_update(version, patch=patch)
    if not patch:
        print("Cleaning main README")
        clean_master_ref_in_model_list()


def post_release_work():
    """Do all the necesarry post-release steps."""
    # First let's get the current version
    current_version = get_version()
    dev_version = f"{current_version.major}.{current_version.minor + 1}.0.dev0"
    current_version = current_version.base_version
    # Get the current commit hash
    repo = git.Repo(".", search_parent_directories=True)
    version_commit = repo.head.object.hexsha[:7]

    # Check with the user we got that right.
    version = input(f"Which version are we developing now? [{dev_version}]")
    commit = input(f"Commit hash to associate to v{current_version}? [{version_commit}]")
    if len(version) == 0:
        version = dev_version
    if len(commit) == 0:
        commit = version_commit

    print(f"Updating version to {version}.")
    global_version_update(version)


def post_patch_work():
    """Do all the necesarry post-patch steps."""
    # Try to guess the right info: last patch in the minor release before current version and its commit hash.
    current_version = get_version()
    repo = git.Repo(".", search_parent_directories=True)
    repo_tags = repo.tags
    default_version = None
    version_commit = None
    for tag in repo_tags:
        if str(tag).startswith(f"v{current_version.major}.{current_version.minor - 1}"):
            if default_version is None:
                default_version = packaging.version.parse(str(tag)[1:])
                version_commit = str(tag.commit)[:7]
            elif packaging.version.parse(str(tag)[1:]) > default_version:
                default_version = packaging.version.parse(str(tag)[1:])
                version_commit = str(tag.commit)[:7]

    # Confirm with the user or ask for the info if not found.
    if default_version is None:
        version = input("Which patch version was just released?")
        commit = input("Commit hash to associated to it?")
    else:
        version = input(f"Which patch version was just released? [{default_version}]")
        commit = input(f"Commit hash to associated to it? [{version_commit}]")
        if len(version) == 0:
            version = default_version
        if len(commit) == 0:
            commit = version_commit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post_release", action="store_true", help="Whether this is pre or post release.")
    parser.add_argument("--patch", action="store_true", help="Whether or not this is a patch release.")
    args = parser.parse_args()
    if not args.post_release:
        pre_release_work(patch=args.patch)
    elif args.patch:
        post_patch_work()
    else:
        post_release_work()
