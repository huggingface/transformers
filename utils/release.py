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

import os
import packaging.version
import re


PATH_TO_EXAMPLES = "examples/"
REPLACE_PATTERNS = {
    "examples": (re.compile(r'^check_min_version\("[^"]+"\)\s*$', re.MULTILINE), 'check_min_version("VERSION")\n'),
    "init": (re.compile(r'^__version__\s+=\s+"([^"]+)"\s*$', re.MULTILINE), '__version__ = "VERSION"\n'),
    "setup": (re.compile(r'^(\s*)version\s*=\s*"[^"]+",', re.MULTILINE), r'\1version="VERSION",'),
    "doc": (re.compile(r"^(\s*)release\s*=\s*u'[^']+'$", re.MULTILINE), "release = u'VERSION'\n"),
}
REPLACE_FILES = {
    "init": "src/transformers/__init__.py",
    "setup": "setup.py",
    "doc": "docs/source/conf.py",
}
README_FILE = "README.md"


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


def global_version_update(version):
    """Update the version in all needed files."""
    for pattern, fname in REPLACE_FILES.items():
        update_version_in_file(fname, version, pattern)
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

    result = []
    current_line = ""
    index = start_index
    
    # Update the lines in the model list.
    while not linesindex].startswith(_end_prompt):
        if lines[index].startswith("1."):
            lines[index] = lines[index].replace(
                "https://huggingface.co/transformers/master/model_doc",
                "https://huggingface.co/transformers/model_doc",
            )
        index += 1
    
    with open(README_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


def _find_text_in_file(filename, start_prompt, end_prompt):
    """
    Find the text in `filename` between a line beginning with `start_prompt` and before `end_prompt`, removing empty
    lines.
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    # Find the start prompt.
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1

    end_index = start_index
    while not lines[end_index].startswith(end_prompt):
        end_index += 1
    end_index -= 1

    while len(lines[start_index]) <= 1:
        start_index += 1
    while len(lines[end_index]) <= 1:
        end_index -= 1
    end_index += 1
    return "".join(lines[start_index:end_index]), start_index, end_index, lines


def pre_release_work():
    """Do all the necessary pre-release steps."""
    # First lest grap the current version in the init.
    with open(REPLACE_FILES["init"], "r") as f:
        code = f.read()
    default_version = REPLACE_PATTERNS["init"][0].search(code).groups()[0]
    default_version = packaging.version.parse(default_version)
    if default_version.is_devrelease:
        default_version = default_version.base_version
    else:
        default_version = f"{default_version.major}.{default_version.minor+1}.0"

    # Now let's ask nicely if that's the right one
    version = input(f"Which version are you releasing? [{default_version}]")
    if len(version) == 0:
        version = default_version

    print(f"Updating version to {version}.")
    # global_version_update(version)
    print("Cleaning main README")
    clean_master_ref_in_model_list()


if __name__ == "__main__":
    pre_release_work()
