# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
import glob
import os
import re
import tempfile


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_copies.py
TRANSFORMERS_PATH = "src/transformers"


def find_code_in_transformers(object_name):
    """ Find and return the code source code of `object_name`."""
    parts = object_name.split(".")
    i = 0

    # First let's find the module where our object lives.
    module = parts[i]
    while i < len(parts) and not os.path.isfile(os.path.join(TRANSFORMERS_PATH, f"{module}.py")):
        i += 1
        module = os.path.join(module, parts[i])
    if i >= len(parts):
        raise ValueError(
            f"`object_name` should begin with the name of a module of transformers but got {object_name}."
        )

    with open(os.path.join(TRANSFORMERS_PATH, f"{module}.py"), "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Now let's find the class / func in the code!
    indent = ""
    line_index = 0
    for name in parts[i + 1 :]:
        while line_index < len(lines) and re.search(f"^{indent}(class|def)\s+{name}", lines[line_index]) is None:
            line_index += 1
        indent += "    "
        line_index += 1

    if line_index >= len(lines):
        raise ValueError(f" {object_name} does not match any function or class in {module}.")

    # We found the beginning of the class / func, now let's find the end (when the indent diminishes).
    start_index = line_index
    while line_index < len(lines) and (lines[line_index].startswith(indent) or len(lines[line_index]) <= 1):
        line_index += 1
    # Clean up empty lines at the end (if any).
    while len(lines[line_index - 1]) <= 1:
        line_index -= 1

    code_lines = lines[start_index:line_index]
    return "".join(code_lines)


_re_copy_warning = re.compile(r"^(\s*)#\s*Copied from\s+transformers\.(\S+\.\S+)\s*($|\S.*$)")
_re_replace_pattern = re.compile(r"with\s+(\S+)->(\S+)(?:\s|$)")


def blackify(code):
    """
    Applies the black part of our `make style` command to `code`.
    """
    has_indent = code.startswith("    ")
    if has_indent:
        code = f"class Bla:\n{code}"
    with tempfile.TemporaryDirectory() as d:
        fname = os.path.join(d, "tmp.py")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(code)
        os.system(f"black -q --line-length 119 --target-version py35 {fname}")
        with open(fname, "r", encoding="utf-8") as f:
            result = f.read()
            return result[len("class Bla:\n") :] if has_indent else result


def is_copy_consistent(filename, overwrite=False):
    """
    Check if the code commented as a copy in `filename` matches the original.

    Return the differences or overwrites the content depending on `overwrite`.
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    found_diff = False
    line_index = 0
    # Not a foor loop cause `lines` is going to change (if `overwrite=True`).
    while line_index < len(lines):
        search = _re_copy_warning.search(lines[line_index])
        if search is None:
            line_index += 1
            continue

        # There is some copied code here, let's retrieve the original.
        indent, object_name, replace_pattern = search.groups()
        theoretical_code = find_code_in_transformers(object_name)
        theoretical_indent = re.search(r"^(\s*)\S", theoretical_code).groups()[0]

        start_index = line_index + 1 if indent == theoretical_indent else line_index + 2
        indent = theoretical_indent
        line_index = start_index

        # Loop to check the observed code, stop when indentation diminishes or if we see a End copy comment.
        should_continue = True
        while line_index < len(lines) and should_continue:
            line_index += 1
            if line_index >= len(lines):
                break
            line = lines[line_index]
            should_continue = (len(line) <= 1 or line.startswith(indent)) and re.search(
                f"^{indent}# End copy", line
            ) is None
        # Clean up empty lines at the end (if any).
        while len(lines[line_index - 1]) <= 1:
            line_index -= 1

        observed_code_lines = lines[start_index:line_index]
        observed_code = "".join(observed_code_lines)

        # Before comparing, use the `replace_pattern` on the original code.
        if len(replace_pattern) > 0:
            search_patterns = _re_replace_pattern.search(replace_pattern)
            if search_patterns is not None:
                obj1, obj2 = search_patterns.groups()
                theoretical_code = re.sub(obj1, obj2, theoretical_code)

        # Test for a diff and act accordingly.
        if observed_code != theoretical_code:
            found_diff = True
            if overwrite:
                lines = lines[:start_index] + [theoretical_code] + lines[line_index:]
                line_index = start_index + 1

    if overwrite and found_diff:
        # Warn the user a file has been modified.
        print(f"Detected changes, rewriting {filename}.")
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return not found_diff


def check_copies(overwrite: bool = False):
    all_files = glob.glob(os.path.join(TRANSFORMERS_PATH, "**/*.py"), recursive=True)
    diffs = []
    for filename in all_files:
        consistent = is_copy_consistent(filename, overwrite)
        if not consistent:
            diffs.append(filename)
    if not overwrite and len(diffs) > 0:
        diff = "\n".join(diffs)
        raise Exception(
            "Found copy inconsistencies in the following files:\n"
            + diff
            + "\nRun `make fix-copies` or `python utils/check_copies --fix_and_overwrite` to fix them."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_copies(args.fix_and_overwrite)
