# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
""" Style utils to preprocess files for doc tests.

    The doc precossing function can be run on a list of files and/org
    directories of files. It will recursively check if the files have
    a python code snippet by looking for a ```python or ```py syntax.
    In the default mode - `remove_new_line==False` the script will
    add a new line before every python code ending ``` line to make
    the docstrings ready for pytest doctests.
    However, we don't want to have empty lines displayed in the
    official documentation which is why the new line command can be
    reversed by adding the flag `--remove_new_line` which sets
    `remove_new_line==True`.

    When debugging the doc tests locally, please make sure to
    always run:

    ```python utils/prepare_for_doc_test.py src docs```

    before running the doc tests:

    ```pytest --doctest-modules $(cat utils/documentation_tests.txt) -sv --doctest-continue-on-failure --doctest-glob="*.mdx"```

    Afterwards you should revert the changes by running

    ```python utils/prepare_for_doc_test.py src docs --remove_new_line```
"""

import argparse
import os


def process_code_block(code, add_new_line=True):
    if add_new_line:
        return maybe_append_new_line(code)
    else:
        return maybe_remove_new_line(code)


def maybe_append_new_line(code):
    """
    Append new line if code snippet is a
    Python code snippet
    """
    lines = code.split("\n")

    if lines[0] in ["py", "python"]:
        # add new line before last line being ```
        last_line = lines[-1]
        lines.pop()
        lines.append("\n" + last_line)

    return "\n".join(lines)


def maybe_remove_new_line(code):
    """
    Remove new line if code snippet is a
    Python code snippet
    """
    lines = code.split("\n")

    if lines[0] in ["py", "python"]:
        # add new line before last line being ```
        lines = lines[:-2] + lines[-1:]

    return "\n".join(lines)


def process_doc_file(code_file, add_new_line=True):
    """
    Process given file.

    Args:
        code_file (`str` or `os.PathLike`): The file in which we want to style the docstring.
    """
    with open(code_file, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()

    # fmt: off
    splits = code.split("```")
    if len(splits) % 2 != 1:
        raise ValueError("The number of occurrences of ``` should be an even number.")

    splits = [s if i % 2 == 0 else process_code_block(s, add_new_line=add_new_line) for i, s in enumerate(splits)]
    clean_code = "```".join(splits)
    # fmt: on

    diff = clean_code != code
    if diff:
        print(f"Overwriting content of {code_file}.")
        with open(code_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean_code)


def process_doc_files(*files, add_new_line=True):
    """
    Applies doc styling or checks everything is correct in a list of files.

    Args:
        files (several `str` or `os.PathLike`): The files to treat.
            Whether to restyle file or just check if they should be restyled.

    Returns:
        List[`str`]: The list of files changed or that should be restyled.
    """
    for file in files:
        # Treat folders
        if os.path.isdir(file):
            files = [os.path.join(file, f) for f in os.listdir(file)]
            files = [f for f in files if os.path.isdir(f) or f.endswith(".mdx") or f.endswith(".py")]
            process_doc_files(*files, add_new_line=add_new_line)
        else:
            try:
                process_doc_file(file, add_new_line=add_new_line)
            except Exception:
                print(f"There is a problem in {file}.")
                raise


def main(*files, add_new_line=True):
    process_doc_files(*files, add_new_line=add_new_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="The file(s) or folder(s) to restyle.")
    parser.add_argument(
        "--remove_new_line",
        action="store_true",
        help="Whether to remove new line after each python code block instead of adding one.",
    )
    args = parser.parse_args()

    main(*args.files, add_new_line=not args.remove_new_line)
