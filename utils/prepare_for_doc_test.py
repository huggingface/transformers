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
"""Style utils to preprocess files for doc tests."""

import argparse
import os


def maybe_append_new_line(docstring):
    """
    Append new line if code snippet is a
    Python code snippet
    """
    lines = docstring.split("\n")

    if lines[0] in ["py", "python"]:
        # add a "\n" before last line
        last_line = lines[-1]
        lines.pop()
        lines.append("\n" + last_line)

    return "\n".join(lines)


def process_doc_file(code_file):
    """
    Process given file.

    Args:
        code_file (`str` or `os.PathLike`): The file in which we want to style the docstring.
    """
    with open(code_file, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()

    # fmt: off
    splits = code.split("```")
    splits = [s if i % 2 == 0 else maybe_append_new_line(s) for i, s in enumerate(splits)]
    clean_code = "```".join(splits)
    # fmt: on

    diff = clean_code != code
    if diff:
        print(f"Overwriting content of {code_file}.")
        with open(code_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean_code)


def process_doc_files(*files):
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
            process_doc_files(*files)
        else:
            try:
                process_doc_file(file)
            except Exception:
                print(f"There is a problem in {file}.")
                raise


def main(*files):
    process_doc_files(*files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="The file(s) or folder(s) to restyle.")
    args = parser.parse_args()

    main(*args.files)
