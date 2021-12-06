# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Checking utils for the docstrings."""

import argparse
import json
import os
from style_doc import _re_doc_ignore, _re_example, get_indent
import subprocess
import tempfile
import warnings


def _check_docstring_example_blocks(text, file=None):
    """Extract code example blocks in a docstring and check if they work"""
    lines = text.split("\n")
    idx = 0

    results = {}

    while idx < len(lines):
        # Detect if the line is the start of a new code-block.
        if _re_example.search(lines[idx]) is not None:  # or _re_code_block_explicit.search(lines[idx]) is not None:
            while len(get_indent(lines[idx])) == 0:
                idx += 1
            start_idx = idx
            start_indent = len(get_indent(lines[start_idx]))
            should_continue = True
            while should_continue:
                idx += 1
                # The condition `len(lines[idx].strip()) == 0` is to keep the lines containing only whitespaces.
                should_continue = (idx < len(lines)) and (len(lines[idx].strip()) == 0 or len(get_indent(lines[idx])) > start_indent)
            end_idx = idx
            code_block_lines = lines[start_idx:end_idx]

            # save un-processed code example block for later use
            orig_code_block = "\n".join(code_block_lines)

            # remove the 1st line (which is the line containing `::`)
            code_block_lines = code_block_lines[1:]

            # remove output lines
            code_block_lines = [x for x in code_block_lines if len(x.strip()) == 0 or x.lstrip().startswith(">>>") or x.lstrip().startswith("...")]

            # remove the 1st line if it is empty
            if code_block_lines and not code_block_lines[0].strip():
                code_block_lines = code_block_lines[1:]

            # check ">>>" and "..." formats
            for x in code_block_lines:
                if ">>>" in x:
                    assert x.strip().startswith(">>> ")
                elif "..." in x:
                    assert x.strip().startswith("... ") or x.lstrip() == "..."  # some possibly empty lines (in LED)

            # remove ">>>" and "..."
            code_block_lines = [x.strip().replace(">>> ", "").replace("... ", "") for x in code_block_lines]

            # deal with lines being "..."
            code_block_lines = [x if x != "..." else "" for x in code_block_lines]

            # put together into a code block
            code_block = "\n".join(code_block_lines)

            # run the code example and capture the error if any
            if len(code_block.strip()) > 0:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    with open(os.path.join(tmpdirname, "tmp.py"), "w", encoding="UTF-8") as fp:
                        fp.write(code_block)
                    result = subprocess.run(f'python {os.path.join(tmpdirname, "tmp.py")}', shell=True, capture_output=True)
                    if result.stderr:
                        error = result.stderr.decode("utf-8").replace("\r\n", "\n")
                        results[orig_code_block] = error
        else:
            idx += 1

    return results


def check_docstring(docstring, file=None):
    """Check code examples in `docstring` work"""
    # Make sure code examples in a docstring work
    return _check_docstring_example_blocks(docstring, file=file)


def check_file_docstrings(code_file):
    """Check code examples in all docstrings in `code_file` work"""
    with open(code_file, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()

    results = {}

    splits = code.split('\"\"\"')
    for i, s in enumerate(splits):
        if i % 2 == 0 or _re_doc_ignore.search(splits[i - 1]) is not None:
            continue
        results.update(check_docstring(s, file=code_file))

    return results


def check_doc_files(*files):
    """Checks docstrings in all `files` and raises an error if there is any example which can't run.
    """
    results = {}
    for file in files:
        # Treat folders
        if os.path.isdir(file):
            files = [os.path.join(file, f) for f in os.listdir(file)]
            files = [f for f in files if os.path.isdir(f) or f.endswith(".py")]
            results.update(check_doc_files(*files))
        elif file.endswith(".py"):
            _results = check_file_docstrings(file)
            if len(_results) > 0:
                if file not in results:
                    results[file] = {}
                results[file].update(_results)
        else:
            warnings.warn(f"Ignoring {file} because it's not a py file or a folder.")

        if file in results:
            with open(file.replace("/", "-").replace("\\", "-").replace(".py", ".json"), "w", encoding="UTF-8") as fp:
                json.dump(results[file], fp, ensure_ascii=False, indent=4)

    with open("results.json", "w", encoding="UTF-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

    convert_json(results)

    return results


def main(*files):
    results = check_doc_files(*files)
    if len(results) > 0:
        n_examples = sum(len(v) for v in results.values())
        raise ValueError(f"{n_examples} docstring examples in {len(results)} .py files should be fixed!")


def convert_json(json_report):

    with open("report.txt", "w", encoding="UTF-8") as fp:
        for file_path in json_report:
            fp.write(file_path + "\n")
            fp.write("=" * len(file_path) + "\n")
            for docstring, error in json_report[file_path].items():
                fp.write("\n")
                fp.write(docstring)
                fp.write("\n")
                indent = get_indent(docstring)
                fp.write(indent + "Errors:\n\n")
                error = "\n".join([" " * 4 + indent + x for x in error.split("\n")])
                fp.write(error + "\n")
                fp.write("-" * len(file_path) + "\n")
            fp.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="The file(s) or folder(s) to check.")
    args = parser.parse_args()

    main(*args.files)
