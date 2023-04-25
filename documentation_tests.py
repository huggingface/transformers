# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
    Utils to run the documentation tests without having to overwrite any files.s

    The doc precossing function can be run on a list of files and/org
    directories of files. It will recursively check if the files have
    a python code snippet by looking for a ```python or ```py syntax.
    The script will add a new line before every python code ending ``` line to make
    the docstrings ready for pytest doctests.
    However, we don't want to have empty lines displayed in the
    official documentation which is why the new code is written to a temporary directory / is tested on the fly depending on the configuration.

    When debugging the doc tests locally, the script should automatically determine which files should
    be processed based on the modified files. It should also run the tests on the fly and delete the
    temp file when finished.

    We will be using the doctest API:


    The question is: do we have to modify the files? 
    - doctest.testfile("./src/transformers/models/whisper/configuration_whisper.py",False,optionflags=doctest.NORMALIZE_WHITESPACE)
    or 
    - doctest.testfile( "whisper/configuration_whisper.py",package = "transformers.models",optionflags=doctest.NORMALIZE_WHITESPACE)
    
    - doctest.testmod(transformers.models.whisper.configuration_whisper)
    
    
    Another way of doing this is 
    
    ```python 
    def load_tests(loader, tests, ignore):
        tests.addTests(doctest.DocTestSuite(my_module_with_doctests))
        return tests
    ```
    - a DocTestFinder() will help us find documentation tests : doctest.DocTestFinder(verbose=False, parser=DocTestParser(), recurse=True, exclude_empty=True)
    - a DocTestParser() can always be passed. We need this in order to process only the part of the code where "python" is stated I guess.
    
    
    
    
    ```python
    >>>  import doctest
    >>> flags = doctest.REPORT_NDIFF|doctest.FAIL_FAST
    >>>  doctest.testfile(
        "roberta_prelayernorm/modeling_roberta_prelayernorm.py",
    package = "transformers.models",
    optionflags=flags)
    ```

    We will just have a pytest wrapper around it:
    ```python
    >>> def test_docstring():
    >>>     doctest_results = doctest.testfile(...)
    >>>     assert doctest_results.failed == 0
    ```
"""


import argparse
import os
import tempfile
import doctest


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

def process_doc_file(code_file, temp_dir = "temp", add_new_line=True):
    """
    Process given file.

    Args:
        code_file (`str` or `os.PathLike`):
            The file in which we want to style the docstring.
        temp_dir (`str` or `os.PathLike`):
            The temporary directory where we want to write to write the tests. Should probably default
            to the cache directory.
    """
    with open(code_file, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()

    # fmt: off
    splits = code.split("```")
    if len(splits) % 2 != 1:
        raise ValueError("The number of occurrences of ``` should be an even number.")

    splits = [s if i % 2 == 0 else maybe_append_new_line(s) for i, s in enumerate(splits)]
    clean_code = "```".join(splits)
    # fmt: on

    # write the code to a temporary file
    with tempfile.NamedTemporaryFile(code_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(clean_code)


def process_doc_files(*files,  temp_dir = "temp", add_new_line=True):
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
            process_doc_files(*files,  temp_dir = temp_dir, add_new_line=add_new_line)
        else:
            try:
                process_doc_file(file, temp_dir = temp_dir,  add_new_line=add_new_line)
            except Exception:
                print(f"There is a problem in {file}.")
                raise


def main(files_to_test_path, temp_dir="temp", add_new_line=True):
    flags = doctest.REPORT_NDIFF|doctest.FAIL_FAST
    with open(files_to_test_path, "r") as f:
        content = f.readlines()
        
    for file_name in content:
        print(f"Processing file: {file_name}")
        package_name = os.path.dirname(file_name).replace("/", ".")
        doctest.testfile("file_name",package = package_name, optionflags=flags)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", default="transformers/utils/documentation_tests.txt", type = str,help="The file(s) or folder(s) to run the doctests on.")
    args = parser.parse_args()

    main(args.files, temp_dir = "temp")
