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
"""
from typing import Iterable
from pytest import DoctestItem
from _pytest.doctest import DoctestModule, Module, _is_mocked, _patch_unwrap_mock_aware, get_optionflags, _get_runner, _get_continue_on_failure, _get_checker, import_path, _is_main_py, _is_setup_py, _is_doctest
import inspect
from _pytest.outcomes import skip

from pathlib import Path
from typing import Optional, Union

from _pytest.doctest import Collector, DoctestModule, DoctestTextfile


import unittest
import argparse
import doctest
import os
import re
from transformers.testing_utils import parse_flag_from_env



def replace_ignore_result(string):
    """Replace all instances of '# doctest: +IGNORE_RESULT' with '# doctest: +SKIP'."""
    return re.sub(r'#\s*doctest:\s*\+IGNORE_RESULT', '# doctest: +SKIP', string)


         
class HfDocTestParser(doctest.DocTestParser):
    # This regular expression is used to find doctest examples in a
    # string.  It defines three groups: `source` is the source code
    # (including leading indentation and prompts); `indent` is the
    # indentation of the first (PS1) line of the source code; and
    # `want` is the expected output (including leading indentation).
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
             (?![ ]*>>>)  # Not a line starting with PS1
             (?:(?!```).)*  # Match any character except '`' until a '```' is found (this is specific to HF because black removes the last line)
             (?:\n|$)  # Match a new line or end of string
          )*)
        ''', re.MULTILINE | re.VERBOSE)
    
    def parse(self, string, name='<string>'):
        processed_text = replace_ignore_result(string)
        return super().parse(processed_text)



   
# DoctestModule
class HfDoctestModule(Module):
    def collect(self) -> Iterable[DoctestItem]:
        
        
        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A hackish doctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456
            https://bugs.python.org/issue25532
            """

            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct
                line number is returned. This will be reported upstream. #8796
                """
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)

                if hasattr(obj, "__wrapped__"):
                    # Get the main obj in case of it being wrapped
                    obj = inspect.unwrap(obj)

                # Type ignored because this is a private function.
                return super()._find_lineno(  # type:ignore[misc]
                    obj,
                    source_lines,
                )

            def _find(
                self, tests, obj, name, module, source_lines, globs, seen
            ) -> None:
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    # Type ignored because this is a private function.
                    super()._find(  # type:ignore[misc]
                        tests, obj, name, module, source_lines, globs, seen
                    )
                    
        if self.path.name == "conftest.py":
            module = self.config.pluginmanager._importconftest(
                self.path,
                self.config.getoption("importmode"),
                rootpath=self.config.rootpath,
            )
        else:
            try:
                module = import_path(
                    self.path,
                    root=self.config.rootpath,
                    mode=self.config.getoption("importmode"),
                )
            except ImportError:
                if self.config.getvalue("doctest_ignore_import_errors"):
                    skip("unable to import module %r" % self.path)
                else:
                    raise
        # Uses internal doctest module parsing mechanism.
        finder = MockAwareDocTestFinder(parser = HfDocTestParser())
        optionflags = get_optionflags(self)
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )

        for test in finder.find(module, module.__name__):
            if test.examples:  # skip empty doctests
                yield DoctestItem.from_parent(
                    self, name=test.name, runner=runner, dtest=test
                )

    
def pytest_collect_file(
    file_path: Path,
    parent: Collector,
) -> Optional[Union["DoctestModule", "DoctestTextfile"]]:
    config = parent.config
    if file_path.suffix == ".py":
        if config.option.doctestmodules and not any(
            (_is_setup_py(file_path), _is_main_py(file_path))
        ):
            mod: DoctestModule = DoctestModule.from_parent(parent, path=file_path)
            return mod
    elif _is_doctest(config, file_path, parent):
        txt: DoctestTextfile = DoctestTextfile.from_parent(parent, path=file_path)
        return txt
    return None





def get_tests_to_run(files_to_test_path):
    """
    Util to run test if the file is called
    """
    flags = doctest.REPORT_NDIFF 
    parser = HfDocTestParser()
    with open(files_to_test_path, "r") as f:
        content = f.read().splitlines()
    test_cases = []    
    for file_name in content:
        if os.path.isdir(file_name):
            continue
        with open(file_name, "r") as f:
            text = f.read()
        test_to_run = parser.get_doctest(text, {}, file_name, None, None)
        test_cases.append(doctest.DocFileCase(test_to_run, optionflags=flags))
    return test_cases



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", action = "store", default=None, type = str, help="The file(s) or folder(s) to run the doctests on.")
    args = parser.parse_args()
    tests = get_tests_to_run(args.files, temp_dir = "temp")
    runner = doctest.DocTestRunner()
    runner.run(tests)



