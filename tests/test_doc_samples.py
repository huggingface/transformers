# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
import unittest
from typing import List, Union

from .utils import require_tf, require_torch, slow


def get_examples_from_file(file):
    examples = []
    example = []
    example_mode = False
    example_indentation = None
    for i, line in enumerate(file):
        if example_mode:
            current_indentation = len(line) - len(line.strip()) - 1

            # Check if the indentation is 0 for the example, so that we don't exit as soon as there's a line return.
            empty_line = example_indentation == 0 and len(line) == 1

            # If we're back to the example indentation or if it's the end of the docstring.
            if (current_indentation == example_indentation and not empty_line) or '"""' in line:
                # Exit the example mode and add the example to the examples list
                example_mode = False
                example_indentation = None
                examples.append(example)
                example = []
            else:
                # If line is not empty, add it to the current example
                if line != "\n":
                    example.append(line[example_indentation + 4 : -1])

        # Detect the example from '::' or 'example::'
        if "example::" in line.lower():
            example_mode = True
            example_indentation = line.lower().find("example::")
        elif "examples::" in line.lower():
            example_mode = True
            example_indentation = line.lower().find("examples::")
        # elif "::" in line.lower() and len(line.strip()) == 2:
        #     example_mode = True
        #     example_indentation = line.lower().find("::")

    examples = ["\n".join(example) for example in examples]
    examples = [example for example in examples if "not runnable" not in example.lower()]

    return examples


@require_torch
@require_tf
@slow
class TestCodeExamples(unittest.TestCase):
    def analyze_directory(
        self, directory: str, identifier: Union[str, None] = None, ignore_files: Union[List[str], None] = None
    ):
        files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

        if identifier is not None:
            files = [file for file in files if identifier in file]

        if ignore_files is not None:
            files = [file for file in files if file not in ignore_files]

        for file in files:
            # Open all files
            with open(os.path.join(directory, file)) as f:
                # Retrieve examples
                examples = get_examples_from_file(f)
                joined_examples = []

                def execute_example(code_example):
                    exec(code_example, {})

                # Some examples are the continuation of others.
                if len(examples) > 0:
                    joined_examples.append(examples[0])
                    joined_examples_index = 0
                    for example in examples[1:]:
                        # If they contain this line, then they're a continuation of the previous script
                        if "# Continuation of the previous script" in example:
                            joined_examples[joined_examples_index] += "\n" + example
                        # If not, create a new example and increment the index
                        else:
                            joined_examples.append(example)
                            joined_examples_index += 1

                print("Testing", file, str(len(joined_examples)) + "/" + str(len(joined_examples)))

                # Execute sub tests with every example.
                for index, code_example in enumerate(joined_examples):
                    with self.subTest(msg=file + " " + str(index) + "/" + str(len(joined_examples)) + code_example):
                        execute_example(code_example)

    def test_configuration_examples(self):
        transformers_directory = "src/transformers"
        configuration_files = "configuration"
        ignore_files = ["configuration_auto.py", "configuration_utils.py"]
        self.analyze_directory(transformers_directory, identifier=configuration_files, ignore_files=ignore_files)

    def test_main_doc_examples(self):
        doc_directory = "docs/source"
        self.analyze_directory(doc_directory)

    def test_modeling_examples(self):
        transformers_directory = "src/transformers"
        modeling_files = "modeling"
        ignore_files = [
            "modeling_auto.py",
            "modeling_t5.py",
            "modeling_tf_auto.py",
            "modeling_utils.py",
            "modeling_tf_t5.py",
        ]
        self.analyze_directory(transformers_directory, identifier=modeling_files, ignore_files=ignore_files)
