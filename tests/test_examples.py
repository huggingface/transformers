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


def get_examples_from_file(file):
    examples = []
    example = []
    example_mode = False
    example_indentation = None
    for i, line in enumerate(file):
        if example_mode:
            current_indentation = len(line) - len(line.strip()) - 1
            if current_indentation == example_indentation or '"""' in line:
                example_mode = False
                example_indentation = None
                examples.append(example)
                example = []
            else:
                if line is not "\n":
                    example.append(line[example_indentation + 4 : -1])
        if "example::" in line.lower():
            example_mode = True
            example_indentation = line.lower().find("example::")

    return ['\n'.join(example) for example in examples]


class TestCodeExamples(unittest.TestCase):
    def test_configuration_examples(self):
        transformers_directory = "../src/transformers"
        configuration_files = [file for file in os.listdir(transformers_directory) if "configuration" in file]

        for configuration_file in configuration_files:
            with open(os.path.join(transformers_directory, configuration_file)) as f:
                examples = get_examples_from_file(f)
                print("Testing", configuration_file, str(len(examples)) + "/" + str(len(examples)))

                def execute_example(code_example):
                    exec(code_example)

                with self.subTest(msg=configuration_file):
                    [execute_example(code_example) for code_example in examples]
