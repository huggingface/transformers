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
from doctest import DocTestSuite
from typing import List, Union

import transformers

from .utils import require_tf, require_torch, slow


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
            print("Testing", file)
            module = file.split(".")[0]
            suite = DocTestSuite(getattr(transformers, module))
            result = unittest.TextTestRunner().run(suite)
            self.assertIs(len(result.failures), 0)

    def test_modeling_examples(self):
        transformers_directory = "src/transformers"
        files = "modeling"
        ignore_files = [
            "modeling_ctrl.py",
            "modeling_tf_ctrl.py",
        ]
        self.analyze_directory(transformers_directory, identifier=files, ignore_files=ignore_files)

    def test_tokenization_examples(self):
        transformers_directory = "src/transformers"
        files = "tokenization"
        ignore_files = [
        ]
        self.analyze_directory(transformers_directory, identifier=files, ignore_files=ignore_files)
