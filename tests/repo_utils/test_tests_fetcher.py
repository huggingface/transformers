# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import sys
import unittest

from git import Repo


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

transformers_path = os.path.join(git_repo_path, "src", "transformers")
# Tests are run against this specific commit for reproducibility
# https://github.com/huggingface/transformers/tree/07f6690206e39ed7a4d9dbc58824314f7089bb38
GIT_TEST_SHA = "07f6690206e39ed7a4d9dbc58824314f7089bb38"

from tests_fetcher import checkout_commit, clean_code, get_module_dependencies  # noqa: E402


class CheckDummiesTester(unittest.TestCase):
    def test_clean_code(self):
        # Clean code removes all strings in triple quotes
        self.assertEqual(clean_code('"""\nDocstring\n"""\ncode\n"""Long string"""\ncode\n'), "code\ncode")
        self.assertEqual(clean_code("'''\nDocstring\n'''\ncode\n'''Long string'''\ncode\n'''"), "code\ncode")

        # Clean code removes all comments
        self.assertEqual(clean_code("code\n# Comment\ncode"), "code\ncode")
        self.assertEqual(clean_code("code  # inline comment\ncode"), "code  \ncode")

    def test_checkout_commit(self):
        repo = Repo(git_repo_path)
        self.assertNotEqual(repo.head.commit.hexsha, GIT_TEST_SHA)
        with checkout_commit(repo, GIT_TEST_SHA):
            self.assertEqual(repo.head.commit.hexsha, GIT_TEST_SHA)
        self.assertNotEqual(repo.head.commit.hexsha, GIT_TEST_SHA)

    def test_get_module_dependencies(self):
        bert_module = os.path.join(transformers_path, "models", "bert", "modeling_bert.py")
        expected_deps = [
            "activations.py",
            "modeling_outputs.py",
            "modeling_utils.py",
            "pytorch_utils.py",
            "models/bert/configuration_bert.py",
        ]
        expected_deps = set(os.path.join(transformers_path, f) for f in expected_deps)
        repo = Repo(git_repo_path)
        with checkout_commit(repo, GIT_TEST_SHA):
            deps = get_module_dependencies(bert_module)
        deps = set(os.path.expanduser(f) for f in deps)
        self.assertEqual(deps, expected_deps)
