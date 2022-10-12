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


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

from tests_fetcher import clean_code  # noqa: E402


class CheckDummiesTester(unittest.TestCase):
    def test_clean_code(self):
        # Clean code removes all strings in triple quotes
        self.assertEqual(clean_code('"""\nDocstring\n"""\ncode\n"""Long string"""\ncode\n'), "code\ncode")
        self.assertEqual(clean_code("'''\nDocstring\n'''\ncode\n'''Long string'''\ncode\n'''"), "code\ncode")

        # Clean code removes all comments
        self.assertEqual(clean_code("code\n# Comment\ncode"), "code\ncode")
        self.assertEqual(clean_code("code  # inline comment\ncode"), "code  \ncode")
