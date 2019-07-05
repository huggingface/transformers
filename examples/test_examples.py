# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
import argparse

try:
    # python 3.4+ can use builtin unittest.mock instead of mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch

import run_bert_squad as rbs

def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    args = parser.parse_args()
    return args.f

class ExamplesTests(unittest.TestCase):

    def test_run_squad(self):
        testargs = ["prog", "-f", "/home/test/setup.py"]
        with patch.object(sys, 'argv', testargs):
            setup = get_setup_file()
            assert setup == "/home/test/setup.py"
            # rbs.main()


if __name__ == "__main__":
    unittest.main()
