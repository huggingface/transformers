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

import unittest
from unittest.mock import patch

from transformers.testing_utils import CaptureStd


class CLITest(unittest.TestCase):
    @patch("sys.argv", ["fakeprogrampath", "env"])
    def test_cli_env(self):
        # test transformers-cli env
        import transformers.commands.transformers_cli

        with CaptureStd() as cs:
            transformers.commands.transformers_cli.main()
        assert "Python version" in cs.out
        assert "Platform" in cs.out
        assert "Using distributed or parallel set-up in script?" in cs.out
