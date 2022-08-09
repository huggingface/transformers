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
import shutil
import unittest
from unittest.mock import patch

from transformers.testing_utils import CaptureStd, is_pt_tf_cross_test


class CLITest(unittest.TestCase):
    @patch("sys.argv", ["fakeprogrampath", "env"])
    def test_cli_env(self):
        # test transformers-cli env
        import transformers.commands.transformers_cli

        with CaptureStd() as cs:
            transformers.commands.transformers_cli.main()
        self.assertIn("Python version", cs.out)
        self.assertIn("Platform", cs.out)
        self.assertIn("Using distributed or parallel set-up in script?", cs.out)

    @is_pt_tf_cross_test
    @patch(
        "sys.argv", ["fakeprogrampath", "pt-to-tf", "--model-name", "hf-internal-testing/tiny-random-gptj", "--no-pr"]
    )
    def test_cli_pt_to_tf(self):
        import transformers.commands.transformers_cli

        shutil.rmtree("/tmp/hf-internal-testing/tiny-random-gptj", ignore_errors=True)  # cleans potential past runs
        transformers.commands.transformers_cli.main()

        # The original repo has no TF weights -- if they exist, they were created by the CLI
        self.assertTrue(os.path.exists("/tmp/hf-internal-testing/tiny-random-gptj/tf_model.h5"))
