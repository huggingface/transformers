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

from transformers.testing_utils import CaptureStd, require_torch


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

    @require_torch
    @patch("sys.argv", ["fakeprogrampath", "download", "hf-internal-testing/tiny-random-gptj", "--cache-dir", "/tmp"])
    def test_cli_download(self):
        import transformers.commands.transformers_cli

        # # remove any previously downloaded model to start clean
        shutil.rmtree("/tmp/models--hf-internal-testing--tiny-random-gptj", ignore_errors=True)

        # run the command
        transformers.commands.transformers_cli.main()

        # check if the model files are downloaded correctly on /tmp/models--hf-internal-testing--tiny-random-gptj
        self.assertTrue(os.path.exists("/tmp/models--hf-internal-testing--tiny-random-gptj/blobs"))
        self.assertTrue(os.path.exists("/tmp/models--hf-internal-testing--tiny-random-gptj/refs"))
        self.assertTrue(os.path.exists("/tmp/models--hf-internal-testing--tiny-random-gptj/snapshots"))

    @require_torch
    @patch(
        "sys.argv",
        [
            "fakeprogrampath",
            "download",
            "hf-internal-testing/test_dynamic_model_with_tokenizer",
            "--trust-remote-code",
            "--cache-dir",
            "/tmp",
        ],
    )
    def test_cli_download_trust_remote(self):
        import transformers.commands.transformers_cli

        # # remove any previously downloaded model to start clean
        shutil.rmtree("/tmp/models--hf-internal-testing--test_dynamic_model_with_tokenizer", ignore_errors=True)

        # run the command
        transformers.commands.transformers_cli.main()

        # check if the model files are downloaded correctly on /tmp/models--hf-internal-testing--test_dynamic_model_with_tokenizer
        self.assertTrue(os.path.exists("/tmp/models--hf-internal-testing--test_dynamic_model_with_tokenizer/blobs"))
        self.assertTrue(os.path.exists("/tmp/models--hf-internal-testing--test_dynamic_model_with_tokenizer/refs"))
        self.assertTrue(
            os.path.exists("/tmp/models--hf-internal-testing--test_dynamic_model_with_tokenizer/snapshots")
        )
