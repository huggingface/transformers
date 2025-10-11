# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest
from unittest.mock import patch

import transformers.commands.transformers_cli as cli
from transformers.commands.chat import ChatArguments, ChatCommand
from transformers.testing_utils import CaptureStd


class ChatCLITest(unittest.TestCase):
    def test_help(self):
        with patch("sys.argv", ["transformers", "chat", "--help"]), CaptureStd() as cs:
            with self.assertRaises(SystemExit):
                cli.main()
        self.assertIn("chat interface", cs.out.lower())

    @patch.object(ChatCommand, "run")
    def test_cli_dispatch_model(self, run_mock):
        """
        Running transformers chat with just a model should work & spawn a serve underneath
        """
        args = ["transformers", "chat", "hf-internal-testing/tiny-random-gpt2"]
        with patch("sys.argv", args):
            cli.main()
        run_mock.assert_called_once()

    def test_cli_dispatch_url(self):
        """
        Running transformers chat with just a URL should not work as a model should additionally be specified
        """
        args = ["transformers", "chat", "localhost:8000"]
        with self.assertRaises(ValueError):
            with patch("sys.argv", args):
                cli.main()

    @patch.object(ChatCommand, "run")
    def test_cli_dispatch_url_and_model(self, run_mock):
        """
        Running transformers chat with a URL and a model should work
        """
        args = ["transformers", "chat", "localhost:8000", "--model_name_or_path=hf-internal-testing/tiny-random-gpt2"]
        with patch("sys.argv", args):
            cli.main()
        run_mock.assert_called_once()

    def test_parsed_args(self):
        with (
            patch.object(ChatCommand, "__init__", return_value=None) as init_mock,
            patch.object(ChatCommand, "run") as run_mock,
            patch(
                "sys.argv",
                [
                    "transformers",
                    "chat",
                    "test-model",
                    "max_new_tokens=64",
                ],
            ),
        ):
            cli.main()
        init_mock.assert_called_once()
        run_mock.assert_called_once()
        parsed_args = init_mock.call_args[0][0]
        self.assertEqual(parsed_args.model_name_or_path_or_address, "test-model")
        self.assertEqual(parsed_args.generate_flags, ["max_new_tokens=64"])


class ChatUtilitiesTest(unittest.TestCase):
    def test_save_and_clear_chat(self):
        tmp_path = tempfile.mkdtemp()

        args = ChatArguments(save_folder=str(tmp_path))
        args.model_name_or_path_or_address = "test-model"

        chat_history = [{"role": "user", "content": "hi"}]
        filename = ChatCommand.save_chat(chat_history, args)
        self.assertTrue(os.path.isfile(filename))

        cleared = ChatCommand.clear_chat_history()
        self.assertEqual(cleared, [])

    def test_parse_generate_flags(self):
        dummy = ChatCommand.__new__(ChatCommand)
        parsed = ChatCommand.parse_generate_flags(dummy, ["temperature=0.5", "max_new_tokens=10"])
        self.assertEqual(parsed["temperature"], 0.5)
        self.assertEqual(parsed["max_new_tokens"], 10)
