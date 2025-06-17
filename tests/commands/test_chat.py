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
    def test_cli_dispatch(self, run_mock):
        args = ["transformers", "chat", "hf-internal-testing/tiny-random-gpt2"]
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
