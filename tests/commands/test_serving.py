import unittest
from unittest.mock import patch

import transformers.commands.transformers_cli as cli
from transformers.commands.serving import ServeCommand
from transformers.testing_utils import CaptureStd


class ServeCLITest(unittest.TestCase):
    def test_help(self):
        with patch("sys.argv", ["transformers", "serve", "--help"]), CaptureStd() as cs:
            with self.assertRaises(SystemExit):
                cli.main()
        self.assertIn("serve", cs.out.lower())

    @patch.object(ServeCommand, "run")
    def test_cli_dispatch(self, run_mock):
        args = ["transformers", "serve", "hf-internal-testing/tiny-random-gpt2"]
        with patch("sys.argv", args):
            cli.main()
        run_mock.assert_called_once()

    def test_parsed_args(self):
        with (
            patch.object(ServeCommand, "__init__", return_value=None) as init_mock,
            patch.object(ServeCommand, "run") as run_mock,
            patch("sys.argv", ["transformers", "serve", "the-model", "--host", "0.0.0.0", "--port", "9000"]),
        ):
            cli.main()
        init_mock.assert_called_once()
        run_mock.assert_called_once()
        parsed_args = init_mock.call_args[0][0]
        self.assertEqual(parsed_args.model_name_or_path, "the-model")
        self.assertEqual(parsed_args.host, "0.0.0.0")
        self.assertEqual(parsed_args.port, 9000)

    def test_build_chunk(self):
        dummy = ServeCommand.__new__(ServeCommand)
        dummy.args = type("Args", (), {"model_name_or_path": "test-model"})()
        chunk = ServeCommand.build_chunk(dummy, "hello", "req0", finish_reason="stop")
        self.assertIn("chat.completion.chunk", chunk)
        self.assertIn("data:", chunk)
        self.assertIn("test-model", chunk)
