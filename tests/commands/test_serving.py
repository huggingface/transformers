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

    def test_parsed_args(self):
        with (
            patch.object(ServeCommand, "__init__", return_value=None) as init_mock,
            patch.object(ServeCommand, "run") as run_mock,
            patch("sys.argv", ["transformers", "serve", "--host", "0.0.0.0", "--port", "9000"]),
        ):
            cli.main()
        init_mock.assert_called_once()
        run_mock.assert_called_once()
        parsed_args = init_mock.call_args[0][0]
        self.assertEqual(parsed_args.host, "0.0.0.0")
        self.assertEqual(parsed_args.port, 9000)

    def test_build_chunk(self):
        dummy = ServeCommand.__new__(ServeCommand)
        dummy.args = type("Args", (), {})()
        chunk = ServeCommand.build_chunk(dummy, "hello", "req0", finish_reason="stop")
        self.assertIn("chat.completion.chunk", chunk)
        self.assertIn("data:", chunk)
