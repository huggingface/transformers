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
import json
import os
import tempfile
import unittest

from typer.testing import CliRunner

import transformers.cli.transformers as cli
from transformers.cli.chat import chat


runner = CliRunner()


class ChatCLITest(unittest.TestCase):
    def test_help(self):
        output = runner.invoke(cli.app, ["chat", "--help"])
        assert output.exit_code == 0
        assert "Chat with a model from the command line." in output.output


class ChatUtilitiesTest(unittest.TestCase):
    def test_save_and_clear_chat(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            filename = chat.save_chat(tmp_path, "test-model", [{"role": "user", "content": "hi"}], {"foo": "bar"})
            assert os.path.isfile(filename)
            with open(filename, "r") as f:
                data = json.load(f)
                assert data["chat_history"] == [{"role": "user", "content": "hi"}]
                assert data["settings"] == {"foo": "bar"}

    def test_clear_chat_history(self):
        assert chat.clear_chat_history() == []
        assert chat.clear_chat_history("prompt") == [{"role": "system", "content": "prompt"}]

    def test_parse_generate_flags(self):
        parsed = chat.parse_generate_flags(["temperature=0.5", "max_new_tokens=10"])
        assert parsed["temperature"] == 0.5
        assert parsed["max_new_tokens"] == 10
