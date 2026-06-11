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

from transformers import GenerationConfig
from transformers.cli.chat import Chat, new_chat_history, parse_generate_flags, save_chat


class DummyInterface:
    def __init__(self):
        self.messages = []

    def print_color(self, text: str, color: str):
        self.messages.append((text, color))


def test_help(cli):
    output = cli("chat", "--help")
    assert output.exit_code == 0
    assert "Chat with a model from the command line." in output.output


def test_save_and_clear_chat():
    with tempfile.TemporaryDirectory() as tmp_path:
        filename = os.path.join(tmp_path, "chat.json")
        save_chat(filename, [{"role": "user", "content": "hi"}], {"foo": "bar"})
        assert os.path.isfile(filename)
        with open(filename, "r") as f:
            data = json.load(f)
            assert data["chat_history"] == [{"role": "user", "content": "hi"}]
            assert data["settings"] == {"foo": "bar"}


def test_save_chat_supports_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    saved_path = save_chat("chat.json", [{"role": "user", "content": "hi"}], {"foo": "bar"})

    assert saved_path == os.path.abspath("chat.json")
    assert os.path.isfile(saved_path)


def test_save_chat_command_supports_custom_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chat = Chat.__new__(Chat)
    chat.save_folder = "chat_history"
    chat.model_id = "org/model"
    chat.settings = {"foo": "bar"}
    chat.system_prompt = None
    interface = DummyInterface()

    chat_history, valid_command, _ = chat.handle_non_exit_user_commands(
        "!save custom-chat.json",
        interface,
        {},
        GenerationConfig(),
        [{"role": "user", "content": "hi"}],
    )

    assert valid_command is True
    assert chat_history == [{"role": "user", "content": "hi"}]
    assert os.path.isfile("custom-chat.json")
    assert interface.messages == [("Chat saved to custom-chat.json!", "green")]


def test_new_chat_history():
    assert new_chat_history() == []
    assert new_chat_history("prompt") == [{"role": "system", "content": "prompt"}]


def test_parse_generate_flags():
    parsed = parse_generate_flags(["temperature=0.5", "max_new_tokens=10"])
    assert parsed["temperature"] == 0.5
    assert parsed["max_new_tokens"] == 10
