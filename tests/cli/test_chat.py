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
from unittest.mock import patch

from transformers.cli.chat import (
    Chat,
    RichInterface,
    management_base_url,
    new_chat_history,
    parse_generate_flags,
    save_chat,
)


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


def test_new_chat_history():
    assert new_chat_history() == []
    assert new_chat_history("prompt") == [{"role": "system", "content": "prompt"}]


def test_parse_generate_flags():
    parsed = parse_generate_flags(["temperature=0.5", "max_new_tokens=10"])
    assert parsed["temperature"] == 0.5
    assert parsed["max_new_tokens"] == 10


def test_management_base_url():
    assert management_base_url("http://localhost:8000/v1") == "http://localhost:8000"
    assert management_base_url("http://localhost:8000/v1/") == "http://localhost:8000"
    assert management_base_url("http://localhost:8000") == "http://localhost:8000"
    assert management_base_url("http://localhost:8000/") == "http://localhost:8000"
    assert management_base_url("https://example.com/proxy/v1") == "https://example.com/proxy"


def test_check_health_targets_service_root():
    with patch("transformers.cli.chat.httpx.get") as get_mock:
        get_mock.return_value.status_code = 200
        Chat.check_health("http://localhost:8000/v1")
    get_mock.assert_called_once_with("http://localhost:8000/health")


def test_model_load_targets_service_root():
    with patch("transformers.cli.chat.requests.post") as post_mock:
        post_mock.return_value.iter_lines.return_value = []
        RichInterface(model_id="m", user_id="u", base_url="http://localhost:8000/v1").print_model_load("m")
    assert post_mock.call_args.args[0] == "http://localhost:8000/load_model"
