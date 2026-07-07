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

from transformers.cli.chat import (
    Chat,
    RichInterface,
    _get_management_base_url,
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


def test_get_management_base_url_strips_v1_api_path():
    assert _get_management_base_url("http://localhost:8000/v1") == "http://localhost:8000"
    assert _get_management_base_url("http://localhost:8000/v1/") == "http://localhost:8000"
    assert _get_management_base_url("http://localhost:8000") == "http://localhost:8000"
    assert _get_management_base_url("http://localhost:8000/proxy/v1") == "http://localhost:8000/proxy/v1"


def test_chat_base_url_with_v1_uses_root_health(monkeypatch):
    seen = {}

    class Response:
        status_code = 200

    def fake_get(url):
        seen["url"] = url
        return Response()

    monkeypatch.setattr("transformers.cli.chat.httpx.get", fake_get)

    Chat.check_health("http://localhost:8000/v1")

    assert seen["url"] == "http://localhost:8000/health"


def test_chat_base_url_with_v1_loads_model_on_root_endpoint(monkeypatch):
    seen = {}

    class Response:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield b'data: {"status": "ready", "cached": true}'

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr("transformers.cli.chat.requests.post", fake_post)

    RichInterface(
        model_id="dummy-model",
        user_id="tester",
        base_url="http://localhost:8000/v1",
    ).print_model_load("dummy-model")

    assert seen["url"] == "http://localhost:8000/load_model"
    assert seen["kwargs"]["json"] == {"model": "dummy-model"}
    assert seen["kwargs"]["stream"] is True
