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

import pytest
import typer

from transformers.cli.chat import (
    get_service_root_url,
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


def test_parse_generate_flags_value_with_equal_sign():
    assert parse_generate_flags(["cache_implementation=a=b"]) == {"cache_implementation": "a=b"}


def test_parse_generate_flags_missing_equal_sign():
    with pytest.raises(typer.BadParameter, match="missing `=` after `do_sample`"):
        parse_generate_flags(["do_sample"])


def test_parse_generate_flags_invalid_json_error_message():
    with pytest.raises(ValueError, match=r"Converted JSON string = \{\"stop_strings\": \[a\]\}"):
        parse_generate_flags(["stop_strings=[a]"])


def test_get_service_root_url():
    assert get_service_root_url("http://localhost:8000/v1") == "http://localhost:8000"
    assert get_service_root_url("http://localhost:8000/v1/") == "http://localhost:8000"
    assert get_service_root_url("http://localhost:8000") == "http://localhost:8000"
    assert get_service_root_url("http://localhost:8000/") == "http://localhost:8000"
    assert get_service_root_url("https://example.com/proxy/v1") == "https://example.com/proxy"
