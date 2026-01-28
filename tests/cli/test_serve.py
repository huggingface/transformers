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
import time
import unittest
from threading import Thread
from unittest.mock import Mock, patch

import httpx
from huggingface_hub import ChatCompletionStreamOutput, InferenceClient, hf_hub_download
from parameterized import parameterized

from transformers import GenerationConfig
from transformers.cli.serve import Modality, Serve
from transformers.testing_utils import require_openai, slow
from transformers.utils.import_utils import is_openai_available


if is_openai_available():
    from openai import APIConnectionError, OpenAI
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseContentPartAddedEvent,
        ResponseContentPartDoneEvent,
        ResponseCreatedEvent,
        ResponseInProgressEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )


@require_openai
def test_help(cli):
    """Minimal test: we can invoke the help command."""
    output = cli("serve", "--help")
    assert output.exit_code == 0
    assert "serve" in output.output


@require_openai
def test_host_port_blocking(cli):
    """Minimal test: we can set arguments through the CLI - blocking"""
    with (
        patch("uvicorn.Config") as ConfigMock,
        patch("uvicorn.Server") as ServerMock,
    ):
        server_instance = Mock()
        ServerMock.return_value = server_instance

        # Call the serve CLI with host/port
        out = cli("serve", "--host", "0.0.0.0", "--port", "9000")
        _, kwargs = ConfigMock.call_args

        assert out.exit_code == 0
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000

        ServerMock.assert_called_once_with(ConfigMock.return_value)
        server_instance.run.assert_called_once()


@require_openai
def test_host_port_non_blocking(cli, caplog):
    """Minimal test: we can set arguments through the CLI - non-blocking"""
    caplog.set_level(100000)
    # ^ hack to avoid an issue happening only in CI. We don't check logs anyway so it's fine.
    #   Source: https://github.com/pallets/click/issues/824#issuecomment-562581313

    with (
        patch("uvicorn.Config") as ConfigMock,
        patch("uvicorn.Server") as ServerMock,
        patch.object(Serve, "start_server") as start_mock,
    ):
        server_instance = Mock()
        ServerMock.return_value = server_instance

        out = cli("serve", "--host", "0.5.0.0", "--port", "9002", "--non-blocking")
        assert out.exit_code == 0

        # Config got the CLI args
        _, kwargs = ConfigMock.call_args
        assert kwargs["host"] == "0.5.0.0"
        assert kwargs["port"] == 9002

        # Non-blocking path uses start_server(), not server.run()
        start_mock.assert_called_once()
        server_instance.run.assert_not_called()


@require_openai
def test_build_chat_completion_chunk():
    """
    Tests that the chunks are correctly built for the Chat Completion API. The `choices` checks implicitly
    confirm that empty fields are not emitted.
    """
    dummy = Serve.__new__(Serve)

    # The keys for these fields must be present in every chunk
    MANDATORY_FIELDS = ["data", "id", "choices", "created", "model", "object", "system_fingerprint"]

    # Case 1: most fields are provided
    chunk = dummy.build_chat_completion_chunk(
        request_id="req0", content="hello", finish_reason="stop", role="user", model="dummy_model@main"
    )
    chunk = dummy.chunk_to_sse_element(chunk)
    for field in MANDATORY_FIELDS:
        assert field in chunk
    assert '"choices":[{"delta":{"content":"hello","role":"user"},"finish_reason":"stop","index":0}]' in chunk

    # Case 2: only the role is provided -- other fields in 'choices' are omitted
    chunk = dummy.build_chat_completion_chunk(request_id="req0", role="user", model="dummy_model@main")
    chunk = dummy.chunk_to_sse_element(chunk)
    for field in MANDATORY_FIELDS:
        assert field in chunk
    assert '"choices":[{"delta":{"role":"user"},"index":0}]' in chunk

    # Case 3: only the content is provided -- other fields in 'choices' are omitted
    chunk = dummy.build_chat_completion_chunk(request_id="req0", content="hello", model="dummy_model@main")
    chunk = dummy.chunk_to_sse_element(chunk)
    for field in MANDATORY_FIELDS:
        assert field in chunk
    assert '"choices":[{"delta":{"content":"hello"},"index":0}]' in chunk

    # Case 4: tool calls support a list of ChoiceDeltaToolCall objects
    tool_call = ChoiceDeltaToolCall(
        index=0,
        function=ChoiceDeltaToolCallFunction(name="foo_bar", arguments='{"foo1": "bar1", "foo2": "bar2"}'),
        type="function",
    )
    chunk = dummy.build_chat_completion_chunk(request_id="req0", tool_calls=[tool_call], model="dummy_model@main")
    chunk = dummy.chunk_to_sse_element(chunk)
    for field in MANDATORY_FIELDS:
        assert field in chunk
    expected_choices_content = (
        'choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"foo1\\": \\"bar1\\", '
        '\\"foo2\\": \\"bar2\\"}","name":"foo_bar"},"type":"function"}]},"index":0}]'
    )
    assert expected_choices_content in chunk


def test_generative_model_list():
    with tempfile.TemporaryDirectory() as cache_dir:
        # "download" a few models, including some non-generative models
        hf_hub_download("Menlo/Jan-nano", "config.json", cache_dir=cache_dir)
        hf_hub_download("Menlo/Jan-nano-128k", "config.json", cache_dir=cache_dir)
        hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "config.json", cache_dir=cache_dir)
        hf_hub_download("HuggingFaceTB/SmolVLM-Instruct", "config.json", cache_dir=cache_dir)
        hf_hub_download("google-bert/bert-base-cased", "config.json", cache_dir=cache_dir)

        expected_results = {
            "HuggingFaceTB/SmolVLM-Instruct": ["HuggingFaceTB", "SmolVLM-Instruct"],
            "Qwen/Qwen2.5-0.5B-Instruct": ["Qwen", "Qwen2.5-0.5B-Instruct"],
            "Menlo/Jan-nano": ["Menlo", "Jan-nano"],
            "Menlo/Jan-nano-128k": ["Menlo", "Jan-nano-128k"],
        }

        # list models
        result = Serve.get_gen_models(cache_dir)
        assert len(expected_results) == len(result)

        local_repos = {repo["id"]: repo["owned_by"] for repo in result}

        for key, value in expected_results.items():
            assert key in local_repos
            assert local_repos[key] == value


@require_openai
def test_build_response_event():
    """
    Tests that the events are correctly built for the Response API.

    Contrarily to the Chat Completion API, the Response API has a wide set of possible output objects. This test
    only checks a few basic assumptions -- we rely on OpenAI's pydantic models to enforce the correct schema.
    """
    dummy = Serve.__new__(Serve)

    response_created = ResponseCreatedEvent(
        type="response.created",
        sequence_number=0,
        response=Response(
            id="resp_0",
            created_at=time.time(),
            status="queued",
            model="dummy_model@main",
            instructions=None,  # <--- is set to None = should NOT be in the output.
            text={"format": {"type": "text"}},
            object="response",
            tools=[],  # <--- empty lists should be in the output (they are often mandatory fields)
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            metadata=None,
        ),
    )

    event = dummy.chunk_to_sse_element(response_created)
    assert event.startswith("data: ")  # Sanity check: event formatting
    assert '"model":"dummy_model@main"' in event  # Sanity check: set field
    assert '"status":"queued"' in event
    assert "tools" in event  # empty lists should be in the output
    assert "output" in event
    assert "instructions" not in event  # None fields should NOT be in the output
    assert "metadata" not in event
    assert "error" not in event  # Unset optional fields should NOT be in the output
    assert "top_p" not in event


def retry(fn, max_attempts=5, delay=2):
    """
    Retry a function up to `max_attempts` times with a `delay` between attempts.
    Useful for testing functions that may fail due to server not being ready.
    """

    def wrapper(*args, **kwargs):
        nb_attempts = 0
        while True:
            nb_attempts += 1
            try:
                return fn(*args, **kwargs)
            except (httpx.HTTPError, APIConnectionError):
                if nb_attempts >= max_attempts:
                    raise
                time.sleep(delay)

    return wrapper


class ServeCompletionsMixin:
    """
    Mixin class for the Completions API tests, to seamlessly replicate tests across the two versions of the API
    (`generate` and `continuous_batching`).
    """

    @retry
    def run_server(self, request):
        with InferenceClient(f"http://localhost:{self.port}") as client:
            return list(client.chat_completion(**request))

    @parameterized.expand(
        [
            ("default_request", {}),
            ("one_token", {"max_tokens": 1}),
            ("different_model", {"model": "HuggingFaceTB/SmolLM2-135M-Instruct"}),
            (
                "tool_call",
                {
                    "tools": [
                        {
                            "function": {
                                "name": "foo_bar",
                                "parameters": {"type": "object"},
                                "description": "Foo bar",
                            },
                            "type": "function",
                        }
                    ]
                },
            ),
        ]
    )
    def test_requests(self, test_name: str, request_flags: dict):
        """Tests that the completions app gracefully handles GOOD requests, producing the expected output payloads."""

        request = {
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "stream": True,  # We don't support "stream": False yet
            "max_tokens": 5,  # Small generation by default
        }
        request.update(request_flags)
        all_payloads = self.run_server(request)

        # If a request is successful, the returned payload needs to follow the schema, which we test here.
        # NOTE: the output of our server is wrapped by `InferenceClient`, which sends fields even when they
        # are empty.

        # Finish reason: the last payload should have a finish reason of "length" or "stop", all others should be empty
        finish_reasons = [payload.choices[0].finish_reason for payload in all_payloads]
        self.assertTrue(finish_reasons[-1] in ["length", "stop"])
        self.assertTrue(all(reason is None for reason in finish_reasons[:-1]))

        # Role: the first payload should have a role of "assistant", all others should be empty
        roles = [payload.choices[0].delta.role for payload in all_payloads]
        self.assertEqual(roles[0], "assistant")
        self.assertTrue(all(role is None for role in roles[1:]))

        # Content: the first and the last payload shouldn't have content (role and finish reason). It may be empty
        # in some other payload positions, e.g. tool calls.
        contents = [payload.choices[0].delta.content for payload in all_payloads]
        self.assertTrue(contents[0] is None and contents[-1] is None)
        self.assertTrue(any(content is not None for content in contents[1:-1]))
        # TODO: add "usage" field to output and test it

    def test_generation_config_in_request(self):
        """Tests that the generation config is correctly passed into the generation call."""
        generation_config = GenerationConfig(do_sample=False, temperature=0.0)
        request = {
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "stream": True,
            "max_tokens": 10,
            "extra_body": {
                "generation_config": generation_config.to_json_string(),
            },
        }
        all_payloads = self.run_server(request)
        contents = [payload.choices[0].delta.content for payload in all_payloads]
        output_text = "".join([text for text in contents if text is not None])
        # The generation config sets greedy decoding, so the output is reproducible. By default, `Qwen/Qwen3-0.6B`
        # sets `do_sample=True`
        self.assertEqual(output_text, '<think>\nOkay, the user just asked, "')

    def test_early_return_due_to_length(self):
        request = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "stream": True,
            "max_tokens": 3,
        }

        all_payloads = self.run_server(request)
        last_payload = all_payloads[-1]
        self.assertTrue(last_payload.choices[0]["finish_reason"] == "length")

    def test_continues_until_stop(self):
        request = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": 'Please only answer with "Hi."'}],
            "stream": True,
            "max_tokens": 30,
        }

        all_payloads = self.run_server(request)
        last_payload = all_payloads[-1]
        self.assertTrue(last_payload.choices[0]["finish_reason"] == "stop")


class ServeCompletionsGenerateMockTests(unittest.TestCase):
    def test_processor_inputs_from_inbound_messages_llm(self):
        modality = Modality.LLM
        messages = expected_outputs = [
            {"role": "user", "content": "How are you doing?"},
            {"role": "assistant", "content": "I'm doing great, thank you for asking! How can I assist you today?"},
            {"role": "user", "content": "Can you help me write tests?"},
        ]
        outputs = Serve.get_processor_inputs_from_inbound_messages(messages, modality)
        self.assertListEqual(expected_outputs, outputs)

        messages_with_type = [
            {"role": "user", "content": [{"type": "text", "text": "How are you doing?"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'm doing great, thank you for asking! How can I assist you today?"}
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Can you help me write tests?"}]},
        ]
        outputs = Serve.get_processor_inputs_from_inbound_messages(messages_with_type, modality)
        self.assertListEqual(expected_outputs, outputs)

        messages_multiple_text = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "How are you doing?"},
                    {"type": "text", "text": "I'm doing great, thank you for asking! How can I assist you today?"},
                ],
            },
        ]
        expected_outputs_multiple_text = [
            {
                "role": "user",
                "content": "How are you doing? I'm doing great, thank you for asking! How can I assist you today?",
            },
        ]
        outputs = Serve.get_processor_inputs_from_inbound_messages(messages_multiple_text, modality)
        self.assertListEqual(expected_outputs_multiple_text, outputs)

    def test_processor_inputs_from_inbound_messages_vlm_text_only(self):
        modality = Modality.VLM
        messages = [
            {"role": "user", "content": "How are you doing?"},
            {"role": "assistant", "content": "I'm doing great, thank you for asking! How can I assist you today?"},
            {"role": "user", "content": "Can you help me write tests?"},
        ]

        expected_outputs = [
            {"role": "user", "content": [{"type": "text", "text": "How are you doing?"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'm doing great, thank you for asking! How can I assist you today?"}
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Can you help me write tests?"}]},
        ]

        outputs = Serve.get_processor_inputs_from_inbound_messages(messages, modality)
        self.assertListEqual(expected_outputs, outputs)

    def test_processor_inputs_from_inbound_messages_vlm_text_and_image_in_base_64(self):
        modality = Modality.VLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "How many pixels are in the image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAASABIAAD/4QBARXhpZgAATU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAABaADAAQAAAABAAAABQAAAAD/7QA4UGhvdG9zaG9wIDMuMAA4QklNBAQAAAAAAAA4QklNBCUAAAAAABDUHYzZjwCyBOmACZjs+EJ+/8AAEQgABQAFAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/bAEMAAQEBAQEBAgEBAgICAgICAwICAgIDBAMDAwMDBAUEBAQEBAQFBQUFBQUFBQYGBgYGBgcHBwcHCAgICAgICAgICP/bAEMBAQEBAgICAwICAwgFBAUICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICP/dAAQAAf/aAAwDAQACEQMRAD8A/v4ooooA/9k="
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "The number of pixels in the image cannot be determined from the provided information.",
            },
            {"role": "user", "content": "Alright"},
        ]

        expected_outputs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "How many pixels are in the image?"},
                    {"type": "image", "url": "/var/folders/4v/64sxdhsd3gz3r8vhhnyc0mqw0000gn/T/tmp50oyghk6.png"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The number of pixels in the image cannot be determined from the provided information.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Alright"}]},
        ]

        outputs = Serve.get_processor_inputs_from_inbound_messages(messages, modality)

        for expected_output, output in zip(expected_outputs, outputs):
            expected_output_content = expected_output["content"]
            output_content = output["content"]

            self.assertEqual(type(expected_output_content), type(output_content))

            if isinstance(expected_output_content, list):
                for expected_output_content_item, output_content_item in zip(expected_output_content, output_content):
                    self.assertIn("type", expected_output_content_item)
                    self.assertIn("type", output_content_item)
                    self.assertTrue(expected_output_content_item["type"] == output_content_item["type"])

                    if expected_output_content_item["type"] == "text":
                        self.assertEqual(expected_output_content_item["text"], output_content_item["text"])

                    if expected_output_content_item["type"] == "image":
                        self.assertTrue(os.path.exists(output_content_item["url"]))
            else:
                raise ValueError("VLMs should only receive content as lists.")


@slow  # server startup time is slow on our push CI
@require_openai
class ServeCompletionsGenerateIntegrationTest(ServeCompletionsMixin, unittest.TestCase):
    """Tests the `generate` version of the Completions API."""

    @classmethod
    def setUpClass(cls):
        """Starts a server for tests to connect to."""
        cls.port = 8001
        cls.server = Serve(port=cls.port, non_blocking=True)

    @classmethod
    def tearDownClass(cls):
        cls.server.kill_server()

    @slow
    def test_tool_call(self):
        """Tests that the tool call is correctly handled and that the payloads are correctly structured."""
        # TODO: move to the mixin when CB also supports tool calls

        request = {
            # This model is a small model that's very eager to call tools
            # TODO: this is a 4B model. Find a smaller model that's eager to call tools
            "model": "Menlo/Jan-nano",
            # The request should produce a tool call
            "messages": [{"role": "user", "content": "Generate an image of a cat."}],
            "stream": True,
            "max_tokens": 50,
            # Reproducibility
            "temperature": 0.0,
            # This tool is a copy from the tool in the original tiny-agents demo
            "tools": [
                {
                    "function": {
                        "name": "flux1_schnell_infer",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "seed": {"type": "number", "description": "numeric value between 0 and 2147483647"},
                                "randomize_seed": {"type": "boolean", "default": True},
                                "width": {
                                    "type": "number",
                                    "description": "numeric value between 256 and 2048",
                                    "default": 1024,
                                },
                                "height": {
                                    "type": "number",
                                    "description": "numeric value between 256 and 2048",
                                    "default": 1024,
                                },
                                "num_inference_steps": {
                                    "type": "number",
                                    "description": "numeric value between 1 and 16",
                                    "default": 4,
                                },
                            },
                        },
                        "description": "Generate an image using the Flux 1 Schnell Image Generator.",
                    },
                    "type": "function",
                }
            ],
        }
        all_payloads = self.run_server(request)

        # The first payload should contain the role
        roles = [payload.choices[0].delta.role for payload in all_payloads]
        self.assertEqual(roles[0], "assistant")
        self.assertTrue(all(role is None for role in roles[1:]))

        # All other payloads (except the last one) should be tool call related, for this specific request
        contents = [payload.choices[0].delta.content for payload in all_payloads]
        self.assertTrue(all(content is None for content in contents))

        # The first tool call delta should contain the tool name. The other tool call deltas should contain the tool
        # arguments.
        tool_calls = [payload.choices[0].delta.tool_calls[0] for payload in all_payloads[1:-1]]
        first_tool_call = tool_calls[0]
        self.assertEqual(first_tool_call["function"]["name"], "flux1_schnell_infer")
        self.assertEqual(first_tool_call["function"]["arguments"], None)
        other_tool_calls = tool_calls[1:]
        self.assertTrue(all(tool_call["function"]["name"] is None for tool_call in other_tool_calls))
        self.assertTrue(all(tool_call["function"]["arguments"] is not None for tool_call in other_tool_calls))

        # Finally, the last payload should contain a finish reason
        finish_reasons = [payload.choices[0].finish_reason for payload in all_payloads]
        # TODO: I think the finish reason for a tool call is different? double check this
        self.assertTrue(finish_reasons[-1] in ["stop", "length"])
        self.assertTrue(all(reason is None for reason in finish_reasons[:-1]))


def _get_scheduler(serve_command):
    # Defensive navigation in case any layer is renamed in the future
    cbm = getattr(serve_command, "running_continuous_batching_manager", None)
    assert cbm is not None, "ServeCommand has no running_continuous_batching_manager"
    bp = getattr(cbm, "batch_processor", None)
    assert bp is not None, "running_continuous_batching_manager has no batch_processor"
    sched = getattr(bp, "scheduler", None)
    assert sched is not None, "batch_processor has no scheduler"
    return sched


def _call_healthcheck(base_url: str):
    response = None
    retries = 10
    while retries > 0:
        try:
            response = httpx.get(f"{base_url}/health")
            break
        except httpx.NetworkError:
            time.sleep(0.1)
            retries -= 1
    return response


def _open_stream_and_cancel(base_url: str, request_id: str):
    with httpx.Client() as s:
        with s.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers={"X-Request-ID": request_id},
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "stream": True,
                "messages": [{"role": "user", "content": "Count slowly so I can cancel you."}],
            },
            timeout=30,
        ) as resp:
            assert resp.status_code == 200

            wait_for_n_chunks = 3
            for i, _ in enumerate(resp.iter_bytes(chunk_size=None)):
                if i >= wait_for_n_chunks:
                    resp.close()
                    break


@slow  # server startup time is slow on our push CI
@require_openai
class ServeCompletionsContinuousBatchingIntegrationTest(ServeCompletionsMixin, unittest.TestCase):
    """Tests the `continuous_batching` version of the Completions API."""

    @classmethod
    def setUpClass(cls):
        """Starts a server for tests to connect to."""
        cls.port = 8002
        cls.server = Serve(
            port=cls.port, continuous_batching=True, attn_implementation="sdpa", default_seed=42, non_blocking=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.kill_server()

    def test_full_request(self):
        """Tests that an inference using the Responses API and Continuous Batching works"""

        request = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a sports assistant designed to craft sports programs."},
                {"role": "user", "content": "Tell me what you can do."},
            ],
            "stream": True,
            "max_tokens": 30,
        }
        all_payloads = self.run_server(request)

        full_text = ""
        for token in all_payloads:
            if isinstance(token, ChatCompletionStreamOutput) and token.choices and len(token.choices) > 0:
                content = token.choices[0].delta.get("content", "")
                full_text += content if content is not None else ""

        # System prompt applied.
        # Expect "sports" mention.
        self.assertTrue(full_text.strip())
        self.assertIn("sports", full_text.lower())

    def test_max_tokens_not_set_in_req(self):
        request = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a sports assistant designed to craft sports programs."},
                {"role": "user", "content": "Tell me what you can do."},
            ],
            "stream": True,
        }
        all_payloads = self.run_server(request)

        full_text = ""
        for token in all_payloads:
            if isinstance(token, ChatCompletionStreamOutput) and token.choices and len(token.choices) > 0:
                content = token.choices[0].delta.get("content", "")
                full_text += content if content is not None else ""

        # System prompt applied.
        # Expect "sports" mention.
        self.assertTrue(full_text.strip())
        self.assertIn("sports", full_text.lower())

    def test_request_cancellation(self):
        """Tests that a request can be cancelled."""

        base_url = f"http://127.0.0.1:{self.port}"
        request_id = "test-cancel"

        # Ensure the server is up before sending a request
        response = _call_healthcheck(base_url)
        self.assertIsNotNone(response, "Failed to connect to the server health endpoint.")
        self.assertEqual(response.status_code, 200)

        _open_stream_and_cancel(base_url, request_id)

        scheduler = _get_scheduler(self.server)

        # Because cancellation is non-blocking, poll for a short, bounded time.
        deadline = time.time() + 8.0  # generous but still CI-friendly
        last_seen = None
        while time.time() < deadline:
            is_cancelled = scheduler.request_is_cancelled(request_id)
            if is_cancelled:
                break
            last_seen = time.time()
            time.sleep(0.1)  # don't spin the CPU

        is_cancelled = scheduler.request_is_cancelled(request_id)
        self.assertTrue(
            is_cancelled,
            f"Request {request_id} still present in scheduler after cancellation "
            f"(last seen at {last_seen}). Check cancellation propagation.",
        )


@require_openai
class ServeResponsesMixin:
    """
    Mixin class for the Completions API tests, to seamlessly replicate tests across the two versions of the API
    (`generate` and `continuous_batching`).
    """

    @retry
    def run_server(self, request):
        client = OpenAI(base_url=f"http://localhost:{self.port}/v1", api_key="<KEY>")
        stream = client.responses.create(**request)

        all_payloads = []
        for payload in stream:
            all_payloads.append(payload)

        return all_payloads

    def test_request(self):
        """Tests that an inference using the Responses API works"""

        request = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "instructions": "You are a helpful assistant.",
            "input": "Hello!",
            "stream": True,
            "max_output_tokens": 1,
        }
        all_payloads = self.run_server(request)

        # Allow variable number of delta events depending on tokenizer/streamer behavior
        self.assertGreaterEqual(len(all_payloads), 8)

        # Start markers
        self.assertIsInstance(all_payloads[0], ResponseCreatedEvent)
        self.assertIsInstance(all_payloads[1], ResponseInProgressEvent)
        self.assertIsInstance(all_payloads[2], ResponseOutputItemAddedEvent)
        self.assertIsInstance(all_payloads[3], ResponseContentPartAddedEvent)

        # At least one delta event during streaming
        self.assertTrue(any(isinstance(p, ResponseTextDeltaEvent) for p in all_payloads[4:-4]))

        # Closing markers
        self.assertIsInstance(all_payloads[-4], ResponseTextDoneEvent)
        self.assertIsInstance(all_payloads[-3], ResponseContentPartDoneEvent)
        self.assertIsInstance(all_payloads[-2], ResponseOutputItemDoneEvent)
        self.assertIsInstance(all_payloads[-1], ResponseCompletedEvent)

    # TODO: one test for each request flag, to confirm it is working as expected
    # TODO: speed-based test to confirm that KV cache is working across requests


@slow  # server startup time is slow on our push CI
@require_openai
class ServeResponsesIntegrationTest(ServeResponsesMixin, unittest.TestCase):
    """Tests the Responses API."""

    @classmethod
    def setUpClass(cls):
        """Starts a server for tests to connect to."""
        cls.port = 8003
        cls.server = Serve(port=cls.port, default_seed=42, non_blocking=True)

    @classmethod
    def tearDownClass(cls):
        cls.server.kill_server()

    @slow
    def test_full_request(self):
        """Tests that an inference using the Responses API works"""

        request = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "instructions": "You are a sports assistant designed to craft sports programs.",
            "input": "Tell me what you can do.",
            "stream": True,
            "max_output_tokens": 30,
            # Disable sampling for deterministic output
            "temperature": 0,
        }
        all_payloads = self.run_server(request)

        full_text = ""
        for token in all_payloads:
            if isinstance(token, ResponseTextDeltaEvent):
                full_text += token.delta

        # Verify that the system prompt went through.
        # With deterministic decoding, exact wording can still vary across versions.
        # Assert non-empty output and that it references sports.
        self.assertTrue(len(full_text) > 0)
        self.assertIn("sports", full_text.lower())

    @slow
    def test_non_streaming_request(self):
        """Tests that an inference using the Responses API with stream=False returns a single Response payload."""
        from openai import OpenAI
        from openai.types.responses import Response as OpenAIResponse

        client = OpenAI(base_url=f"http://localhost:{self.port}/v1", api_key="<KEY>")
        resp = client.responses.create(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            instructions="You are a helpful assistant.",
            input="Hello!",
            stream=False,
            max_output_tokens=5,
        )

        # Should be a single Response object with completed status and one output item containing text
        self.assertIsInstance(resp, OpenAIResponse)
        self.assertEqual(resp.status, "completed")
        self.assertTrue(len(resp.output) >= 1)
        first_item = resp.output[0]
        self.assertEqual(first_item.type, "message")
        self.assertEqual(first_item.status, "completed")
        self.assertTrue(len(first_item.content) >= 1)
        first_part = first_item.content[0]
        self.assertEqual(first_part.type, "output_text")
        self.assertIsInstance(first_part.text, str)


class ServeInfrastructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 8042
        thread = Thread(target=Serve, kwargs={"port": cls.port})
        thread.daemon = True
        thread.start()

    def test_healthcheck(self):
        """Tests that the healthcheck endpoint works."""
        response = _call_healthcheck(f"http://localhost:{self.port}")
        self.assertIsNotNone(response, "Failed to connect to the server health endpoint.")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
