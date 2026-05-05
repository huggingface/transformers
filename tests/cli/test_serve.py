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
"""
Tests for the serving layer.

"""

import asyncio
import json
import socket
import time
import unittest
from unittest.mock import MagicMock

import httpx

from transformers.cli.serve import Serve
from transformers.cli.serving.chat_completion import ChatCompletionHandler
from transformers.cli.serving.completion import CompletionHandler
from transformers.cli.serving.model_manager import ModelManager, TimedModel
from transformers.cli.serving.response import ResponseHandler, compute_usage
from transformers.cli.serving.server import build_server
from transformers.cli.serving.transcription import TranscriptionHandler
from transformers.cli.serving.utils import (
    _TOOL_CALL_FALLBACKS,
    BaseHandler,
    GenerationState,
    Modality,
    get_tool_call_config,
    parse_tool_calls,
)
from transformers.testing_utils import (
    require_librosa,
    require_multipart,
    require_serve,
    require_torch_accelerator,
    require_torchcodec,
    require_vision,
    slow,
)
from transformers.utils.chat_parsing_utils import recursive_parse
from transformers.utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException
    from openai import OpenAI
    from openai.types.responses import Response, ResponseCreatedEvent


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _start_serve(**kwargs) -> tuple["Serve", int]:
    """Start a non-blocking Serve instance on a free port and wait until healthy.

    Returns ``(serve, port)``.
    """
    port = _find_free_port()
    serve = Serve(port=port, non_blocking=True, **kwargs)
    for _ in range(30):
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return serve, port
        except Exception:  # noqa: S110
            pass
        time.sleep(1)
    raise RuntimeError(f"Server on port {port} did not become healthy in time")


@require_serve
def test_host_port_blocking(cli):
    """CLI args --host and --port are passed to uvicorn.Config, and server.run() is called."""
    from unittest.mock import Mock, patch

    with (
        patch("uvicorn.Config") as ConfigMock,
        patch("uvicorn.Server") as ServerMock,
    ):
        server_instance = Mock()
        ServerMock.return_value = server_instance

        out = cli("serve", "--host", "0.0.0.0", "--port", "9000")
        _, kwargs = ConfigMock.call_args

        assert out.exit_code == 0
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000
        ServerMock.assert_called_once_with(ConfigMock.return_value)
        server_instance.run.assert_called_once()


class TestProcessorInputsFromMessages(unittest.TestCase):
    def test_llm_string_content(self):
        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = get_processor_inputs_from_messages(messages, Modality.LLM)
        self.assertEqual(result, [{"role": "user", "content": "Hello"}])

    def test_llm_list_content_text_only(self):
        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        messages = [{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}]
        result = get_processor_inputs_from_messages(messages, Modality.LLM)
        self.assertEqual(result, [{"role": "user", "content": "A B"}])

    def test_vlm_string_content_wrapped(self):
        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = get_processor_inputs_from_messages(messages, Modality.VLM)
        self.assertEqual(result, [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])

    def test_vlm_text_and_image_url(self):
        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        result = get_processor_inputs_from_messages(messages, Modality.VLM)
        self.assertEqual(len(result[0]["content"]), 2)
        self.assertEqual(result[0]["content"][0]["type"], "text")
        self.assertEqual(result[0]["content"][1], {"type": "image", "url": "https://example.com/img.png"})

    def test_llm_multi_turn_conversation(self):
        """Multi-turn conversation with string content should pass through as-is."""

        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        messages = [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm great!"},
            {"role": "user", "content": "Help me write tests?"},
        ]
        result = get_processor_inputs_from_messages(messages, Modality.LLM)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["content"], "How are you?")
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(result[2]["content"], "Help me write tests?")

    def test_llm_list_content_with_type(self):
        """LLM messages with typed content list should extract text and join."""

        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]}
        ]
        result = get_processor_inputs_from_messages(messages, Modality.LLM)
        self.assertEqual(result[0]["content"], "Hello world")

    @require_vision
    def test_vlm_base64_image_passed_through(self):
        """Base64 image URLs should be passed through as-is for the processor to handle."""

        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        # Minimal valid 1x1 PNG as base64
        base64_url = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": base64_url}},
                ],
            }
        ]
        result = get_processor_inputs_from_messages(messages, Modality.VLM)
        image_item = result[0]["content"][1]
        self.assertEqual(image_item["type"], "image")
        self.assertEqual(image_item["url"], base64_url)

    def test_vlm_multi_turn(self):
        """VLM multi-turn: string content should be wrapped in text type."""

        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        messages = [
            {"role": "user", "content": "Describe the image"},
            {"role": "assistant", "content": "It shows a cat"},
            {"role": "user", "content": "What color?"},
        ]
        result = get_processor_inputs_from_messages(messages, Modality.VLM)
        self.assertEqual(len(result), 3)
        for msg in result:
            self.assertIsInstance(msg["content"], list)
            self.assertEqual(msg["content"][0]["type"], "text")

    def test_llm_tool_use_fields_forwarded(self):
        """Tool-use fields (tool_calls, tool_call_id) should be forwarded to processor inputs."""

        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
        ]
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "tool_calls": tool_calls},
            {"role": "tool", "content": "22°C, sunny", "tool_call_id": "call_1"},
        ]
        result = get_processor_inputs_from_messages(messages, Modality.LLM)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1]["tool_calls"], tool_calls)
        self.assertNotIn("tool_calls", result[0])
        self.assertEqual(result[2]["tool_call_id"], "call_1")
        self.assertNotIn("tool_call_id", result[0])

    def test_vlm_tool_use_fields_forwarded(self):
        """Tool-use fields should be forwarded for VLM modality as well."""

        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "describe", "arguments": "{}"}}]
        messages = [
            {"role": "user", "content": "Describe this"},
            {"role": "assistant", "tool_calls": tool_calls},
            {"role": "tool", "content": "A landscape photo", "tool_call_id": "call_1"},
        ]
        result = get_processor_inputs_from_messages(messages, Modality.VLM)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1]["tool_calls"], tool_calls)
        self.assertEqual(result[2]["tool_call_id"], "call_1")

    def test_multimodal_base64_input_audio_converted_to_data_uri(self):
        """input_audio with base64 data should be converted to a data URI for the processor to handle."""
        import base64

        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
        audio_bytes = httpx.get(audio_url, follow_redirects=True).content
        audio_b64 = base64.b64encode(audio_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you hear?"},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                ],
            }
        ]
        result = get_processor_inputs_from_messages(messages, Modality.MULTIMODAL)
        self.assertEqual(len(result[0]["content"]), 2)
        self.assertEqual(result[0]["content"][0]["type"], "text")
        audio_item = result[0]["content"][1]
        self.assertEqual(audio_item["type"], "audio")
        self.assertTrue(audio_item["url"].startswith("data:audio/mp3;base64,"))
        self.assertIn(audio_b64, audio_item["url"])

    def test_vlm_ignores_audio_content(self):
        """VLM models should ignore audio content parts."""
        import base64

        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        audio_b64 = base64.b64encode(b"fake audio").decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you hear?"},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                ],
            }
        ]
        result = get_processor_inputs_from_messages(messages, Modality.VLM)
        self.assertEqual(len(result[0]["content"]), 1)
        self.assertEqual(result[0]["content"][0]["type"], "text")

    def test_video_url_converted_to_hf_video_format(self):
        """video_url content should be converted to HF video format for both VLM and MULTIMODAL."""
        get_processor_inputs_from_messages = BaseHandler.get_processor_inputs_from_messages

        video_src = "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_src}},
                    {"type": "text", "text": "What is happening?"},
                ],
            }
        ]
        for modality in (Modality.VLM, Modality.MULTIMODAL):
            with self.subTest(modality=modality):
                result = get_processor_inputs_from_messages(messages, modality)
                self.assertEqual(len(result[0]["content"]), 2)
                video_item = result[0]["content"][0]
                self.assertEqual(video_item, {"type": "video", "url": video_src})


class TestGenerativeModelList(unittest.TestCase):
    def test_lists_only_generative_models(self):
        """Should list LLMs and VLMs but not non-generative models like BERT."""
        import tempfile

        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory() as cache_dir:
            # Download config.json for a few models
            hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "config.json", cache_dir=cache_dir)
            hf_hub_download("google-bert/bert-base-cased", "config.json", cache_dir=cache_dir)

            result = ModelManager.get_gen_models(cache_dir)
            model_ids = {r["id"] for r in result}

            self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", model_ids)
            self.assertNotIn("google-bert/bert-base-cased", model_ids)


@require_serve
class TestBuildGenerationConfig(unittest.TestCase):
    def _make_handler(self):
        return ChatCompletionHandler(model_manager=MagicMock(), generation_state=GenerationState())

    def test_max_tokens(self):
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({"max_tokens": 7}, GenerationConfig())
        self.assertEqual(result.max_new_tokens, 7)

    def test_temperature_zero_disables_sampling(self):
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({"temperature": 0.0}, GenerationConfig(do_sample=True))
        self.assertFalse(result.do_sample)

    def test_frequency_penalty(self):
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({"frequency_penalty": 0.5}, GenerationConfig())
        self.assertAlmostEqual(result.repetition_penalty, 1.5)

    def test_logit_bias_tuple_keys(self):
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({"logit_bias": {"42": 1.0}}, GenerationConfig())
        self.assertEqual(result.sequence_bias, {(42,): 1.0})

    def test_stop_strings(self):
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({"stop": ["<END>"]}, GenerationConfig())
        self.assertEqual(result.stop_strings, ["<END>"])

    def test_generation_config_json_overrides(self):
        from transformers import GenerationConfig

        custom = GenerationConfig(max_new_tokens=5, do_sample=False)
        result = self._make_handler()._build_generation_config(
            {"generation_config": custom.to_json_string()}, GenerationConfig(max_new_tokens=100)
        )
        self.assertEqual(result.max_new_tokens, 5)
        self.assertFalse(result.do_sample)

    def test_generation_config_json_no_defaults_applied(self):
        """When generation_config JSON is passed, serving defaults should NOT be applied."""
        from transformers import GenerationConfig

        custom = GenerationConfig(max_new_tokens=10)
        result = self._make_handler()._build_generation_config(
            {"generation_config": custom.to_json_string()}, GenerationConfig()
        )
        # Should keep 10, not bump to 1024
        self.assertEqual(result.max_new_tokens, 10)

    def test_default_bumps_short_max_new_tokens(self):
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({}, GenerationConfig(max_new_tokens=20))
        self.assertEqual(result.max_new_tokens, 1024)

    def test_user_max_tokens_overrides_default(self):
        """User's max_tokens should win over the serving default."""
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({"max_tokens": 50}, GenerationConfig(max_new_tokens=20))
        self.assertEqual(result.max_new_tokens, 50)


@require_serve
class TestValidation(unittest.TestCase):
    def _make_handler(self):
        return ChatCompletionHandler(model_manager=MagicMock(), generation_state=GenerationState())

    def test_valid_request_passes(self):
        handler = self._make_handler()
        # Should not raise
        handler._validate_request({"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True})

    def test_unexpected_keys_rejected(self):
        handler = self._make_handler()
        with self.assertRaises(HTTPException) as ctx:
            handler._validate_request({"model": "x", "messages": [], "bogus_field": True})
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("bogus_field", ctx.exception.detail)

    def test_unsupported_fields_warns(self):
        handler = self._make_handler()
        with self.assertLogs("transformers", level="WARNING") as cm:
            handler._validate_request({"model": "x", "messages": [], "audio": {}})
        self.assertTrue(any("audio" in msg for msg in cm.output))


class TestResolveModel(unittest.TestCase):
    def _make_handler(self, force_model=None):
        mm = MagicMock()
        mm.force_model = force_model
        mm.process_model_name.side_effect = ModelManager.process_model_name
        mm.load_model_and_processor.return_value = (MagicMock(), MagicMock())
        return ChatCompletionHandler(model_manager=mm, generation_state=GenerationState())

    def test_force_model_overrides_when_model_omitted(self):
        handler = self._make_handler(force_model="org/pinned")
        body = {}
        model_id, _, _ = handler._resolve_model(body)
        self.assertEqual(model_id, "org/pinned@main")
        self.assertEqual(body["model"], "org/pinned")

    def test_force_model_allows_matching_request(self):
        handler = self._make_handler(force_model="org/pinned")
        body = {"model": "org/pinned"}
        model_id, _, _ = handler._resolve_model(body)
        self.assertEqual(model_id, "org/pinned@main")

    def test_force_model_rejects_mismatched_request(self):
        handler = self._make_handler(force_model="org/pinned")
        with self.assertRaises(HTTPException) as ctx:
            handler._resolve_model({"model": "other/model"})
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("org/pinned", ctx.exception.detail)
        self.assertIn("other/model", ctx.exception.detail)


class TestModelManager(unittest.TestCase):
    def test_process_model_name_adds_main(self):
        self.assertEqual(ModelManager.process_model_name("org/model"), "org/model@main")

    def test_process_model_name_preserves_revision(self):
        self.assertEqual(ModelManager.process_model_name("org/model@dev"), "org/model@dev")

    def test_quantization_config_4bit(self):
        mm = ModelManager(quantization="bnb-4bit")
        cfg = mm.get_quantization_config()
        self.assertTrue(cfg.load_in_4bit)

    def test_quantization_config_8bit(self):
        mm = ModelManager(quantization="bnb-8bit")
        cfg = mm.get_quantization_config()
        self.assertTrue(cfg.load_in_8bit)

    def test_quantization_config_none(self):
        mm = ModelManager()
        self.assertIsNone(mm.get_quantization_config())


class TestTimedModel(unittest.TestCase):
    def test_delete_model(self):
        mock_model = MagicMock()
        deleted = []
        timed = TimedModel(
            mock_model, timeout_seconds=9999, processor=MagicMock(), on_unload=lambda: deleted.append(True)
        )
        self.assertIsNotNone(timed.model)
        timed.delete_model()
        self.assertIsNone(timed.model)
        self.assertEqual(len(deleted), 1)

    def test_timeout_zero_no_delete(self):
        mock_model = MagicMock()
        timed = TimedModel(mock_model, timeout_seconds=0, processor=MagicMock())
        timed._timeout_reached()
        self.assertIsNotNone(timed.model)
        timed._timer.cancel()


@require_serve
class TestChunkSSE(unittest.TestCase):
    def _make_handler(self):
        return ChatCompletionHandler(model_manager=MagicMock(), generation_state=GenerationState())

    def test_build_chunk_sse_content(self):
        handler = self._make_handler()
        sse = handler._build_chunk_sse(request_id="req1", content="hi", model="m")
        self.assertTrue(sse.startswith("data: "))
        self.assertTrue(sse.endswith("\n\n"))
        parsed = json.loads(sse[len("data: ") :].strip())
        self.assertEqual(parsed["choices"][0]["delta"]["content"], "hi")

    def test_build_chunk_sse_role(self):
        handler = self._make_handler()
        sse = handler._build_chunk_sse(request_id="req1", role="assistant", model="m")
        parsed = json.loads(sse[len("data: ") :].strip())
        self.assertEqual(parsed["choices"][0]["delta"]["role"], "assistant")
        self.assertNotIn("content", parsed["choices"][0]["delta"])

    def test_build_chunk_sse_finish_reason(self):
        handler = self._make_handler()
        sse = handler._build_chunk_sse(request_id="req1", finish_reason="stop", model="m")
        parsed = json.loads(sse[len("data: ") :].strip())
        self.assertEqual(parsed["choices"][0]["finish_reason"], "stop")

    def test_chunk_to_sse_string_passthrough(self):
        result = BaseHandler.chunk_to_sse("data: already formatted\n\n")
        self.assertEqual(result, "data: already formatted\n\n")

    def test_chunk_to_sse_wraps_plain_string(self):
        result = BaseHandler.chunk_to_sse("hello")
        self.assertEqual(result, "data: hello\n\n")


@require_serve
@require_serve
class TestAppRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_manager = MagicMock(spec=ModelManager)
        cls.model_manager.get_gen_models.return_value = [
            {"id": "test/model", "owned_by": "test", "object": "model", "created": 0}
        ]
        cls.chat_handler = MagicMock(spec=ChatCompletionHandler)
        cls.completion_handler = MagicMock(spec=CompletionHandler)
        cls.response_handler = MagicMock(spec=ResponseHandler)
        cls.transcription_handler = MagicMock(spec=TranscriptionHandler)
        cls.app = build_server(
            cls.model_manager,
            cls.chat_handler,
            cls.completion_handler,
            cls.response_handler,
            cls.transcription_handler,
            generation_state=GenerationState(),
        )
        cls.transport = httpx.ASGITransport(app=cls.app)

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(transport=self.transport, base_url="http://test") as c:
            return await c.request(method, path, **kwargs)

    def test_health(self):
        resp = asyncio.get_event_loop().run_until_complete(self._request("GET", "/health"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_models_list(self):
        resp = asyncio.get_event_loop().run_until_complete(self._request("GET", "/v1/models"))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 1)

    def test_request_id_generated(self):
        resp = asyncio.get_event_loop().run_until_complete(self._request("GET", "/health"))
        self.assertIn("x-request-id", resp.headers)
        self.assertEqual(len(resp.headers["x-request-id"]), 36)  # UUID length

    def test_request_id_passthrough(self):
        resp = asyncio.get_event_loop().run_until_complete(
            self._request("GET", "/health", headers={"x-request-id": "my-id"})
        )
        self.assertEqual(resp.headers["x-request-id"], "my-id")


@slow
@require_serve
class TestChatCompletion(unittest.TestCase):
    """Integration tests for /v1/chat/completions with a real model."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve()
        cls.base_url = f"http://localhost:{port}"
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    def test_non_streaming(self):
        resp = self.client.chat.completions.create(
            model=self.MODEL, messages=[{"role": "user", "content": "Say hello"}]
        )
        self.assertIsNotNone(resp.choices[0].message.content)
        self.assertIn(resp.choices[0].finish_reason, ("stop", "length"))

    def test_streaming(self):
        text = ""
        for chunk in self.client.chat.completions.create(
            model=self.MODEL, messages=[{"role": "user", "content": "Say hello"}], stream=True
        ):
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        self.assertTrue(len(text) > 0)

    def test_early_return_due_to_length(self):
        """When max_tokens is hit, finish_reason should be 'length'."""
        chunks = list(
            self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                stream=True,
                max_tokens=3,
            )
        )
        last = chunks[-1]
        self.assertEqual(last.choices[0].finish_reason, "length")

    def test_continues_until_stop(self):
        """When model stops naturally, finish_reason should be 'stop'."""
        chunks = list(
            self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": 'Please only answer with "Hi."'}],
                stream=True,
                max_tokens=30,
            )
        )
        last = chunks[-1]
        self.assertEqual(last.choices[0].finish_reason, "stop")

    def test_stop_strings(self):
        resp = self.client.chat.completions.create(
            model=self.MODEL, messages=[{"role": "user", "content": "Count to 10"}], stop=["5"]
        )
        self.assertNotIn("6", resp.choices[0].message.content)

    def test_multi_turn(self):
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Nice to meet you!"},
                {"role": "user", "content": "What is my name?"},
            ],
        )
        self.assertIn("Alice", resp.choices[0].message.content)

    def test_multiple_models_on_demand(self):
        """Load two different models via separate requests — both should work."""
        model_a = "Qwen/Qwen2.5-0.5B-Instruct"
        model_b = "HuggingFaceTB/SmolLM2-135M-Instruct"
        prompt = [{"role": "user", "content": "Say hello"}]

        resp_a = self.client.chat.completions.create(model=model_a, messages=prompt)
        self.assertIn(model_a, resp_a.model)
        self.assertIsNotNone(resp_a.choices[0].message.content)

        resp_b = self.client.chat.completions.create(model=model_b, messages=prompt)
        self.assertIn(model_b, resp_b.model)
        self.assertIsNotNone(resp_b.choices[0].message.content)

    def test_non_streaming_usage(self):
        resp = self.client.chat.completions.create(
            model=self.MODEL, messages=[{"role": "user", "content": "Say hello"}]
        )
        self.assertIsNotNone(resp.usage)
        self.assertGreater(resp.usage.prompt_tokens, 0)
        self.assertGreater(resp.usage.completion_tokens, 0)
        self.assertEqual(resp.usage.total_tokens, resp.usage.prompt_tokens + resp.usage.completion_tokens)

    def test_streaming_usage(self):
        chunks = list(
            self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": "Say hello"}],
                stream=True,
            )
        )
        # Last chunk should have usage
        last = chunks[-1]
        self.assertIsNotNone(last.usage)
        self.assertGreater(last.usage.prompt_tokens, 0)
        self.assertGreater(last.usage.completion_tokens, 0)
        self.assertEqual(last.usage.total_tokens, last.usage.prompt_tokens + last.usage.completion_tokens)

    def test_concurrent_non_streaming(self):
        """Two concurrent non-streaming requests should both complete without interference."""
        import concurrent.futures

        prompts = [
            [{"role": "user", "content": "Say hello"}],
            [{"role": "user", "content": "Say goodbye"}],
        ]
        results = [None, None]

        def request_in_thread(index):
            client = OpenAI(base_url=f"{self.base_url}/v1", api_key="unused")
            results[index] = client.chat.completions.create(model=self.MODEL, messages=prompts[index])

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request_in_thread, i) for i in range(2)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()  # re-raise exceptions

        for i in range(2):
            self.assertIsNotNone(results[i])
            self.assertIsNotNone(results[i].choices[0].message.content)
            self.assertTrue(len(results[i].choices[0].message.content) > 0)

    def test_concurrent_streaming(self):
        """Two concurrent streaming requests should both produce complete, non-empty output."""
        import concurrent.futures

        prompts = [
            [{"role": "user", "content": "Say hello"}],
            [{"role": "user", "content": "Say goodbye"}],
        ]
        results = [None, None]

        def stream_in_thread(index):
            client = OpenAI(base_url=f"{self.base_url}/v1", api_key="unused")
            text = ""
            for chunk in client.chat.completions.create(model=self.MODEL, messages=prompts[index], stream=True):
                if chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
            results[index] = text

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(stream_in_thread, i) for i in range(2)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        for i in range(2):
            self.assertIsNotNone(results[i])
            self.assertTrue(len(results[i]) > 0, f"Request {i} produced empty output")

    def test_request_cancellation(self):
        """Closing a stream early doesn't crash and the server stays healthy."""

        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.MODEL,
                "stream": True,
                "messages": [{"role": "user", "content": "Count slowly so I can cancel you."}],
                "max_tokens": 500,
            },
            timeout=30,
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            chunks_read = 0
            for _ in resp.iter_lines():
                chunks_read += 1
                if chunks_read >= 3:
                    break

        # Server should still be healthy and serve subsequent requests
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=10,
        )
        self.assertIsNotNone(resp.choices[0].message.content)


@slow
@require_serve
class TestCompletion(unittest.TestCase):
    """Integration tests for /v1/completions with a real model.

    Covers sequential and continuous-batching generation, both streaming and
    non-streaming, plus finish_reason, usage, stop strings, suffix, and
    cancellation behaviour.
    """

    MODEL = "Qwen/Qwen2.5-0.5B"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve()
        cls.base_url = f"http://localhost:{port}"
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    # ----- non-streaming -----

    def test_non_streaming(self):
        from openai.types import Completion as OpenAICompletion

        result = self.client.completions.create(
            model=self.MODEL,
            prompt="The capital of France is",
            max_tokens=5,
        )

        self.assertIsInstance(result, OpenAICompletion)
        self.assertEqual(result.object, "text_completion")
        self.assertIsInstance(result.choices[0].text, str)
        self.assertTrue(len(result.choices[0].text) > 0)
        self.assertIn(result.choices[0].finish_reason, ("stop", "length"))

    def test_non_streaming_usage(self):
        result = self.client.completions.create(
            model=self.MODEL,
            prompt="The capital of France is",
            max_tokens=5,
        )
        self.assertIsNotNone(result.usage)
        self.assertGreater(result.usage.prompt_tokens, 0)
        self.assertGreater(result.usage.completion_tokens, 0)
        self.assertEqual(result.usage.total_tokens, result.usage.prompt_tokens + result.usage.completion_tokens)

    def test_finish_reason_length(self):
        result = self.client.completions.create(model=self.MODEL, prompt="Hello", max_tokens=1)
        self.assertEqual(result.choices[0].finish_reason, "length")

    def test_finish_reason_stop(self):
        result = self.client.completions.create(model=self.MODEL, prompt="Hello", max_tokens=1024)
        self.assertEqual(result.choices[0].finish_reason, "stop")

    def test_stop_strings(self):
        result = self.client.completions.create(
            model=self.MODEL,
            prompt="one two three four",
            max_tokens=20,
            stop=["six"],
        )
        self.assertNotIn("seven", result.choices[0].text)

    def test_suffix(self):
        """suffix should be appended to generated text."""
        suffix = " [END]"
        result = self.client.completions.create(
            model=self.MODEL,
            prompt="The capital of France is",
            max_tokens=5,
            suffix=suffix,
        )
        self.assertTrue(result.choices[0].text.endswith(suffix))

    # ----- streaming -----

    def test_streaming(self):
        chunks = list(
            self.client.completions.create(
                model=self.MODEL,
                prompt="The capital of France is",
                max_tokens=5,
                stream=True,
            )
        )
        texts = [c.choices[0].text for c in chunks]
        self.assertTrue(any(t != "" for t in texts))
        self.assertIn(chunks[-1].choices[0].finish_reason, ("stop", "length"))

    def test_streaming_usage(self):
        chunks = list(
            self.client.completions.create(
                model=self.MODEL,
                prompt="The capital of France is",
                max_tokens=5,
                stream=True,
            )
        )
        last = chunks[-1]
        self.assertIsNotNone(last.usage)
        self.assertGreater(last.usage.prompt_tokens, 0)
        self.assertGreater(last.usage.completion_tokens, 0)
        self.assertEqual(last.usage.total_tokens, last.usage.prompt_tokens + last.usage.completion_tokens)

    def test_streaming_finish_reason_length(self):
        chunks = list(self.client.completions.create(model=self.MODEL, prompt="Hello", max_tokens=1, stream=True))
        self.assertEqual(chunks[-1].choices[0].finish_reason, "length")

    def test_streaming_suffix(self):
        """suffix should be emitted as a final text chunk before finish_reason."""
        suffix = " [END]"
        chunks = list(
            self.client.completions.create(
                model=self.MODEL,
                prompt="The capital of France is",
                max_tokens=5,
                suffix=suffix,
                stream=True,
            )
        )
        all_text = "".join(c.choices[0].text for c in chunks)
        self.assertTrue(all_text.endswith(suffix))

    def test_request_cancellation(self):
        """Closing a stream early doesn't crash and the server stays healthy."""
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/completions",
            json={"model": self.MODEL, "prompt": "Count slowly:", "max_tokens": 500, "stream": True},
            timeout=30,
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            chunks_read = 0
            for _ in resp.iter_lines():
                chunks_read += 1
                if chunks_read >= 3:
                    break

        result = self.client.completions.create(model=self.MODEL, prompt="Hello", max_tokens=5)
        self.assertIsNotNone(result.choices[0].text)

    # ----- continuous batching -----

    @require_torch_accelerator
    def test_cb_streaming(self):
        """Streaming completion with CB produces text."""
        serve, port = _start_serve(
            force_model=self.MODEL,
            device="cuda:0",
            continuous_batching=True,
            attn_implementation="sdpa",
        )
        try:
            client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="unused")
            chunks = list(
                client.completions.create(
                    model=self.MODEL, prompt="The capital of France is", max_tokens=10, stream=True
                )
            )
            texts = [c.choices[0].text for c in chunks]
            self.assertTrue(any(t != "" for t in texts))
            self.assertIn(chunks[-1].choices[0].finish_reason, ("stop", "length"))
        finally:
            serve.kill_server()

    @require_torch_accelerator
    def test_cb_non_streaming(self):
        """Non-streaming completion with CB returns a full response."""
        serve, port = _start_serve(
            force_model=self.MODEL,
            device="cuda:0",
            continuous_batching=True,
            attn_implementation="sdpa",
        )
        try:
            client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="unused")
            result = client.completions.create(model=self.MODEL, prompt="The capital of France is", max_tokens=10)
            self.assertIsInstance(result.choices[0].text, str)
            self.assertTrue(len(result.choices[0].text) > 0)
            self.assertIn(result.choices[0].finish_reason, ("stop", "length"))
        finally:
            serve.kill_server()


@require_serve
class TestResponseInputConversion(unittest.TestCase):
    def _make_handler(self):
        return ResponseHandler(model_manager=MagicMock(), generation_state=GenerationState())

    def test_string_input(self):
        handler = self._make_handler()
        msgs = handler._normalize_input({"input": "Hello"})
        self.assertEqual(msgs, [{"role": "user", "content": "Hello"}])

    def test_string_input_with_instructions(self):
        handler = self._make_handler()
        msgs = handler._normalize_input({"input": "Hello", "instructions": "Be brief"})
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0], {"role": "system", "content": "Be brief"})
        self.assertEqual(msgs[1], {"role": "user", "content": "Hello"})

    def test_list_input(self):
        handler = self._make_handler()
        msgs = handler._normalize_input(
            {"input": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]}
        )
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "A")

    def test_list_input_with_instructions_prepends_system(self):
        handler = self._make_handler()
        msgs = handler._normalize_input({"input": [{"role": "user", "content": "Hi"}], "instructions": "Be helpful"})
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "Be helpful")

    def test_list_input_with_instructions_replaces_existing_system(self):
        handler = self._make_handler()
        msgs = handler._normalize_input(
            {"input": [{"role": "system", "content": "Old"}, {"role": "user", "content": "Hi"}], "instructions": "New"}
        )
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "New")

    def test_flat_content_list(self):
        """Flat content list (Responses API native) is wrapped as a single user message."""
        handler = self._make_handler()
        flat_input = [
            {"type": "input_text", "text": "Hello"},
            {"type": "input_image", "image_url": "https://example.com/img.jpg"},
        ]
        msgs = handler._normalize_input({"input": flat_input})
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], flat_input)

    def test_flat_content_list_with_instructions(self):
        """Flat content list with instructions prepends a system message."""
        handler = self._make_handler()
        flat_input = [{"type": "input_text", "text": "Hello"}]
        msgs = handler._normalize_input({"input": flat_input, "instructions": "Be brief"})
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0], {"role": "system", "content": "Be brief"})
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], flat_input)


@require_serve
class TestResponseValidation(unittest.TestCase):
    def _make_handler(self):
        return ResponseHandler(model_manager=MagicMock(), generation_state=GenerationState())

    def test_unsupported_fields_warns(self):
        handler = self._make_handler()
        with self.assertLogs("transformers", level="WARNING") as cm:
            handler._validate_request({"model": "x", "input": "hi", "previous_response_id": "abc"})
        self.assertTrue(any("previous_response_id" in msg for msg in cm.output))

    def test_valid_request_passes(self):
        handler = self._make_handler()
        # Should not raise
        handler._validate_request({"model": "x", "input": "hi"})


@require_serve
class TestResponseGenerationConfig(unittest.TestCase):
    def _make_handler(self):
        return ResponseHandler(model_manager=MagicMock(), generation_state=GenerationState())

    def test_max_output_tokens(self):
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({"max_output_tokens": 42}, GenerationConfig())
        self.assertEqual(result.max_new_tokens, 42)

    def test_default_bumps_short_max_new_tokens(self):
        from transformers import GenerationConfig

        result = self._make_handler()._build_generation_config({}, GenerationConfig(max_new_tokens=20))
        self.assertEqual(result.max_new_tokens, 1024)


@require_serve
class TestResponseUsage(unittest.TestCase):
    def testcompute_usage(self):
        usage = compute_usage(input_tokens=100, output_tokens=50)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.total_tokens, 150)
        self.assertEqual(usage.input_tokens_details.cached_tokens, 0)
        self.assertEqual(usage.output_tokens_details.reasoning_tokens, 0)

    def test_usage_in_completed_response(self):
        """Usage should serialize correctly inside a Response."""

        usage = compute_usage(10, 5)
        response = Response(
            id="resp_test",
            created_at=0,
            status="completed",
            model="test",
            output=[],
            object="response",
            tools=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            usage=usage,
        )
        dumped = response.model_dump(exclude_none=True)
        self.assertEqual(dumped["usage"]["input_tokens"], 10)
        self.assertEqual(dumped["usage"]["output_tokens"], 5)
        self.assertEqual(dumped["usage"]["total_tokens"], 15)


@require_serve
class TestResponseSSEFormat(unittest.TestCase):
    def test_sse_format(self):
        event = ResponseCreatedEvent(
            type="response.created",
            sequence_number=0,
            response=Response(
                id="resp_test",
                created_at=0,
                status="queued",
                model="test",
                text={"format": {"type": "text"}},
                object="response",
                tools=[],
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
            ),
        )
        result = BaseHandler.chunk_to_sse(event)
        self.assertTrue(result.startswith("data: "))
        self.assertTrue(result.endswith("\n\n"))
        parsed = json.loads(result[len("data: ") :].strip())
        self.assertEqual(parsed["type"], "response.created")
        self.assertEqual(parsed["response"]["status"], "queued")


@slow
@require_serve
class TestResponsesIntegration(unittest.TestCase):
    """Integration tests for /v1/responses with a real model."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve()
        cls.base_url = f"http://localhost:{port}"
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    def test_streaming(self):
        events = list(
            self.client.responses.create(
                model=self.MODEL,
                input="Say hello",
                stream=True,
                max_output_tokens=1,
            )
        )
        # At least 8 events: created, in_progress, output_item_added, content_part_added,
        # delta(s), text_done, content_part_done, output_item_done, completed
        self.assertGreaterEqual(len(events), 8)

        # Start markers (fixed order)
        self.assertEqual(events[0].type, "response.created")
        self.assertEqual(events[1].type, "response.in_progress")
        self.assertEqual(events[2].type, "response.output_item.added")
        self.assertEqual(events[3].type, "response.content_part.added")

        # At least one delta
        self.assertTrue(any(e.type == "response.output_text.delta" for e in events[4:-4]))

        # Closing markers (fixed order from the end)
        self.assertEqual(events[-4].type, "response.output_text.done")
        self.assertEqual(events[-3].type, "response.content_part.done")
        self.assertEqual(events[-2].type, "response.output_item.done")
        self.assertEqual(events[-1].type, "response.completed")

    def test_non_streaming(self):
        resp = self.client.responses.create(
            model=self.MODEL,
            input="Say hello",
            stream=False,
        )
        self.assertEqual(resp.status, "completed")
        self.assertTrue(len(resp.output) > 0)
        self.assertTrue(len(resp.output[0].content[0].text) > 0)

    def test_flat_content_list(self):
        """Flat content list input (Responses API native format)."""
        resp = self.client.responses.create(
            model=self.MODEL,
            input=[{"type": "input_text", "text": "Say hello"}],
            stream=False,
        )
        self.assertEqual(resp.status, "completed")
        self.assertTrue(len(resp.output[0].content[0].text) > 0)

    def test_non_streaming_usage(self):
        resp = self.client.responses.create(
            model=self.MODEL,
            input="Say hello",
            stream=False,
        )
        self.assertIsNotNone(resp.usage)
        self.assertGreater(resp.usage.input_tokens, 0)
        self.assertGreater(resp.usage.output_tokens, 0)
        self.assertEqual(resp.usage.total_tokens, resp.usage.input_tokens + resp.usage.output_tokens)

    def test_streaming_usage(self):
        events = list(
            self.client.responses.create(
                model=self.MODEL,
                input="Say hello",
                stream=True,
                max_output_tokens=5,
            )
        )
        completed = events[-1]
        self.assertEqual(completed.type, "response.completed")
        usage = completed.response.usage
        self.assertIsNotNone(usage)
        self.assertGreater(usage.input_tokens, 0)
        self.assertGreater(usage.output_tokens, 0)
        self.assertEqual(usage.total_tokens, usage.input_tokens + usage.output_tokens)

    def test_multi_turn(self):
        """Multi-turn conversation via list input."""
        resp = self.client.responses.create(
            model=self.MODEL,
            input=[
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Nice to meet you!"},
                {"role": "user", "content": "What is my name?"},
            ],
            stream=False,
        )
        self.assertEqual(resp.status, "completed")
        self.assertIn("Alice", resp.output[0].content[0].text)

    def test_concurrent_non_streaming(self):
        """Two concurrent non-streaming responses requests should both complete."""
        import concurrent.futures

        inputs = ["Say hello", "Say goodbye"]
        results = [None, None]

        def request_in_thread(index):
            client = OpenAI(base_url=f"{self.base_url}/v1", api_key="unused")
            results[index] = client.responses.create(model=self.MODEL, input=inputs[index], stream=False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request_in_thread, i) for i in range(2)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        for i in range(2):
            self.assertIsNotNone(results[i])
            self.assertEqual(results[i].status, "completed")
            self.assertTrue(len(results[i].output[0].content[0].text) > 0)

    def test_concurrent_streaming(self):
        """Two concurrent streaming responses requests should both produce complete event streams."""
        import concurrent.futures

        inputs = ["Say hello", "Say goodbye"]
        results = [None, None]

        def stream_in_thread(index):
            client = OpenAI(base_url=f"{self.base_url}/v1", api_key="unused")
            results[index] = list(client.responses.create(model=self.MODEL, input=inputs[index], stream=True))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(stream_in_thread, i) for i in range(2)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        for i in range(2):
            types = [e.type for e in results[i]]
            self.assertIn("response.created", types, f"Request {i} missing created event")
            self.assertIn("response.output_text.delta", types, f"Request {i} missing delta events")
            self.assertIn("response.completed", types, f"Request {i} missing completed event")


def _parse_sse_events(response):
    """Parse SSE lines from a streaming httpx response into a list of dicts."""
    events = []
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        events.append(json.loads(line[len("data: ") :]))
    return events


@slow
@require_serve
class TestLoadModel(unittest.TestCase):
    """Integration tests for POST /load_model SSE endpoint."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve()
        cls.base_url = f"http://localhost:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    def setUp(self):
        # Clear model cache so each test starts fresh
        self.serve.reset_loaded_models()

    def _load_model(self, model: str):
        with httpx.stream("POST", f"{self.base_url}/load_model", json={"model": model}, timeout=120) as resp:
            events = _parse_sse_events(resp)
        return resp, events

    def test_load_model_fresh(self):
        """POST /load_model returns SSE events ending with ready."""
        response, events = self._load_model(self.MODEL)

        self.assertEqual(response.status_code, 200)

        stages = [e["stage"] for e in events if e["status"] == "loading" and "stage" in e]
        self.assertIn("processor", stages)
        self.assertIn("weights", stages)

        last = events[-1]
        self.assertEqual(last["status"], "ready")
        self.assertFalse(last["cached"])

        for event in events:
            self.assertIn("status", event)
            self.assertIn("model", event)

    def test_load_model_cached(self):
        """Loading an already-loaded model returns a single ready event with cached: true."""
        self._load_model(self.MODEL)

        _, events = self._load_model(self.MODEL)

        ready_events = [e for e in events if e["status"] == "ready"]
        self.assertEqual(len(ready_events), 1)
        self.assertTrue(ready_events[0]["cached"])

        loading_events = [e for e in events if e["status"] == "loading"]
        self.assertEqual(len(loading_events), 0)

    def test_load_model_error(self):
        """Loading a nonexistent model produces an error event."""
        _, events = self._load_model("nonexistent/model-that-does-not-exist")

        error_events = [e for e in events if e["status"] == "error"]
        self.assertGreaterEqual(len(error_events), 1)
        self.assertIn("message", error_events[0])

    def test_load_model_missing_field(self):
        """POST /load_model with no model field returns 422."""

        response = httpx.post(f"{self.base_url}/load_model", json={}, timeout=30)
        self.assertEqual(response.status_code, 422)

    def test_load_model_event_schema(self):
        """Every event conforms to the expected schema."""
        _, events = self._load_model(self.MODEL)

        for event in events:
            self.assertIsInstance(event["status"], str)
            self.assertIsInstance(event["model"], str)

            if event["status"] == "loading":
                self.assertIn("stage", event)
                if event["stage"] in ("download", "weights") and "progress" in event:
                    progress = event["progress"]
                    self.assertIn("current", progress)
                    self.assertIn("total", progress)
                    self.assertIsInstance(progress["current"], int)

            if event["status"] == "ready":
                self.assertIn("cached", event)
                self.assertIsInstance(event["cached"], bool)

    def test_load_model_stage_ordering(self):
        """Stages appear in the expected order."""
        _, events = self._load_model(self.MODEL)

        stages = [e["stage"] for e in events if e["status"] == "loading" and "stage" in e]
        seen = set()
        unique_stages = []
        for s in stages:
            if s not in seen:
                seen.add(s)
                unique_stages.append(s)

        expected_order = ["processor", "config", "download", "weights"]
        expected_present = [s for s in expected_order if s in unique_stages]
        self.assertEqual(unique_stages, expected_present, "Stages appeared out of order")

    def test_concurrent_load_same_model(self):
        """Two concurrent /load_model requests both get events and a ready event."""
        import concurrent.futures

        results = [None, None]

        def load_in_thread(index):
            with httpx.stream("POST", f"{self.base_url}/load_model", json={"model": self.MODEL}, timeout=120) as resp:
                events = _parse_sse_events(resp)
                results[index] = (resp.status_code, events)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(load_in_thread, i) for i in range(2)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        for i in range(2):
            status_code, events = results[i]
            self.assertEqual(status_code, 200, f"Caller {i} got non-200 status")
            self.assertTrue(len(events) > 0, f"Caller {i} received no events")
            ready_events = [e for e in events if e["status"] == "ready"]
            self.assertEqual(len(ready_events), 1, f"Caller {i} should get exactly one ready event")

    def test_concurrent_load_second_caller_gets_cached(self):
        """If the first /load_model finishes before the second, the second gets cached: true."""
        _, events1 = self._load_model(self.MODEL)
        ready1 = [e for e in events1 if e["status"] == "ready"]
        self.assertEqual(len(ready1), 1)
        self.assertFalse(ready1[0]["cached"])

        _, events2 = self._load_model(self.MODEL)
        ready2 = [e for e in events2 if e["status"] == "ready"]
        self.assertEqual(len(ready2), 1)
        self.assertTrue(ready2[0]["cached"])

        loading2 = [e for e in events2 if e["status"] == "loading"]
        self.assertEqual(len(loading2), 0)

    def test_load_model_weights_progress_complete(self):
        """Weights progress should go from 1 to total, with total matching across events."""
        _, events = self._load_model(self.MODEL)

        weights_events = [e for e in events if e.get("stage") == "weights" and "progress" in e]
        self.assertGreater(len(weights_events), 0, "No weights progress events emitted")

        # All events should have the same total
        totals = {e["progress"]["total"] for e in weights_events}
        self.assertEqual(len(totals), 1, f"Inconsistent totals: {totals}")
        total = totals.pop()
        self.assertIsNotNone(total)
        self.assertGreater(total, 0)

        # First should be 1, last should be total
        self.assertEqual(weights_events[0]["progress"]["current"], 1)
        self.assertEqual(weights_events[-1]["progress"]["current"], total)

        # Progress should be monotonically increasing
        currents = [e["progress"]["current"] for e in weights_events]
        self.assertEqual(currents, sorted(currents))

    def test_load_model_exactly_one_ready(self):
        """A fresh load should produce exactly one ready event as the last event."""
        _, events = self._load_model(self.MODEL)

        ready_events = [e for e in events if e["status"] == "ready"]
        self.assertEqual(len(ready_events), 1)
        self.assertEqual(events[-1]["status"], "ready")

    def test_load_model_usable_after_load(self):
        """After /load_model completes, the model should be usable for inference."""
        self._load_model(self.MODEL)

        client = OpenAI(base_url=f"{self.base_url}/v1", api_key="unused")
        resp = client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=5,
        )
        self.assertIsNotNone(resp.choices[0].message.content)
        self.assertTrue(len(resp.choices[0].message.content) > 0)

    def test_load_model_model_field_matches(self):
        """The model field in every event should match the canonical model ID."""
        _, events = self._load_model(self.MODEL)

        for event in events:
            self.assertTrue(
                event["model"].startswith(self.MODEL),
                f"Event model '{event['model']}' doesn't match '{self.MODEL}'",
            )

    def test_concurrent_non_streaming(self):
        """Two concurrent non-streaming responses requests should both complete."""
        import concurrent.futures

        inputs = ["Say hello", "Say goodbye"]
        results = [None, None]

        def request_in_thread(index):
            client = OpenAI(base_url=f"{self.base_url}/v1", api_key="unused")
            results[index] = client.responses.create(model=self.MODEL, input=inputs[index], stream=False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request_in_thread, i) for i in range(2)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        for i in range(2):
            self.assertIsNotNone(results[i])
            self.assertEqual(results[i].status, "completed")
            self.assertTrue(len(results[i].output[0].content[0].text) > 0)

    def test_concurrent_streaming(self):
        """Two concurrent streaming responses requests should both produce complete event streams."""
        import concurrent.futures

        inputs = ["Say hello", "Say goodbye"]
        results = [None, None]

        def stream_in_thread(index):
            client = OpenAI(base_url=f"{self.base_url}/v1", api_key="unused")
            events = list(client.responses.create(model=self.MODEL, input=inputs[index], stream=True))
            results[index] = events

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(stream_in_thread, i) for i in range(2)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        for i in range(2):
            types = [e.type for e in results[i]]
            self.assertIn("response.created", types, f"Request {i} missing created event")
            self.assertIn("response.output_text.delta", types, f"Request {i} missing delta events")
            self.assertIn("response.completed", types, f"Request {i} missing completed event")


# Real image URL for VLM tests (person + dog on a beach)
_DOG_IMAGE_URL = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"


@slow
@require_vision
@require_serve
class TestVLM(unittest.TestCase):
    """Integration tests for VLM (vision-language model) support. Requires torchvision."""

    MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve()
        cls.base_url = f"http://localhost:{port}"
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    def test_chat_completion_with_image(self):
        """Chat completions should accept image_url content and produce a meaningful response."""
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {"type": "image_url", "image_url": {"url": _DOG_IMAGE_URL}},
                    ],
                }
            ],
            max_tokens=50,
        )
        text = resp.choices[0].message.content
        self.assertIsNotNone(text)
        self.assertTrue(
            any(word in text.lower() for word in ["dog", "beach", "person"]),
            f"Expected dog/beach/person in response, got: {text}",
        )

    def test_responses_with_image(self):
        """Responses API should accept input_image content and produce a meaningful response."""
        resp = self.client.responses.create(
            model=self.MODEL,
            input=[
                {"type": "input_text", "text": "What do you see in this image?"},
                {"type": "input_image", "image_url": _DOG_IMAGE_URL},
            ],
            stream=False,
            max_output_tokens=50,
        )
        self.assertEqual(resp.status, "completed")
        text = resp.output[0].content[0].text
        self.assertTrue(
            any(word in text.lower() for word in ["dog", "beach", "person"]),
            f"Expected dog/beach/person in response, got: {text}",
        )


_AUDIO_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
)
_VIDEO_URL = "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"


@slow
@require_serve
class TestMultimodalLM(unittest.TestCase):
    """Integration tests for multimodal (audio, video) chat completions with Gemma 4."""

    MODEL = "google/gemma-4-E2B-it"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve()
        cls.base_url = f"http://localhost:{port}"
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    def _get_audio_messages(self):
        import base64

        audio_bytes = httpx.get(_AUDIO_URL, follow_redirects=True).content
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                ],
            }
        ]

    def _get_audio_flat_input(self):
        """Flat content list format for Responses API."""
        import base64

        audio_bytes = httpx.get(_AUDIO_URL, follow_redirects=True).content
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return [
            {"type": "input_text", "text": "Transcribe this audio."},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
        ]

    def _assert_audio_transcription(self, text):
        self.assertIsNotNone(text)
        self.assertIn("chicago", text.lower(), f"Expected 'chicago' in transcription, got: {text}")

    def _assert_video_description(self, text):
        self.assertIsNotNone(text)
        self.assertTrue(
            any(word in text.lower() for word in ["concert", "music", "stage", "perform"]),
            f"Expected concert/music/stage/perform in response, got: {text}",
        )

    def test_chat_completion_with_audio(self):
        """Chat completions should accept input_audio (base64) content and transcribe audio."""
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self._get_audio_messages(),
            max_tokens=200,
        )
        self._assert_audio_transcription(resp.choices[0].message.content)

    def test_chat_completion_with_audio_streaming(self):
        """Streaming chat completions should accept input_audio (base64) content and transcribe audio."""
        stream = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self._get_audio_messages(),
            max_tokens=200,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        self._assert_audio_transcription("".join(chunks))

    @require_torchcodec
    def test_chat_completion_with_video(self):
        """Chat completions should accept video_url content and describe video."""
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": _VIDEO_URL}},
                        {"type": "text", "text": "What is happening in the video?"},
                    ],
                }
            ],
            max_tokens=200,
        )
        self._assert_video_description(resp.choices[0].message.content)

    @require_torchcodec
    def test_chat_completion_with_video_streaming(self):
        """Streaming chat completions should accept video_url content and describe video."""
        stream = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": _VIDEO_URL}},
                        {"type": "text", "text": "What is happening in the video?"},
                    ],
                }
            ],
            max_tokens=200,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        self._assert_video_description("".join(chunks))

    def test_responses_with_audio(self):
        """Responses API should accept input_audio (base64) content and transcribe audio."""
        resp = self.client.responses.create(
            model=self.MODEL,
            input=self._get_audio_flat_input(),
            stream=False,
            max_output_tokens=200,
        )
        self.assertEqual(resp.status, "completed")
        self._assert_audio_transcription(resp.output[0].content[0].text)

    def test_responses_with_audio_streaming(self):
        """Streaming responses API should accept input_audio (base64) content and transcribe audio."""
        stream = self.client.responses.create(
            model=self.MODEL,
            input=self._get_audio_flat_input(),
            stream=True,
            max_output_tokens=200,
        )
        text = ""
        for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
        self._assert_audio_transcription(text)

    @require_torchcodec
    def test_responses_with_video(self):
        """Responses API should accept video_url content and describe video."""
        resp = self.client.responses.create(
            model=self.MODEL,
            input=[
                {"type": "input_text", "text": "What is happening in the video?"},
                {"type": "video_url", "video_url": {"url": _VIDEO_URL}},
            ],
            stream=False,
            max_output_tokens=200,
        )
        self.assertEqual(resp.status, "completed")
        self._assert_video_description(resp.output[0].content[0].text)

    @require_torchcodec
    def test_responses_with_video_streaming(self):
        """Streaming responses API should accept video_url content and describe video."""
        stream = self.client.responses.create(
            model=self.MODEL,
            input=[
                {"type": "input_text", "text": "What is happening in the video?"},
                {"type": "video_url", "video_url": {"url": _VIDEO_URL}},
            ],
            stream=True,
            max_output_tokens=200,
        )
        text = ""
        for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
        self._assert_video_description(text)


class TestToolCallUnit(unittest.TestCase):
    """Unit tests for tool call parsing utilities (no server needed)."""

    def test_get_tool_call_config_fallback(self):
        """Fallback config is returned for known model families (Qwen)."""
        model = MagicMock()
        model.config.model_type = "qwen2"
        processor = MagicMock(spec=["convert_tokens_to_ids"])
        processor.convert_tokens_to_ids.return_value = 151657
        config = get_tool_call_config(processor, model)
        self.assertIsNotNone(config)
        self.assertEqual(config["stc_id"], 151657)
        self.assertEqual(config["etc_id"], 151657)

    def test_get_tool_call_config_unsupported(self):
        """None is returned for models without tool call support."""
        model = MagicMock()
        model.config.model_type = "llama"
        processor = MagicMock(spec=[])
        self.assertIsNone(get_tool_call_config(processor, model))

    def test_parse_tool_calls_from_text(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        processor = MagicMock()
        processor.parse_response = lambda t, s: recursive_parse(t, s)
        calls = parse_tool_calls(processor, text, _TOOL_CALL_FALLBACKS["qwen"]["schema"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")

    def test_parse_multiple_tool_calls_from_text(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'
        )
        processor = MagicMock()
        processor.parse_response = lambda t, s: recursive_parse(t, s)
        calls = parse_tool_calls(processor, text, _TOOL_CALL_FALLBACKS["qwen"]["schema"])
        self.assertEqual(len(calls), 2)


class TestCBWorkerDeadServerIntegration(unittest.TestCase):
    """End-to-end FastAPI behavior when the CB worker has died.

    Asserts the wiring from a ``CBWorkerDeadError`` raised in a request handler
    (or a dead CB worker observed by ``/health``) to a 503 response carrying the cause —
    the contract orchestrators rely on to recycle the pod.
    """

    def _build_app(self, generation_state, chat_handler=None):
        from transformers.cli.serving.server import build_server

        return build_server(
            model_manager=MagicMock(),
            chat_handler=chat_handler or MagicMock(),
            completion_handler=MagicMock(),
            response_handler=MagicMock(),
            transcription_handler=MagicMock(),
            generation_state=generation_state,
        )

    def test_health_returns_503_when_cb_dead(self):
        from fastapi.testclient import TestClient

        from transformers.cli.serving.utils import CBGenerateManager

        state = GenerationState(continuous_batching=True)
        # Manager whose underlying CB has a fatal_error set -> is_alive() returns False.
        mgr = CBGenerateManager()
        mgr._cb = MagicMock()
        mgr._cb.fatal_error = RuntimeError("CUDA illegal memory access")
        state._cb_manager = mgr

        resp = TestClient(self._build_app(state)).get("/health")

        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json(), {"status": "unhealthy", "reason": "cb_worker_dead"})

    def test_chat_endpoint_returns_503_with_cause(self):
        """A CBWorkerDeadError raised from the chat route maps to 503 with the cause in the body."""
        from fastapi.testclient import TestClient

        from transformers.cli.serving.utils import CBWorkerDeadError

        chat_handler = MagicMock()

        async def handle_request(_body, _request_id):
            raise CBWorkerDeadError("CB worker is dead and cannot accept request: CUDA illegal memory access")

        chat_handler.handle_request = handle_request

        client = TestClient(self._build_app(GenerationState(continuous_batching=True), chat_handler=chat_handler))
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

        self.assertEqual(resp.status_code, 503)
        # Body carries the original cause so the client knows why the server is broken.
        self.assertIn("CUDA illegal memory access", resp.json()["error"])


class _TestToolCallBase:
    """Base class for tool call integration tests. Subclasses set MODEL and inherit all tests."""

    MODEL: str

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve()
        cls.base_url = f"http://localhost:{port}"
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    def _get_tool_def(self):
        return {
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                "description": "Get the weather for a city.",
            },
            "type": "function",
        }

    def test_chat_non_streaming(self):
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            stream=False,
            max_tokens=50,
            temperature=0.0,
            tools=[self._get_tool_def()],
        )
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertIsNotNone(resp.choices[0].message.tool_calls)
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "get_weather")
        parsed_args = json.loads(tc.function.arguments)
        self.assertIsInstance(parsed_args, dict)

    def test_chat_streaming(self):
        chunks = list(
            self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": "What is the weather in Paris?"}],
                stream=True,
                max_tokens=50,
                temperature=0.0,
                tools=[self._get_tool_def()],
            )
        )
        tool_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]
        self.assertGreater(len(tool_chunks), 0, "Model did not produce a tool call")
        first_tool = tool_chunks[0].choices[0].delta.tool_calls[0]
        self.assertEqual(first_tool.function.name, "get_weather")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")
        parsed_args = json.loads(first_tool.function.arguments)
        self.assertIsInstance(parsed_args, dict)

    def test_chat_multiple_tool_calls_non_streaming(self):
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "What is the weather in Paris and London?"}],
            stream=False,
            max_tokens=100,
            temperature=0.0,
            tools=[self._get_tool_def()],
        )
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(len(resp.choices[0].message.tool_calls), 2)

    def test_chat_multiple_tool_calls_streaming(self):
        chunks = list(
            self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": "What is the weather in Paris and London?"}],
                stream=True,
                max_tokens=100,
                temperature=0.0,
                tools=[self._get_tool_def()],
            )
        )
        tool_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]
        self.assertEqual(len(tool_chunks), 2, f"Expected 2 tool calls, got {len(tool_chunks)}")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

    def test_chat_multi_turn_non_streaming(self):
        tool_def = self._get_tool_def()
        resp1 = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            stream=False,
            max_tokens=50,
            temperature=0.0,
            tools=[tool_def],
        )
        self.assertEqual(resp1.choices[0].finish_reason, "tool_calls")
        tc = resp1.choices[0].message.tool_calls[0]

        resp2 = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                resp1.choices[0].message,
                {"role": "tool", "tool_call_id": tc.id, "content": '{"temperature": 22, "condition": "sunny"}'},
            ],
            stream=False,
            max_tokens=100,
            temperature=0.0,
            tools=[tool_def],
        )
        self.assertIn(resp2.choices[0].finish_reason, ("stop", "length"))
        content = resp2.choices[0].message.content
        self.assertIsNotNone(content)
        self.assertTrue(
            "22" in content.lower() or "sunny" in content.lower(),
            f"Expected model to reference tool result, got: {content}",
        )

    def test_chat_multi_turn_streaming(self):
        tool_def = self._get_tool_def()

        # Turn 1: streaming — accumulate tool call from deltas
        chunks = list(
            self.client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": "What is the weather in Paris?"}],
                stream=True,
                max_tokens=50,
                temperature=0.0,
                tools=[tool_def],
            )
        )
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")
        tool_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]
        self.assertGreater(len(tool_chunks), 0)
        tc = tool_chunks[0].choices[0].delta.tool_calls[0]

        # Reconstruct assistant message from deltas
        content = "".join(c.choices[0].delta.content for c in chunks if c.choices[0].delta.content)
        assistant_msg = {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"id": tc.id, "type": "function", "function": tc.function.model_dump()}],
        }

        # Turn 2: streaming — send back tool result
        chunks2 = list(
            self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "user", "content": "What is the weather in Paris?"},
                    assistant_msg,
                    {"role": "tool", "tool_call_id": tc.id, "content": '{"temperature": 22, "condition": "sunny"}'},
                ],
                stream=True,
                max_tokens=100,
                temperature=0.0,
                tools=[tool_def],
            )
        )
        content = "".join(c.choices[0].delta.content for c in chunks2 if c.choices[0].delta.content)
        self.assertTrue(
            "22" in content.lower() or "sunny" in content.lower(),
            f"Expected model to reference tool result, got: {content}",
        )

    def test_responses_non_streaming(self):
        resp = self.client.responses.create(
            model=self.MODEL,
            input="What is the weather in Paris?",
            stream=False,
            max_output_tokens=50,
            tools=[self._get_tool_def()],
        )
        self.assertEqual(resp.status, "completed")
        fc_items = [o for o in resp.output if o.type == "function_call"]
        self.assertGreater(len(fc_items), 0)
        self.assertEqual(fc_items[0].name, "get_weather")
        parsed = json.loads(fc_items[0].arguments)
        self.assertIsInstance(parsed, dict)

    def test_responses_streaming(self):
        events = list(
            self.client.responses.create(
                model=self.MODEL,
                input="What is the weather in Paris?",
                stream=True,
                max_output_tokens=50,
                tools=[self._get_tool_def()],
            )
        )
        types = [e.type for e in events]
        self.assertIn("response.created", types)
        self.assertIn("response.completed", types)
        self.assertIn("response.function_call_arguments.done", types)

        args_done = [e for e in events if e.type == "response.function_call_arguments.done"]
        self.assertGreater(len(args_done), 0)
        self.assertEqual(args_done[0].name, "get_weather")
        parsed = json.loads(args_done[0].arguments)
        self.assertIsInstance(parsed, dict)

    def test_responses_multiple_tool_calls_non_streaming(self):
        resp = self.client.responses.create(
            model=self.MODEL,
            input="What is the weather in Paris and London?",
            stream=False,
            max_output_tokens=100,
            tools=[self._get_tool_def()],
        )
        self.assertEqual(resp.status, "completed")
        fc_items = [o for o in resp.output if o.type == "function_call"]
        self.assertEqual(len(fc_items), 2, f"Expected 2 tool calls, got {len(fc_items)}")

    def test_responses_multiple_tool_calls_streaming(self):
        events = list(
            self.client.responses.create(
                model=self.MODEL,
                input="What is the weather in Paris and London?",
                stream=True,
                max_output_tokens=100,
                tools=[self._get_tool_def()],
            )
        )
        args_done = [e for e in events if e.type == "response.function_call_arguments.done"]
        self.assertEqual(len(args_done), 2, f"Expected 2 tool calls, got {len(args_done)}")
        self.assertEqual(events[-1].type, "response.completed")

    def test_responses_multi_turn_non_streaming(self):
        tool_def = self._get_tool_def()
        resp1 = self.client.responses.create(
            model=self.MODEL,
            input="What is the weather in Paris?",
            stream=False,
            max_output_tokens=50,
            tools=[tool_def],
        )
        self.assertEqual(resp1.status, "completed")
        fc_items = [o for o in resp1.output if o.type == "function_call"]
        self.assertGreater(len(fc_items), 0)

        input_list = [{"role": "user", "content": "What is the weather in Paris?"}]
        input_list += resp1.output
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": fc_items[0].call_id,
                "output": '{"temperature": 22, "condition": "sunny"}',
            }
        )
        resp2 = self.client.responses.create(
            model=self.MODEL,
            input=input_list,
            stream=False,
            max_output_tokens=100,
            tools=[tool_def],
        )
        self.assertEqual(resp2.status, "completed")
        msg_items = [o for o in resp2.output if o.type == "message"]
        self.assertGreater(len(msg_items), 0)
        content = msg_items[0].content[0].text
        self.assertTrue(
            "22" in content.lower() or "sunny" in content.lower(),
            f"Expected model to reference tool result, got: {content}",
        )

    def test_responses_multi_turn_streaming(self):
        tool_def = self._get_tool_def()

        # Turn 1: streaming — get completed response with tool calls
        events = list(
            self.client.responses.create(
                model=self.MODEL,
                input="What is the weather in Paris?",
                stream=True,
                max_output_tokens=50,
                tools=[tool_def],
            )
        )
        completed = [e for e in events if e.type == "response.completed"]
        self.assertEqual(len(completed), 1)
        resp1_output = completed[0].response.output
        fc_items = [o for o in resp1_output if o.type == "function_call"]
        self.assertGreater(len(fc_items), 0)

        # Turn 2: streaming — send back tool result
        input_list = [{"role": "user", "content": "What is the weather in Paris?"}]
        input_list += resp1_output
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": fc_items[0].call_id,
                "output": '{"temperature": 22, "condition": "sunny"}',
            }
        )
        events2 = list(
            self.client.responses.create(
                model=self.MODEL,
                input=input_list,
                stream=True,
                max_output_tokens=100,
                tools=[tool_def],
            )
        )
        content = "".join(e.delta for e in events2 if e.type == "response.output_text.delta")
        self.assertTrue(
            "22" in content.lower() or "sunny" in content.lower(),
            f"Expected model to reference tool result, got: {content}",
        )


@slow
@require_serve
@require_torch_accelerator
class TestToolCallQwen(_TestToolCallBase, unittest.TestCase):
    """Tool call tests with Qwen (fallback config, no response_schema)."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@slow
@require_serve
@require_torch_accelerator
class TestToolCallGemma(_TestToolCallBase, unittest.TestCase):
    """Tool call tests with Gemma 4 (response_schema + stc/etc special tokens)."""

    MODEL = "google/gemma-4-E2B-it"


@slow
@require_librosa
@require_multipart
@require_serve
class TestTranscription(unittest.TestCase):
    """Integration tests for POST /v1/audio/transcriptions with whisper-tiny."""

    MODEL = "openai/whisper-tiny"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve()
        cls.base_url = f"http://localhost:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    @classmethod
    def _get_audio_bytes(cls):
        """Download the MLK 'I have a dream' speech sample from HF Hub."""
        if not hasattr(cls, "_audio_bytes"):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download("Narsil/asr_dummy", "mlk.flac", repo_type="dataset")
            with open(path, "rb") as f:
                cls._audio_bytes = f.read()
        return cls._audio_bytes

    def test_transcription_returns_text(self):
        """POST /v1/audio/transcriptions with real speech returns meaningful transcription."""

        audio_bytes = self._get_audio_bytes()
        resp = httpx.post(
            f"{self.base_url}/v1/audio/transcriptions",
            files={"file": ("mlk.flac", audio_bytes, "audio/flac")},
            data={"model": self.MODEL},
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("text", data)
        self.assertIsInstance(data["text"], str)
        # Whisper-tiny should recognize at least "dream" from the MLK speech
        self.assertIn("dream", data["text"].lower())

    def test_transcription_openai_client(self):
        """Transcription should work via the OpenAI Python client."""
        audio_bytes = self._get_audio_bytes()
        client = OpenAI(base_url=f"{self.base_url}/v1", api_key="unused")
        result = client.audio.transcriptions.create(
            model=self.MODEL,
            file=("mlk.flac", audio_bytes),
        )
        self.assertIsInstance(result.text, str)
        self.assertTrue(len(result.text) > 10)

    def test_transcription_streaming(self):
        """Streaming transcription should yield text chunks via SSE."""

        audio_bytes = self._get_audio_bytes()
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/audio/transcriptions",
            files={"file": ("mlk.flac", audio_bytes, "audio/flac")},
            data={"model": self.MODEL, "stream": "true"},
            timeout=120,
        ) as resp:
            self.assertEqual(resp.status_code, 200)

            chunks = []
            for line in resp.iter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line[len("data: ") :])

        self.assertGreater(len(chunks), 0, "No streaming chunks received")
        full_text = "".join(chunks)
        self.assertIn("dream", full_text.lower())

    def test_transcription_missing_file(self):
        """POST without a file should fail."""

        resp = httpx.post(
            f"{self.base_url}/v1/audio/transcriptions",
            data={"model": self.MODEL},
            timeout=30,
        )
        self.assertNotEqual(resp.status_code, 200)


@slow
@require_serve
@require_torch_accelerator
class TestContinuousBatchingChatCompletion(unittest.TestCase):
    """Integration tests for /v1/chat/completions with continuous batching."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve(
            force_model=cls.MODEL,
            device="cuda:0",
            continuous_batching=True,
            attn_implementation="sdpa",
            default_seed=42,
        )
        cls.base_url = f"http://localhost:{port}"
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    def test_streaming(self):
        """Streaming chat completion with CB produces text."""
        text = ""
        for chunk in self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": "You are a sports assistant designed to craft sports programs."},
                {"role": "user", "content": "Tell me what you can do."},
            ],
            stream=True,
            max_tokens=30,
        ):
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        self.assertTrue(len(text) > 0)

    def test_non_streaming(self):
        """Non-streaming chat completion with CB returns a full response."""
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=20,
        )
        self.assertIsNotNone(resp.choices[0].message.content)
        self.assertTrue(len(resp.choices[0].message.content) > 0)

    def test_non_streaming_response_json_format(self):
        """Non-streaming CB responses return proper JSON objects, not double-encoded strings."""
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            stream=False,
            max_tokens=5,
        )
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.id)
        self.assertIsNotNone(response.choices)
        self.assertEqual(len(response.choices), 1)

        choice = response.choices[0]
        self.assertIsNotNone(choice.message)
        self.assertIsNotNone(choice.message.content)
        self.assertEqual(choice.message.role, "assistant")
        self.assertIsInstance(choice.message.content, str)

    def test_multi_turn(self):
        """Multi-turn conversation works with CB."""
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Nice to meet you!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=20,
        )
        self.assertIn("Alice", resp.choices[0].message.content)

    def test_request_cancellation(self):
        """Opening a stream and closing it early triggers CB cancellation."""

        request_id = "test-cb-cancel"

        # Open a streaming request and close after a few chunks
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            headers={"X-Request-ID": request_id},
            json={
                "model": self.MODEL,
                "stream": True,
                "messages": [{"role": "user", "content": "Count slowly so I can cancel you."}],
            },
            timeout=30,
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            chunks_read = 0
            for _ in resp.iter_lines():
                chunks_read += 1
                if chunks_read >= 3:
                    break

        # Poll for cancellation in the CB scheduler
        scheduler = self.serve._generation_state._cb_manager.scheduler
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if scheduler.request_is_cancelled(request_id):
                break
            time.sleep(0.1)

        self.assertTrue(
            scheduler.request_is_cancelled(request_id),
            f"Request {request_id} not cancelled in scheduler after stream close.",
        )

        # Server should still be healthy and serve subsequent requests
        resp = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=10,
        )
        self.assertIsNotNone(resp.choices[0].message.content)


@slow
@require_serve
@require_torch_accelerator
class TestContinuousBatchingResponses(unittest.TestCase):
    """Integration tests for /v1/responses with continuous batching."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.serve, port = _start_serve(
            force_model=cls.MODEL,
            device="cuda:0",
            continuous_batching=True,
            attn_implementation="sdpa",
            default_seed=42,
        )
        cls.base_url = f"http://localhost:{port}"
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key="unused")

    @classmethod
    def tearDownClass(cls):
        cls.serve.kill_server()

    def test_streaming(self):
        """Streaming response with CB produces text."""
        text = ""
        stream = self.client.responses.create(
            model=self.MODEL,
            input="Say hello in one sentence.",
            stream=True,
            max_output_tokens=30,
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
        self.assertTrue(len(text) > 0)

    def test_non_streaming(self):
        """Non-streaming response with CB returns text."""
        resp = self.client.responses.create(
            model=self.MODEL,
            input="Say hello in one sentence.",
            stream=False,
            max_output_tokens=30,
        )
        content = resp.output[0].content[0].text
        self.assertTrue(len(content) > 0)

    def test_multi_turn(self):
        """Multi-turn conversation works with CB via Responses API."""
        resp = self.client.responses.create(
            model=self.MODEL,
            input=[
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Nice to meet you!"},
                {"role": "user", "content": "What is my name?"},
            ],
            stream=False,
            max_output_tokens=20,
        )
        content = resp.output[0].content[0].text
        self.assertIn("Alice", content)

    def test_request_cancellation(self):
        """Opening a stream and closing it early triggers CB cancellation."""

        request_id = "test-cb-resp-cancel"

        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/responses",
            headers={"X-Request-ID": request_id},
            json={
                "model": self.MODEL,
                "stream": True,
                "input": "Count slowly so I can cancel you.",
                "max_output_tokens": 500,
            },
            timeout=30,
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            # Read enough data to ensure CB generation has started, then close.
            received = b""
            for chunk in resp.iter_bytes(chunk_size=512):
                received += chunk
                if b"output_text.delta" in received:
                    break

        # Poll for cancellation in the CB scheduler
        scheduler = self.serve._generation_state._cb_manager.scheduler
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if scheduler.request_is_cancelled(request_id):
                break
            time.sleep(0.1)

        self.assertTrue(
            scheduler.request_is_cancelled(request_id),
            f"Request {request_id} not cancelled in scheduler after stream close.",
        )

        # Server should still serve subsequent requests
        resp = self.client.responses.create(
            model=self.MODEL,
            input="Say hi",
            stream=False,
            max_output_tokens=10,
        )
        self.assertTrue(len(resp.output[0].content[0].text) > 0)
