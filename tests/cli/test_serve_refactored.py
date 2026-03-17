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
Tests for the refactored serving layer (Phase 1: chat completions).

Run: pytest tests/cli/test_serve_refactored.py -x -v
Integration tests (need GPU): RUN_SLOW=1 pytest tests/cli/test_serve_refactored.py -x -v -k "Integration"
"""

import asyncio
import json
import os
import time
import unittest
from unittest.mock import MagicMock

from transformers.testing_utils import require_openai, slow
from transformers.utils.import_utils import is_openai_available, is_vision_available


if is_openai_available():
    from openai import OpenAI

run_slow = os.environ.get("RUN_SLOW", "0") == "1"


# ---------------------------------------------------------------------------
# 1. CLI tests — verify CLI args reach uvicorn
# ---------------------------------------------------------------------------


@require_openai
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


# ---------------------------------------------------------------------------
# 2. Unit tests — message parsing
# ---------------------------------------------------------------------------


class TestProcessorInputsFromMessages(unittest.TestCase):
    def test_llm_string_content(self):
        from transformers.cli.serving.utils import Modality, get_processor_inputs_from_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = get_processor_inputs_from_messages(messages, Modality.LLM)
        self.assertEqual(result, [{"role": "user", "content": "Hello"}])

    def test_llm_list_content_text_only(self):
        from transformers.cli.serving.utils import Modality, get_processor_inputs_from_messages

        messages = [{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}]
        result = get_processor_inputs_from_messages(messages, Modality.LLM)
        self.assertEqual(result, [{"role": "user", "content": "A B"}])

    def test_vlm_string_content_wrapped(self):
        from transformers.cli.serving.utils import Modality, get_processor_inputs_from_messages

        messages = [{"role": "user", "content": "Hello"}]
        result = get_processor_inputs_from_messages(messages, Modality.VLM)
        self.assertEqual(result, [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])

    def test_vlm_text_and_image_url(self):
        from transformers.cli.serving.utils import Modality, get_processor_inputs_from_messages

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
        from transformers.cli.serving.utils import Modality, get_processor_inputs_from_messages

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
        from transformers.cli.serving.utils import Modality, get_processor_inputs_from_messages

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]}
        ]
        result = get_processor_inputs_from_messages(messages, Modality.LLM)
        self.assertEqual(result[0]["content"], "Hello world")

    @unittest.skipUnless(is_vision_available(), "Requires PIL")
    def test_vlm_base64_image_creates_temp_file(self):
        """Base64 image URLs should be decoded and saved to a temp file."""
        import os

        from transformers.cli.serving.utils import Modality, get_processor_inputs_from_messages

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
        self.assertTrue(os.path.exists(image_item["url"]))  # temp file was created

    def test_vlm_multi_turn(self):
        """VLM multi-turn: string content should be wrapped in text type."""
        from transformers.cli.serving.utils import Modality, get_processor_inputs_from_messages

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


class TestGenerativeModelList(unittest.TestCase):
    def test_lists_only_generative_models(self):
        """Should list LLMs and VLMs but not non-generative models like BERT."""
        import tempfile

        from huggingface_hub import hf_hub_download

        from transformers.cli.serving.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as cache_dir:
            # Download config.json for a few models
            hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "config.json", cache_dir=cache_dir)
            hf_hub_download("google-bert/bert-base-cased", "config.json", cache_dir=cache_dir)

            result = ModelManager.get_gen_models(cache_dir)
            model_ids = {r["id"] for r in result}

            self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", model_ids)
            self.assertNotIn("google-bert/bert-base-cased", model_ids)


# ---------------------------------------------------------------------------
# 2. Unit tests — generation config mapping
# ---------------------------------------------------------------------------


@require_openai
class TestBuildGenerationConfig(unittest.TestCase):
    def _make_handler(self):
        from transformers.cli.serving.chat_completion import ChatCompletionHandler

        return ChatCompletionHandler(model_manager=MagicMock())

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


# ---------------------------------------------------------------------------
# 3. Unit tests — validation
# ---------------------------------------------------------------------------


@require_openai
class TestValidation(unittest.TestCase):
    def _make_handler(self):
        from transformers.cli.serving.chat_completion import ChatCompletionHandler

        return ChatCompletionHandler(model_manager=MagicMock())

    def test_valid_request_passes(self):
        from fastapi import HTTPException

        handler = self._make_handler()
        # Should not raise
        handler._validate_request({"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True})

    def test_unexpected_keys_rejected(self):
        from fastapi import HTTPException

        handler = self._make_handler()
        with self.assertRaises(HTTPException) as ctx:
            handler._validate_request({"model": "x", "messages": [], "bogus_field": True})
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("bogus_field", ctx.exception.detail)

    def test_unsupported_fields_rejected(self):
        from fastapi import HTTPException

        handler = self._make_handler()
        with self.assertRaises(HTTPException) as ctx:
            handler._validate_request({"model": "x", "messages": [], "audio": {}})
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("audio", ctx.exception.detail)


# ---------------------------------------------------------------------------
# 4. Unit tests — model manager
# ---------------------------------------------------------------------------


class TestModelManager(unittest.TestCase):
    def test_process_model_name_adds_main(self):
        from transformers.cli.serving.model_manager import ModelManager

        self.assertEqual(ModelManager.process_model_name("org/model"), "org/model@main")

    def test_process_model_name_preserves_revision(self):
        from transformers.cli.serving.model_manager import ModelManager

        self.assertEqual(ModelManager.process_model_name("org/model@dev"), "org/model@dev")

    def test_quantization_config_4bit(self):
        from transformers.cli.serving.model_manager import ModelManager

        mm = ModelManager(quantization="bnb-4bit")
        cfg = mm.get_quantization_config()
        self.assertTrue(cfg.load_in_4bit)

    def test_quantization_config_8bit(self):
        from transformers.cli.serving.model_manager import ModelManager

        mm = ModelManager(quantization="bnb-8bit")
        cfg = mm.get_quantization_config()
        self.assertTrue(cfg.load_in_8bit)

    def test_quantization_config_none(self):
        from transformers.cli.serving.model_manager import ModelManager

        mm = ModelManager()
        self.assertIsNone(mm.get_quantization_config())


class TestTimedModel(unittest.TestCase):
    def test_delete_model(self):
        from transformers.cli.serving.model_manager import TimedModel

        mock_model = MagicMock()
        timed = TimedModel(mock_model, timeout_seconds=9999, processor=MagicMock())
        self.assertFalse(timed.is_deleted())
        timed.delete_model()
        self.assertTrue(timed.is_deleted())

    def test_timeout_zero_no_delete(self):
        from transformers.cli.serving.model_manager import TimedModel

        mock_model = MagicMock()
        timed = TimedModel(mock_model, timeout_seconds=0, processor=MagicMock())
        timed._timeout_reached()
        self.assertFalse(timed.is_deleted())
        timed._timer.cancel()


# ---------------------------------------------------------------------------
# 5. Unit tests — SSE formatting
# ---------------------------------------------------------------------------


@require_openai
class TestChunkSSE(unittest.TestCase):
    def _make_handler(self):
        from transformers.cli.serving.chat_completion import ChatCompletionHandler

        return ChatCompletionHandler(model_manager=MagicMock())

    def test_build_chunk_sse_content(self):
        handler = self._make_handler()
        sse = handler._build_chunk_sse(request_id="req1", content="hi", model="m")
        self.assertTrue(sse.startswith("data: "))
        self.assertTrue(sse.endswith("\n\n"))
        parsed = json.loads(sse[len("data: "):].strip())
        self.assertEqual(parsed["choices"][0]["delta"]["content"], "hi")

    def test_build_chunk_sse_role(self):
        handler = self._make_handler()
        sse = handler._build_chunk_sse(request_id="req1", role="assistant", model="m")
        parsed = json.loads(sse[len("data: "):].strip())
        self.assertEqual(parsed["choices"][0]["delta"]["role"], "assistant")
        self.assertNotIn("content", parsed["choices"][0]["delta"])

    def test_build_chunk_sse_finish_reason(self):
        handler = self._make_handler()
        sse = handler._build_chunk_sse(request_id="req1", finish_reason="stop", model="m")
        parsed = json.loads(sse[len("data: "):].strip())
        self.assertEqual(parsed["choices"][0]["finish_reason"], "stop")

    def test_chunk_to_sse_string_passthrough(self):
        handler = self._make_handler()
        result = handler._chunk_to_sse("data: already formatted\n\n")
        self.assertEqual(result, "data: already formatted\n\n")

    def test_chunk_to_sse_wraps_plain_string(self):
        handler = self._make_handler()
        result = handler._chunk_to_sse("hello")
        self.assertEqual(result, "data: hello\n\n")


# ---------------------------------------------------------------------------
# 6. App-level tests with ASGI test client (no real model)
# ---------------------------------------------------------------------------


@require_openai
class TestAppRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers.cli.serving.chat_completion import ChatCompletionHandler
        from transformers.cli.serving.model_manager import ModelManager
        from transformers.cli.serving.server import build_server

        cls.model_manager = MagicMock(spec=ModelManager)
        cls.model_manager.get_gen_models.return_value = [
            {"id": "test/model", "owned_by": "test", "object": "model", "created": 0}
        ]
        cls.chat_handler = MagicMock(spec=ChatCompletionHandler)
        cls.app = build_server(cls.model_manager, cls.chat_handler)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_health(self):
        from httpx import ASGITransport, AsyncClient

        async def _test():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test") as c:
                resp = await c.get("/health")
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json(), {"status": "ok"})

        self._run(_test())

    def test_models_list(self):
        from httpx import ASGITransport, AsyncClient

        async def _test():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test") as c:
                resp = await c.get("/v1/models")
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["object"], "list")
                self.assertEqual(len(data["data"]), 1)

        self._run(_test())

    def test_request_id_generated(self):
        from httpx import ASGITransport, AsyncClient

        async def _test():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test") as c:
                resp = await c.get("/health")
                self.assertIn("x-request-id", resp.headers)
                self.assertEqual(len(resp.headers["x-request-id"]), 36)  # UUID length

        self._run(_test())

    def test_request_id_passthrough(self):
        from httpx import ASGITransport, AsyncClient

        async def _test():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test") as c:
                resp = await c.get("/health", headers={"x-request-id": "my-id"})
                self.assertEqual(resp.headers["x-request-id"], "my-id")

        self._run(_test())


# ---------------------------------------------------------------------------
# 7. Integration tests (need GPU + model)
#    Only test what requires a real model. Everything else is above with mocks.
# ---------------------------------------------------------------------------


@unittest.skipUnless(run_slow, "Set RUN_SLOW=1 to run integration tests")
@require_openai
class TestChatCompletion(unittest.TestCase):
    """Integration tests for /v1/chat/completions with a real model."""

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    PORT = 8877

    @classmethod
    def setUpClass(cls):
        from transformers.cli.serve_refactored import Serve

        cls.serve = Serve(port=cls.PORT, non_blocking=True, log_level="warning")
        import requests

        for _ in range(30):
            try:
                if requests.get(f"http://localhost:{cls.PORT}/health", timeout=1).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)

        cls.client = OpenAI(base_url=f"http://localhost:{cls.PORT}/v1", api_key="unused")

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
