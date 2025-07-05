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
import asyncio
import time
import unittest
from threading import Thread
from unittest.mock import patch

import aiohttp.client_exceptions
from huggingface_hub import AsyncInferenceClient
from parameterized import parameterized

import transformers.commands.transformers_cli as cli
from transformers import GenerationConfig
from transformers.commands.serving import ServeArguments, ServeCommand
from transformers.testing_utils import CaptureStd


class ServeCLITest(unittest.TestCase):
    def test_help(self):
        """Minimal test: we can invoke the help command."""
        with patch("sys.argv", ["transformers", "serve", "--help"]), CaptureStd() as cs:
            with self.assertRaises(SystemExit):
                cli.main()
        self.assertIn("serve", cs.out.lower())

    def test_parsed_args(self):
        """Minimal test: we can set arguments through the CLI."""
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

    def test_completions_build_chunk(self):
        """Tests that the chunks are correctly built for the Completions API."""
        dummy = ServeCommand.__new__(ServeCommand)
        dummy.args = type("Args", (), {})()

        # Case 1: most fields are provided
        chunk = ServeCommand.build_chunk(dummy, request_id="req0", content="hello", finish_reason="stop", role="user")
        self.assertIn("chat.completion.chunk", chunk)
        self.assertIn("data:", chunk)
        self.assertIn(
            '"choices": [{"delta": {"content": "hello", "role": "user"}, "index": 0, "finish_reason": "stop"}]', chunk
        )

        # Case 2: only the role is provided -- other fields in 'choices' are omitted
        chunk = ServeCommand.build_chunk(dummy, request_id="req0", role="user")
        self.assertIn("chat.completion.chunk", chunk)
        self.assertIn("data:", chunk)
        self.assertIn('"choices": [{"delta": {"role": "user"}, "index": 0}]', chunk)

        # Case 3: only the content is provided -- other fields in 'choices' are omitted
        chunk = ServeCommand.build_chunk(dummy, request_id="req0", content="hello")
        self.assertIn("chat.completion.chunk", chunk)
        self.assertIn("data:", chunk)
        self.assertIn('"choices": [{"delta": {"content": "hello"}, "index": 0}]', chunk)

        # Case 4: tool calls support a list of nested dictionaries
        chunk = ServeCommand.build_chunk(dummy, request_id="req0", tool_calls=[{"foo1": "bar1", "foo2": "bar2"}])
        self.assertIn("chat.completion.chunk", chunk)
        self.assertIn("data:", chunk)
        self.assertIn('"choices": [{"delta": {"tool_calls": [{"foo1": "bar1", "foo2": "bar2"}]}, "index": 0}]', chunk)


def async_retry(fn, max_attempts=5, delay=1):
    """
    Retry a function up to `max_attempts` times with a `delay` between attempts.
    Useful for testing async functions that may fail due to server not being ready.
    """

    async def wrapper(*args, **kwargs):
        for _ in range(max_attempts):
            try:
                return await fn(*args, **kwargs)
            except aiohttp.client_exceptions.ClientConnectorError:
                time.sleep(delay)

    return wrapper


class ServeCompletionsMixin:
    """
    Mixin class for the Completions API tests, to seamlessly replicate tests across the two versions of the API
    (`generate` and `continuous_batching`).
    """

    @async_retry
    async def run_server(self, request):
        client = AsyncInferenceClient("http://localhost:8000")
        stream = client.chat_completion(**request)

        all_payloads = []
        async for payload in await stream:
            all_payloads.append(payload)

        await client.close()
        return all_payloads

    @parameterized.expand(
        [
            ("default_request", {}),
            ("one_token", {"max_tokens": 1}),
            ("different_model", {"model": "HuggingFaceTB/SmolLM2-135M-Instruct"}),
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
        all_payloads = asyncio.run(self.run_server(request))

        # If a request is successful, the returned payload needs to follow the schema, which we test here.
        # NOTE: the output of our server is wrapped by `AsyncInferenceClient`, which sends fields even when they
        # are empty.

        # Finish reason: the last payload should have a finish reason of "stop", all others should be empty
        # TODO: we may add other finish reasons in the future, and this may need more logic
        finish_reasons = [payload.choices[0].finish_reason for payload in all_payloads]
        self.assertEqual(finish_reasons[-1], "stop")
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
        # TODO: tool tests
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
        all_payloads = asyncio.run(self.run_server(request))
        contents = [payload.choices[0].delta.content for payload in all_payloads]
        output_text = "".join([text for text in contents if text is not None])
        # The generation config sets greedy decoding, so the output is reproducible. By default, `Qwen/Qwen3-0.6B`
        # sets `do_sample=True`
        self.assertEqual(output_text, '<think>\nOkay, the user just asked, "')

    # TODO: implement API-compliant error handling, and then test it
    # See https://platform.openai.com/docs/guides/error-codes,
    # TODO: one test for each request flag, to confirm it is working as expected
    # TODO: speed-based test to confirm that KV cache is working across requests


class ServeCompletionsGenerateTest(ServeCompletionsMixin, unittest.TestCase):
    """Tests the `generate` version of the Completions API."""

    @classmethod
    def setUpClass(cls):
        """Starts a server for tests to connect to."""
        args = ServeArguments()
        serve_command = ServeCommand(args)
        thread = Thread(target=serve_command.run)
        thread.daemon = True
        thread.start()


class ServeCompletionsContinuousBatchingTest(ServeCompletionsMixin, unittest.TestCase):
    """Tests the `continuous_batching` version of the Completions API."""

    @classmethod
    def setUpClass(cls):
        """Starts a server for tests to connect to."""
        args = ServeArguments(attn_implementation="sdpa_paged")  # important: toggle continuous batching
        serve_command = ServeCommand(args)
        thread = Thread(target=serve_command.run)
        thread.daemon = True
        thread.start()
