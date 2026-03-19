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
Shared types, constants, and utilities for the serving layer.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import enum
import json
import re
import tempfile
import threading
from concurrent.futures import Future
from io import BytesIO
from queue import Queue

from transformers.utils.import_utils import is_openai_available, is_vision_available


if is_vision_available():
    from PIL import Image

if is_openai_available():
    from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

    class TransformersCompletionCreateParamsStreaming(CompletionCreateParamsStreaming, total=False):
        generation_config: str
        processor: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

X_REQUEST_ID = "x-request-id"

# Fields accepted by the OpenAI schema but not yet supported.
# Receiving these raises an error to avoid silent misbehaviour.
# NOTE: "stop" is NOT in this set — we map it to stop_strings.
UNUSED_CHAT_COMPLETION_FIELDS = {
    "audio",
    "function_call",
    "functions",
    "logprobs",
    "max_completion_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "service_tier",
    "store",
    "stream_options",
    "tool_choice",
    "top_logprobs",
    "user",
    "web_search_options",
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class Modality(enum.Enum):
    LLM = "LLM"
    VLM = "VLM"
    STT = "STT"
    TTS = "TTS"


class _StreamError:
    """Sentinel to signal an error from the generate thread."""

    def __init__(self, msg: str):
        self.msg = msg


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class DirectStreamer:
    """Streamer that decodes tokens incrementally and pushes text to an asyncio.Queue.

    Uses the Rust ``DecodeStream.step()`` for O(1) per-token decode, unlike
    ``TextIteratorStreamer`` which re-decodes the full sequence each time.

    Args:
        processor: A HuggingFace processor or tokenizer (must have a ``._tokenizer`` attribute).
        loop: The asyncio event loop to push results to.
        queue: The asyncio.Queue to push decoded text chunks to.
        skip_special_tokens: Whether to skip special tokens during decoding.
    """

    def __init__(self, processor, loop, queue, skip_special_tokens: bool = True):
        from tokenizers.decoders import DecodeStream

        self._tokenizer = processor._tokenizer  # raw tokenizers.Tokenizer
        self._loop = loop
        self._queue = queue
        self._decode_stream = DecodeStream([], skip_special_tokens)
        self._first = True
        self.total_tokens = 0

    def put(self, value) -> None:
        if self._first:
            self._first = False
            return  # skip prompt tokens
        if len(value.shape) > 1:
            value = value[0]
        for token_id in value.tolist():
            self.total_tokens += 1
            text = self._decode_stream.step(self._tokenizer, token_id)
            if text is not None:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, text)

    def end(self) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)


# ---------------------------------------------------------------------------
# Torch helpers
# ---------------------------------------------------------------------------


def set_torch_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def reset_torch_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class InferenceThread:
    """A single persistent thread that runs all model.generate() calls.

    torch.compile with ``mode="reduce-overhead"`` uses CUDA graphs, which store
    state in thread-local storage (TLS). If generate() is called from different
    threads (e.g. a new Thread per streaming request), the CUDA graph state is
    lost or corrupted — causing silent wrong output or crashes.

    This class ensures all inference runs on the **same thread**, matching what
    vLLM does with its engine loop. Both streaming and non-streaming requests
    submit work here.
    """

    def __init__(self):
        self._queue: Queue = Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            fn, args, kwargs, future = self._queue.get()
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    def submit(self, fn, *args, **kwargs) -> Future:
        """Submit a callable to run on the inference thread. Returns a Future."""
        future: Future = Future()
        self._queue.put((fn, args, kwargs, future))
        return future


# ---------------------------------------------------------------------------
# Base handler
# ---------------------------------------------------------------------------


class BaseHandler:
    """Shared logic for chat completion and responses handlers.

    Subclasses implement ``_streaming`` and ``_non_streaming`` for their
    specific SSE / JSON formats, plus ``_validate_request``.
    """

    def __init__(
        self,
        model_manager,
        force_model=None,
        force_processor=None,
        inference_thread=None,
        compile=False,
    ):
        self.model_manager = model_manager
        self.force_model = force_model
        self.force_processor = force_processor
        self._inference_thread = inference_thread or InferenceThread()
        self._compile = compile

    @staticmethod
    def chunk_to_sse(chunk) -> str:
        """Format a pydantic model or string as an SSE ``data:`` line."""
        if isinstance(chunk, str):
            return chunk if chunk.startswith("data: ") else f"data: {chunk}\n\n"
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def _resolve_model(self, body: dict):
        """Apply force_model, load model + processor. Returns (model_id, model, processor)."""
        if self.force_model is not None:
            body["model"] = self.force_model

        model_id = self.model_manager.process_model_name(body["model"])
        processor_id = self.force_processor or body.get("processor")
        model, processor = self.model_manager.load_model_and_processor(model_id, processor_id=processor_id)

        return model_id, model, processor

    def _build_generation_config(self, body: dict, model_generation_config, processor=None):
        """Build a GenerationConfig from shared params (temperature, top_p, seed, generation_config JSON).

        Subclasses should call ``super()._build_generation_config(...)`` then apply
        endpoint-specific params (``max_tokens``, ``max_output_tokens``, etc.).
        """
        from transformers import GenerationConfig

        if body.get("generation_config") is not None:
            generation_config = GenerationConfig(**json.loads(body["generation_config"]))
        else:
            generation_config = copy.deepcopy(model_generation_config)
            if generation_config.max_new_tokens is None or generation_config.max_new_tokens < 1024:
                generation_config.max_new_tokens = 1024

        # GGUF models may not have eos/pad token IDs set — sync from processor
        if processor is not None:
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            if generation_config.eos_token_id is None and hasattr(tokenizer, "eos_token_id"):
                generation_config.eos_token_id = tokenizer.eos_token_id
            if generation_config.pad_token_id is None and hasattr(tokenizer, "pad_token_id"):
                generation_config.pad_token_id = tokenizer.pad_token_id

        if body.get("temperature") is not None:
            generation_config.temperature = float(body["temperature"])
            if float(body["temperature"]) == 0.0:
                generation_config.do_sample = False
        if body.get("top_p") is not None:
            generation_config.top_p = float(body["top_p"])
        if body.get("seed") is not None:
            set_torch_seed(body["seed"])

        # --compile flag: use static cache + torch.compile for faster decode
        if self._compile and generation_config.cache_implementation is None:
            generation_config.cache_implementation = "static"

        return generation_config

    def _start_streaming(self, model, processor, inputs, gen_config):
        """Set up DirectStreamer + queue, submit generate to inference thread.

        Returns ``(queue, streamer)`` — caller reads from queue to build SSE events.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        streamer = DirectStreamer(processor, loop, queue, skip_special_tokens=True)
        gen_kwargs = {**inputs, "streamer": streamer, "generation_config": gen_config, "tokenizer": processor}

        def _run():
            try:
                model.generate(**gen_kwargs)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, _StreamError(str(e)))

        self._inference_thread.submit(_run)
        return queue, streamer

    def _generate_non_streaming(self, model, processor, inputs, gen_config):
        """Run generate on the inference thread, decode output. Returns ``(text, input_len, generated_ids)``."""
        future = self._inference_thread.submit(
            model.generate, **inputs, generation_config=gen_config, tokenizer=processor
        )
        sequences = future.result()
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = sequences[0, input_len:]
        text = processor.decode(generated_ids, skip_special_tokens=True)
        return text, input_len, generated_ids


# ---------------------------------------------------------------------------
# Message preprocessing: OpenAI messages → processor-compatible format
# ---------------------------------------------------------------------------


def get_processor_inputs_from_messages(messages: list[dict], modality: Modality) -> list[dict]:
    """Convert OpenAI-format messages to the format expected by HF processors."""
    processor_inputs = []

    for message in messages:
        parsed = {"role": message["role"], "content": []}

        if modality == Modality.LLM:
            if isinstance(message["content"], str):
                parsed["content"] = message["content"]
            elif isinstance(message["content"], list):
                texts = [c["text"] for c in message["content"] if c["type"] == "text"]
                parsed["content"] = " ".join(texts)

        elif modality == Modality.VLM:
            if isinstance(message["content"], str):
                parsed["content"].append({"type": "text", "text": message["content"]})
            else:
                for content in message["content"]:
                    if content["type"] == "text":
                        parsed["content"].append(content)
                    elif content["type"] == "image_url":
                        url = content["image_url"]["url"]
                        if "base64" in url:
                            image_data = re.sub("^data:image/.+;base64,", "", url)
                            image = Image.open(BytesIO(base64.b64decode(image_data)))
                            file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                            image.save(file.name)
                            url = file.name
                        parsed["content"].append({"type": "image", "url": url})

        processor_inputs.append(parsed)
    return processor_inputs
