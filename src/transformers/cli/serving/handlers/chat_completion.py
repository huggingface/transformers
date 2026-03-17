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
Handler for the /v1/chat/completions endpoint.

Supports streaming (SSE via DirectStreamer) and non-streaming (JSON) responses.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import AsyncGenerator
from threading import Thread
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast, ProcessorMixin
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from tokenizers.decoders import DecodeStream

from transformers import GenerationConfig, PreTrainedModel
from transformers.generation.streamers import BaseStreamer

from ....utils import logging
from ..model_manager import ModelManager
from ..protocol import (
    UNUSED_CHAT_COMPLETION_FIELDS,
    TransformersCompletionCreateParamsStreaming,
    get_processor_inputs_from_messages,
)
from ..utils import _StreamError, set_torch_seed
from .base import BaseHandler


logger = logging.get_logger(__name__)


class DirectStreamer(BaseStreamer):
    """Streamer that decodes tokens incrementally and pushes text to an asyncio.Queue.

    Uses the Rust `DecodeStream.step()` for O(1) per-token decode, unlike
    `TextIteratorStreamer` which re-decodes the full sequence each time.

    Args:
        processor: A HuggingFace processor or tokenizer (must have a `._tokenizer` attribute).
        loop: The asyncio event loop to push results to.
        queue: The asyncio.Queue to push decoded text chunks to.
        skip_special_tokens: Whether to skip special tokens during decoding.
    """

    def __init__(
        self,
        processor: ProcessorMixin | PreTrainedTokenizerFast,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        skip_special_tokens: bool = True,
    ):
        self._tokenizer = processor._tokenizer  # raw tokenizers.Tokenizer
        self._loop = loop
        self._queue = queue
        self._decode_stream = DecodeStream([], skip_special_tokens)
        self._first = True
        self.total_tokens = 0

    def put(self, value: torch.Tensor) -> None:
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


class ChatCompletionHandler(BaseHandler):
    """Handler for the `/v1/chat/completions` endpoint.

    Supports both streaming (SSE) and non-streaming (JSON) responses.

    Args:
        model_manager: The model manager to load models from.
        force_model: If set, override the model field in every request.
    """

    def __init__(self, model_manager: ModelManager, force_model: str | None = None):
        super().__init__(model_manager)
        self.force_model = force_model

    def _validate_request(self, body: dict) -> None:
        """Validate a chat completion request. Raises HTTPException if invalid."""
        logger.debug(f"Validating request: {body}")

        input_keys = set(body.keys())
        unexpected = input_keys - TransformersCompletionCreateParamsStreaming.__mutable_keys__
        if unexpected:
            logger.error(f"Unexpected keys in the request: {unexpected}")
            raise HTTPException(status_code=422, detail=f"Unexpected keys in the request: {unexpected}")

        # TODO: add back strict Pydantic validation (input_validation flag)
        unused = input_keys & UNUSED_CHAT_COMPLETION_FIELDS
        if unused:
            logger.error(f"Unsupported fields in the request: {unused}")
            raise HTTPException(status_code=422, detail=f"Unsupported fields in the request: {unused}")

    def _build_generation_config(self, body: dict, model_generation_config: GenerationConfig) -> GenerationConfig:
        """Map Chat Completions API params to a GenerationConfig.

        If `body` contains a `generation_config` JSON string, it is used as baseline
        (overriding the model default). Other body params are applied on top.
        """
        if body.get("generation_config") is not None:
            generation_config = GenerationConfig(**json.loads(body["generation_config"]))
        else:
            generation_config = copy.deepcopy(model_generation_config)
            self._apply_default_generation_config(generation_config)

        if body.get("max_tokens") is not None:
            generation_config.max_new_tokens = int(body["max_tokens"])
        if body.get("frequency_penalty") is not None:
            generation_config.repetition_penalty = 1.0 + float(body["frequency_penalty"])
        if body.get("logit_bias") is not None:
            generation_config.sequence_bias = {(int(k),): v for k, v in body["logit_bias"].items()}
        if body.get("stop") is not None:
            generation_config.stop_strings = body["stop"]
        if body.get("temperature") is not None:
            generation_config.temperature = float(body["temperature"])
            if float(body["temperature"]) == 0.0:
                generation_config.do_sample = False
        if body.get("top_p") is not None:
            generation_config.top_p = float(body["top_p"])
        if body.get("seed") is not None:
            set_torch_seed(body["seed"])

        return generation_config

    def handle_request(self, body: dict, request_id: str) -> StreamingResponse | JSONResponse:
        """Validate the request, load the model, and dispatch to streaming or non-streaming."""
        self._validate_request(body)

        if self.force_model is not None:
            body["model"] = self.force_model

        messages = body["messages"]

        # HACK: tiny-agents sends requests ending with assistant message — skip
        if messages and messages[-1]["role"] == "assistant":
            return JSONResponse({}, status_code=200)

        model_id = self.model_manager.process_model_name(body["model"])
        model, processor = self.model_manager.load_model_and_processor(model_id)

        modality = self.model_manager.get_model_modality(model, processor=processor)
        processor_inputs = get_processor_inputs_from_messages(messages, modality)

        inputs = processor.apply_chat_template(
            processor_inputs,
            add_generation_prompt=True,
            tools=body.get("tools"),
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(model.device)

        gen_config = self._build_generation_config(body, model.generation_config)

        if body.get("stream"):
            return self._streaming(request_id, model, processor, model_id, inputs, gen_config)
        return self._non_streaming(request_id, model, processor, model_id, inputs, gen_config)

    # ----- streaming -----

    def _streaming(
        self,
        request_id: str,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizerFast,
        model_id: str,
        inputs: dict[str, torch.Tensor],
        gen_config: GenerationConfig,
    ) -> StreamingResponse:
        """Run generation in a background thread, stream tokens as SSE via DirectStreamer."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        streamer = DirectStreamer(processor, loop, queue, skip_special_tokens=True)
        gen_kwargs = {**inputs, "streamer": streamer, "generation_config": gen_config}

        def _run() -> None:
            try:
                model.generate(**gen_kwargs)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, _StreamError(str(e)))

        Thread(target=_run, daemon=True).start()

        async def sse_gen() -> AsyncGenerator[str, None]:
            # First chunk: tell the client the assistant is about to speak (OpenAI protocol)
            yield self._build_chunk_sse(request_id, role="assistant", model=model_id)

            # Stream tokens as they arrive from the background generate thread
            while True:
                text = await queue.get()
                if text is None:
                    break  # generation done
                elif isinstance(text, _StreamError):
                    yield f'data: {{"error": "{text.msg}"}}\n\n'
                    return

                yield self._build_chunk_sse(request_id, content=text, model=model_id)

            # Last chunk: tell the client why generation stopped
            hit_max = gen_config.max_new_tokens is not None and streamer.total_tokens >= gen_config.max_new_tokens
            if hit_max:
                logger.warning(
                    f"Generation hit max_new_tokens={gen_config.max_new_tokens}. "
                    "Output may be truncated. Use `max_tokens` to increase the limit."
                )
            yield self._build_chunk_sse(request_id, finish_reason="length" if hit_max else "stop", model=model_id)

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    # ----- non-streaming -----

    def _non_streaming(
        self,
        request_id: str,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizerFast,
        model_id: str,
        inputs: dict[str, torch.Tensor],
        gen_config: GenerationConfig,
    ) -> JSONResponse:
        """Run generation synchronously and return a JSONResponse."""
        sequences = model.generate(**inputs, generation_config=gen_config)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = sequences[0, input_len:]
        content = processor.decode(generated_ids, skip_special_tokens=True)

        hit_max = gen_config.max_new_tokens is not None and len(generated_ids) >= gen_config.max_new_tokens
        if hit_max:
            logger.warning(
                f"Generation hit max_new_tokens={gen_config.max_new_tokens}. "
                "Output may be truncated. Use `max_tokens` to increase the limit."
            )
        return JSONResponse(
            self._build_completion(request_id, content, model_id, finish_reason="length" if hit_max else "stop"),
            media_type="application/json",
        )

    # ----- response builders -----

    def _build_completion(self, request_id: str, content: str, model_id: str, finish_reason: str) -> dict:
        """Build a non-streaming ChatCompletion response dict."""
        result = ChatCompletion(
            id=request_id,
            created=int(time.time()),
            object="chat.completion",
            model=model_id,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(content=content, role="assistant"),
                    finish_reason=finish_reason,
                )
            ],
        )
        return result.model_dump(exclude_none=True)

    def _build_chunk_sse(
        self,
        request_id: str = "",
        content: str | None = None,
        model: str | None = None,
        role: str | None = None,
        finish_reason: str | None = None,
        tool_calls: list | None = None,
    ) -> str:
        """Build a streaming ChatCompletionChunk and format it as an SSE event."""
        chunk = ChatCompletionChunk(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChoiceChunk(
                    delta=ChoiceDelta(content=content, role=role, tool_calls=tool_calls),
                    index=0,
                    finish_reason=finish_reason,
                )
            ],
            system_fingerprint="",
            object="chat.completion.chunk",
        )
        return self.chunk_to_sse(chunk)
