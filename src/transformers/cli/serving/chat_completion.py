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

import time
from collections.abc import AsyncGenerator
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
from openai.types.completion_usage import CompletionUsage

from transformers import GenerationConfig, PreTrainedModel

from ...utils import logging
from .utils import (
    UNUSED_CHAT_COMPLETION_FIELDS,
    BaseHandler,
    TransformersCompletionCreateParamsStreaming,
    _StreamError,
    get_processor_inputs_from_messages,
)


logger = logging.get_logger(__name__)


class ChatCompletionHandler(BaseHandler):
    """Handler for the `/v1/chat/completions` endpoint.

    Supports both streaming (SSE) and non-streaming (JSON) responses.
    """

    # ----- entry point -----

    def handle_request(self, body: dict, request_id: str) -> StreamingResponse | JSONResponse:
        """Validate the request, load the model, and dispatch to streaming or non-streaming."""
        self._validate_request(body)

        messages = body["messages"]

        # HACK: tiny-agents sends requests ending with assistant message — skip
        if messages and messages[-1]["role"] == "assistant":
            return JSONResponse({}, status_code=200)

        model_id, model, processor = self._resolve_model(body)

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

        gen_config = self._build_generation_config(body, model.generation_config, processor)

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
        """Stream tokens as SSE via DirectStreamer."""
        queue, streamer = self._start_streaming(model, processor, inputs, gen_config)
        input_len = inputs["input_ids"].shape[-1]

        async def sse_gen() -> AsyncGenerator[str, None]:
            yield self._build_chunk_sse(request_id, role="assistant", model=model_id)

            while True:
                text = await queue.get()
                if text is None:
                    break
                elif isinstance(text, _StreamError):
                    yield f'data: {{"error": "{text.msg}"}}\n\n'
                    return
                yield self._build_chunk_sse(request_id, content=text, model=model_id)

            hit_max = gen_config.max_new_tokens is not None and streamer.total_tokens >= gen_config.max_new_tokens
            usage = CompletionUsage(
                prompt_tokens=input_len,
                completion_tokens=streamer.total_tokens,
                total_tokens=input_len + streamer.total_tokens,
            )
            yield self._build_chunk_sse(
                request_id,
                finish_reason="length" if hit_max else "stop",
                model=model_id,
                usage=usage,
            )

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
        """Run generation and return a JSONResponse."""
        content, input_len, generated_ids = self._generate_non_streaming(model, processor, inputs, gen_config)

        hit_max = gen_config.max_new_tokens is not None and len(generated_ids) >= gen_config.max_new_tokens
        completion_tokens = len(generated_ids)
        usage = CompletionUsage(
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            total_tokens=input_len + completion_tokens,
        )
        return JSONResponse(
            self._build_completion(
                request_id,
                content,
                model_id,
                finish_reason="length" if hit_max else "stop",
                usage=usage,
            ),
            media_type="application/json",
        )

    # ----- helpers -----

    def _validate_request(self, body: dict) -> None:
        """Validate a chat completion request. Raises HTTPException if invalid."""
        input_keys = set(body.keys())
        unexpected = input_keys - TransformersCompletionCreateParamsStreaming.__mutable_keys__
        if unexpected:
            raise HTTPException(status_code=422, detail=f"Unexpected keys in the request: {unexpected}")

        unused = input_keys & UNUSED_CHAT_COMPLETION_FIELDS
        if unused:
            raise HTTPException(status_code=422, detail=f"Unsupported fields in the request: {unused}")

    def _build_generation_config(self, body: dict, model_generation_config, processor=None):
        """Chat Completions params on top of base config."""
        generation_config = super()._build_generation_config(body, model_generation_config, processor)

        if body.get("max_tokens") is not None:
            generation_config.max_new_tokens = int(body["max_tokens"])
        if body.get("frequency_penalty") is not None:
            generation_config.repetition_penalty = 1.0 + float(body["frequency_penalty"])
        if body.get("logit_bias") is not None:
            generation_config.sequence_bias = {(int(k),): v for k, v in body["logit_bias"].items()}
        if body.get("stop") is not None:
            generation_config.stop_strings = body["stop"]

        return generation_config

    # ----- response builders -----

    def _build_completion(
        self,
        request_id: str,
        content: str,
        model_id: str,
        finish_reason: str,
        usage: CompletionUsage | None = None,
    ) -> dict:
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
            usage=usage,
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
        usage: CompletionUsage | None = None,
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
            usage=usage,
            system_fingerprint="",
            object="chat.completion.chunk",
        )
        return self.chunk_to_sse(chunk)
