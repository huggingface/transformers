# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from ...utils import logging
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi.responses import JSONResponse, StreamingResponse
    from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta, ChoiceDeltaToolCall
    from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
    from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
    from openai.types.completion_usage import CompletionUsage

from .utils import (
    BaseGenerateManager,
    BaseHandler,
    ToolCallParser,
    _StreamError,
    detect_tool_format,
)


if TYPE_CHECKING:
    from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin


class TransformersCompletionCreateParamsStreaming(CompletionCreateParamsStreaming, total=False):
    generation_config: str
    seed: int


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


logger = logging.get_logger(__name__)


class ChatCompletionHandler(BaseHandler):
    """Handler for the `/v1/chat/completions` endpoint.

    Supports both streaming (SSE) and non-streaming (JSON) responses.
    """

    _valid_params_class = TransformersCompletionCreateParamsStreaming
    _unused_fields = UNUSED_CHAT_COMPLETION_FIELDS

    async def handle_request(self, body: dict, request_id: str) -> StreamingResponse | JSONResponse:
        """Validate the request, load the model, and dispatch to streaming or non-streaming.

        Args:
            body (`dict`): The raw JSON request body (OpenAI chat completion format).
            request_id (`str`): Unique request identifier (from header or auto-generated).

        Returns:
            `StreamingResponse | JSONResponse`: SSE stream or JSON depending on ``body["stream"]``.
        """
        self._validate_request(body)

        model_id, model, processor = self._resolve_model(body)
        modality = self.model_manager.get_model_modality(model, processor=processor)
        use_cb = self.generation_state.use_continuous_batching(model, modality)
        logger.warning(f"[Request received] Model: {model_id}, CB: {use_cb}")
        gen_manager = self.generation_state.get_manager(model_id, use_cb=use_cb)
        processor_inputs = self.get_processor_inputs_from_messages(body["messages"], modality)

        inputs = processor.apply_chat_template(
            processor_inputs,
            add_generation_prompt=True,
            tools=body.get("tools"),
            return_tensors=None if use_cb else "pt",
            return_dict=True,
            tokenize=True,
        )
        if not use_cb:
            inputs = inputs.to(model.device)

        gen_config = self._build_generation_config(body, model.generation_config, use_cb=use_cb)
        # TODO: remove when CB supports per-request generation config
        if use_cb:
            gen_manager.init_cb(model, gen_config)

        # Detect tool support for the loaded model
        # TODO: after tool_call start token, use constrained generation to:
        # 1. force generation to pick from the available tool names
        # 2. force generation to produce valid JSON matching the tool's parameter schema
        tool_format = detect_tool_format(model) if body.get("tools") else None

        streaming = body.get("stream")
        if streaming:
            return self._streaming(
                request_id,
                model,
                processor,
                model_id,
                inputs,
                gen_config,
                gen_manager=gen_manager,
                tool_format=tool_format,
            )
        else:
            return await self._non_streaming(
                request_id,
                model,
                processor,
                model_id,
                inputs,
                gen_config,
                gen_manager=gen_manager,
                tool_format=tool_format,
            )

    # ----- streaming -----

    def _streaming(
        self,
        request_id: str,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        model_id: str,
        inputs: dict,
        gen_config: "GenerationConfig",
        gen_manager: BaseGenerateManager,
        tool_format: dict | None = None,
    ) -> StreamingResponse:
        """Stream tokens as SSE via DirectStreamer."""
        queue, streamer = gen_manager.generate_streaming(model, processor, inputs, gen_config, request_id=request_id)
        input_ids = inputs["input_ids"]
        # CB returns plain lists, regular path returns tensors
        input_len = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]
        parser = ToolCallParser(tool_format) if tool_format else None

        async def sse_gen() -> AsyncGenerator[str, None]:
            has_tool_calls = False
            try:
                yield self._build_chunk_sse(request_id, role="assistant", model=model_id)

                done = False
                while not done:
                    text = await queue.get()
                    batch = [text]
                    try:
                        while True:
                            batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        pass

                    sse_parts: list[str] = []
                    for text in batch:
                        if text is None:
                            done = True
                            break
                        if isinstance(text, _StreamError):
                            sse_parts.append(f'data: {{"error": "{text.msg}"}}\n\n')
                            yield "".join(sse_parts)
                            return

                        # Tool call parsing: None = normal text, CONSUMED = buffering, else = tool call dict
                        chunk_kwargs = {"content": text}
                        if parser is not None and (result := parser.feed(text)) is not None:
                            if result is ToolCallParser.CONSUMED:
                                continue
                            has_tool_calls = True
                            chunk_kwargs = {
                                "tool_calls": [
                                    ChoiceDeltaToolCall(
                                        index=0,
                                        type="function",
                                        id=f"{request_id}_tool_call",
                                        function={"name": result["name"], "arguments": result["arguments"]},
                                    )
                                ]
                            }

                        sse_parts.append(self._build_chunk_sse(request_id, model=model_id, **chunk_kwargs))

                    if sse_parts:
                        yield "".join(sse_parts)

                hit_max = gen_config.max_new_tokens is not None and streamer.total_tokens >= gen_config.max_new_tokens
                if has_tool_calls:
                    finish_reason = "tool_calls"
                elif hit_max:
                    finish_reason = "length"
                else:
                    finish_reason = "stop"
                usage = CompletionUsage(
                    prompt_tokens=input_len,
                    completion_tokens=streamer.total_tokens,
                    total_tokens=input_len + streamer.total_tokens,
                )
                yield self._build_chunk_sse(
                    request_id,
                    finish_reason=finish_reason,
                    model=model_id,
                    usage=usage,
                )
            except (GeneratorExit, asyncio.CancelledError):
                # Client disconnected — abort generation to free GPU.
                # Re-raise is mandatory: Python raises RuntimeError if GeneratorExit is swallowed.
                streamer.cancel()
                raise

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    # ----- non-streaming -----

    async def _non_streaming(
        self,
        request_id: str,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        model_id: str,
        inputs: dict,
        gen_config: "GenerationConfig",
        gen_manager: BaseGenerateManager,
        tool_format: dict | None = None,
    ) -> JSONResponse:
        """Run generation and return a JSONResponse."""
        content, input_len, generated_ids = await gen_manager.generate_non_streaming(
            model, processor, inputs, gen_config, request_id=request_id
        )

        hit_max = gen_config.max_new_tokens is not None and len(generated_ids) >= gen_config.max_new_tokens
        completion_tokens = len(generated_ids)
        usage = CompletionUsage(
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            total_tokens=input_len + completion_tokens,
        )

        # Parse tool calls from the generated text
        tool_calls = None
        if tool_format is not None:
            parsed = ToolCallParser.parse(content, tool_format)
            if parsed is not None:
                tool_calls = [
                    ChatCompletionMessageToolCall(
                        id=f"{request_id}_tool_call",
                        type="function",
                        function={"name": tc["name"], "arguments": tc["arguments"]},
                    )
                    for tc in parsed
                ]

        if tool_calls is not None:
            finish_reason = "tool_calls"
        elif hit_max:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return JSONResponse(
            self._build_completion(
                request_id,
                content,
                model_id,
                finish_reason=finish_reason,
                usage=usage,
                tool_calls=tool_calls,
            ),
            media_type="application/json",
        )

    # ----- helpers -----

    def _build_generation_config(self, body: dict, model_generation_config: "GenerationConfig", use_cb: bool = False):
        """Apply Chat Completions params (``max_tokens``, ``frequency_penalty``, ``logit_bias``,
        ``stop``) on top of the base generation config."""
        generation_config = super()._build_generation_config(body, model_generation_config, use_cb=use_cb)

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
        tool_calls: list[dict] | None = None,
    ) -> dict:
        """Build a non-streaming ChatCompletion response dict.

        Args:
            request_id (`str`): Unique request identifier.
            content (`str`): The generated text.
            model_id (`str`): Model ID to include in the response.
            finish_reason (`str`): Why generation stopped (``"stop"``, ``"length"``, ``"tool_calls"``).
            usage (`CompletionUsage`, *optional*): Token usage statistics.
            tool_calls (`list[dict]`, *optional*): Parsed tool calls, if any.

        Returns:
            `dict`: Serialized ``ChatCompletion`` ready for JSON response.
        """
        message = ChatCompletionMessage(content=content, role="assistant", tool_calls=tool_calls)
        result = ChatCompletion(
            id=request_id,
            created=int(time.time()),
            object="chat.completion",
            model=model_id,
            choices=[
                Choice(
                    index=0,
                    message=message,
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
        """Build a streaming ``ChatCompletionChunk`` and format it as an SSE ``data:`` line.

        Args:
            request_id (`str`): Unique request identifier.
            content (`str`, *optional*): Text content delta.
            model (`str`, *optional*): Model ID.
            role (`str`, *optional*): Role (only sent in the first chunk).
            finish_reason (`str`, *optional*): Set on the final chunk.
            tool_calls (`list`, *optional*): Tool call deltas.
            usage (`CompletionUsage`, *optional*): Token usage (sent with the final chunk).

        Returns:
            `str`: A formatted SSE event string.
        """
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
