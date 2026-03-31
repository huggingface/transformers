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
Handler for the /v1/responses endpoint (OpenAI Responses API).

Supports streaming (SSE) and non-streaming (JSON) responses.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from ...utils import logging
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseContentPartAddedEvent,
        ResponseContentPartDoneEvent,
        ResponseCreatedEvent,
        ResponseError,
        ResponseErrorEvent,
        ResponseFailedEvent,
        ResponseFunctionCallArgumentsDoneEvent,
        ResponseFunctionToolCall,
        ResponseInProgressEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )
    from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

from .utils import (
    BaseGenerateManager,
    BaseHandler,
    ToolCallParser,
    _StreamError,
    detect_tool_format,
)


if TYPE_CHECKING:
    from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin


logger = logging.get_logger(__name__)


class TransformersResponseCreateParamsStreaming(ResponseCreateParamsStreaming, total=False):
    generation_config: str
    seed: int


UNUSED_RESPONSE_FIELDS = {
    "background",
    "include",
    "max_tool_calls",
    "previous_response_id",
    "prompt",
    "reasoning",
    "service_tier",
    "store",
    "text",
    "tool_choice",
    "top_logprobs",
    "truncation",
    "user",
}


class ResponseHandler(BaseHandler):
    """Handler for the ``/v1/responses`` endpoint."""

    _valid_params_class = TransformersResponseCreateParamsStreaming
    _unused_fields = UNUSED_RESPONSE_FIELDS

    async def handle_request(self, body: dict, request_id: str) -> StreamingResponse | JSONResponse:
        """Validate, load model, dispatch to streaming or non-streaming.

        Args:
            body (`dict`): The raw JSON request body (OpenAI Responses API format).
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

        # Two-step input conversion (chat completions skips step 1 since messages are already standard):
        # 1. Normalize Responses API input (string/list/dict + instructions) → standard messages list
        # 2. Transform message content for the HF processor (VLM image handling, text joining, etc.)
        messages = self._input_to_messages(body)
        processor_inputs = self.get_processor_inputs_from_messages(messages, modality)

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
        tool_format = detect_tool_format(model) if body.get("tools") else None

        streaming = body.get("stream", True)
        if streaming:
            return self._streaming(
                request_id,
                model,
                processor,
                model_id,
                body,
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
                body,
                inputs,
                gen_config,
                gen_manager=gen_manager,
                tool_format=tool_format,
            )

    # ----- input conversion -----

    @staticmethod
    def _input_to_messages(body: dict) -> list[dict]:
        """Convert the Responses API ``input`` field to a list of chat messages.

        Handles string, list, and dict inputs. If ``instructions`` is provided, it is
        prepended as a system message (or replaces an existing one).

        Args:
            body (`dict`): The raw request body containing ``input`` and optionally ``instructions``.

        Returns:
            `list[dict]`: Standard OpenAI-format chat messages.
        """
        inp = body["input"]
        instructions = body.get("instructions")

        if isinstance(inp, str):
            messages = [{"role": "system", "content": instructions}] if instructions else []
            messages.append({"role": "user", "content": inp})
        elif isinstance(inp, list):
            if instructions:
                if inp[0]["role"] != "system":
                    messages = [{"role": "system", "content": instructions}, *inp]
                else:
                    messages = list(inp)
                    messages[0]["content"] = instructions
            else:
                messages = inp
        elif isinstance(inp, dict):
            messages = [{"role": "system", "content": instructions}] if instructions else []
            messages.append(inp)
        else:
            raise HTTPException(status_code=422, detail="'input' must be a string, list, or dict")

        return messages

    # ----- streaming -----

    def _streaming(
        self,
        request_id: str,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        model_id: str,
        body: dict,
        inputs: dict,
        gen_config: "GenerationConfig",
        gen_manager: BaseGenerateManager,
        tool_format: dict | None = None,
    ) -> StreamingResponse:
        """Generate a streaming Responses API reply (SSE) using DirectStreamer."""
        queue, streamer = gen_manager.generate_streaming(model, processor, inputs, gen_config, request_id=request_id)
        input_ids = inputs["input_ids"]
        # CB returns plain lists, regular path returns tensors
        input_len = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]
        parser = ToolCallParser(tool_format) if tool_format else None

        seq = 0
        output_index = 0
        created_at = time.time()
        resp_id = f"resp_{request_id}"
        msg_id = f"msg_{request_id}"

        response_base = {
            "id": resp_id,
            "created_at": created_at,
            "model": model_id,
            "object": "response",
            # Required by pydantic but not used — echo request config back
            "tools": [],
            "parallel_tool_calls": body.get("parallel_tool_calls", False),
            "tool_choice": "auto",
        }

        async def event_stream() -> AsyncGenerator[str, None]:
            nonlocal seq, output_index

            try:
                # 1. Created + In progress
                yield self.chunk_to_sse(
                    ResponseCreatedEvent(
                        type="response.created",
                        sequence_number=seq,
                        response=Response(**response_base, status="queued", output=[]),
                    )
                )
                seq += 1
                yield self.chunk_to_sse(
                    ResponseInProgressEvent(
                        type="response.in_progress",
                        sequence_number=seq,
                        response=Response(**response_base, status="in_progress", output=[]),
                    )
                )
                seq += 1

                # 2. Output item added (message)
                yield self.chunk_to_sse(
                    ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=seq,
                        output_index=output_index,
                        item=ResponseOutputMessage(
                            id=msg_id,
                            type="message",
                            status="in_progress",
                            role="assistant",
                            content=[],
                        ),
                    )
                )
                seq += 1

                # 3. Content part added
                yield self.chunk_to_sse(
                    ResponseContentPartAddedEvent(
                        type="response.content_part.added",
                        item_id=msg_id,
                        sequence_number=seq,
                        output_index=output_index,
                        content_index=0,
                        part=ResponseOutputText(type="output_text", text="", annotations=[]),
                    )
                )
                seq += 1

                # 4. Stream tokens — drain queue to batch HTTP writes
                full_text = ""
                tool_calls = []
                done = False

                while not done:
                    text = await queue.get()
                    # Drain all available tokens for one batched HTTP write
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
                            logger.error(f"Exception in response generation: {text.msg}")
                            sse_parts.append(
                                self.chunk_to_sse(
                                    ResponseErrorEvent(type="error", sequence_number=seq, message=text.msg)
                                )
                            )
                            seq += 1
                            sse_parts.append(
                                self.chunk_to_sse(
                                    ResponseFailedEvent(
                                        type="response.failed",
                                        sequence_number=seq,
                                        response=Response(
                                            **response_base,
                                            status="failed",
                                            output=[],
                                            error=ResponseError(code="server_error", message=text.msg),
                                        ),
                                    )
                                )
                            )
                            yield "".join(sse_parts)
                            return

                        # Tool call parsing
                        if parser is not None and (result := parser.feed(text)) is not None:
                            if result is not ToolCallParser.CONSUMED:
                                tc_id = f"{request_id}_tool_call"
                                name = result["name"]
                                arguments = result["arguments"]
                                tc_item = ResponseFunctionToolCall(
                                    id=tc_id,
                                    call_id=tc_id,
                                    type="function_call",
                                    name=name,
                                    arguments=arguments,
                                    status="completed",
                                )
                                tool_calls.append(tc_item)
                                output_index += 1
                                sse_parts.append(
                                    self.chunk_to_sse(
                                        ResponseOutputItemAddedEvent(
                                            type="response.output_item.added",
                                            sequence_number=seq,
                                            output_index=output_index,
                                            item=tc_item,
                                        )
                                    )
                                )
                                seq += 1
                                sse_parts.append(
                                    self.chunk_to_sse(
                                        ResponseFunctionCallArgumentsDoneEvent(
                                            type="response.function_call_arguments.done",
                                            sequence_number=seq,
                                            item_id=tc_id,
                                            output_index=output_index,
                                            arguments=arguments,
                                            name=name,
                                        )
                                    )
                                )
                                seq += 1
                                sse_parts.append(
                                    self.chunk_to_sse(
                                        ResponseOutputItemDoneEvent(
                                            type="response.output_item.done",
                                            sequence_number=seq,
                                            output_index=output_index,
                                            item=tc_item,
                                        )
                                    )
                                )
                                seq += 1
                            continue

                        full_text += text
                        sse_parts.append(
                            self.chunk_to_sse(
                                ResponseTextDeltaEvent(
                                    type="response.output_text.delta",
                                    item_id=msg_id,
                                    sequence_number=seq,
                                    output_index=0,
                                    content_index=0,
                                    delta=text,
                                    logprobs=[],
                                )
                            )
                        )
                        seq += 1

                    if sse_parts:
                        yield "".join(sse_parts)

                # 5. Close text output
                output_text_part = ResponseOutputText(type="output_text", text=full_text, annotations=[])
                yield self.chunk_to_sse(
                    ResponseTextDoneEvent(
                        type="response.output_text.done",
                        item_id=msg_id,
                        sequence_number=seq,
                        output_index=0,
                        content_index=0,
                        text=full_text,
                        logprobs=[],
                    )
                )
                seq += 1
                yield self.chunk_to_sse(
                    ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        item_id=msg_id,
                        sequence_number=seq,
                        output_index=0,
                        content_index=0,
                        part=output_text_part,
                    )
                )
                seq += 1

                msg_item = ResponseOutputMessage(
                    id=msg_id,
                    type="message",
                    status="completed",
                    role="assistant",
                    content=[output_text_part],
                    annotations=[],
                )
                yield self.chunk_to_sse(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=seq,
                        output_index=0,
                        item=msg_item,
                    )
                )
                seq += 1

                # 6. Completed
                all_output = [msg_item] + list(tool_calls)
                usage = compute_usage(input_len, streamer.total_tokens)
                yield self.chunk_to_sse(
                    ResponseCompletedEvent(
                        type="response.completed",
                        sequence_number=seq,
                        response=Response(**response_base, status="completed", output=all_output, usage=usage),
                    )
                )
                seq += 1
            except (GeneratorExit, asyncio.CancelledError):
                # Client disconnected — abort generation to free GPU.
                # Re-raise is mandatory: Python raises RuntimeError if GeneratorExit is swallowed.
                streamer.cancel()
                raise

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ----- non-streaming -----

    async def _non_streaming(
        self,
        request_id: str,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        model_id: str,
        body: dict,
        inputs: dict,
        gen_config: "GenerationConfig",
        gen_manager: BaseGenerateManager,
        tool_format: dict | None = None,
    ) -> JSONResponse:
        """Generate a non-streaming Responses API reply (single JSON)."""
        full_text, input_len, generated_ids = await gen_manager.generate_non_streaming(
            model, processor, inputs, gen_config, request_id=request_id
        )

        output_items = [
            ResponseOutputMessage(
                id=f"msg_{request_id}",
                type="message",
                status="completed",
                role="assistant",
                content=[ResponseOutputText(type="output_text", text=full_text, annotations=[])],
                annotations=[],
            )
        ]

        # Parse tool calls from the generated text
        if tool_format is not None:
            parsed_calls = ToolCallParser.parse(full_text, tool_format)
            if parsed_calls is not None:
                for i, tc in enumerate(parsed_calls):
                    tc_id = f"{request_id}_tool_call"
                    output_items.append(
                        ResponseFunctionToolCall(
                            id=tc_id,
                            call_id=tc_id,
                            type="function_call",
                            name=tc["name"],
                            arguments=tc["arguments"],
                            status="completed",
                        )
                    )

        usage = compute_usage(input_len, len(generated_ids))
        response = Response(
            id=f"resp_{request_id}",
            created_at=time.time(),
            status="completed",
            model=model_id,
            output=output_items,
            object="response",
            usage=usage,
            # Required by pydantic but not used — echo request config back
            tools=[],
            parallel_tool_calls=body.get("parallel_tool_calls", False),
            tool_choice="auto",
        )
        return JSONResponse(response.model_dump(exclude_none=True))

    # ----- helpers -----

    def _build_generation_config(self, body: dict, model_generation_config: "GenerationConfig", use_cb: bool = False):
        """Apply Responses API params (``max_output_tokens``) on top of the base generation config."""
        generation_config = super()._build_generation_config(body, model_generation_config, use_cb=use_cb)

        if body.get("max_output_tokens") is not None:
            generation_config.max_new_tokens = int(body["max_output_tokens"])

        return generation_config


def compute_usage(input_tokens: int, output_tokens: int) -> ResponseUsage:
    """Build a ``ResponseUsage`` object for a Responses API reply.

    Args:
        input_tokens (`int`): Number of prompt tokens.
        output_tokens (`int`): Number of generated tokens.

    Returns:
        `ResponseUsage`: Usage statistics with zero-filled detail fields.
    """
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
