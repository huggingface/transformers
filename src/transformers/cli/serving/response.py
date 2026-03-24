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
Handler for the /v1/responses endpoint (OpenAI Responses API).

Supports streaming (SSE) and non-streaming (JSON) responses.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

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
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

from ...utils import logging
from .utils import BaseHandler, ToolCallParser, _StreamError, detect_tool_format, get_processor_inputs_from_messages


if TYPE_CHECKING:
    from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin


logger = logging.get_logger(__name__)

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

    # ----- entry point -----

    def handle_request(self, body: dict, request_id: str) -> StreamingResponse | JSONResponse:
        """Validate, load model, dispatch to streaming or non-streaming."""
        self._validate_request(body)

        model_id, model, processor = self._resolve_model(body)

        # Two-step input conversion (chat completions skips step 1 since messages are already standard):
        # 1. Normalize Responses API input (string/list/dict + instructions) → standard messages list
        # 2. Transform message content for the HF processor (VLM image handling, text joining, etc.)
        messages = self._input_to_messages(body)
        modality = self.model_manager.get_model_modality(model, processor=processor)
        processor_inputs = get_processor_inputs_from_messages(messages, modality)

        inputs = processor.apply_chat_template(
            processor_inputs,
            add_generation_prompt=True,
            tools=body.get("tools"),
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        gen_config = self._build_generation_config(body, model.generation_config, processor)
        tool_format = detect_tool_format(model) if body.get("tools") else None

        if body.get("stream", True):
            return self._streaming(
                request_id,
                model,
                processor,
                model_id,
                body,
                inputs,
                gen_config,
                tool_format=tool_format,
            )
        return self._non_streaming(
            request_id,
            model,
            processor,
            model_id,
            body,
            inputs,
            gen_config,
            tool_format=tool_format,
        )

    # ----- input conversion -----

    @staticmethod
    def _input_to_messages(body: dict) -> list[dict]:
        """Convert the Responses API ``input`` field to a list of chat messages."""
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
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizerFast,
        model_id: str,
        body: dict,
        inputs: dict,
        gen_config: GenerationConfig,
        tool_format: dict | None = None,
    ) -> StreamingResponse:
        """Generate a streaming Responses API reply (SSE) using DirectStreamer."""
        queue, streamer = self._start_streaming(model, processor, inputs, gen_config)
        input_len = inputs["input_ids"].shape[-1]
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

            # 4. Stream tokens
            full_text = ""
            tool_calls = []

            while True:
                text = await queue.get()
                if text is None:
                    break
                if isinstance(text, _StreamError):
                    logger.error(f"Exception in response generation: {text.msg}")
                    yield self.chunk_to_sse(ResponseErrorEvent(type="error", sequence_number=seq, message=text.msg))
                    seq += 1
                    yield self.chunk_to_sse(
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
                    return

                # Tool call parsing
                if parser is not None and (result := parser.feed(text)) is not None:
                    if result is not ToolCallParser.CONSUMED:
                        # Emit tool call as a function_call output item
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
                        # output[0] = message, output[1..N] = tool calls (required by OpenAI SSE spec)
                        output_index += 1
                        yield self.chunk_to_sse(
                            ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=seq,
                                output_index=output_index,
                                item=tc_item,
                            )
                        )
                        seq += 1
                        yield self.chunk_to_sse(
                            ResponseFunctionCallArgumentsDoneEvent(
                                type="response.function_call_arguments.done",
                                sequence_number=seq,
                                item_id=tc_id,
                                output_index=output_index,
                                arguments=arguments,
                                name=name,
                            )
                        )
                        seq += 1
                        yield self.chunk_to_sse(
                            ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=seq,
                                output_index=output_index,
                                item=tc_item,
                            )
                        )
                        seq += 1
                    continue

                full_text += text
                yield self.chunk_to_sse(
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
                seq += 1

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
            usage = _make_usage(input_len, streamer.total_tokens)
            yield self.chunk_to_sse(
                ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=seq,
                    response=Response(**response_base, status="completed", output=all_output, usage=usage),
                )
            )
            seq += 1

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ----- non-streaming -----

    def _non_streaming(
        self,
        request_id: str,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizerFast,
        model_id: str,
        body: dict,
        inputs: dict,
        gen_config: GenerationConfig,
        tool_format: dict | None = None,
    ) -> JSONResponse:
        """Generate a non-streaming Responses API reply (single JSON)."""
        full_text, input_len, generated_ids = self._generate_non_streaming(model, processor, inputs, gen_config)

        created_at = time.time()
        resp_id = f"resp_{request_id}"
        msg_id = f"msg_{request_id}"
        output_tokens = len(generated_ids)

        output_items = [
            ResponseOutputMessage(
                id=msg_id,
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

        usage = _make_usage(input_len, output_tokens)
        response = Response(
            id=resp_id,
            created_at=created_at,
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

    def _validate_request(self, body: dict) -> None:
        """Validate a Responses API request."""
        unused = set(body.keys()) & UNUSED_RESPONSE_FIELDS
        if unused:
            raise HTTPException(status_code=422, detail=f"Unsupported fields in the request: {unused}")

    def _build_generation_config(self, body: dict, model_generation_config, processor=None):
        """Responses API params on top of base config."""
        generation_config = super()._build_generation_config(body, model_generation_config, processor)

        if body.get("max_output_tokens") is not None:
            generation_config.max_new_tokens = int(body["max_output_tokens"])

        return generation_config


def _make_usage(input_tokens: int, output_tokens: int) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
