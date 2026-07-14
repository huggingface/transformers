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
        ResponseReasoningItem,
        ResponseReasoningTextDeltaEvent,
        ResponseReasoningTextDoneEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )
    from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage


from .utils import (
    BaseGenerateManager,
    BaseHandler,
    Modality,
    ReasoningText,
    _StreamError,
    get_reasoning_config,
    get_tool_call_config,
    parse_reasoning,
    parse_tool_calls,
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
    "service_tier",
    "store",
    "text",
    "tool_choice",
    "top_logprobs",
    "truncation",
    "user",
}


class _ResponseStreamBuilder:
    """Builds SSE events for one streaming Responses API generation."""

    def __init__(self, *, request_id: str, response_defaults: dict):
        self._response_defaults = response_defaults
        self.resp_id = f"resp_{request_id}"
        self.msg_id = f"msg_{request_id}"
        self.reasoning_id = f"rs_{request_id}"
        self.seq = 0
        self.output_index = 0
        self.full_text = ""
        self.full_reasoning = ""
        self.reasoning_open = False
        self.message_open = False
        self.reasoning_item: ResponseReasoningItem | None = None
        self.message_item: ResponseOutputMessage | None = None
        self.tool_calls: list[ResponseFunctionToolCall] = []

    def _emit(self, event) -> str:
        sse = BaseHandler.chunk_to_sse(event)
        self.seq += 1
        return sse

    def _response(self, status: str, **extra) -> "Response":
        return Response(**self._response_defaults, status=status, **extra)

    def start_response(self) -> list[str]:
        return [
            self._emit(
                ResponseCreatedEvent(
                    type="response.created",
                    sequence_number=self.seq,
                    response=self._response("queued", output=[]),
                )
            ),
            self._emit(
                ResponseInProgressEvent(
                    type="response.in_progress",
                    sequence_number=self.seq,
                    response=self._response("in_progress", output=[]),
                )
            ),
        ]

    def start_reasoning(self) -> list[str]:
        self.reasoning_open = True
        return [
            self._emit(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    item=ResponseReasoningItem(
                        id=self.reasoning_id, type="reasoning", summary=[], content=[], status="in_progress"
                    ),
                )
            )
        ]

    def reasoning_delta(self, text: str) -> list[str]:
        self.full_reasoning += text
        return [
            self._emit(
                ResponseReasoningTextDeltaEvent(
                    type="response.reasoning_text.delta",
                    item_id=self.reasoning_id,
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    content_index=0,
                    delta=text,
                )
            )
        ]

    def finish_reasoning(self) -> list[str]:
        self.reasoning_item = ResponseReasoningItem(
            id=self.reasoning_id,
            type="reasoning",
            summary=[],
            content=[{"type": "reasoning_text", "text": self.full_reasoning}],
            status="completed",
        )
        parts = [
            self._emit(
                ResponseReasoningTextDoneEvent(
                    type="response.reasoning_text.done",
                    item_id=self.reasoning_id,
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    content_index=0,
                    text=self.full_reasoning,
                )
            ),
            self._emit(
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    item=self.reasoning_item,
                )
            ),
        ]
        self.reasoning_open = False
        self.output_index += 1
        return parts

    def start_message(self) -> list[str]:
        self.message_open = True
        return [
            self._emit(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    item=ResponseOutputMessage(
                        id=self.msg_id, type="message", status="in_progress", role="assistant", content=[]
                    ),
                )
            ),
            self._emit(
                ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    item_id=self.msg_id,
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    content_index=0,
                    part=ResponseOutputText(type="output_text", text="", annotations=[]),
                )
            ),
        ]

    def text_delta(self, text: str) -> list[str]:
        self.full_text += text
        return [
            self._emit(
                ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    item_id=self.msg_id,
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    content_index=0,
                    delta=text,
                    logprobs=[],
                )
            )
        ]

    def finish_message(self) -> list[str]:
        output_text_part = ResponseOutputText(type="output_text", text=self.full_text, annotations=[])
        self.message_item = ResponseOutputMessage(
            id=self.msg_id,
            type="message",
            status="completed",
            role="assistant",
            content=[output_text_part],
            annotations=[],  # type: ignore[call-arg]
        )
        parts = [
            self._emit(
                ResponseTextDoneEvent(
                    type="response.output_text.done",
                    item_id=self.msg_id,
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    content_index=0,
                    text=self.full_text,
                    logprobs=[],
                )
            ),
            self._emit(
                ResponseContentPartDoneEvent(
                    type="response.content_part.done",
                    item_id=self.msg_id,
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    content_index=0,
                    part=output_text_part,
                )
            ),
            self._emit(
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    item=self.message_item,
                )
            ),
        ]
        self.message_open = False
        return parts

    def tool_call(self, tc_id: str, name: str, arguments: str) -> list[str]:
        self.output_index += 1
        item = ResponseFunctionToolCall(
            id=tc_id,
            call_id=tc_id,
            type="function_call",
            name=name,
            arguments=arguments,
            status="completed",
        )
        self.tool_calls.append(item)
        return [
            self._emit(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    item=item,
                )
            ),
            self._emit(
                ResponseFunctionCallArgumentsDoneEvent(
                    type="response.function_call_arguments.done",
                    sequence_number=self.seq,
                    item_id=tc_id,
                    output_index=self.output_index,
                    arguments=arguments,
                    name=name,
                )
            ),
            self._emit(
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=self.seq,
                    output_index=self.output_index,
                    item=item,
                )
            ),
        ]

    def error(self, msg: str) -> list[str]:
        return [
            self._emit(ResponseErrorEvent(type="error", sequence_number=self.seq, message=msg)),
            self._emit(
                ResponseFailedEvent(
                    type="response.failed",
                    sequence_number=self.seq,
                    response=self._response(
                        "failed", output=[], error=ResponseError(code="server_error", message=msg)
                    ),
                )
            ),
        ]

    def completed(self, usage) -> list[str]:
        all_output: list = []
        if self.reasoning_item is not None:
            all_output.append(self.reasoning_item)
        all_output.append(self.message_item)
        all_output.extend(self.tool_calls)
        return [
            self._emit(
                ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=self.seq,
                    response=self._response("completed", output=all_output, usage=usage),
                )
            )
        ]


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
        messages = self._normalize_input(body)
        processor_inputs = self.get_processor_inputs_from_messages(messages, modality)

        has_video = any(
            c.get("type") == "video"
            for msg in processor_inputs
            for c in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        )

        # Default to 32 frames for video (Gemma 4 default); some processors load all frames otherwise.
        # Merge order (later wins): custom default -> server default → request-level kwargs.
        chat_template_kwargs: dict = {}
        if has_video:
            chat_template_kwargs["num_frames"] = 32
        chat_template_kwargs.update(self.chat_template_kwargs)
        chat_template_kwargs.update(body.get("chat_template_kwargs") or {})
        # updates the flat tool structure to the one expected by the `apply_chat_template` method.
        tools = self._normalize_tools(body.get("tools"))
        inputs = processor.apply_chat_template(
            processor_inputs,
            add_generation_prompt=True,
            tools=tools,
            return_tensors=None if use_cb else "pt",
            return_dict=True,
            tokenize=True,
            load_audio_from_video=modality == Modality.MULTIMODAL and has_video,
            **chat_template_kwargs,
        )
        if not use_cb:
            inputs = inputs.to(model.device)  # type: ignore[union-attr]

        gen_config = self._build_generation_config(body, model.generation_config, use_cb=use_cb)
        # TODO: remove when CB supports per-request generation config
        if use_cb:
            gen_manager.init_cb(model, gen_config)
        tool_config = get_tool_call_config(processor, model) if body.get("tools") else None
        reasoning_config = get_reasoning_config(processor, model, inputs["input_ids"])

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
                tool_config=tool_config,
                reasoning_config=reasoning_config,
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
                tool_config=tool_config,
                reasoning_config=reasoning_config,
            )

    # ----- input conversion -----

    @staticmethod
    def _normalize_tools(tools: list[dict] | None) -> list[dict] | None:
        """Normalize Responses API tool definitions for ``apply_chat_template``.

        The Responses API uses a flat format: ``{"type": "function", "name": ..., "parameters": ...}``
        while ``apply_chat_template`` expects a nested format:
        ``{"type": "function", "function": {"name": ..., "parameters": ...}}``.
        Already-nested tools are passed through unchanged.
        """
        if not tools:
            return tools
        return [
            {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}} if "function" not in t else t
            for t in tools
        ]

    @staticmethod
    def _normalize_input(body: dict) -> list[dict]:
        """Normalize the Responses API ``input`` field into chat messages.

        The Responses API accepts multiple input formats. This method converts them
        into a structure close to what ``apply_chat_template`` expects (messages with
        ``role``, ``content``, ``tool_calls``, ``tool_call_id``). Further processing
        is done by ``get_processor_inputs_from_messages``.

        NOTE: if this conversion logic grows too complex, consider having separate
        ``get_processor_inputs_from_messages`` implementations for chat completions
        and the Responses API instead of funneling both through the same path.

        Formats handled:
            - **String** → single user message.
            - **Flat content list** (``input_text``, ``input_image``, no ``role``) → user message.
            - **Multi-turn list** — messages and tool call items (``function_call``,
              ``function_call_output``) from a previous response, converted via
              :meth:`_normalize_response_items`.

        If ``instructions`` is present, it is prepended as a system message.
        """
        inp = body["input"]
        instructions = body.get("instructions")

        if isinstance(inp, str):
            messages = [{"role": "user", "content": inp}]
        elif isinstance(inp, list):
            if inp and "role" not in inp[0]:
                # Flat content list (single-turn, e.g. input_text/input_image)
                messages = [{"role": "user", "content": inp}]
            else:
                messages = ResponseHandler._normalize_response_items(inp)
        else:
            raise HTTPException(status_code=422, detail="'input' must be a string or list")

        # Prepend instructions as a system message
        if instructions:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = instructions
            else:
                messages.insert(0, {"role": "system", "content": instructions})

        return messages

    @staticmethod
    def _normalize_response_items(items: list[dict]) -> list[dict]:
        """Convert a list of Responses API items into chat messages.

        Input items may be a mix of:
            - Messages (``EasyInputMessageParam`` with ``role``, or ``type: "message"``).
            - ``reasoning`` — buffered and attached as ``reasoning_content`` to the next assistant message.
            - ``function_call`` — merged as ``tool_calls`` onto the preceding assistant message.
            - ``function_call_output`` — converted to ``role: "tool"`` messages.
        """
        messages = []
        pending_reasoning: str | None = None

        for item in items:
            item_type = item.get("type")

            if item_type == "reasoning":
                pending_reasoning = "".join(c["text"] for c in item.get("content") or [])
                continue

            if "role" in item:
                msg = {"role": item["role"], "content": item.get("content", "")}
                if pending_reasoning is not None and item["role"] == "assistant":
                    msg["reasoning_content"] = pending_reasoning
                    pending_reasoning = None
                messages.append(msg)

            elif item_type == "function_call":
                tc = {
                    "id": item["call_id"],
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(tc)
                else:
                    messages.append({"role": "assistant", "tool_calls": [tc]})

            elif item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": item["output"],
                    }
                )

            else:
                raise HTTPException(status_code=422, detail=f"Unsupported input item type: {item_type!r}")

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
        tool_config: dict | None = None,
        reasoning_config: dict | None = None,
    ) -> StreamingResponse:
        """Generate a streaming Responses API reply (SSE) using DirectStreamer."""
        queue, streamer = gen_manager.generate_streaming(
            model,
            processor,
            inputs,
            gen_config,
            request_id=request_id,
            tool_config=tool_config,
            reasoning_config=reasoning_config,
        )
        input_ids = inputs["input_ids"]
        # CB returns plain lists, regular path returns tensors
        input_len = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]

        response_defaults = {
            "id": f"resp_{request_id}",
            "created_at": time.time(),
            "model": model_id,
            "object": "response",
            # Required by pydantic but not used — echo request config back
            "tools": [],
            "parallel_tool_calls": body.get("parallel_tool_calls", False),
            "tool_choice": "auto",
        }

        async def event_stream() -> AsyncGenerator[str, None]:
            builder = _ResponseStreamBuilder(request_id=request_id, response_defaults=response_defaults)
            try:
                yield "".join(builder.start_response())

                # Stream tokens — items are opened lazily so reasoning (if any)
                # appears as a separate output item before the message item.
                done = False
                while not done:
                    batch = [await queue.get()]
                    try:
                        while True:
                            batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        pass

                    parts: list[str] = []
                    for text in batch:
                        if text is None:
                            done = True
                            break
                        if isinstance(text, _StreamError):
                            logger.error(f"Exception in response generation: {text.msg}")
                            parts.extend(builder.error(text.msg))
                            yield "".join(parts)
                            return
                        if isinstance(text, ReasoningText):
                            if not builder.reasoning_open:
                                parts.extend(builder.start_reasoning())
                            parts.extend(builder.reasoning_delta(text))
                        else:
                            if builder.reasoning_open:
                                parts.extend(builder.finish_reasoning())
                            if not builder.message_open:
                                parts.extend(builder.start_message())
                            parts.extend(builder.text_delta(text))

                    if parts:
                        yield "".join(parts)

                # Close any open reasoning, then ensure a message section exists.
                if builder.reasoning_open:
                    yield "".join(builder.finish_reasoning())
                if not builder.message_open:
                    yield "".join(builder.start_message())
                yield "".join(builder.finish_message())

                # Tool calls are parsed after generation completes (not during streaming),
                # because the full token sequence is needed for reliable parsing.
                if tool_config:
                    parsed = parse_tool_calls(processor, streamer.generated_token_ids, tool_config["schema"])
                    if parsed:
                        for i, tc in enumerate(parsed):
                            yield "".join(
                                builder.tool_call(f"{request_id}_tool_call_{i}", tc["name"], tc["arguments"])
                            )

                yield "".join(builder.completed(compute_usage(input_len, streamer.total_tokens)))
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
        tool_config: dict | None = None,
        reasoning_config: dict | None = None,
    ) -> JSONResponse:
        """Generate a non-streaming Responses API reply (single JSON)."""
        full_text, input_len, generated_ids = await gen_manager.generate_non_streaming(
            model, processor, inputs, gen_config, request_id=request_id
        )

        output_items = []
        if reasoning_config is not None:
            full_text, reasoning_content = parse_reasoning(processor, generated_ids, full_text, reasoning_config)
            if reasoning_content is not None:
                output_items.append(
                    ResponseReasoningItem(
                        id=f"rs_{request_id}",
                        type="reasoning",
                        summary=[],
                        content=[{"type": "reasoning_text", "text": reasoning_content}],
                        status="completed",
                    )
                )

        output_items.append(
            ResponseOutputMessage(
                id=f"msg_{request_id}",
                type="message",
                status="completed",
                role="assistant",
                content=[ResponseOutputText(type="output_text", text=full_text, annotations=[])],
                annotations=[],  # type: ignore[call-arg]
            )
        )

        if tool_config is not None:
            parsed = parse_tool_calls(processor, generated_ids, tool_config["schema"])
            if parsed:
                for i, tc in enumerate(parsed):
                    tc_id = f"{request_id}_tool_call_{i}"
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
        input_tokens_details=InputTokensDetails(
            cached_tokens=0,
            **{"cache_write_tokens": 0} if "cache_write_tokens" in InputTokensDetails.model_fields else {},
        ),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
