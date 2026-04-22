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
Handler for the /v1/completions endpoint (OpenAI legacy Completions API).

Accepts a freeform text prompt (no chat template) and returns generated text
in choices[].text. Supports streaming and non-streaming modes, and suffix for
fill-in-the-middle text insertion.
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
    from openai.types import Completion, CompletionChoice, CompletionUsage
    from openai.types.completion_create_params import CompletionCreateParamsBase


from .utils import BaseGenerateManager, BaseHandler, _StreamError


if TYPE_CHECKING:
    from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin


class TransformersTextCompletionCreateParams(CompletionCreateParamsBase, total=False):
    generation_config: str
    seed: int
    stream: bool


# Fields accepted by the OpenAI schema but not yet supported.
UNUSED_LEGACY_COMPLETION_FIELDS = {
    "best_of",
    "echo",
    "logprobs",
    "n",
    "presence_penalty",
    "stream_options",
    "user",
}


logger = logging.get_logger(__name__)


class CompletionHandler(BaseHandler):
    """Handler for the `/v1/completions` endpoint.

    Takes a raw text ``prompt`` (no chat template) and generates text returned in
    ``choices[].text``. Supports streaming (SSE) and non-streaming (JSON) responses,
    and ``suffix`` for fill-in-the-middle insertion.
    """

    _valid_params_class = TransformersTextCompletionCreateParams
    _unused_fields = UNUSED_LEGACY_COMPLETION_FIELDS

    async def handle_request(self, body: dict, request_id: str) -> "StreamingResponse | JSONResponse":
        """Validate the request, load the model, and dispatch to streaming or non-streaming.

        Args:
            body (`dict`): The raw JSON request body (OpenAI legacy completions format).
            request_id (`str`): Unique request identifier (from header or auto-generated).

        Returns:
            `StreamingResponse | JSONResponse`: SSE stream or JSON depending on ``body["stream"]``.
        """
        self._validate_request(body)

        prompt = body.get("prompt", "")
        if not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="prompt must be a string.")

        model_id, model, processor = self._resolve_model(body)
        modality = self.model_manager.get_model_modality(model, processor=processor)
        use_cb = self.generation_state.use_continuous_batching(model, modality)
        logger.warning(f"[Request received] Model: {model_id}, CB: {use_cb}")
        gen_manager = self.generation_state.get_manager(model_id, use_cb=use_cb)

        tokenizer = getattr(processor, "tokenizer", processor)
        inputs = tokenizer(prompt, return_tensors=None if use_cb else "pt")
        if not use_cb:
            inputs = inputs.to(model.device)

        gen_config = self._build_generation_config(body, model.generation_config, use_cb=use_cb)
        if use_cb:
            gen_manager.init_cb(model, gen_config)

        suffix = body.get("suffix")
        streaming = body.get("stream")

        if streaming:
            return self._streaming(request_id, model, processor, model_id, inputs, gen_config, gen_manager, suffix)
        else:
            return await self._non_streaming(
                request_id, model, processor, model_id, inputs, gen_config, gen_manager, suffix
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
        suffix: str | None = None,
    ) -> "StreamingResponse":
        """Stream tokens as SSE."""
        queue, streamer = gen_manager.generate_streaming(model, processor, inputs, gen_config, request_id=request_id)
        input_ids = inputs["input_ids"]
        input_len = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]

        async def sse_gen() -> AsyncGenerator[str, None]:
            try:
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

                        sse_parts.append(self._build_chunk_sse(request_id, model_id, text=text))

                    if sse_parts:
                        yield "".join(sse_parts)

                hit_max = gen_config.max_new_tokens is not None and streamer.total_tokens >= gen_config.max_new_tokens
                finish_reason = "length" if hit_max else "stop"

                if suffix is not None:
                    yield self._build_chunk_sse(request_id, model_id, text=suffix)
                usage = CompletionUsage(
                    prompt_tokens=input_len,
                    completion_tokens=streamer.total_tokens,
                    total_tokens=input_len + streamer.total_tokens,
                )
                yield self._build_chunk_sse(request_id, model_id, finish_reason=finish_reason, usage=usage)
            except (GeneratorExit, asyncio.CancelledError):
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
        suffix: str | None = None,
    ) -> "JSONResponse":
        """Run generation and return a JSONResponse."""
        text, input_len, generated_ids = await gen_manager.generate_non_streaming(
            model, processor, inputs, gen_config, request_id=request_id
        )

        if suffix is not None:
            text = text + suffix

        completion_tokens = len(generated_ids)
        hit_max = gen_config.max_new_tokens is not None and completion_tokens >= gen_config.max_new_tokens
        finish_reason = "length" if hit_max else "stop"

        usage = CompletionUsage(
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            total_tokens=input_len + completion_tokens,
        )

        result = Completion(
            id=request_id,
            created=int(time.time()),
            model=model_id,
            choices=[
                CompletionChoice(
                    text=text,
                    index=0,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            object="text_completion",
            usage=usage,
        )

        return JSONResponse(result.model_dump(exclude_none=True), media_type="application/json")

    # ----- helpers -----

    def _build_chunk_sse(
        self,
        request_id: str,
        model_id: str,
        text: str = "",
        finish_reason: str | None = None,
        usage: "CompletionUsage | None" = None,
    ) -> str:
        """Build a streaming ``Completion`` chunk and format it as an SSE ``data:`` line.

        Uses ``model_construct`` to bypass pydantic validation so that ``finish_reason``
        can be ``None`` for mid-stream chunks (the OpenAI SDK's ``CompletionChoice`` only
        accepts literal values).
        """
        chunk = Completion.model_construct(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model_id,
            choices=[
                CompletionChoice.model_construct(
                    text=text,
                    index=0,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )
        return self.chunk_to_sse(chunk)

    # ----- generation config -----

    def _build_generation_config(self, body: dict, model_generation_config: "GenerationConfig", use_cb: bool = False):
        """Apply legacy completion params (``max_tokens``, ``frequency_penalty``, ``stop``) on top of base config."""
        generation_config = super()._build_generation_config(body, model_generation_config, use_cb=use_cb)

        if body.get("max_tokens") is not None:
            generation_config.max_new_tokens = int(body["max_tokens"])
        if body.get("frequency_penalty") is not None:
            generation_config.repetition_penalty = 1.0 + float(body["frequency_penalty"])
        if body.get("stop") is not None:
            generation_config.stop_strings = body["stop"]

        return generation_config
