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

import json
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from openai.types import Completion, CompletionChoice, CompletionUsage
    from openai.types.completion_create_params import CompletionCreateParamsBase


from .utils import BaseHandler, EventStream, _StreamError, prompt_length


if TYPE_CHECKING:
    from transformers import GenerationConfig

    from .model_manager import EngineClient


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
    "user",
}


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

        engine = self._get_engine(body)
        tokenizer = getattr(engine.processor, "tokenizer", engine.processor)
        inputs = await self.run_in_thread(tokenizer, prompt, return_tensors=None if engine.use_cb else "pt")
        generation_kwargs = self._build_generation_kwargs(body, engine.generation_config)

        suffix = body.get("suffix")
        streaming = body.get("stream")
        self._log_request(request_id, engine, inputs, generation_kwargs, streaming, body)

        if streaming:
            return self._streaming(request_id, engine, inputs, generation_kwargs, suffix, body.get("stream_options"))
        else:
            return await self._non_streaming(request_id, engine, inputs, generation_kwargs, suffix)

    # ----- streaming -----

    def _streaming(
        self,
        request_id: str,
        engine: "EngineClient",
        inputs: dict,
        generation_kwargs: dict,
        suffix: str | None = None,
        stream_options: dict | None = None,
    ) -> "StreamingResponse":
        """Stream tokens as SSE. With ``stream_options={"include_usage": True}``, a last chunk has the usage."""
        include_usage = bool(stream_options and stream_options.get("include_usage"))
        model_id = engine.model_id
        queue, streamer = engine.generate_streaming(inputs, generation_kwargs)
        input_len = prompt_length(inputs)
        text_template = self.sse_template(self._build_chunk_sse(request_id, model_id, text="<text>"), text="<text>")

        async def sse_gen() -> AsyncGenerator[str, None]:
            try:
                done = False
                while not done:
                    new_outputs = await self.get_new_outputs(queue)
                    sse_parts: list[str] = []
                    for text in new_outputs:
                        if text is None:
                            done = True
                            break
                        if isinstance(text, _StreamError):
                            sse_parts.append(f"data: {json.dumps({'error': text.msg})}\n\n")
                            yield "".join(sse_parts)
                            return

                        sse_parts.append(text_template.format(text=json.dumps(text, ensure_ascii=False)))

                    if sse_parts:
                        yield "".join(sse_parts)

                finish_reason = self._get_finish_reason(generation_kwargs, streamer.total_tokens)

                if suffix is not None:
                    yield self._build_chunk_sse(request_id, model_id, text=suffix)
                yield self._build_chunk_sse(request_id, model_id, finish_reason=finish_reason)
                if include_usage:
                    usage = CompletionUsage(
                        prompt_tokens=input_len,
                        completion_tokens=streamer.total_tokens,
                        total_tokens=input_len + streamer.total_tokens,
                    )
                    yield self._build_chunk_sse(request_id, model_id, usage=usage)
            finally:  # stops the generation if the client left
                streamer.cancel()

        return EventStream(sse_gen(), media_type="text/event-stream")

    # ----- non-streaming -----

    async def _non_streaming(
        self,
        request_id: str,
        engine: "EngineClient",
        inputs: dict,
        generation_kwargs: dict,
        suffix: str | None = None,
    ) -> "JSONResponse":
        """Run generation and return a JSONResponse."""
        model_id = engine.model_id
        text, input_len, generated_ids = await engine.generate_non_streaming(inputs, generation_kwargs)

        if suffix is not None:
            text = text + suffix

        completion_tokens = len(generated_ids)
        finish_reason = self._get_finish_reason(generation_kwargs, completion_tokens)

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
        accepts literal values). With `usage`, it's the usage chunk, the last one, with no choices
        (``stream_options.include_usage``).
        """
        choice = CompletionChoice.model_construct(text=text, index=0, logprobs=None, finish_reason=finish_reason)
        chunk = Completion.model_construct(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model_id,
            choices=[] if usage is not None else [choice],
            usage=usage,
        )
        return self.chunk_to_sse(chunk)

    # ----- generation config -----

    def _build_generation_kwargs(self, body: dict, default: "GenerationConfig") -> dict:
        """Adds the legacy completion params (``max_tokens``, ``frequency_penalty``, ``stop``) to the shared ones."""
        generation_kwargs = super()._build_generation_kwargs(body, default)

        if body.get("max_tokens") is not None:
            generation_kwargs["max_new_tokens"] = int(body["max_tokens"])
        if body.get("frequency_penalty") is not None:
            generation_kwargs["repetition_penalty"] = 1.0 + float(body["frequency_penalty"])
        if body.get("stop") is not None:
            generation_kwargs["stop_strings"] = body["stop"]

        return generation_kwargs
