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
Handler for the /v1/audio/transcriptions endpoint.
"""

import io
from typing import TYPE_CHECKING

from ...utils import logging
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from openai.types.audio.transcription_create_params import TranscriptionCreateParamsBase

from .model_manager import ModelManager
from .utils import DirectStreamer, GenerateManager, GenerationState, _StreamError


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin


logger = logging.get_logger(__name__)


class TransformersTranscriptionCreateParams(TranscriptionCreateParamsBase, total=False):
    stream: bool


UNUSED_TRANSCRIPTION_FIELDS = {
    "chunking_strategy",
    "include",
    "language",
    "prompt",
    "response_format",
    "temperature",
    "timestamp_granularities",
}


class TranscriptionHandler:
    """Handler for ``POST /v1/audio/transcriptions``.

    Accepts a multipart/form-data request with an audio file and model name,
    runs speech-to-text, and returns an OpenAI-compatible Transcription response.

    Standalone (does not extend :class:`BaseHandler`) because audio requests use
    multipart form data, not JSON bodies, and don't need generation config or
    validation. Shares the :class:`GenerationState` for thread safety.
    """

    def __init__(self, model_manager: ModelManager, generation_state: GenerationState):
        """
        Args:
            model_manager (`ModelManager`): Handles model loading, caching, and lifecycle.
            generation_state (`GenerationState`): Shared generation state for thread safety.
        """
        self.model_manager = model_manager
        self.generation_state = generation_state

    def _validate_request(self, form_keys: set[str]) -> None:
        """Validate transcription request fields."""
        unexpected = form_keys - TransformersTranscriptionCreateParams.__mutable_keys__
        if unexpected:
            raise HTTPException(status_code=422, detail=f"Unexpected fields in the request: {unexpected}")
        unused = form_keys & UNUSED_TRANSCRIPTION_FIELDS
        if unused:
            logger.warning_once(f"Ignoring unsupported fields in the request: {unused}")

    async def handle_request(self, request: Request) -> JSONResponse | StreamingResponse:
        """Parse multipart form, run transcription, return result.

        Args:
            request (`Request`): FastAPI request containing multipart form data with
                ``file`` (audio bytes), ``model`` (model ID), and optional ``stream`` flag.

        Returns:
            `JSONResponse | StreamingResponse`: Transcription result or SSE stream.
        """
        from transformers.utils.import_utils import is_librosa_available, is_multipart_available

        if not is_librosa_available():
            raise ImportError("Missing librosa dependency for audio transcription. Install with `pip install librosa`")
        if not is_multipart_available():
            raise ImportError(
                "Missing python-multipart dependency for file uploads. Install with `pip install python-multipart`"
            )

        async with request.form() as form:
            self._validate_request(set(form.keys()))
            file_bytes = await form["file"].read()
            model = form["model"]
            stream = str(form.get("stream", "false")).lower() == "true"

        model_id_and_revision = self.model_manager.process_model_name(model)
        audio_model, audio_processor = self.model_manager.load_model_and_processor(model_id_and_revision)
        gen_manager = self.generation_state.get_manager(model_id_and_revision)
        audio_inputs = self._prepare_audio_inputs(file_bytes, audio_processor, audio_model)

        if stream:
            return self._streaming(gen_manager, audio_model, audio_processor, audio_inputs)
        return await self._non_streaming(gen_manager, audio_model, audio_processor, audio_inputs)

    @staticmethod
    def _prepare_audio_inputs(
        file_bytes: bytes, audio_processor: "ProcessorMixin", audio_model: "PreTrainedModel"
    ) -> dict:
        """Load audio bytes and convert to model inputs."""
        import librosa

        sampling_rate = audio_processor.feature_extractor.sampling_rate
        audio_array, _ = librosa.load(io.BytesIO(file_bytes), sr=sampling_rate, mono=True)
        audio_inputs = audio_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").to(
            audio_model.device
        )
        audio_inputs["input_features"] = audio_inputs["input_features"].to(audio_model.dtype)
        return audio_inputs

    async def _non_streaming(
        self,
        gen_manager: GenerateManager,
        audio_model: "PreTrainedModel",
        audio_processor: "ProcessorMixin",
        audio_inputs: dict,
    ) -> JSONResponse:
        # Audio models have different inputs (input_features) and decode (batch_decode)
        # than text models, so we use async_submit() directly instead of
        # generate_non_streaming()
        from openai.types.audio import Transcription

        generated_ids = await gen_manager.async_submit(audio_model.generate, **audio_inputs)
        text = audio_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return JSONResponse(Transcription(text=text).model_dump(exclude_none=True))

    def _streaming(
        self,
        gen_manager: GenerateManager,
        audio_model: "PreTrainedModel",
        audio_processor: "ProcessorMixin",
        audio_inputs: dict,
    ) -> StreamingResponse:
        # Same as _non_streaming — uses submit() directly because audio inputs
        # differ from text.
        import asyncio

        tokenizer = audio_processor.tokenizer if hasattr(audio_processor, "tokenizer") else audio_processor
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        streamer = DirectStreamer(tokenizer._tokenizer, loop, queue, skip_special_tokens=True)
        gen_kwargs = {**audio_inputs, "streamer": streamer}

        def _run():
            try:
                audio_model.generate(**gen_kwargs)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, _StreamError(str(e)))

        gen_manager.submit(_run)

        async def sse_gen():
            while True:
                text = await queue.get()
                if text is None:
                    break
                if isinstance(text, _StreamError):
                    yield f'data: {{"error": "{text.msg}"}}\n\n'
                    return
                yield f"data: {text}\n\n"

        return StreamingResponse(sse_gen(), media_type="text/event-stream")
