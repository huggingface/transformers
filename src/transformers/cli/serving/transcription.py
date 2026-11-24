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

import base64
from typing import TYPE_CHECKING

from ...audio_utils import load_audio
from ...utils import logging
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from openai.types.audio.transcription_create_params import TranscriptionCreateParamsBase

from .model_manager import ModelManager
from .utils import _StreamError, transcribe


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from .model_manager import EngineClient


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
    multipart form data, not JSON bodies. The model runs in its own process (see
    `ModelManager.get_engine`).
    """

    def __init__(self, model_manager: ModelManager):
        """
        Args:
            model_manager (`ModelManager`): The models the server runs.
        """
        self.model_manager = model_manager

    def _validate_request(self, form_keys: set[str]) -> None:
        """Validate transcription request fields."""
        unexpected = form_keys - getattr(TransformersTranscriptionCreateParams, "__mutable_keys__", set())
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
        from transformers.utils.import_utils import is_multipart_available

        if not is_multipart_available():
            raise ImportError(
                "Missing python-multipart dependency for file uploads. Install with `pip install python-multipart`"
            )

        async with request.form() as form:
            self._validate_request(set(form.keys()))
            file_field = form["file"]
            if isinstance(file_field, str):
                raise HTTPException(status_code=422, detail="Expected file upload, got string")
            file_bytes = await file_field.read()
            model = form["model"]
            if not isinstance(model, str):
                raise HTTPException(status_code=422, detail="Expected model name as string")
            stream = str(form.get("stream", "false")).lower() == "true"

        engine = self.model_manager.get_engine(model)
        audio_inputs = self._prepare_audio_inputs(file_bytes, engine.processor)
        if stream:
            return self._streaming(engine, audio_inputs)
        return await self._non_streaming(engine, audio_inputs)

    @staticmethod
    def _prepare_audio_inputs(file_bytes: bytes, audio_processor: "ProcessorMixin") -> dict:
        """Load audio bytes and convert to model inputs."""
        sampling_rate = audio_processor.feature_extractor.sampling_rate
        audio_array = load_audio(base64.b64encode(file_bytes).decode("ascii"), sampling_rate=sampling_rate)
        return dict(audio_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt"))

    async def _non_streaming(self, engine: "EngineClient", audio_inputs: dict) -> JSONResponse:
        from openai.types.audio import Transcription

        generated_ids = await engine.submit(transcribe, audio_inputs)
        text = engine.processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
        return JSONResponse(Transcription(text=text).model_dump(exclude_none=True))

    def _streaming(self, engine: "EngineClient", audio_inputs: dict) -> StreamingResponse:
        queue, streamer = engine.submit_streaming(transcribe, audio_inputs)

        async def sse_gen():
            try:
                while True:
                    text = await queue.get()
                    if text is None:
                        break
                    if isinstance(text, _StreamError):
                        yield f'data: {{"error": "{text.msg}"}}\n\n'
                        return
                    yield f"data: {text}\n\n"
            finally:  # stops the transcription if the client left
                streamer.cancel()

        return StreamingResponse(sse_gen(), media_type="text/event-stream")
