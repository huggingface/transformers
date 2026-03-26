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
Handler for the /v1/audio/transcriptions endpoint.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse, StreamingResponse

from ...utils import logging
from .model_manager import ModelManager
from .utils import DirectStreamer, GenerationState, _StreamError


if TYPE_CHECKING:
    from fastapi import Request

logger = logging.get_logger(__name__)


class TranscriptionHandler:
    """Handler for ``POST /v1/audio/transcriptions``.

    Accepts a multipart/form-data request with an audio file and model name,
    runs speech-to-text, and returns an OpenAI-compatible Transcription response.

    Standalone (does not extend :class:`BaseHandler`) because audio requests use
    multipart form data, not JSON bodies, and don't need generation config or
    validation. Shares the :class:`GenerationState` for thread safety.
    """

    def __init__(self, model_manager: ModelManager, generation_state: GenerationState):
        self.model_manager = model_manager
        self._generation_state = generation_state

    async def handle_request(self, request: Request) -> JSONResponse | StreamingResponse:
        """Parse multipart form, run transcription, return result."""
        from transformers.utils.import_utils import is_librosa_available

        if not is_librosa_available():
            raise ImportError("Missing librosa dependency for audio transcription. Install with `pip install librosa`")

        import librosa

        async with request.form() as form:
            file_bytes = await form["file"].read()
            model = form["model"]
            stream = str(form.get("stream", "false")).lower() == "true"

        model_id_and_revision = self.model_manager.process_model_name(model)
        audio_model, audio_processor = self.model_manager.load_model_and_processor(model_id_and_revision)

        # Read audio with librosa at the model's expected sampling rate
        model_sampling_rate = audio_processor.feature_extractor.sampling_rate
        audio_array, _ = librosa.load(io.BytesIO(file_bytes), sr=model_sampling_rate, mono=True)
        audio_inputs = audio_processor(audio_array, sampling_rate=model_sampling_rate, return_tensors="pt").to(
            audio_model.device
        )
        audio_inputs["input_features"] = audio_inputs["input_features"].to(audio_model.dtype)

        # Transcription uses the per-model InferenceThread (no CB for audio).
        gen_manager = self._generation_state.get_manager(model_id_and_revision, use_cb=False)
        tokenizer = audio_processor.tokenizer if hasattr(audio_processor, "tokenizer") else audio_processor

        if stream:
            return self._streaming(gen_manager, audio_model, tokenizer, audio_inputs)
        return await self._non_streaming(gen_manager, audio_model, audio_processor, audio_inputs)

    async def _non_streaming(self, gen_manager, audio_model, audio_processor, audio_inputs) -> JSONResponse:
        # Audio models have different inputs (input_features) and decode (batch_decode)
        # than text models, so we use async_submit() directly instead of
        # generate_non_streaming(). TODO: add generate_audio_non_streaming() when
        # more audio modalities are supported.
        from openai.types.audio import Transcription

        generated_ids = await gen_manager.async_submit(audio_model.generate, **audio_inputs)
        text = audio_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return JSONResponse(Transcription(text=text).model_dump(exclude_none=True))

    def _streaming(self, gen_manager, audio_model, tokenizer, audio_inputs) -> StreamingResponse:
        # Same as _non_streaming — uses submit() directly because audio inputs
        # differ from text. TODO: add generate_audio_streaming() when more audio
        # modalities are supported.
        import asyncio

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
