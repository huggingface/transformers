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

Supports streaming and non-streaming responses.
"""

import asyncio
import copy
import inspect
import io
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import numpy as np

from ...utils import logging
from ...utils.import_utils import is_multipart_available, is_serve_available, is_torchcodec_available, requires
from .utils import BaseHandler, GenerateManager, _StreamError, speaker_labels, words_to_diarized_segments


if is_serve_available():
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from openai.types.audio import (
        Transcription,
        TranscriptionDiarized,
        TranscriptionVerbose,
        TranscriptionWord,
    )
    from openai.types.audio.transcription_create_params import TranscriptionCreateParamsBase


if TYPE_CHECKING:
    from ...generation import GenerationConfig
    from ...modeling_utils import PreTrainedModel
    from ...processing_utils import ProcessorMixin


logger = logging.get_logger(__name__)


class TransformersTranscriptionCreateParams(TranscriptionCreateParamsBase, total=False):
    stream: bool


UNUSED_TRANSCRIPTION_FIELDS = {
    "chunking_strategy",
    "include",
    "known_speaker_names",
    "known_speaker_references",
    "language",
    "temperature",
}

# The model card's instruction for speaker tags, which diarized_json sends when the request has no prompt: Nemotron 3.5
# Transcribe (`nemo_speechlm`) only tells speakers apart when asked to
SPEAKER_PROMPTS = {
    "nemo_speechlm": (
        "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. "
        "Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the "
        "transcription."
    ),
}


def split_audio(audio: np.ndarray, sampling_rate: int, max_seconds: float) -> list[np.ndarray]:
    """Split audio into chunks of at most `max_seconds`, each cut at the quietest 0.1 s of its last second."""
    chunk_size, window = int(max_seconds * sampling_rate), sampling_rate // 10
    chunks, start = [], 0
    while len(audio) - start > chunk_size:
        search_start = start + chunk_size - sampling_rate
        # The 0.1 s windows of the last second, without its final one
        windows = audio[search_start : search_start + 9 * window].reshape(9, window)
        cut = search_start + int(np.argmin(np.mean(windows**2, axis=1))) * window
        chunks.append(audio[start:cut])
        start = cut
    chunks.append(audio[start:])
    return chunks


class TranscriptionHandler(BaseHandler):
    """Handler for the `/v1/audio/transcriptions` endpoint.

    Supports both streaming and non-streaming responses.
    """

    _valid_params_class = TransformersTranscriptionCreateParams
    _unused_fields = UNUSED_TRANSCRIPTION_FIELDS

    async def handle_request(self, request: "Request") -> "JSONResponse | PlainTextResponse | StreamingResponse":
        """Read and validate the request, load the model, and dispatch to streaming or non-streaming.

        Args:
            request (`Request`): The multipart form request (OpenAI transcription format), with the audio `file`.

        Returns:
            `JSONResponse | PlainTextResponse | StreamingResponse`: The transcript, or a stream of it if the request
            sets `stream`.
        """
        if not is_multipart_available():
            raise ImportError("File uploads need python-multipart: `pip install python-multipart`.")
        # The form's fields, with the uploaded file as bytes
        async with request.form() as form:
            body = {}
            for key in form:
                values = [value if isinstance(value, str) else await value.read() for value in form.getlist(key)]
                # The OpenAI client sends arrays as `key[]`
                body[key.removesuffix("[]")] = values if key.endswith("[]") else values[-1]
        self._validate_request(body)

        model_id, model, processor = self._resolve_model(body)
        self._validate_model_support(body, model, processor)
        if self.generation_state._continuous_batching:
            logger.warning_once(
                "/v1/audio/transcriptions does not support continuous batching. Falling back to sequential generation."
            )
        gen_manager = self.generation_state.get_manager(model_id, use_cb=False)  # type: ignore[assignment]
        gen_config = self._get_generation_config(model)
        streaming = body.get("stream") == "true"

        audio = self._load_audio(body["file"], processor)
        inputs, num_chunks = self._prepare_inputs(audio, model, processor, body.get("prompt"))

        if streaming:
            return self._streaming(model, processor, inputs, num_chunks, gen_config, gen_manager)
        else:
            return await self._non_streaming(body, model, processor, audio, inputs, gen_config, gen_manager)

    def _validate_request(self, body: dict) -> None:
        """Validate the transcription-specific fields on top of the base validation."""
        super()._validate_request(body)
        if not isinstance(body.get("file"), bytes):
            raise HTTPException(status_code=422, detail="`file` must be an uploaded audio file.")
        if body.get("stream", "false") not in ("true", "false"):
            raise HTTPException(status_code=422, detail="`stream` must be 'true' or 'false'.")
        if body.setdefault("response_format", "json") not in ("json", "text", "verbose_json", "diarized_json"):
            raise HTTPException(
                status_code=400, detail="Supported response formats are json, text, verbose_json and diarized_json."
            )
        if body["response_format"] in ("verbose_json", "diarized_json") and body.get("stream") == "true":
            raise HTTPException(
                status_code=400, detail=f"A stream sends the transcript only, not {body['response_format']}."
            )
        if body.get("timestamp_granularities") and body["response_format"] != "verbose_json":
            logger.warning_once(
                f"`timestamp_granularities` is ignored with response_format={body['response_format']}: only "
                "verbose_json returns word timestamps."
            )
        # diarized_json times its speaker turns with the words, and the formats other than verbose_json have none
        if body["response_format"] == "diarized_json":
            body["timestamp_granularities"] = ["word"]
        elif body["response_format"] != "verbose_json":
            body["timestamp_granularities"] = []
        if set(body.setdefault("timestamp_granularities", [])) - {"word"}:
            raise HTTPException(status_code=400, detail="Only word timestamps are supported.")

    @staticmethod
    def _validate_model_support(body: dict, model: "PreTrainedModel", processor: "ProcessorMixin") -> None:
        """Validate the fields that depend on the loaded model. The prompt of a model that takes none is ignored, and
        diarized_json asks a model that only tells speakers apart when asked for speaker tags."""
        if body["response_format"] == "diarized_json" and model.config.model_type in SPEAKER_PROMPTS:
            body.setdefault("prompt", SPEAKER_PROMPTS[model.config.model_type])
        # Word timestamps come from the model's own alignment (e.g. Nemotron 3.5 Transcribe's timestamp head)
        if "word" in body["timestamp_granularities"] and not hasattr(processor, "align_word_timestamps"):
            raise HTTPException(
                status_code=400, detail="This model does not support word timestamps, which diarized_json also needs."
            )
        apply_transcription_request = getattr(processor, "apply_transcription_request", None)
        if "prompt" in body and not (
            apply_transcription_request and "prompt" in inspect.signature(apply_transcription_request).parameters
        ):
            logger.warning_once("This model takes no prompt: `prompt` is ignored.")
            del body["prompt"]

    def _streaming(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin",
        inputs: dict,
        num_chunks: int,
        gen_config: "GenerationConfig",
        gen_manager: GenerateManager,
    ) -> "StreamingResponse":
        """Stream the transcript as SSE events while it is generated."""
        queue, streamer = gen_manager.generate_transcription_streaming(
            model, processor, inputs, num_chunks, gen_config
        )

        async def sse_gen() -> AsyncGenerator[str, None]:
            try:
                while (item := await queue.get()) is not None:
                    if isinstance(item, _StreamError):
                        yield self.chunk_to_sse(json.dumps({"error": item.msg}))
                        return
                    yield self.chunk_to_sse(item)
            except (GeneratorExit, asyncio.CancelledError):
                # Client disconnected — abort generation to free GPU.
                # Re-raise is mandatory: Python raises RuntimeError if GeneratorExit is swallowed.
                streamer.cancel()
                raise

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    async def _non_streaming(
        self,
        body: dict,
        model: "PreTrainedModel",
        processor: "ProcessorMixin",
        audio: np.ndarray,
        inputs: dict,
        gen_config: "GenerationConfig",
        gen_manager: GenerateManager,
    ) -> "JSONResponse | PlainTextResponse":
        """Run generation and return the transcript, with verbose_json also the audio's duration and the words, and
        with diarized_json the duration and the speaker turns."""
        word_timestamps = "word" in body["timestamp_granularities"]
        text, words = await gen_manager.generate_transcription_non_streaming(
            model, processor, inputs, gen_config, word_timestamps=word_timestamps
        )
        if body["response_format"] == "text":
            return PlainTextResponse(text)
        if body["response_format"] == "json":
            return JSONResponse(Transcription(text=text).model_dump(exclude_none=True))
        duration = len(audio) / processor.feature_extractor.sampling_rate
        if body["response_format"] == "diarized_json":
            response = TranscriptionDiarized(
                duration=duration, segments=words_to_diarized_segments(words), task="transcribe", text=text
            )
            return JSONResponse(response.model_dump(exclude_none=True))
        # OpenAI requires `language`, which is null: not every model reports the language it hears
        response = TranscriptionVerbose.model_construct(text=text, duration=duration, language=None)
        if word_timestamps:
            # `speaker` is an extension of OpenAI's word, labeled as in diarized_json, which only times speaker turns
            response.words = [
                TranscriptionWord.model_construct(
                    word=word["text"], start=round(start, 3), end=round(end, 3), speaker=speaker
                )
                for word, speaker in zip(words, speaker_labels(words))
                for start, end in [word["timestamp"]]
            ]
        return JSONResponse(response.model_dump(exclude_unset=True))

    @staticmethod
    @requires(backends=("librosa",))
    def _load_audio(file_bytes: bytes, processor: "ProcessorMixin") -> np.ndarray:
        """Load the uploaded audio as mono samples at the processor's sampling rate, with librosa, or with torchcodec
        for the formats librosa cannot read (e.g. m4a, mp4, webm)."""
        import librosa

        sampling_rate = processor.feature_extractor.sampling_rate
        try:
            return librosa.load(io.BytesIO(file_bytes), sr=sampling_rate, mono=True)[0]
        except RuntimeError:
            if not is_torchcodec_available():
                raise HTTPException(
                    status_code=400, detail="Unsupported audio file. Install torchcodec to read more formats."
                ) from None
        from torchcodec.decoders import AudioDecoder

        try:
            return (
                AudioDecoder(file_bytes, sample_rate=sampling_rate, num_channels=1).get_all_samples().data[0].numpy()
            )
        except RuntimeError:
            raise HTTPException(status_code=400, detail="Invalid or unsupported audio file.") from None

    @staticmethod
    def _get_generation_config(model: "PreTrainedModel") -> "GenerationConfig":
        """The model's generation config. A model that sets no length generates until it stops or fills its context,
        instead of `generate()`'s default of 20 tokens."""
        generation_config = copy.deepcopy(model.generation_config)
        if generation_config.max_new_tokens is None and generation_config.max_length is None:
            # TODO: only `max_position_embeddings` is read, while some configs name their context length otherwise
            # (e.g. `n_positions`, `max_target_positions`)
            generation_config.max_length = model.config.get_text_config(decoder=True).max_position_embeddings
        return generation_config

    @staticmethod
    def _prepare_inputs(
        audio: np.ndarray, model: "PreTrainedModel", processor: "ProcessorMixin", prompt: str | None
    ) -> tuple[dict, int]:
        """Convert the audio to the inputs of ``model.generate()``: one row per chunk of audio, cut at the window the
        model declares.

        Returns:
            `tuple[dict, int]`: The inputs, and the number of chunks.
        """
        window = getattr(processor.feature_extractor, "chunk_length", None)
        chunks = split_audio(audio, processor.feature_extractor.sampling_rate, window) if window else [audio]
        apply_transcription_request = getattr(processor, "apply_transcription_request", None)
        if apply_transcription_request is None:
            inputs = processor(chunks, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
        elif prompt is None:
            inputs = apply_transcription_request(audio=chunks)
        else:
            # The prompt replaces the model's instruction, e.g. to ask Nemotron 3.5 Transcribe for speaker tags
            inputs = apply_transcription_request(audio=chunks, prompt=prompt)
        # The processor returns float32 features, which the audio encoder only takes in the model's dtype
        return inputs.to(model.device, model.dtype), len(chunks)
