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
Handler for the /v1/audio/speech endpoint.

Supports streaming and non-streaming responses.
"""

import asyncio
import base64
import inspect
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import torch

from ...utils import logging
from ...utils.import_utils import is_serve_available
from .utils import BaseHandler, GenerateManager, _encode_audio, _StreamError, streaming_wav_header


if is_serve_available():
    from fastapi import HTTPException
    from fastapi.responses import Response, StreamingResponse
    from openai.types.audio.speech_create_params import SpeechCreateParams


if TYPE_CHECKING:
    from ...generation import GenerationConfig
    from ...modeling_utils import PreTrainedModel
    from ...processing_utils import ProcessorMixin


logger = logging.get_logger(__name__)


class TransformersSpeechCreateParams(SpeechCreateParams, total=False):
    ref_audio: str
    ref_text: str
    generation_config: str
    seed: int


UNUSED_SPEECH_FIELDS = {"speed"}

# OpenAI's speech formats and their media types
AUDIO_FORMATS = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
}


class SpeechHandler(BaseHandler):
    """Handler for the `/v1/audio/speech` endpoint.

    Supports both streaming and non-streaming responses.
    """

    _valid_params_class = TransformersSpeechCreateParams
    _unused_fields = UNUSED_SPEECH_FIELDS

    async def handle_request(self, body: dict, request_id: str) -> "Response | StreamingResponse":
        """Validate the request, load the model, and dispatch to streaming or non-streaming.

        Args:
            body (`dict`): The raw JSON request body (OpenAI speech format).
            request_id (`str`): Unique request identifier (from header or auto-generated).

        Returns:
            `Response | StreamingResponse`: The audio file, or a stream of PCM if ``body["stream_format"]`` is set.
        """
        self._validate_request(body)

        model_id, model, processor = self._resolve_model(body)
        if self.generation_state._continuous_batching:
            logger.warning_once(
                "/v1/audio/speech does not support continuous batching. Falling back to sequential generation."
            )
        gen_manager = self.generation_state.get_manager(model_id, use_cb=False)  # type: ignore[assignment]
        processor_inputs = self.get_processor_inputs_from_request(body, model.config.model_type)
        # Some processors encode `ref_audio` with a codec model (e.g. Higgs Audio V2), which needs no gradients.
        # They return the audio in float32, so it is cast to the model's dtype, as in the model docs.
        # TODO: models such as VibeVoice could cast their audio inputs themselves, like CLIP does for `pixel_values`
        with torch.inference_mode():
            inputs = processor.apply_chat_template(
                processor_inputs,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(model.device, model.dtype)  # type: ignore[union-attr]

        gen_config = self._build_generation_config(body, model.generation_config)
        # Describe the audio, which raw PCM does not carry
        headers = {"X-Audio-Sample-Rate": str(processor.feature_extractor.sampling_rate), "X-Audio-Channels": "1"}

        streaming = "stream_format" in body
        # Streaming needs a model whose `generate()` accepts an `audio_streamer`
        if streaming and "audio_streamer" not in inspect.signature(model._sample).parameters:
            raise HTTPException(status_code=400, detail="This model does not support streaming audio.")

        if streaming:
            return self._streaming(body, model, processor, inputs, gen_config, gen_manager, headers)
        else:
            return await self._non_streaming(body, model, processor, inputs, gen_config, gen_manager, headers)

    def _validate_request(self, body: dict) -> None:
        """Validate the speech-specific fields on top of the base validation."""
        super()._validate_request(body)
        if body.setdefault("response_format", "wav") not in AUDIO_FORMATS:
            raise HTTPException(status_code=400, detail=f"Supported response formats are {', '.join(AUDIO_FORMATS)}.")
        # Setting `stream_format` asks for streaming, of WAV (a header, then PCM) or PCM: the other formats are encoded
        # across chunks
        if "stream_format" in body:
            if body["stream_format"] not in ("audio", "sse"):
                raise HTTPException(status_code=400, detail="`stream_format` must be 'audio' or 'sse'.")
            if body["response_format"] not in ("wav", "pcm"):
                logger.warning_once("Streaming sends WAV or PCM: `response_format` is replaced by 'wav'.")
                body["response_format"] = "wav"
        if not isinstance(body.get("ref_audio", ""), str):
            raise HTTPException(status_code=422, detail="`ref_audio` must be base64 audio, a URL or a file path.")
        if not isinstance(body.get("ref_text", ""), str):
            raise HTTPException(status_code=422, detail="`ref_text` must be the transcript of `ref_audio`.")

    def _build_generation_config(
        self, body: dict, model_generation_config: "GenerationConfig", use_cb: bool = False
    ) -> "GenerationConfig":
        """Request ``generate()``'s dict output, which carries the audio, on top of the base generation config."""
        generation_config = super()._build_generation_config(body, model_generation_config, use_cb=use_cb)
        generation_config.return_dict_in_generate = True
        return generation_config

    def _streaming(
        self,
        body: dict,
        model: "PreTrainedModel",
        processor: "ProcessorMixin",
        inputs: dict,
        gen_config: "GenerationConfig",
        gen_manager: GenerateManager,
        headers: dict,
    ) -> "StreamingResponse":
        """Stream audio via AudioStreamer, as WAV or PCM, in raw bytes or SSE events."""
        queue, audio_streamer = gen_manager.generate_audio_streaming(model, inputs, gen_config)
        input_len = inputs["input_ids"].shape[-1]
        # A WAV stream starts with its header, then carries the same PCM chunks
        wav_header = None
        if body["response_format"] == "wav":
            wav_header = streaming_wav_header(processor.feature_extractor.sampling_rate)

        async def sse_gen() -> AsyncGenerator[str, None]:
            try:
                if wav_header:
                    yield self.chunk_to_sse(
                        json.dumps(
                            {"type": "speech.audio.delta", "audio": base64.b64encode(wav_header).decode("ascii")}
                        )
                    )
                while (item := await queue.get()) is not None:
                    if isinstance(item, _StreamError):
                        yield self.chunk_to_sse(json.dumps({"type": "error", "message": item.msg}))
                        return
                    # Each item is a chunk of 16-bit PCM, sent base64-encoded
                    yield self.chunk_to_sse(
                        json.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(item).decode("ascii")})
                    )
                usage = {
                    "input_tokens": input_len,
                    "output_tokens": audio_streamer.total_tokens,
                    "total_tokens": input_len + audio_streamer.total_tokens,
                }
                yield self.chunk_to_sse(json.dumps({"type": "speech.audio.done", "usage": usage}))
            except (GeneratorExit, asyncio.CancelledError):
                # Client disconnected — abort generation to free GPU.
                # Re-raise is mandatory: Python raises RuntimeError if GeneratorExit is swallowed.
                audio_streamer.cancel()
                raise

        async def audio_gen() -> AsyncGenerator[bytes, None]:
            try:
                if wav_header:
                    yield wav_header
                while (item := await queue.get()) is not None:
                    if isinstance(item, _StreamError):
                        # Raw audio cannot carry an error: abort the response, so the client sees it is incomplete
                        raise RuntimeError(item.msg)
                    yield item
            except (GeneratorExit, asyncio.CancelledError):
                # Client disconnected — abort generation to free GPU.
                # Re-raise is mandatory: Python raises RuntimeError if GeneratorExit is swallowed.
                audio_streamer.cancel()
                raise

        if body["stream_format"] == "sse":
            return StreamingResponse(
                sse_gen(), media_type="text/event-stream", headers={**headers, "X-Accel-Buffering": "no"}
            )
        return StreamingResponse(audio_gen(), media_type=AUDIO_FORMATS[body["response_format"]], headers=headers)

    async def _non_streaming(
        self,
        body: dict,
        model: "PreTrainedModel",
        processor: "ProcessorMixin",
        inputs: dict,
        gen_config: "GenerationConfig",
        gen_manager: GenerateManager,
        headers: dict,
    ) -> "Response":
        """Run generation and return the encoded audio."""
        waveform = await gen_manager.generate_audio_non_streaming(model, processor, inputs, gen_config)
        if waveform is None:
            raise RuntimeError("The model did not generate audio.")
        audio = _encode_audio(waveform, body["response_format"], processor.feature_extractor.sampling_rate)
        return Response(audio, media_type=AUDIO_FORMATS[body["response_format"]], headers=headers)

    @staticmethod
    def get_processor_inputs_from_request(body: dict, model_type: str) -> list[dict]:
        """Convert an OpenAI speech request to the messages expected by HF processors.

        Args:
            body (`dict`): The OpenAI speech request.
            model_type (`str`): The model type.

        Returns:
            `list[dict]`: Processor-compatible messages.
        """
        text = {"type": "text", "text": body["input"]}
        voice = None
        if body.get("ref_audio"):
            voice = {"type": "audio", "url": body["ref_audio"]}
        instructions = body.get("instructions")

        if model_type == "vibevoice":
            # Roles are speaker IDs
            return [{"role": "0", "content": [text, voice] if voice else [text]}]

        if model_type == "higgs_audio_v2":
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]}
            ]
            # The instructions go in the scene, which the template expects right after the system turn
            scene = []
            if instructions:
                scene.append({"type": "text", "text": instructions})
            example = []
            if voice:
                if body.get("ref_text"):
                    # With its transcript, the voice is an example turn, as in the model docs
                    example = [
                        {"role": "user", "content": [{"type": "text", "text": body["ref_text"]}]},
                        {"role": "assistant", "content": [voice]},
                    ]
                else:
                    # Without it, the voice goes in the scene
                    scene.append(voice)
            if scene:
                messages.append({"role": "scene", "content": scene})
            messages += example
            messages.append({"role": "user", "content": [text]})
            return messages

        return [{"role": "user", "content": [text, voice] if voice else [text]}]
