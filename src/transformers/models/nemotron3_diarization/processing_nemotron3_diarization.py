# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for Nemotron3Diarization."""

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import auto_docstring


class Nemotron3DiarizationProcessorKwargs(ProcessingKwargs, total=False):
    # Defaults travel with the checkpoint, in `processor_config.json`.
    _defaults = {}


@auto_docstring
class Nemotron3DiarizationProcessor(ProcessorMixin):
    r"""
    Wraps the feature extractor of a Nemotron3Diarization checkpoint and sizes the chunks of a streaming session.

    The model consumes `chunk_length` encoder frames per step and looks `chunk_right_context` frames ahead, so each
    chunk of a streaming session carries `num_mel_frames_per_audio_chunk` mel frames and the next chunk starts
    `num_mel_frames_per_step` frames later, the look-ahead frames of the previous chunk opening it.
    """

    def __init__(self, feature_extractor, chunk_length=340, chunk_right_context=40, subsampling_factor=8):
        r"""
        chunk_length (`int`, *optional*, defaults to 340):
            Number of encoder frames the model processes per streaming step, mirroring
            `Nemotron3DiarizationConfig.chunk_length`.
        chunk_right_context (`int`, *optional*, defaults to 40):
            Number of look-ahead encoder frames appended to each chunk, mirroring
            `Nemotron3DiarizationConfig.chunk_right_context`.
        subsampling_factor (`int`, *optional*, defaults to 8):
            Number of mel frames per encoder frame, mirroring `Nemotron3DiarizationConfig.subsampling_factor`.
        """
        self.chunk_length = chunk_length
        self.chunk_right_context = chunk_right_context
        self.subsampling_factor = subsampling_factor
        super().__init__(feature_extractor)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: int | None = None,
        is_streaming: bool = False,
        is_first_audio_chunk: bool = True,
        **kwargs: Unpack[Nemotron3DiarizationProcessorKwargs],
    ) -> BatchFeature:
        r"""
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. Validated against the feature extractor's expected sampling
            rate (16000 Hz) when provided.
        is_streaming (`bool`, *optional*, defaults to `False`):
            Whether the audio is one chunk of a streaming session, `is_first_audio_chunk` telling the first chunk
            from the later ones.
        is_first_audio_chunk (`bool`, *optional*, defaults to `True`):
            Whether this is the first chunk of a streaming session. The feature extractor centers the analysis
            windows (`center=True`) for the first chunk and for offline use, and does not (`center=False`) for the
            later chunks, so that the per-chunk spectrogram reproduces, frame for frame, a single full-utterance
            pass. Must be `True` when `is_streaming=False`.

        Returns:
            [`BatchFeature`]: the feature extractor outputs, `input_features` and `attention_mask`. In streaming mode
            the trailing frames whose analysis window reaches past the chunk are dropped, so `input_features` holds
            exactly the frames of the chunk and can be passed to the model as is.
        """
        if not is_streaming and not is_first_audio_chunk:
            raise ValueError("In non-streaming mode (`is_streaming=False`), `is_first_audio_chunk` must be `True`.")

        audio = make_list_of_audio(audio)
        output_kwargs = self._merge_kwargs(Nemotron3DiarizationProcessorKwargs, **kwargs)
        inputs = self.feature_extractor(
            audio, sampling_rate=sampling_rate, center=is_first_audio_chunk, **output_kwargs["audio_kwargs"]
        )
        if is_streaming:
            num_frames = int(inputs["attention_mask"].sum(-1).max())
            inputs["input_features"] = inputs["input_features"][:, :num_frames]
            inputs["attention_mask"] = inputs["attention_mask"][:, :num_frames]
        return inputs

    def set_streaming_profile(self, chunk_length: int, chunk_right_context: int):
        """
        Selects a latency profile, re-deriving every chunk size below. The same values must reach the model, whose
        `chunk_length` and `chunk_right_context` are configuration fields:
        `Nemotron3DiarizationForAudioFrameClassification.from_pretrained(..., chunk_length=..., chunk_right_context=...)`.
        """
        self.chunk_length = chunk_length
        self.chunk_right_context = chunk_right_context

    @property
    def streaming_latency_ms(self) -> int:
        """
        Input buffer latency (ms) of the current profile: the model emits a chunk once its look-ahead frames have
        arrived, i.e. after `(chunk_length + chunk_right_context)` encoder frames.
        """
        encoder_frame_ms = (
            self.subsampling_factor * self.feature_extractor.hop_length / self.feature_extractor.sampling_rate * 1000
        )
        return round((self.chunk_length + self.chunk_right_context) * encoder_frame_ms)

    @property
    def num_mel_frames_per_audio_chunk(self) -> int:
        """Number of mel frames each streaming chunk carries: its own frames plus the look-ahead frames."""
        return (self.chunk_length + self.chunk_right_context) * self.subsampling_factor

    @property
    def num_mel_frames_per_step(self) -> int:
        """
        Number of mel frames the model emits per streaming chunk, i.e. how far the frame cursor advances between two
        chunks (`num_mel_frames_per_audio_chunk` minus the look-ahead frames).
        """
        return self.chunk_length * self.subsampling_factor

    @property
    def num_samples_first_audio_chunk(self) -> int:
        """
        Number of audio samples to feed for the first chunk of a session (`is_first_audio_chunk=True`, centered
        windows) so that the processor returns exactly `num_mel_frames_per_audio_chunk` frames.
        """
        return (self.num_mel_frames_per_audio_chunk - 1) * self.feature_extractor.hop_length + (
            self.feature_extractor.win_length // 2
        )

    @property
    def num_samples_per_audio_chunk(self) -> int:
        """
        Number of audio samples to feed for a later chunk of a session (`is_first_audio_chunk=False`, uncentered
        windows) so that the processor returns exactly `num_mel_frames_per_audio_chunk` frames.
        """
        return (
            self.num_mel_frames_per_audio_chunk * self.feature_extractor.hop_length + self.feature_extractor.win_length
        )

    def audio_chunk_start(self, mel_frame_idx: int) -> int:
        """
        First audio sample of the chunk starting at `mel_frame_idx`. An uncentered window starts half a transform
        before the frame it belongs to, so a chunk starts `n_fft // 2` samples before its first frame.
        """
        return mel_frame_idx * self.feature_extractor.hop_length - self.feature_extractor.n_fft // 2

    @property
    def model_input_names(self):
        return self.feature_extractor.model_input_names


__all__ = ["Nemotron3DiarizationProcessor"]
