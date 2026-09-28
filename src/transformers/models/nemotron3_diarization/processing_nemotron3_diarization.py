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
from ...utils import auto_docstring, is_torch_available
from ...utils.import_utils import requires


if is_torch_available():
    import torch


DEFAULT_STREAMING_MODES = {
    "low_latency": (9, 4),  # 1.04 s
    "very_low_latency": (6, 2),  # 0.64 s
    "ultra_low_latency": (3, 1),  # 0.32 s
}


class Nemotron3DiarizationProcessorKwargs(ProcessingKwargs, total=False):
    # Defaults travel with the checkpoint, in `processor_config.json`.
    _defaults = {}


@auto_docstring
class Nemotron3DiarizationProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, subsampling_factor=8, streaming_modes=None, streaming_mode="low_latency"):
        r"""
        subsampling_factor (`int`, *optional*, defaults to 8):
            Number of mel frames per encoder frame, mirroring `Nemotron3DiarizationAudioConfig.subsampling_factor`.
        streaming_modes (`dict[str, tuple[int, int]]`, *optional*):
            Streaming modes the checkpoint supports, name to `(chunk_length, chunk_right_context)` in encoder frames.
            The processor is the single source of truth for this set:
            [`~Nemotron3DiarizationProcessor.set_streaming_mode`] validates against it. Defaults to the model-card
            modes, `"low_latency"` (9, 4), `"very_low_latency"` (6, 2) and `"ultra_low_latency"` (3, 1).
        streaming_mode (`str`, *optional*, defaults to `"low_latency"`):
            Streaming mode of the sessions, one of `streaming_modes`; change it with
            [`~Nemotron3DiarizationProcessor.set_streaming_mode`]. Offline use ignores it.
        """
        self.subsampling_factor = subsampling_factor
        self.streaming_modes = streaming_modes if streaming_modes is not None else DEFAULT_STREAMING_MODES
        self.set_streaming_mode(streaming_mode)
        super().__init__(feature_extractor)

    def set_streaming_mode(self, streaming_mode: str):
        """
        Selects the streaming mode of the sessions among `streaming_modes`, `"low_latency"`, `"very_low_latency"` or
        `"ultra_low_latency"` for the released checkpoint: every chunk size below and the `num_lookahead_frames` given
        to the model re-derive from it.
        """
        if streaming_mode not in self.streaming_modes:
            raise ValueError(
                f"Unknown `streaming_mode` {streaming_mode!r}, expected one of {list(self.streaming_modes)}."
            )
        self.streaming_mode = streaming_mode

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: int | None = None,
        is_streaming: bool = False,
        is_first_audio_chunk: bool = True,
        is_last_audio_chunk: bool = False,
        **kwargs: Unpack[Nemotron3DiarizationProcessorKwargs],
    ) -> BatchFeature:
        r"""
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. Validated against the feature extractor's expected sampling
            rate (16000 Hz) when provided.
        is_streaming (`bool`, *optional*, defaults to `False`):
            Whether the audio is one chunk of a streaming session, `is_first_audio_chunk` and `is_last_audio_chunk`
            telling the first and the last chunks from the others. The chunk sizes are those of `streaming_mode`,
            changed with [`~Nemotron3DiarizationProcessor.set_streaming_mode`]. Every chunk but the last must hold exactly
            `num_samples_first_audio_chunk` audio samples for the first one and `num_samples_per_audio_chunk` for
            the later ones.
        is_first_audio_chunk (`bool`, *optional*, defaults to `True`):
            Whether this is the first chunk of a streaming session. The feature extractor centers the analysis
            windows (`center=True`) for the first chunk and for offline use, and does not (`center=False`) for the
            later chunks, so that the per-chunk spectrogram reproduces, frame for frame, a single full-utterance
            pass. Must be `True` when `is_streaming=False`.
        is_last_audio_chunk (`bool`, *optional*, defaults to `False`):
            Whether this chunk ends the streaming session. A chunk of a session ends with `chunk_right_context`
            look-ahead encoder frames that the model scores at the next step only, and that its next chunk opens
            with. The last chunk has no next step, so every one of its frames is scored, whatever their number. Must
            be `False` when `is_streaming=False`.

        Returns:
            [`BatchFeature`]: the feature extractor outputs, `input_features` and `attention_mask`. In streaming mode
            the trailing frames whose analysis window reaches past the chunk are dropped, so `input_features` holds
            exactly the frames of the chunk and can be passed to the model as is, and every chunk but the last also
            carries `num_lookahead_frames`, the number of its trailing look-ahead encoder frames, which puts the
            model in streaming mode.
        """
        if not is_streaming:
            if not is_first_audio_chunk or is_last_audio_chunk:
                raise ValueError(
                    "In non-streaming mode (`is_streaming=False`), `is_first_audio_chunk` must be `True` and "
                    "`is_last_audio_chunk` must be `False`."
                )

        audio = make_list_of_audio(audio)
        output_kwargs = self._merge_kwargs(Nemotron3DiarizationProcessorKwargs, **kwargs)
        inputs = self.feature_extractor(
            audio, sampling_rate=sampling_rate, center=is_first_audio_chunk, **output_kwargs["audio_kwargs"]
        )
        if is_streaming:
            num_frames = int(inputs["attention_mask"].sum(-1).max())
            inputs["input_features"] = inputs["input_features"][:, :num_frames]
            inputs["attention_mask"] = inputs["attention_mask"][:, :num_frames]
            if not is_last_audio_chunk:
                expected_num_frames = self.num_mel_frames_per_audio_chunk
                if num_frames != expected_num_frames:
                    which = "num_samples_first_audio_chunk" if is_first_audio_chunk else "num_samples_per_audio_chunk"
                    raise ValueError(
                        f"A `{self.streaming_mode}` chunk must hold {expected_num_frames} mel frames, got "
                        f"{num_frames}: feed `{which}` audio samples, or pass `is_last_audio_chunk=True` for the last "
                        "chunk of the session."
                    )
                inputs["num_lookahead_frames"] = self._streaming_chunk_sizes[1]
        return inputs

    @property
    def _streaming_chunk_sizes(self) -> tuple[int, int]:
        """`(chunk_length, chunk_right_context)` of the streaming mode, in encoder frames."""
        # a list once reloaded from `processor_config.json`
        chunk_length, chunk_right_context = self.streaming_modes[self.streaming_mode]
        return chunk_length, chunk_right_context

    @property
    def streaming_latency_ms(self) -> int:
        """
        Input buffer latency (ms) of the streaming mode: the model emits a chunk once its look-ahead frames have
        arrived, i.e. after `(chunk_length + chunk_right_context)` encoder frames.
        """
        chunk_length, chunk_right_context = self._streaming_chunk_sizes
        encoder_frame_ms = (
            self.subsampling_factor * self.feature_extractor.hop_length / self.feature_extractor.sampling_rate * 1000
        )
        return round((chunk_length + chunk_right_context) * encoder_frame_ms)

    @property
    def num_mel_frames_per_audio_chunk(self) -> int:
        """Number of mel frames each streaming chunk carries: its own frames plus the look-ahead frames."""
        chunk_length, chunk_right_context = self._streaming_chunk_sizes
        return (chunk_length + chunk_right_context) * self.subsampling_factor

    @property
    def num_mel_frames_per_step(self) -> int:
        """
        Number of mel frames the model emits per streaming chunk, i.e. how far the frame cursor advances between two
        chunks (`num_mel_frames_per_audio_chunk` minus the look-ahead frames).
        """
        chunk_length, _ = self._streaming_chunk_sizes
        return chunk_length * self.subsampling_factor

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

    @requires(backends=("torch",))
    def extract_speaker_dict(
        self,
        logits: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        threshold: float = 0.5,
    ) -> list[list[dict]]:
        """
        Turns the per-frame speaker logits of [`Nemotron3DiarizationForAudioFrameClassification`] into speech
        segments, in the format of [`VibeVoiceAsrProcessor.extract_speaker_dict`] without the transcription.

        Args:
            logits (`torch.Tensor` of shape `(batch_size, num_frames, num_speakers)`):
                Logits returned by the model, at the spectrogram frame rate (one per 10 ms). For a streaming
                session, the concatenated logits of its chunks.
            attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
                Valid frames of each sample of a padded batch, as returned by the processor.
            threshold (`float`, *optional*, defaults to 0.5):
                Speaker probability above which a frame counts as speech of that speaker.

        Returns:
            `list[list[dict]]`: for each sample, its speech segments sorted by start time, each a dict with the
            `"Start"` and `"End"` times in seconds and the `"Speaker"` index, speakers being numbered in order of
            first arrival. Overlapping speech gives overlapping segments.

        Example:

        ```python
        >>> processor.extract_speaker_dict(logits, inputs.attention_mask)
        [[{'Start': 0.0, 'End': 15.43, 'Speaker': 0}, {'Start': 15.43, 'End': 21.05, 'Speaker': 1}]]
        ```
        """
        frame_duration = self.feature_extractor.hop_length / self.feature_extractor.sampling_rate
        active = logits.sigmoid() > threshold
        if attention_mask is not None:
            active = active & attention_mask.to(device=active.device, dtype=torch.bool)[..., None]
        active = active.int()
        boundary = active.new_zeros(active.shape[0], 1, active.shape[2])
        # +1 where a speaker starts talking, -1 where they stop
        changes = torch.cat([boundary, active, boundary], dim=1).diff(dim=1)

        speaker_dicts = []
        for sample_changes in changes:
            segments = []
            for speaker in range(sample_changes.shape[1]):
                starts = (sample_changes[:, speaker] == 1).nonzero()[:, 0].tolist()
                ends = (sample_changes[:, speaker] == -1).nonzero()[:, 0].tolist()
                segments.extend(
                    {
                        "Start": round(start * frame_duration, 2),
                        "End": round(end * frame_duration, 2),
                        "Speaker": speaker,
                    }
                    for start, end in zip(starts, ends)
                )
            segments.sort(key=lambda segment: (segment["Start"], segment["Speaker"]))
            speaker_dicts.append(segments)
        return speaker_dicts

    def audio_chunk_start(self, mel_frame_idx: int) -> int:
        """
        First audio sample of the chunk starting at `mel_frame_idx`. An uncentered window starts half a transform
        before the frame it belongs to, so a chunk starts `n_fft // 2` samples before its first frame.
        """
        return mel_frame_idx * self.feature_extractor.hop_length - self.feature_extractor.n_fft // 2

    @property
    def model_input_names(self):
        return self.feature_extractor.model_input_names + ["num_lookahead_frames"]


__all__ = ["Nemotron3DiarizationProcessor"]
