# Copyright 2026 The rednote-hilab team and the HuggingFace Inc. team. All rights reserved.
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
"""Exact waveform feature extraction for the Dots3-Note audio encoder."""

from __future__ import annotations

import base64
import binascii
import math
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from ...audio_utils import mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torch_available, logging
from ...utils.import_utils import requires
from .configuration_dots3_note_omni import Dots3NoteOmniAudioConfig


if is_torch_available():
    import torch
    import torch.nn.functional as F


logger = logging.get_logger(__name__)


def compute_audio_token_length(
    num_samples: int,
    *,
    chunk_samples: int = 960_000,
    token_stride: int = 1_280,
) -> int:
    """Return the exact number of AE embeddings emitted for one waveform."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if chunk_samples <= 0 or token_stride <= 0:
        raise ValueError("chunk_samples and token_stride must be positive")
    return sum(
        math.ceil(min(chunk_samples, num_samples - start) / token_stride)
        for start in range(0, num_samples, chunk_samples)
    )


@requires(backends=("torch",))
def _pad_or_trim(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.shape[-1] > length:
        waveform = waveform[..., :length]
    if waveform.shape[-1] < length:
        waveform = F.pad(waveform, (0, length - waveform.shape[-1]))
    return waveform


@requires(backends=("torch",))
class Dots3NoteOmniFeatureExtractor(SequenceFeatureExtractor):
    """Convert 16 kHz channel-first waveforms into Dots3 log-mel chunks."""

    model_input_names = [
        "input_features",
        "chunk_sample_lengths",
        "chunk_token_lengths",
        "audio_token_lengths",
        "audio_chunk_counts",
    ]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_seconds: int = 60,
        conv_temporal_stride: int = 8,
        merge_factor: int = 1,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_seconds = chunk_seconds
        self.conv_temporal_stride = conv_temporal_stride
        self.merge_factor = merge_factor
        self.chunk_samples = chunk_seconds * sampling_rate
        self.token_stride = hop_length * conv_temporal_stride * merge_factor
        filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=float(sampling_rate) / 2.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = np.ascontiguousarray(filters.T, dtype=np.float32)

    @classmethod
    def from_audio_config(cls, config: Dots3NoteOmniAudioConfig) -> Dots3NoteOmniFeatureExtractor:
        return cls(
            feature_size=config.feature_size,
            sampling_rate=config.sampling_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            chunk_seconds=config.chunk_seconds,
            conv_temporal_stride=config.conv_temporal_stride,
            merge_factor=config.merge_factor,
        )

    def token_length(self, num_samples: int) -> int:
        return compute_audio_token_length(
            num_samples,
            chunk_samples=self.chunk_samples,
            token_stride=self.token_stride,
        )

    @staticmethod
    def _resolve_audio_source(source: str):
        if source.startswith(("http://", "https://")):
            with urlopen(source, timeout=30) as response:
                return BytesIO(response.read())
        if source.startswith("data:"):
            try:
                _, payload = source.split(",", 1)
            except ValueError as error:
                raise ValueError("invalid data URI audio input") from error
            return BytesIO(base64.b64decode(payload))
        path = Path(source)
        if path.is_file():
            return str(path)
        try:
            return BytesIO(base64.b64decode(source, validate=True))
        except (ValueError, binascii.Error) as error:
            raise ValueError("audio string must be a path, URL, data URI, or base64 payload") from error

    def fetch_audio(self, audio_url_or_urls, sampling_rate: int | None = None):
        """Decode with torchaudio and preserve the serving path's channel-0 rule."""
        if isinstance(audio_url_or_urls, (list, tuple)) and audio_url_or_urls:
            if not isinstance(audio_url_or_urls[0], (int, float, np.integer, np.floating)):
                return [self.fetch_audio(item, sampling_rate=sampling_rate) for item in audio_url_or_urls]
        if not isinstance(audio_url_or_urls, str):
            return audio_url_or_urls

        import torchaudio

        target_rate = sampling_rate or self.sampling_rate
        waveform, source_rate = torchaudio.load(self._resolve_audio_source(audio_url_or_urls))
        if waveform.shape[-1] <= 0 or source_rate <= 0:
            raise ValueError(f"invalid decoded audio: samples={waveform.shape[-1]}, sample_rate={source_rate}")
        if source_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=source_rate, new_freq=target_rate)
        return waveform[0].to(torch.float32).cpu().numpy()

    def _extract_log_mel(self, waveforms: torch.Tensor, device: str) -> torch.Tensor:
        waveforms = waveforms.to(device=device, dtype=torch.float32)
        stft = torch.stft(
            waveforms,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32, device=device),
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().pow(2)
        mel_spec = torch.from_numpy(self.mel_filters).to(device) @ magnitudes
        log_spec = mel_spec.clamp_min(1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0

    @staticmethod
    def _as_clips(raw_speech) -> list:
        if isinstance(raw_speech, (np.ndarray, torch.Tensor)):
            if raw_speech.ndim > 2:
                raise ValueError("a single audio input must be 1-D or channel-first 2-D")
            return [raw_speech]
        if isinstance(raw_speech, (list, tuple)):
            if not raw_speech:
                raise ValueError("received an empty audio batch")
            if isinstance(raw_speech[0], (int, float, np.integer, np.floating)):
                return [raw_speech]
            return list(raw_speech)
        raise TypeError(f"unsupported audio input type: {type(raw_speech)}")

    @staticmethod
    def _select_first_channel(clip) -> torch.Tensor:
        waveform = clip if isinstance(clip, torch.Tensor) else torch.as_tensor(np.asarray(clip))
        waveform = waveform.to(torch.float32)
        if waveform.ndim == 2:
            waveform = waveform[0]
        elif waveform.ndim != 1:
            raise ValueError(f"audio input must be 1-D or channel-first 2-D, got {tuple(waveform.shape)}")
        if waveform.numel() == 0:
            raise ValueError("audio waveform must contain at least one sample")
        return waveform.contiguous()

    def __call__(
        self,
        raw_speech,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = "pt",
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"expected {self.sampling_rate} Hz audio, received {sampling_rate} Hz; "
                "resample before feature extraction"
            )
        if sampling_rate is None:
            logger.warning_once("Pass sampling_rate=16000 to avoid silent audio errors.")

        clips = [self._select_first_channel(clip) for clip in self._as_clips(raw_speech)]
        chunks = []
        chunk_sample_lengths = []
        chunk_token_lengths = []
        audio_token_lengths = []
        audio_chunk_counts = []
        chunk_audio_indices = []

        for audio_index, waveform in enumerate(clips):
            per_audio_tokens = 0
            per_audio_chunks = 0
            for start in range(0, waveform.numel(), self.chunk_samples):
                chunk = waveform[start : start + self.chunk_samples]
                sample_length = int(chunk.numel())
                token_length = math.ceil(sample_length / self.token_stride)
                chunks.append(_pad_or_trim(chunk, self.chunk_samples))
                chunk_sample_lengths.append(sample_length)
                chunk_token_lengths.append(token_length)
                chunk_audio_indices.append(audio_index)
                per_audio_tokens += token_length
                per_audio_chunks += 1
            audio_token_lengths.append(per_audio_tokens)
            audio_chunk_counts.append(per_audio_chunks)

        input_features = self._extract_log_mel(torch.stack(chunks), device=device)
        data = {
            "input_features": input_features,
            "chunk_sample_lengths": torch.tensor(chunk_sample_lengths, dtype=torch.long, device=device),
            "chunk_token_lengths": torch.tensor(chunk_token_lengths, dtype=torch.long, device=device),
            "audio_token_lengths": torch.tensor(audio_token_lengths, dtype=torch.long, device=device),
            "audio_chunk_counts": torch.tensor(audio_chunk_counts, dtype=torch.long, device=device),
            "chunk_audio_indices": torch.tensor(chunk_audio_indices, dtype=torch.long, device=device),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Dots3NoteOmniFeatureExtractor"]
