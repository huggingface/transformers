# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Feature extractor class for TML audio models."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


def _to_exact_int(value: float, name: str, tolerance: float = 1e-6) -> int:
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


def _resample(samples: np.ndarray, src_sample_rate: int, sample_rate: int) -> np.ndarray:
    audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    resampled = F.resample(audio, orig_freq=src_sample_rate, new_freq=sample_rate)
    return resampled.detach().cpu().numpy().astype(np.float32, copy=False)


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    """Slaney mel scale, matching the librosa/torchaudio convention."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = frequencies / f_sp
    log = min_log_mel + np.log(np.maximum(frequencies, min_log_hz) / min_log_hz) / logstep
    return np.where(frequencies >= min_log_hz, log, linear)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    linear = mels * f_sp
    log = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log, linear)


def _mel_filter_bank(sampling_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    fft_bins = n_fft // 2 + 1
    fft_freqs = np.arange(fft_bins, dtype=np.float64) * sampling_rate / n_fft
    mel_edges = _mel_to_hz(
        np.linspace(
            _hz_to_mel(np.array([0.0]))[0],
            _hz_to_mel(np.array([sampling_rate / 2.0]))[0],
            n_mels + 2,
            dtype=np.float64,
        )
    )
    mel_widths = np.diff(mel_edges)
    lower = (fft_freqs[None, :] - mel_edges[:-2, None]) / mel_widths[:-1, None]
    upper = (mel_edges[2:, None] - fft_freqs[None, :]) / mel_widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalization.
    weights *= (2.0 / (mel_edges[2:] - mel_edges[:-2]))[:, None]
    return torch.from_numpy(weights.astype(np.float32, copy=False)).contiguous()


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class InklingFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a TML audio feature extractor, which converts raw audio waveforms into discretized dMel
    bins: a log-mel spectrogram whose values are quantized into `num_dmel_bins` equal-width bins.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]
    which contains most of the main methods. Users should refer to this superclass for more information
    regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features, i.e. the number of mel filterbanks.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value used to pad the dMel bin sequences to the same length in a batch.
        num_dmel_bins (`int`, *optional*, defaults to 16):
            Number of discrete bins each (clamped) log-mel value is quantized into.
        dmel_min_value (`float`, *optional*, defaults to -7.0):
            Lower clamp bound, in log10 space, used for dMel quantization.
        dmel_max_value (`float`, *optional*, defaults to 2.0):
            Upper clamp bound, in log10 space, used for dMel quantization.
        audio_token_duration_s (`float`, *optional*, defaults to 0.05):
            Duration, in seconds, represented by a single audio token, i.e. the STFT hop length.
        window_size_multiplier (`float`, *optional*, defaults to 2.0):
            Multiplier applied to `audio_token_duration_s` to obtain the STFT window length.
        n_fft (`int`, *optional*):
            FFT size. Defaults to the window length (`audio_token_duration_s * window_size_multiplier *
            sampling_rate`) when not provided.
    """

    model_input_names = ["dmel_bins", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        num_dmel_bins: int = 16,
        dmel_min_value: float = -7.0,
        dmel_max_value: float = 2.0,
        audio_token_duration_s: float = 0.05,
        window_size_multiplier: float = 2.0,
        n_fft: int | None = None,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.num_dmel_bins = num_dmel_bins
        self.dmel_min_value = dmel_min_value
        self.dmel_max_value = dmel_max_value
        self.audio_token_duration_s = audio_token_duration_s
        self.window_size_multiplier = window_size_multiplier

        self.hop_length = _to_exact_int(
            audio_token_duration_s * sampling_rate, "audio_token_duration_s * sampling_rate"
        )
        self.window_size = _to_exact_int(
            audio_token_duration_s * window_size_multiplier * sampling_rate,
            "audio_token_duration_s * window_size_multiplier * sampling_rate",
        )
        self.n_fft = n_fft or self.window_size
        if self.hop_length <= 0 or self.window_size <= 0 or self.n_fft <= 0:
            raise ValueError("hop_length, window_size, and n_fft must all be positive")

        # Precomputed once at init, mirrors e.g. WhisperFeatureExtractor.mel_filters.
        self.window = torch.hann_window(self.window_size, periodic=True, dtype=torch.float32)
        self.mel_filters = _mel_filter_bank(sampling_rate, self.n_fft, feature_size)
        self.bin_centers = torch.linspace(dmel_min_value, dmel_max_value, num_dmel_bins, dtype=torch.float64)

    def to_dict(self) -> dict[str, Any]:
        # Keep tensors out of preprocessor_config.json; they're cheaply recomputed from the
        # scalar config fields on load, same pattern as Whisper's `mel_filters`.
        output = super().to_dict()
        for key in ("window", "mel_filters", "bin_centers"):
            output.pop(key, None)
        return output

    def _extract_dmel_bins(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() == 0:
            return torch.empty((0, self.feature_size), dtype=torch.float32)

        hop_length, window_size, n_fft = self.hop_length, self.window_size, self.n_fft
        right_pad = math.ceil(waveform.numel() / hop_length) * hop_length - waveform.numel()
        left_pad = max(n_fft - hop_length, 0)
        waveform = F.pad(waveform, (left_pad, right_pad))

        spec = torch.stft(
            waveform.unsqueeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_size,
            window=self.window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec_ri = torch.view_as_real(spec)
        magnitude = (spec_ri[..., 0].square() + spec_ri[..., 1].square()).clamp_min(1e-10).sqrt().squeeze(0)

        mel = self.mel_filters.matmul(magnitude).clamp_min(1e-10).log10()
        mel = mel.to(torch.float64).clamp(min=self.dmel_min_value, max=self.dmel_max_value)

        dmel_bins = (mel.unsqueeze(-1) - self.bin_centers).abs().argmin(dim=-1)
        return dmel_bins.to(torch.float32).T.contiguous()  # (T, feature_size)

    def _normalize_input(self, raw_speech) -> list[torch.Tensor]:
        """Coerce `raw_speech` into a list of 1-D float32 waveform tensors at `self.sampling_rate`."""
        decoded_audio = []
        for item in raw_speech:
            decoded_audio.append(self.fetch_audio(item, sampling_rate=self.sampling_rate))

        if isinstance(decoded_audio, np.ndarray) and decoded_audio.ndim == 1:
            decoded_audio = [decoded_audio]
        elif isinstance(decoded_audio, np.ndarray) and decoded_audio.ndim == 2:
            decoded_audio = list(raw_speech)
        elif not isinstance(decoded_audio, (list, tuple)):
            decoded_audio = [decoded_audio]
        return decoded_audio

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]] | bytes | str | Sequence,
        sampling_rate: int | None = None,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = True,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Extract dMel bin features from one or several audio clip(s).

        Args:
            raw_speech:
                A single waveform (`np.ndarray`/`List[float]`), a batch of waveforms, or raw audio
                bytes / a local path / `file://` URI / file-like object (or a list thereof). Waveforms
                are assumed to already be at `self.sampling_rate`; bytes/paths are decoded and resampled
                internally.
            sampling_rate (`int`, *optional*):
                The sampling rate of `raw_speech`, used only to validate against `self.sampling_rate`
                when `raw_speech` is passed as already-decoded array(s).
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor was trained using a sampling "
                    f"rate of {self.sampling_rate}. Please make sure that the provided audio input "
                    f"was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning_once(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        waveforms = self._normalize_input(raw_speech)
        dmel_bins = [self._extract_dmel_bins(w) for w in waveforms]
        num_audio_tokens = [bins.shape[0] for bins in dmel_bins]

        batch_input = BatchFeature(data={"dmel_bins": [bins.numpy() for bins in dmel_bins]})
        batch_input = self.pad(
            batch_input,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )
        batch_input["num_audio_tokens"] = num_audio_tokens
        return batch_input


__all__ = ["InklingFeatureExtractor"]
