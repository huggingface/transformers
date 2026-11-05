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
"""Feature extractor class for LFM3-Audio."""

from dataclasses import dataclass

import numpy as np
import torch

from ...audio_utils import mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


@dataclass
class Lfm3AudioFeatureStreamingCache:
    """Unconsumed mono samples and frame count for one input stream."""

    samples: torch.Tensor | None = None
    num_samples: int = 0
    num_frames: int = 0
    finished: bool = False


class Lfm3AudioFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Computes the log-mel spectrogram the LFM3-Audio speech encoder was trained on: a 25 ms Hann window every 10 ms
    without centering, 128 Slaney mel filters, a base-10 logarithm clamped from below and an affine normalization.
    Clips are right-padded so that no trailing samples are dropped, and every 8 frames become one encoder token.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The number of mel filters.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate of the audio.
        n_fft (`int`, *optional*, defaults to 400):
            The FFT size.
        hop_length (`int`, *optional*, defaults to 160):
            The hop between frames, in samples.
        win_length (`int`, *optional*, defaults to 400):
            The window length, in samples.
        log_floor (`float`, *optional*, defaults to -12.0):
            The lower clamp of the log-mel values.
        normalization_offset (`float`, *optional*, defaults to 4.0):
            The value added to the log-mel features before dividing by `normalization_divisor`.
        normalization_divisor (`float`, *optional*, defaults to 4.0):
            The divisor of the normalization.
        encoder_stride (`int`, *optional*, defaults to 8):
            The number of spectrogram frames per encoder token.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value used to pad the spectrograms of a batch.
    """

    model_input_names = ["input_features", "feature_attention_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        log_floor: float = -12.0,
        normalization_offset: float = 4.0,
        normalization_divisor: float = 4.0,
        encoder_stride: int = 8,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.log_floor = log_floor
        self.normalization_offset = normalization_offset
        self.normalization_divisor = normalization_divisor
        self.encoder_stride = encoder_stride
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def num_frames(self, num_samples: int) -> int:
        padded = self.pad_length(num_samples)
        return 1 + (padded - self.n_fft) // self.hop_length

    def pad_length(self, num_samples: int) -> int:
        if num_samples < self.n_fft:
            return self.n_fft
        return num_samples + (-(num_samples - self.n_fft) % self.hop_length)

    def num_encoder_tokens(self, num_frames: int) -> int:
        return -(num_frames // -self.encoder_stride)

    def _log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=waveform.device)
        spectrogram = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        )
        power = spectrogram.abs().square()
        mel_filters = torch.from_numpy(self.mel_filters).to(device=waveform.device, dtype=power.dtype)
        log_mel = (mel_filters.T @ power).log10().clamp(min=self.log_floor)
        return (log_mel + self.normalization_offset) / self.normalization_divisor

    @torch.no_grad()
    def stream(
        self,
        raw_speech: np.ndarray | torch.Tensor | list[float],
        cache: Lfm3AudioFeatureStreamingCache | None = None,
        is_final: bool = False,
        sampling_rate: int | None = None,
    ) -> tuple[BatchFeature, Lfm3AudioFeatureStreamingCache]:
        """Extract completed windows from a mono chunk, retaining overlap. Finalization pads the last window."""
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected {self.sampling_rate} Hz, got {sampling_rate} Hz")
        if cache is None:
            cache = Lfm3AudioFeatureStreamingCache()
        if cache.finished:
            raise ValueError("This stream is finalized; start with a new cache")
        chunk = torch.as_tensor(raw_speech, dtype=torch.float32)
        if chunk.ndim != 1:
            raise ValueError("stream expects one mono waveform")
        samples = chunk if cache.samples is None else torch.cat((cache.samples, chunk))
        num_samples = cache.num_samples + chunk.numel()
        if is_final:
            count = self.num_frames(num_samples) - cache.num_frames
            if count:
                size = (count - 1) * self.hop_length + self.n_fft
                samples = torch.nn.functional.pad(samples, (0, size - samples.numel()))
        else:
            count = max(0, 1 + (samples.numel() - self.n_fft) // self.hop_length)
        features = self._log_mel(samples) if count else samples.new_empty((self.feature_size, 0))
        # the cache changes only once the chunk's features exist, so a failed call can be retried
        cache.samples = samples[count * self.hop_length :].clone() if not is_final else None
        cache.num_samples = num_samples
        cache.num_frames += count
        cache.finished = is_final
        return BatchFeature(
            {
                "input_features": features[None],
                "feature_attention_mask": torch.ones((1, count), device=chunk.device, dtype=torch.long),
            }
        ), cache

    def __call__(
        self,
        raw_speech: np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor] | list[float],
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        r"""
        raw_speech (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
            One mono clip or a list of mono clips, as float samples at `sampling_rate`.
        sampling_rate (`int`, *optional*):
            The sampling rate of `raw_speech`, checked against the extractor's.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
            If set, will return tensors instead of numpy arrays.
        """
        if kwargs:
            raise TypeError(
                f"{self.__class__.__name__} does not implement {sorted(kwargs)}: clips are always right-padded to "
                "whole frames and `feature_attention_mask` is always returned"
            )
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{self.__class__.__name__} expects audio sampled at {self.sampling_rate} Hz, got {sampling_rate} Hz"
            )
        if sampling_rate is None:
            logger.warning_once(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()` "
                "to avoid silent errors."
            )
        if isinstance(raw_speech, (list, tuple)) and len(raw_speech) == 0:
            raise ValueError(f"{self.__class__.__name__} needs at least one clip")
        is_batched = isinstance(raw_speech, (list, tuple)) and hasattr(raw_speech[0], "__len__")
        is_batched = is_batched or (isinstance(raw_speech, (np.ndarray, torch.Tensor)) and raw_speech.ndim == 2)
        clips = list(raw_speech) if is_batched else [raw_speech]
        clips = [
            clip.detach().cpu().float()
            if isinstance(clip, torch.Tensor)
            else torch.as_tensor(np.asarray(clip, dtype=np.float32))
            for clip in clips
        ]
        for clip in clips:
            if clip.ndim != 1:
                raise ValueError(
                    f"{self.__class__.__name__} expects mono clips, got a clip of shape {tuple(clip.shape)}"
                )

        features = []
        for clip in clips:
            clip = torch.nn.functional.pad(clip, (0, self.pad_length(clip.shape[0]) - clip.shape[0]))
            features.append(self._log_mel(clip))
        lengths = [feature.shape[-1] for feature in features]
        max_length = max(lengths)
        input_features = torch.full((len(features), self.feature_size, max_length), float(self.padding_value))
        feature_attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        for index, feature in enumerate(features):
            input_features[index, :, : feature.shape[-1]] = feature
            feature_attention_mask[index, : feature.shape[-1]] = 1

        data = {
            "input_features": input_features.numpy(),
            "feature_attention_mask": feature_attention_mask.numpy(),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Lfm3AudioFeatureExtractor", "Lfm3AudioFeatureStreamingCache"]
