# Copyright 2025 The HuggingFace Inc. team.
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

import numpy as np
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_processing_base import make_legacy_audio_processor_alias
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class Gemma3nAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`Gemma3nAudioProcessorNumpy`]. Unfold-based STFT framed at
    `win_length + 1` samples so `_apply_frame_processing` can apply HTK-style preemphasis
    before reducing to `win_length`."""

    sample_rate = 16000
    force_mono = True
    max_length = 480000  # 30 seconds
    truncation = True
    pad_to_multiple_of = 128
    preemphasis_htk_flavor = True

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            win_length=512,
            hop_length=160,
            power=1.0,
            center=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=125.0,
            f_max=7600.0,
            mel_scale="htk",
            matmul_order="features_first",
        ),
        mel_floor=1e-5,
        log_mode="log",
        preemphasis=0.97,
        computation_dtype="float64",
    )

    def __init__(self, per_bin_mean=None, per_bin_stddev=None, **kwargs):
        super().__init__(**kwargs)

        win_length = self.spectrogram_config.stft_config.win_length
        # Match the numpy sibling's manual hann formula to keep the windows bit-equivalent
        # before being cast across backends.
        hann_arange = np.arange(win_length, dtype=np.float32)
        window_np = (0.5 * (1 - np.cos(2 * np.pi * hann_arange / win_length))).astype(np.float32)
        self.window = torch.from_numpy(window_np)

        n_mels = self.spectrogram_config.mel_scale_config.n_mels
        self.per_bin_mean = torch.as_tensor(per_bin_mean).reshape(1, n_mels) if per_bin_mean is not None else None
        self.per_bin_stddev = torch.as_tensor(per_bin_stddev).reshape(1, n_mels) if per_bin_stddev is not None else None

    def _apply_frame_processing(self, frames, *, spectrogram_config, **kwargs):
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None and preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first = frames[..., :1] * (1.0 - preemphasis)
                rest = frames[..., 1:-1] - preemphasis * frames[..., :-2]
                return torch.cat([first, rest], dim=-1)
            return frames[..., 1:] - preemphasis * frames[..., :-1]
        return frames[..., :-1]

    def _stft(self, audio, *, spectrogram_config, **kwargs):
        stft_cfg = spectrogram_config.stft_config
        frame_size_for_unfold = stft_cfg.win_length + 1
        # `audio.unfold` returns (..., num_frames, frame_size). After rfft along the last axis
        # we have (..., num_frames, freq); transpose to the canonical (..., freq, num_frames)
        # layout that the base `_apply_mel_scale` expects (it transposes again internally for
        # the `features_first` matmul).
        frames = audio.unfold(-1, frame_size_for_unfold, stft_cfg.hop_length)
        frames = self._apply_frame_processing(frames, spectrogram_config=spectrogram_config, **kwargs)
        window = self.window.to(device=audio.device, dtype=frames.dtype)
        frames = frames * window
        stft = torch.fft.rfft(frames, n=stft_cfg.n_fft, dim=-1)
        return stft.abs().transpose(-2, -1)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        result = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        return result.to(torch.float32)

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        hop_length = spectrogram_config.stft_config.hop_length
        if include_center_frame:
            frame_size = spectrogram_config.stft_config.win_length + 1
            return (audio_lengths - frame_size) // hop_length + 1
        return (audio_lengths + hop_length - 1) // hop_length


Gemma3nAudioFeatureExtractor = make_legacy_audio_processor_alias(Gemma3nAudioProcessor, "Gemma3nAudioFeatureExtractor")


__all__ = ["Gemma3nAudioProcessor", "Gemma3nAudioFeatureExtractor"]
