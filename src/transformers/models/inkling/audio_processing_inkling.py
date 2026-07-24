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

import math

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class InklingAudioProcessor(TorchAudioBackend):
    """Audio processor for Inkling.

    Produces log10 mel-filterbank features (mel energies in log10 space). Uses the base
    `_standard_mel_banks` (slaney norm + slaney mel scale — no librosa), a magnitude (not power)
    spectrogram, and Inkling's fixed framing: `center=False` with a left pad of `n_fft - hop`
    and a right pad up to a multiple of `hop`. Downstream dMel quantization is done by
    `InklingProcessor`, not here.
    """

    sample_rate = 16000
    force_mono = True
    model_input_names = ["input_features", "input_features_mask"]
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1600,
            hop_length=800,
            win_length=1600,
            window_fn="hann_window",
            power=1.0,
            center=False,
            periodic=True,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            norm="slaney",
            mel_scale="slaney",
        ),
        log_mode="log10",
        mel_floor=1e-10,
    )

    def _stft(self, audio, *, spectrogram_config, audio_ranges=None, **kwargs):
        # Inkling's fixed framing: left-pad (n_fft - hop) and right-pad up to a hop multiple, center=False.
        stft_cfg = spectrogram_config.stft_config
        hop, n_fft = stft_cfg.hop_length, stft_cfg.n_fft
        right_pad = math.ceil(audio.shape[-1] / hop) * hop - audio.shape[-1]
        left_pad = max(n_fft - hop, 0)
        audio = torch.nn.functional.pad(audio, (left_pad, right_pad))
        return super()._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        magnitudes = torch.view_as_real(stft_out)
        magnitudes = magnitudes.pow(2).sum(-1).clamp_min(1e-10).sqrt()
        if power != 1.0:
            magnitudes = magnitudes.pow(power)
        return magnitudes

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Pointwise log10 (ADR 0005), then transpose to (batch, frames, mels).
        features = features.clamp_min(spectrogram_config.mel_floor).log10()
        return features.transpose(1, 2)

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        # Inkling emits ceil(audio_length / hop) frames (its right-pad rounds up to a hop multiple).
        hop = spectrogram_config.stft_config.hop_length
        return (audio_lengths + hop - 1) // hop

    def _postprocess_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        # No normalization; zero padded frames and emit the legacy keys the model consumes.
        # The mask is named `input_features_mask` so it doesn't collide with a text `attention_mask`.
        features = output.pop("audio_features")
        mask = output.pop("audio_features_mask", None)
        if mask is not None:
            features = features * mask.unsqueeze(-1).to(features.dtype)
            output["input_features_mask"] = mask
        output["input_features"] = features
        return output


__all__ = ["InklingAudioProcessor"]
