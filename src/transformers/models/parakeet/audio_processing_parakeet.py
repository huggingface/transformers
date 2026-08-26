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

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class ParakeetAudioProcessorMixin:
    force_mono = True
    sampling_rate = 16000
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            hop_length=160,
            win_length=400,
            window_fn="hann_window",
            power=2.0,
            pad_mode="constant",
            periodic=False,
            magnitude_mode="sqrt_sum_squares",
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=0.0,
            norm="slaney",
            mel_scale="slaney",
            matmul_order="filters_first_matmul",
            bank_rounding="librosa",
        ),
        preemphasis=0.97,
        preemphasis_mode="waveform",
        log_mode="log",
        mel_floor=0.0,
        pre_log_offset=2**-24,
        transpose_features=True,
    )


class ParakeetAudioProcessor(ParakeetAudioProcessorMixin, TorchAudioBackend):
    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output

        features = output["audio_features"]
        stft_cfg = self.spectrogram_config.stft_config
        audio_lengths = torch.tensor([end - start for start, end in audio_ranges])
        features_lengths = torch.floor_divide(
            audio_lengths + stft_cfg.n_fft // 2 * 2 - stft_cfg.n_fft, stft_cfg.hop_length
        )
        attention_mask = torch.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = attention_mask.unsqueeze(-1)
        mel_masked = features * mask
        mean = (mel_masked.sum(dim=1) / features_lengths.unsqueeze(-1)).unsqueeze(1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        output["audio_features"] = (features - mean) / (std + 1e-5) * mask
        return output


__all__ = ["ParakeetAudioProcessor"]
