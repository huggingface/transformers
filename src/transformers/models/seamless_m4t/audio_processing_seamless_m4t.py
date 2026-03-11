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

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, mel_filter_bank, spectrogram, window_function
from ...feature_extraction_utils import BatchFeature


class SeamlessM4tAudioProcessor(NumpyAudioBackend):
    sample_rate = 16000
    force_mono = True
    stride = 2

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn="povey",
            power=2.0,
            center=False,
            periodic=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=20.0,
            f_max=8000.0,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        ),
        log_mode="log",
        preemphasis=0.97,
        remove_dc_offset=True,
        mel_floor=1.192092955078125e-07,
        waveform_scale=32768.0,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window = window_function(400, "povey", periodic=False)

    def _mel_filter_bank(self, spectrogram_config):
        mel_cfg = spectrogram_config.mel_scale_config
        return mel_filter_bank(
            num_frequency_bins=257,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate // 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
            triangularize_in_mel_space=mel_cfg.triangularize_in_mel_space,
        )

    def _extract_fbank_features(self, waveform):
        waveform = np.squeeze(waveform) * (2**15)  # Kaldi compliance: 16-bit signed integers
        features = spectrogram(
            waveform,
            self.window,
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            preemphasis=0.97,
            mel_filters=self.mel_filters,
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        ).T
        return features

    def feature_normalize(self, features):
        # Per-mel-bin normalization with ddof=1 for variance
        normalized = []
        for f in features:
            mean = np.expand_dims(f.mean(axis=0), 0)
            var = np.expand_dims(f.var(axis=0, ddof=1), 0)
            normalized.append((f - mean) / np.sqrt(var + 1e-7))
        return normalized

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Extract Kaldi-style features matching the FE exactly
        features = [self._extract_fbank_features(waveform) for waveform in audio]

        # Per-mel-bin normalization
        features = self.feature_normalize(features)

        # Pad features to longest (pad_to_multiple_of=2 for stride)
        max_len = max(f.shape[0] for f in features)
        if max_len % self.stride != 0:
            max_len = ((max_len // self.stride) + 1) * self.stride
        padded = []
        for f in features:
            if f.shape[0] < max_len:
                pad_amount = max_len - f.shape[0]
                f = np.pad(f, ((0, pad_amount), (0, 0)), mode="constant", constant_values=0.0)
            padded.append(f)

        stacked = np.stack(padded, axis=0)  # (batch, frames, n_mels)
        batch_size, num_frames, num_channels = stacked.shape

        # Stride concatenation
        remainder = num_frames % self.stride
        if remainder != 0:
            stacked = stacked[:, : num_frames - remainder, :]
            num_frames = num_frames - remainder

        stacked = stacked.reshape(batch_size, num_frames // self.stride, num_channels * self.stride)

        output_key = "audio_features"
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


__all__ = ["SeamlessM4tAudioProcessor"]
