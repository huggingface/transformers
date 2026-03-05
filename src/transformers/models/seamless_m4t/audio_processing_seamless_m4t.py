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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
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
            window_fn="povey_window",
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

    def feature_normalize(self, features, *, feature_normalization_config):
        # Per-mel-bin normalization with ddof=1 for variance
        normalized = []
        for f in features:
            mean = np.expand_dims(f.mean(axis=0), 0)
            var = np.expand_dims(f.var(axis=0, ddof=1), 0)
            normalized.append((f - mean) / np.sqrt(var + 1e-7))
        return normalized

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Extract Kaldi-style features via generic config-based API
        features = self.extract_spectrogram(audio, spectrogram_config=self.spectrogram_config)

        # Generic extract_spectrogram returns (n_mels, frames); transpose to (frames, n_mels)
        features = [f.T for f in features]

        # Per-mel-bin normalization
        features = self.feature_normalize(features, feature_normalization_config=None)

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

        output_key = self.model_input_names[0]
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


__all__ = ["SeamlessM4tAudioProcessor"]
