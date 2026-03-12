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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, spectrogram, window_function
from ...feature_extraction_utils import BatchFeature


class UnivNetAudioProcessor(NumpyAudioBackend):
    sample_rate = 24000
    force_mono = True
    n_fft = 1024
    hop_length = 256
    n_mels = 100
    fmin = 0.0
    fmax = 12000.0
    mel_floor = 1e-9
    compression_clip_val = 1e-5
    compression_factor = 1.0
    do_normalize = False
    normalize_min = -11.512925148010254
    normalize_max = 2.3143386840820312
    max_length_s = 10
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(n_fft=1024),
        mel_scale_config=MelScaleConfig(
            n_mels=100,
            f_min=0.0,
            f_max=12000.0,
            mel_scale="slaney",
            norm="slaney",
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_max_samples = self.max_length_s * self.sample_rate
        self.window = window_function(self.n_fft, "hann", periodic=True)

    def mel_spectrogram(self, waveform):
        # Reflect-pad waveform
        pad_amount = int((self.n_fft - self.hop_length) / 2)
        waveform = np.pad(waveform, (pad_amount, pad_amount), mode="reflect")

        # Complex spectrogram
        complex_spec = spectrogram(
            waveform,
            window=self.window,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            fft_length=self.n_fft,
            power=None,
            center=False,
            mel_filters=None,
            mel_floor=None,
        )

        # Custom amplitude spectrogram: sqrt(real^2 + imag^2 + mel_floor)
        amplitude_spec = np.sqrt(np.real(complex_spec) ** 2 + np.imag(complex_spec) ** 2 + self.mel_floor)

        # Apply mel filter bank
        mel_spec = np.matmul(self.mel_filters.T, amplitude_spec)

        # Log compression
        log_mel = np.log(np.clip(mel_spec, a_min=self.compression_clip_val, a_max=None) * self.compression_factor)

        return log_mel.T  # (frames, n_mels)

    def normalize(self, spectrogram_data):
        return 2 * ((spectrogram_data - self.normalize_min) / (self.normalize_max - self.normalize_min)) - 1

    def extract_spectrogram(self, audio, *, spectrogram_config):
        features = []
        for waveform in audio:
            waveform = np.squeeze(waveform)
            mel = self.mel_spectrogram(waveform)
            if self.do_normalize:
                mel = self.normalize(mel)
            features.append(mel.astype(np.float32))
        return features

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, generator=None, **kwargs):
        # Pad raw audio
        if padding:
            audio, _audio_ranges = self.pad(audio, padding=True, max_length=max_length)

        # Extract mel spectrograms
        features = self.extract_spectrogram(audio, spectrogram_config=None)

        # Pad features
        max_feat_len = max(f.shape[0] for f in features)
        padded = []
        for f in features:
            if f.shape[0] < max_feat_len:
                pad_amount = max_feat_len - f.shape[0]
                f = np.pad(f, ((0, pad_amount), (0, 0)), mode="constant", constant_values=0.0)
            padded.append(f)

        output_key = "audio_features"
        stacked = np.stack(padded, axis=0)

        # Generate noise sequence matching the FE
        if generator is None:
            generator = np.random.default_rng()
        noise = [
            generator.standard_normal((f.shape[0], 64), dtype=np.float32)
            for f in padded
        ]

        return BatchFeature(
            data={output_key: stacked, "noise_sequence": noise},
            tensor_type=return_tensors,
        )


__all__ = ["UnivNetAudioProcessor"]
