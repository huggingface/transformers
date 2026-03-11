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


class ClvpAudioProcessor(NumpyAudioBackend):
    sample_rate = 22050
    force_mono = True
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    max_length = 132300  # 6 seconds at 22050 Hz

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            hop_length=256,
            window_fn="hann_window",
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            norm="slaney",
            mel_scale="htk",
        ),
        log_mode="log",
        mel_floor=1e-5,
    )

    def __init__(self, mel_norms=None, **kwargs):
        super().__init__(**kwargs)
        self.mel_norms = mel_norms

    def _mel_filter_bank(self, spectrogram_config):
        mel_cfg = spectrogram_config.mel_scale_config
        stft_cfg = spectrogram_config.stft_config
        return mel_filter_bank(
            num_frequency_bins=1 + (stft_cfg.n_fft // 2),
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else 8000.0,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
        )

    def extract_spectrogram(self, audio, *, spectrogram_config=None, **kwargs):
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        if not isinstance(audio, list):
            audio = [audio]

        stft_cfg = spectrogram_config.stft_config
        features = []
        for waveform in audio:
            waveform = np.squeeze(waveform)
            log_spec = spectrogram(
                waveform,
                window_function(stft_cfg.n_fft, "hann"),
                frame_length=stft_cfg.n_fft,
                hop_length=stft_cfg.hop_length,
                power=2.0,
                mel_filters=self.mel_filters,
                log_mel=None,
            )
            log_spec = np.log(np.clip(log_spec, a_min=1e-5, a_max=None))

            if self.mel_norms is not None:
                log_spec = log_spec / np.array(self.mel_norms)[:, None]

            features.append(log_spec)

        return features

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Determine the raw-audio target length
        if max_length is None:
            max_length = self.max_length

        # Truncate to max_length first
        audio = [a[..., :max_length] for a in audio]

        # Pad raw audio: if padding=True, pad to longest in batch; otherwise pad to max_length
        if padding is True or padding == "longest":
            pad_length = max(a.shape[-1] for a in audio)
        else:
            pad_length = max_length
        audio = self.pad(audio, padding=True, max_length=pad_length)

        # Extract spectrogram via audio_utils (with mel_norms applied)
        features = self.extract_spectrogram(audio, spectrogram_config=self.spectrogram_config)

        # Cast to float32 to match the legacy FeatureExtractor
        features = [f.astype(np.float32) for f in features]

        output_key = "audio_features"
        stacked = np.stack(features, axis=0)
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


__all__ = ["ClvpAudioProcessor"]
