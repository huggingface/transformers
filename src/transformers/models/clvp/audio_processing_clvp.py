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


class ClvpAudioProcessor(NumpyAudioBackend):
    sample_rate = 22050
    force_mono = True
    max_length = 132300  # 6 seconds at 22050 Hz
    truncation = True

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
            frequency_bin_mode="linspace",
        ),
        log_mode="log",
        mel_floor=1e-5,
    )

    def __init__(self, mel_norms=None, **kwargs):
        super().__init__(**kwargs)
        self.mel_norms = mel_norms

    def extract_spectrogram(self, audio, *, spectrogram_config=None, **kwargs):
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = [audio[i] for i in range(audio.shape[0])]
        elif not isinstance(audio, list):
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

            features.append(log_spec.astype(np.float32))

        return np.stack(features, axis=0) if len(features) > 1 else features

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        """CLVP uses raw-audio-level mask even for spectrogram output."""
        mask = np.zeros((len(audio_ranges), padded_length), dtype=np.int32)
        for i, (start, end) in enumerate(audio_ranges):
            mask[i, start:end] = 1
        return {"audio_features_mask": mask}


__all__ = ["ClvpAudioProcessor"]
