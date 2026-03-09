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

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class WhisperAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    truncation = True
    max_length = 480000  # 30 seconds at 16000 Hz
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=400,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            mel_scale="slaney",
            norm="slaney",
        ),
        log_mode="log10",
    )

    def extract_spectrogram(self, audio, **kwargs):
        import torch

        features = super().extract_spectrogram(audio, **kwargs)
        spectrogram_config = kwargs.get("spectrogram_config", self.spectrogram_config)
        mel_floor = spectrogram_config.mel_floor
        processed = []
        for spec in features:
            log_spec = torch.clamp(spec, min=mel_floor).log10()
            max_val = log_spec.max()
            log_spec = torch.maximum(log_spec, max_val - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            processed.append(log_spec)
        return processed


__all__ = ["WhisperAudioProcessor"]
