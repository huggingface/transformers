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
from ...feature_extraction_utils import BatchFeature


class GraniteSpeechAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
        ),
        log_mode="log10",
    )

    def extract_spectrogram(self, audio, *, spectrogram_config):
        import torch

        # Use parent's extract_spectrogram for basic mel spectrogram
        # Parent returns list of (n_mels, frames) tensors with log10 + (x+4)/4 normalization
        features = super().extract_spectrogram(audio, spectrogram_config=spectrogram_config)

        # Transpose each: (n_mels, frames) -> (frames, n_mels)
        features = [f.permute(1, 0) for f in features]

        # Remove last frame if odd
        features = [f[:-1] if f.shape[0] % 2 == 1 else f for f in features]

        # Frame stacking: (frames, n_mels) -> (frames//2, 2*n_mels)
        features = [f.reshape(-1, 2 * f.shape[-1]) for f in features]

        return features

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        import torch

        # Pad raw audio values
        if padding:
            audio = self.pad_values(
                audio, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of
            )

        # Extract spectrogram with frame stacking
        features = self.extract_spectrogram(audio, spectrogram_config=self.spectrogram_config)

        # Pad features to same length
        max_feat_len = max(f.shape[0] for f in features)
        padded = []
        for f in features:
            if f.shape[0] < max_feat_len:
                pad_amount = max_feat_len - f.shape[0]
                f = torch.nn.functional.pad(f, (0, 0, 0, pad_amount), mode="constant", value=0.0)
            padded.append(f)

        output_key = self.model_input_names[0]
        stacked = torch.stack(padded, dim=0)
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


__all__ = ["GraniteSpeechAudioProcessor"]
