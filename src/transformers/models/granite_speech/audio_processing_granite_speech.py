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

import math

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...feature_extraction_utils import BatchFeature


class GraniteSpeechAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    projector_window_size = 15
    projector_downsample_rate = 5
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
        ),
        log_mode="log10",
    )

    def extract_spectrogram(self, audio, **kwargs):
        features = super().extract_spectrogram(audio, **kwargs)

        processed = []
        for f in features:
            # f is (n_mels, frames) from base; transpose to (frames, n_mels)
            f = f.T

            # Apply max-8 normalization matching the FE
            mx = f.amax(dim=(-2, -1), keepdim=True)
            f = torch.maximum(f, mx - 8.0)
            f = f / 4.0 + 1.0

            # Remove last frame if odd
            if f.shape[0] % 2 == 1:
                f = f[:-1]

            # Stack pairs of frames: (frames//2, n_mels*2)
            f = f.reshape(-1, 2 * f.shape[-1])
            processed.append(f)

        return processed

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors,
                    spectrogram_config=None, do_extract_spectrogram=None, **kwargs):
        hop_length = self.spectrogram_config.stft_config.hop_length

        # Record original lengths before padding
        audio_lengths = [a.shape[-1] for a in audio]

        # Pad audio to longest in batch
        audio = self.pad(audio, padding=True, max_length=max_length)

        # Stack and extract spectrogram
        audio_stacked = torch.stack(audio)
        features = self.extract_spectrogram(audio_stacked, spectrogram_config=spectrogram_config)

        # Compute audio_embed_sizes matching the FE
        effective_window_size = self.projector_window_size // self.projector_downsample_rate
        audio_embed_sizes = []
        for raw_length in audio_lengths:
            mel_length = raw_length // hop_length + 1
            encoder_length = mel_length // 2
            nblocks = math.ceil(encoder_length / self.projector_window_size)
            projector_length = nblocks * effective_window_size
            audio_embed_sizes.append(projector_length)

        # Build input_features_mask matching the FE
        input_features_mask = torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(
            audio_embed_sizes
        ).view(-1, 1)

        data = {
            "audio_features": features,
            "audio_embed_sizes": audio_embed_sizes,
            "input_features_mask": input_features_mask,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["GraniteSpeechAudioProcessor"]
