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

import librosa
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...feature_extraction_utils import BatchFeature

LOG_ZERO_GUARD_VALUE = 2**-24
EPSILON = 1e-5


class ParakeetAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    preemphasis = 0.97
    n_fft = 512
    hop_length = 160
    win_length = 400
    n_mels = 80

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use librosa for mel filters to match the FeatureExtractor exactly
        # (mel_filter_bank uses float64 internally, causing numerical differences)
        mel_filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=self.sample_rate / 2,
            norm="slaney",
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(torch.float32)

    def _torch_extract_fbank_features(self, waveform, device="cpu"):
        """Extract log-mel spectrogram features, matching the FE implementation."""
        window = torch.hann_window(self.win_length, periodic=False, device=device)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        # Match original implementation: view_as_real then sqrt(sum of squares)
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        magnitudes = magnitudes.pow(2)

        # Log mel spectrogram
        mel_filters = self.mel_filters.to(device)
        mel_spec = mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)

        # (batch, n_mels, frames) -> (batch, frames, n_mels)
        mel_spec = mel_spec.permute(0, 2, 1)
        return mel_spec

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        device = "cpu"

        # Record original audio lengths before padding
        audio_lengths = torch.tensor([a.shape[-1] for a in audio])

        # Pad values to longest
        audio = self.pad_values(audio, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of)

        # Stack into batch tensor
        waveform = torch.stack(audio, dim=0).to(torch.float32)

        # Preemphasis (mask-aware, matching FE)
        if self.preemphasis is not None:
            timemask = torch.arange(waveform.shape[1], device=device).unsqueeze(0) < audio_lengths.unsqueeze(1)
            waveform = torch.cat(
                [waveform[:, :1], waveform[:, 1:] - self.preemphasis * waveform[:, :-1]], dim=1
            )
            waveform = waveform.masked_fill(~timemask, 0.0)

        # Extract log-mel spectrogram
        input_features = self._torch_extract_fbank_features(waveform, device)

        # Compute feature lengths (matching FE formula)
        features_lengths = torch.floor_divide(
            audio_lengths + self.n_fft // 2 * 2 - self.n_fft, self.hop_length
        )

        # Build attention mask over feature frames
        attention_mask = torch.arange(input_features.shape[1], device=device)[None, :] < features_lengths[:, None]

        # Mask-aware normalization (matching FE exactly)
        mask = attention_mask.unsqueeze(-1)
        input_features_masked = input_features * mask
        mean = input_features_masked.sum(dim=1) / features_lengths.unsqueeze(-1)
        mean = mean.unsqueeze(1)
        variance = ((input_features_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        input_features = (input_features - mean) / (std + EPSILON)
        input_features *= mask

        output_key = self.model_input_names[0]
        return BatchFeature(data={output_key: input_features}, tensor_type=return_tensors)


__all__ = ["ParakeetAudioProcessor"]
