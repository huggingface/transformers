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
        mel_filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=self.sample_rate / 2,
            norm="slaney",
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(torch.float32)

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Pad raw audio
        lengths = [a.shape[-1] for a in audio]
        audio = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)

        # Stack into batch
        waveform = torch.stack(audio)  # (batch, length)
        audio_lengths = torch.tensor(lengths)

        # Preemphasis with masking for padded regions
        if self.preemphasis is not None:
            timemask = torch.arange(waveform.shape[1]).unsqueeze(0) < audio_lengths.unsqueeze(1)
            waveform = torch.cat(
                [waveform[:, :1], waveform[:, 1:] - self.preemphasis * waveform[:, :-1]], dim=1
            )
            waveform = waveform.masked_fill(~timemask, 0.0)

        # STFT
        window = torch.hann_window(self.win_length, periodic=False)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        # Match FE: view_as_real -> pow(2).sum(-1).sqrt().pow(2)
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        magnitudes = magnitudes.pow(2)

        # Mel spectrogram + log
        mel_spec = self.mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)

        # (batch, mels, frames) -> (batch, frames, mels)
        mel_spec = mel_spec.permute(0, 2, 1)

        # Per-utterance normalization
        features_lengths = torch.floor_divide(
            audio_lengths + self.n_fft // 2 * 2 - self.n_fft, self.hop_length
        )
        attention_mask = torch.arange(mel_spec.shape[1])[None, :] < features_lengths[:, None]
        mask = attention_mask.unsqueeze(-1)
        mel_masked = mel_spec * mask
        mean = mel_masked.sum(dim=1) / features_lengths.unsqueeze(-1)
        mean = mean.unsqueeze(1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        mel_spec = (mel_spec - mean) / (std + EPSILON)
        mel_spec *= mask

        return BatchFeature(
            data={"audio_features": mel_spec},
            tensor_type=return_tensors,
        )


__all__ = ["ParakeetAudioProcessor"]
