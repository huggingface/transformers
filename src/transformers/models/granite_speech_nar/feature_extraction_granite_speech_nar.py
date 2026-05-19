# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Feature extraction for Granite Speech NAR."""

from ...feature_extraction_utils import FeatureExtractionMixin
from ...utils import is_torch_available, is_torchaudio_available
from ...utils.import_utils import requires_backends


if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio


class GraniteSpeechNarFeatureExtractor(FeatureExtractionMixin):
    """Extracts log-mel spectrogram features for GraniteSpeechNar.

    Produces stacked pairs of 80-band mel frames, yielding 160-dim features
    at half the original frame rate.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        **kwargs,
    ):
        requires_backends(self, ["torch", "torchaudio"])
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def _extract_features(self, raw_audio: torch.Tensor) -> torch.Tensor:
        mel_transform = self.mel_transform.to(raw_audio.device)
        B, T = raw_audio.shape
        l = 2 * (T // (2 * self.hop_length))
        mel = mel_transform(raw_audio.float())[..., :l]
        logmel = mel.transpose(-1, -2).clamp_min_(1e-10).log10_()
        mx = logmel.amax(dim=(-2, -1), keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
        return logmel.reshape(B, -1, 2 * self.n_mels)

    def __call__(
        self,
        audios: torch.Tensor | list[torch.Tensor],
        device: str | torch.device | None = None,
    ) -> dict:
        if isinstance(audios, torch.Tensor):
            if audios.ndim == 1:
                audios = [audios]
            elif audios.ndim == 2:
                audios = [audios[i] for i in range(audios.shape[0])]
            else:
                raise ValueError(f"Expected 1-D or 2-D tensor, got {audios.ndim}-D")

        raw_lengths = [a.shape[-1] for a in audios]
        encoder_frame_counts = [l // (2 * self.hop_length) for l in raw_lengths]

        raw_audio = torch.nn.utils.rnn.pad_sequence(
            [a.squeeze(0) if a.ndim > 1 else a for a in audios],
            batch_first=True,
            padding_value=0.0,
        )
        if device is not None:
            raw_audio = raw_audio.to(device)

        input_features = self._extract_features(raw_audio)

        max_enc_frames = input_features.shape[1]
        x_sizes = torch.tensor(encoder_frame_counts, dtype=torch.long)
        attention_mask = torch.arange(max_enc_frames).unsqueeze(0) < x_sizes.unsqueeze(1)

        if device is not None:
            input_features = input_features.to(device)
            attention_mask = attention_mask.to(device)

        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }


__all__ = ["GraniteSpeechNarFeatureExtractor"]
