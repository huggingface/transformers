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
from ...utils import PaddingStrategy


class ClapAudioProcessor(NumpyAudioBackend):
    sample_rate = 48000
    force_mono = True
    max_length = 480000
    truncation_mode = "rand_trunc"  # "fusion" or "rand_trunc"

    _mel_configs = {
        "rand_trunc": MelScaleConfig(n_mels=64, f_min=50, f_max=14000, mel_scale="slaney", norm="slaney", frequency_bin_mode="linspace"),
        "fusion": MelScaleConfig(n_mels=64, f_min=50, f_max=14000, mel_scale="htk", frequency_bin_mode="linspace"),
    }

    def __init__(self, **kwargs):
        truncation_mode = kwargs.pop("truncation_mode", self.truncation_mode)
        self.truncation_mode = truncation_mode
        self.spectrogram_config = SpectrogramConfig(
            stft_config=StftConfig(n_fft=1024, hop_length=480, power=2.0),
            mel_scale_config=self._mel_configs[truncation_mode],
            log_mode="dB",
        )
        super().__init__(**kwargs)
        # rand_trunc: base class truncates via pad() → _truncate_single (random offset)
        # fusion: no pre-truncation; full mel is extracted then chunked
        self.truncation = truncation_mode == "rand_trunc"

    def _get_padding_strategies(self, padding=False, max_length=None):
        # CLAP always pads to max_length, not to the longest in the batch
        if padding is True and max_length is not None:
            return PaddingStrategy.MAX_LENGTH
        return super()._get_padding_strategies(padding=padding, max_length=max_length)

    def pad(self, audio, *args, **kwargs):
        self._is_longer_flags = []
        return super().pad(audio, *args, **kwargs)

    def _truncate_single(self, audio_el, max_length):
        """Random-offset truncation for rand_trunc mode, also tracks which samples were longer."""
        self._is_longer_flags.append(audio_el.shape[-1] > max_length)
        if audio_el.shape[-1] > max_length:
            idx = np.random.randint(0, audio_el.shape[-1] - max_length + 1)
            return audio_el[..., idx : idx + max_length]
        return audio_el

    def extract_spectrogram(self, audio, *, spectrogram_config=None, audio_ranges=None, **kwargs):
        """Extract mel spectrogram and shape output (1 view for rand_trunc, 4 for fusion)."""
        is_fusion = self.truncation_mode == "fusion"
        chunk_frames = self.max_length // self.spectrogram_config.stft_config.hop_length + 1

        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            waveforms = list(audio)
        elif isinstance(audio, np.ndarray) and audio.ndim == 1:
            waveforms = [audio]
        else:
            waveforms = audio

        mels = []
        is_longer = []
        for waveform in waveforms:
            mel = super().extract_spectrogram(waveform, spectrogram_config=self.spectrogram_config).T  # (time, n_mels)
            total_frames = mel.shape[0]

            if is_fusion and total_frames > chunk_frames:
                mels.append(self._random_mel_fusion(mel, total_frames, chunk_frames))
                is_longer.append(True)
            elif is_fusion:
                mels.append(np.stack([mel, mel, mel, mel], axis=0))
                is_longer.append(False)
            else:
                mels.append(mel[np.newaxis])
                is_longer.append(False)

        if is_fusion:
            self._is_longer_flags = is_longer
        return mels

    def _random_mel_fusion(self, mel, total_frames, chunk_frames):
        import torch

        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
        if len(ranges[1]) == 0:
            ranges[1] = [0]
        if len(ranges[2]) == 0:
            ranges[2] = [0]
        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])

        mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
        mel_chunk_middle = mel[idx_middle : idx_middle + chunk_frames, :]
        mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]

        mel_tensor = torch.tensor(mel[None, None, :])
        mel_shrink = torch.nn.functional.interpolate(
            mel_tensor, size=[chunk_frames, 64], mode="bilinear", align_corners=False
        )
        mel_shrink = mel_shrink[0][0].numpy()
        return np.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], axis=0)

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        """Return CLAP's is_longer flag instead of a standard attention mask."""
        is_longer = getattr(self, "_is_longer_flags", None) or [False] * len(audio_ranges)
        if self.truncation_mode == "fusion" and sum(is_longer) == 0:
            rand_idx = np.random.randint(0, len(is_longer))
            is_longer[rand_idx] = True
        return {"is_longer": [[longer] for longer in is_longer]}


__all__ = ["ClapAudioProcessor"]
