# Copyright 2026 The HuggingFace Inc. team.
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


class ClapAudioProcessorMixin:
    """Backend-agnostic CLAP logic shared by the numpy and torch siblings: `rand_trunc`
    (single view) and `fusion` (4-view chunking with a bilinear-downsampled global view)
    truncation modes. Random offsets come from `np.random` on both backends."""

    sampling_rate = 48000
    force_mono = True
    max_length = 480000
    truncation_mode = "rand_trunc"  # "fusion" or "rand_trunc"
    return_padding_mask = False  # CLAP returns is_longer instead of a padding mask

    # computation_dtype="float64": the legacy FE builds its filter banks in float64
    _mel_configs = {
        "rand_trunc": MelScaleConfig(
            n_mels=64,
            f_min=50,
            f_max=14000,
            mel_scale="slaney",
            norm="slaney",
            frequency_bin_mode="linspace",
            computation_dtype="float64",
        ),
        "fusion": MelScaleConfig(
            n_mels=64,
            f_min=50,
            f_max=14000,
            mel_scale="htk",
            frequency_bin_mode="linspace",
            computation_dtype="float64",
        ),
    }

    def _set_attributes(self, **kwargs):
        # an explicitly passed spectrogram_config wins over the per-mode default
        if kwargs.get("spectrogram_config") is None:
            self.spectrogram_config = SpectrogramConfig(
                stft_config=StftConfig(n_fft=1024, hop_length=480, power=2.0),
                mel_scale_config=self._mel_configs[self.truncation_mode],
                log_mode="dB",
                computation_dtype="float64",
            )
        super()._set_attributes(**kwargs)
        # fusion extracts the full mel then chunks, so no pre-truncation
        self.truncation = self.truncation_mode == "rand_trunc"

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

        if not isinstance(audio, list):
            audio = list(audio) if audio.ndim == 2 else [audio]
        waveforms = [self._as_backend_array(w) for w in audio]

        mels = []
        is_longer = []
        for waveform in waveforms:
            mel = super().extract_spectrogram(waveform, spectrogram_config=self.spectrogram_config).swapaxes(-2, -1)
            total_frames = mel.shape[0]

            if is_fusion and total_frames > chunk_frames:
                mels.append(self._random_mel_fusion(mel, total_frames, chunk_frames))
                is_longer.append(True)
            elif is_fusion:
                mels.append(self._stack([mel, mel, mel, mel]))
                is_longer.append(False)
            else:
                mels.append(mel[None])
                is_longer.append(False)

        if is_fusion:
            self._is_longer_flags = is_longer
        return mels

    def _random_mel_fusion(self, mel, total_frames, chunk_frames):
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
        mel_shrink = self._bilinear_shrink(mel, chunk_frames)  # downsampled "global" view
        return self._stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back])

    def _postprocess_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        """Add CLAP's is_longer flag to the output (returned instead of a standard attention mask)."""
        ranges = audio_ranges if audio_ranges is not None else feature_ranges
        is_longer = getattr(self, "_is_longer_flags", None) or [False] * len(ranges)
        if self.truncation_mode == "fusion" and sum(is_longer) == 0:
            rand_idx = np.random.randint(0, len(is_longer))
            is_longer[rand_idx] = True
        output["is_longer"] = [[longer] for longer in is_longer]
        return output


class ClapAudioProcessorNumpy(ClapAudioProcessorMixin, NumpyAudioBackend):
    """NumPy sibling of [`ClapAudioProcessor`]."""

    def _bilinear_shrink(self, mel, chunk_frames):
        # legacy numpy dtype path: float64 straight through interpolate (torch sibling
        # round-trips through float32 instead)
        import torch

        mel_tensor = torch.tensor(mel[None, None, :])
        mel_shrink = torch.nn.functional.interpolate(
            mel_tensor, size=[chunk_frames, 64], mode="bilinear", align_corners=False
        )
        return mel_shrink[0][0].numpy()


__all__ = ["ClapAudioProcessorNumpy"]
