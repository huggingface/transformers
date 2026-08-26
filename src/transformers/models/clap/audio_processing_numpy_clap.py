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

from dataclasses import replace

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...utils import PaddingStrategy


class ClapAudioProcessorMixin:
    sampling_rate = 48000
    force_mono = True
    max_length = 480000
    truncation_mode = "rand_trunc"  # "fusion" or "rand_trunc"
    return_padding_mask = False  # CLAP returns is_longer instead of a padding mask
    # How audio shorter than `max_length` is filled: "repeatpad" (tile, then zero-pad the
    # remainder), "repeat" (tile past `max_length` and cut) or "pad" (zero-pad only). Both
    # released checkpoints use "repeatpad"; the legacy FE spelled this its `padding` argument.
    padding_mode = "repeatpad"

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(n_fft=1024, hop_length=480, power=2.0),
        mel_scale_config=MelScaleConfig(
            n_mels=64,
            f_min=50,
            f_max=14000,
            mel_scale="slaney",
            norm="slaney",
            frequency_bin_mode="linspace",
            computation_dtype="float64",
        ),
        log_mode="dB",
        computation_dtype="float64",
    )
    # The two modes share every STFT and log setting and differ *only* in how the mel bank is
    # built: `rand_trunc` audio reaches HTSAT as a waveform and is mel'd by its torchlibrosa
    # front-end (librosa defaults, i.e. the slaney scale + slaney norm above), while `fusion`
    # mels are precomputed with torchaudio defaults instead. The checkpoints are therefore
    # trained on differently scaled mels and are not interchangeable.
    _fusion_mel_overrides = {"mel_scale": "htk", "norm": None}

    # Legacy hub configs spell the mode as `truncation`. `top_db` only lines up with
    # `clip_max_offset` because CLAP's `log_mode` is "dB" (Whisper's offset is in log10 units).
    legacy_field_mapping = {
        "truncation": "truncation_mode",
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
        "fft_window_size": "spectrogram_config.stft_config.n_fft",
        "top_db": "spectrogram_config.clip_max_offset",
        "nb_max_samples": "max_length",
        "chunk_length_s": None,  # duplicates nb_max_samples
    }

    def _set_attributes(self, **kwargs):
        if self.truncation_mode == "fusion":
            mel_scale_config = replace(self.spectrogram_config.mel_scale_config, **self._fusion_mel_overrides)
            self.spectrogram_config = replace(self.spectrogram_config, mel_scale_config=mel_scale_config)
        super()._set_attributes(**kwargs)
        # fusion extracts the full mel then chunks, so no pre-truncation
        self.truncation = self.truncation_mode == "rand_trunc"

    def _get_padding_strategies(self, padding=False, max_length=None):
        if padding in ("repeatpad", "repeat", "pad"):
            # legacy spelling: `padding` named the fill method for short audio, not the target length
            self.padding_mode, padding = padding, True
        # CLAP always pads to max_length, not to the longest in the batch
        if padding is True and max_length is not None:
            return PaddingStrategy.MAX_LENGTH
        return super()._get_padding_strategies(padding=padding, max_length=max_length)

    def pad(self, audio, *args, **kwargs):
        self._is_longer_flags = []
        return super().pad(audio, *args, **kwargs)

    def _to_batch(self, audio):
        # `extract_spectrogram` mels each waveform on its own, so leave them as a list: in fusion
        # mode nothing truncates to `max_length` and clips longer than it stay ragged.
        return audio

    def _pad_single(self, audio, max_length):
        """Tile short audio before the base class zero-pads whatever remains."""
        current_length = audio.shape[-1]
        if current_length < max_length and self.padding_mode in ("repeat", "repeatpad"):
            n_repeat = max_length // current_length
            if self.padding_mode == "repeat":
                audio = self._tile(audio, n_repeat + 1)[..., :max_length]
            else:
                audio = self._tile(audio, n_repeat)
        return super()._pad_single(audio, max_length)

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

    # `is_longer` replaces the padding mask CLAP does not use.
    extra_model_input_names = ["is_longer"]

    def _tile(self, audio, n_repeat):
        return np.tile(audio, n_repeat)

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
