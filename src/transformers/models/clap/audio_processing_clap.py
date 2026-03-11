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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, mel_filter_bank, spectrogram, window_function
from ...feature_extraction_utils import BatchFeature


class ClapAudioProcessor(NumpyAudioBackend):
    sample_rate = 48000
    force_mono = True
    max_length_s = 10
    truncation_mode = "rand_trunc"  # "fusion" or "rand_trunc"
    padding_mode = "repeatpad"  # "repeatpad", "repeat", or "pad"

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            hop_length=480,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=64,
            f_min=50,
            f_max=14000,
            mel_scale="slaney",
            norm="slaney",
        ),
        log_mode="dB",
    )

    # Fusion mode uses a different mel filter bank (htk scale, no norm)
    spectrogram_config_fusion = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1024,
            hop_length=480,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=64,
            f_min=0,
            f_max=14000,
            mel_scale="htk",
        ),
        log_mode="dB",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nb_max_samples = self.max_length_s * self.sample_rate
        self.mel_filters_fusion = self._mel_filter_bank(self.spectrogram_config_fusion)

    def _pad_single_clap(self, audio: np.ndarray, max_length: int, padding_mode: str) -> np.ndarray:
        """
        CLAP-specific padding: handles "repeat" and "repeatpad" modes.
        This is separate from the standard _pad_single used by the base class.
        """
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio

        if padding_mode == "repeat":
            # Repeat the audio enough times to cover max_length
            n_repeat = int(max_length / current_length)
            audio = np.tile(audio, n_repeat + 1)[:max_length]
            return audio
        elif padding_mode == "repeatpad":
            # Repeat then pad with zeros
            n_repeat = int(max_length / current_length)
            audio = np.tile(audio, n_repeat)
            remaining = max_length - audio.shape[-1]
            if remaining > 0:
                audio = np.pad(audio, (0, remaining), mode="constant", constant_values=0)
            return audio
        else:
            # For other modes, use standard padding via parent's _pad_single
            return super()._pad_single(audio, max_length)

    def _mel_filter_bank(self, spectrogram_config):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        return mel_filter_bank(
            num_frequency_bins=(stft_cfg.n_fft // 2) + 1,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
        )

    def _extract_single_mel(self, waveform, spectrogram_config=None):
        """Extract mel spectrogram for a single waveform using audio_utils.spectrogram."""
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config
        stft_cfg = spectrogram_config.stft_config

        # Use the correct mel filters for this config
        if spectrogram_config is self.spectrogram_config_fusion:
            mel_filters = self.mel_filters_fusion
        else:
            mel_filters = self.mel_filters

        log_mel_spectrogram = spectrogram(
            waveform,
            window_function(stft_cfg.n_fft, "hann"),
            frame_length=stft_cfg.n_fft,
            hop_length=stft_cfg.hop_length,
            power=2.0,
            mel_filters=mel_filters,
            log_mel="dB",
        )
        return log_mel_spectrogram.T

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

    def _get_input_mel(self, waveform, max_length, truncation):
        hop_length = self.spectrogram_config.stft_config.hop_length

        if waveform.shape[0] > max_length:
            if truncation == "rand_trunc":
                longer = True
                overflow = len(waveform) - max_length
                idx = np.random.randint(0, overflow + 1)
                waveform = waveform[idx : idx + max_length]
                input_mel = self._extract_single_mel(waveform)[None, :]
            elif truncation == "fusion":
                mel = self._extract_single_mel(waveform, spectrogram_config=self.spectrogram_config_fusion)
                chunk_frames = max_length // hop_length + 1
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    input_mel = np.stack([mel, mel, mel, mel], axis=0)
                    longer = False
                else:
                    input_mel = self._random_mel_fusion(mel, total_frames, chunk_frames)
                    longer = True
            else:
                raise NotImplementedError(f"data_truncating {truncation} not implemented")
        else:
            longer = False
            if truncation == "fusion":
                input_mel = self._extract_single_mel(waveform, spectrogram_config=self.spectrogram_config_fusion)
                input_mel = np.stack([input_mel, input_mel, input_mel, input_mel], axis=0)
            else:
                input_mel = self._extract_single_mel(waveform)[None, :]

        return input_mel, longer

    def _preprocess(
        self,
        audio,
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        spectrogram_config=None,
        do_extract_spectrogram=None,
        do_batch_spectrogram=True,
        **kwargs,
    ):
        # Use instance defaults when not explicitly provided (matching feature extractor behavior)
        truncation_mode = self.truncation_mode if truncation is None else truncation
        # For padding: use instance default only when not provided (None or False)
        # When padding=True is passed, use it directly (feature extractor behavior)
        if padding is None or padding is False:
            padding_mode = self.padding_mode
        else:
            padding_mode = padding
        nb_max_samples = max_length if isinstance(max_length, int) and max_length > 0 else self.nb_max_samples

        # Handle truncation: only apply if boolean truncation=True OR if using CLAP-specific string modes
        # Note: CLAP's _get_input_mel handles truncation internally based on truncation_mode
        # We only do pre-truncation here for standard boolean truncation=True case
        if truncation is True:
            if nb_max_samples is None:
                raise ValueError("When setting `truncation=True`, make sure that `max_length` is defined.")
            trunc_length = nb_max_samples
            if pad_to_multiple_of is not None and (trunc_length % pad_to_multiple_of != 0):
                trunc_length = ((trunc_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            audio = [self._truncate_single(audio_el, max_length=trunc_length) for audio_el in audio]

        # Handle padding: CLAP-specific modes ("repeat", "repeatpad") vs standard modes
        if padding_mode in ("repeat", "repeatpad"):
            # Use CLAP's custom _pad_single_clap which handles repeat/repeatpad
            audio = [self._pad_single_clap(audio_el, max_length=nb_max_samples, padding_mode=padding_mode) for audio_el in audio]
        elif padding is not False and padding_mode is not False:
            # Use standard padding flow for "longest", "max_length", True, etc.
            from ...utils import PaddingStrategy
            if padding_mode is True and nb_max_samples is not None:
                # When padding=True and we have a max length, use MAX_LENGTH strategy
                # (matching feature extractor behavior that pads to max_length)
                padding_strategy = PaddingStrategy.MAX_LENGTH
            elif isinstance(padding_mode, str) and padding_mode not in ("longest", "max_length", "do_not_pad"):
                padding_strategy = PaddingStrategy.LONGEST  # Default to longest for unknown string values
            else:
                padding_strategy = padding_mode
            audio = self.pad(audio, padding_strategy, nb_max_samples, truncation=False, pad_to_multiple_of=pad_to_multiple_of)

        # Process each waveform through CLAP's mel extraction (handles truncation internally)
        padded_inputs = [
            self._get_input_mel(np.squeeze(waveform), nb_max_samples, truncation_mode)
            for waveform in audio
        ]

        input_mel = []
        is_longer = []
        for mel, longer in padded_inputs:
            input_mel.append(mel)
            is_longer.append(longer)

        if truncation_mode == "fusion" and sum(is_longer) == 0:
            rand_idx = np.random.randint(0, len(input_mel))
            is_longer[rand_idx] = True

        is_longer = [[longer] for longer in is_longer]

        input_features = {"audio_features": input_mel, "is_longer": is_longer}
        input_features = BatchFeature(input_features)

        if return_tensors is not None:
            input_features = input_features.convert_to_tensors(return_tensors)

        return input_features


__all__ = ["ClapAudioProcessor"]
