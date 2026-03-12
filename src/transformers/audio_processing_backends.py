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


import sys
from pathlib import Path

import numpy as np

from .audio_processing_utils import BaseAudioProcessor
from .audio_utils import SpectrogramConfig, amplitude_to_db, power_to_db
from .feature_extraction_utils import BatchFeature
from .utils import is_torch_available, logging


logger = logging.get_logger(__name__)

_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[3])
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from spectrograms import numpy_mel_spectrogram as _np_spec


if is_torch_available():
    import torch
    from spectrograms import torch_mel_spectrogram as _torch_spec


class NumpyAudioBackend(BaseAudioProcessor):
    """NumPy backend for portable CPU-only audio processing."""

    @property
    def backend(self) -> str:
        return "numpy"

    def _process_audio(self, audio_el):
        """
        Process a single raw audio input into a np.ndarray.

        Handles mono conversion (averaging channels) and numpy conversion.
        Closely mirrors the torch backend logic: expects channel-first.
        """
        if not isinstance(audio_el, np.ndarray):
            audio_el = np.asarray(audio_el)

        if audio_el.ndim > 1:
            # Expecting channel-first: (channels, samples)
            if self.force_mono and audio_el.shape[0] > 1:
                audio_el = audio_el.mean(axis=0)
            elif audio_el.shape[0] == 1:
                audio_el = np.squeeze(audio_el, axis=0)
            else:
                raise ValueError("Audio has more than one channel but force_mono is False")
        return audio_el

    def _pad_single(self, audio: np.ndarray, max_length: int) -> np.ndarray:
        """Pad a single audio array to a target length using np.pad."""
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio

        if self.padding_value is None:
            raise ValueError(
                "Asking to pad but the audio processor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `audio_processor.padding_value = 0.0`."
            )

        pad_length = max_length - current_length
        if self.padding_side == "right":
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, pad_length)]
        elif self.padding_side == "left":
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(pad_length, 0)]
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")

        return np.pad(audio, pad_width, mode="constant", constant_values=self.padding_value)

    def _extract_spectrogram(
        self,
        audio: list[np.ndarray],
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list[np.ndarray]:
        """Compute the (power) spectrogram via STFT using the numpy backend."""
        stft_cfg = spectrogram_config.stft_config

        return _np_spec._extract_spectrogram(
                audio,
                self.sample_rate,
                n_fft=stft_cfg.n_fft,
                win_length=stft_cfg.win_length,
                hop_length=stft_cfg.hop_length,
                window_fn=stft_cfg.window_fn,
                power=stft_cfg.power,
                center=stft_cfg.center,
                pad_mode=stft_cfg.pad_mode,
                normalized=stft_cfg.normalized,
                pad=stft_cfg.pad,
                periodic=stft_cfg.periodic,
                preemphasis=spectrogram_config.preemphasis,
                remove_dc_offset=spectrogram_config.remove_dc_offset,
            )

    def _apply_mel_scale(
        self,
        features: list[np.ndarray],
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list[np.ndarray]:
        """Apply mel filterbank to spectrogram features using the numpy backend."""
        return _np_spec._apply_mel_scale(features, self.mel_filters, mel_floor=spectrogram_config.mel_floor)

    def _normalize_magnitude(
        self,
        features: np.ndarray,
        *,
        spectrogram_config: SpectrogramConfig,
        reference: float = 1.0,
        min_value: float = 1e-10,
        db_range: float | None = None,
        dtype: np.dtype = np.float32,
        **kwargs,
    ) -> np.ndarray:
        """Apply magnitude normalization (log, log10, or dB scaling) to spectrogram features.

        Accepts a single or batched spectrogram (not a list).
        Mirrors the normalization logic in `audio_utils.spectrogram()`.
        """
        log_mel = spectrogram_config.log_mode
        mel_floor = spectrogram_config.mel_floor
        power = spectrogram_config.stft_config.power

        if log_mel is None:
            return features

        # Clamp to mel_floor before taking log
        result = np.maximum(mel_floor, features)

        if log_mel == "log":
            result = np.log(result).astype(dtype)
        elif log_mel == "log10":
            result = np.log10(result).astype(dtype)
        elif log_mel == "dB":
            if power == 1.0:
                result = amplitude_to_db(result, reference, min_value, db_range).astype(dtype)
            elif power == 2.0:
                result = power_to_db(result, reference, min_value, db_range).astype(dtype)
            else:
                raise ValueError(f"Cannot use log_mel option 'dB' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        return result

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        return _np_spec.mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
            triangularize_in_mel_space=mel_cfg.triangularize_in_mel_space,
            frequency_bin_mode=mel_cfg.frequency_bin_mode,
        )

    def _to_batch(self, audio):
        return np.stack(audio)

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        if do_extract_spectrogram:
            spec_cfg = spectrogram_config or self.spectrogram_config
            audio_lengths = np.array([end - start for start, end in audio_ranges])
            features_lengths = self._get_features_lengths(audio_lengths, spec_cfg)
            n_features = self._get_features_lengths(padded_length, spec_cfg, include_center_frame=True)
            mask = (np.arange(n_features)[None, :] < features_lengths[:, None]).astype(np.int32)
            return {"audio_features_mask": mask}
        else:
            mask = np.zeros((len(audio_ranges), padded_length), dtype=np.int32)
            for i, (start, end) in enumerate(audio_ranges):
                mask[i, start:end] = 1
            return {"audio_values_mask": mask}


class TorchAudioBackend(BaseAudioProcessor):
    """Torch backend for audio processing."""

    @property
    def backend(self) -> str:
        return "torch"

    def _process_audio(self, audio_el):
        """
        Process a single raw audio input into a torch.Tensor.

        Handles mono conversion (averaging channels) and numpy-to-torch conversion.
        """
        import torch

        if isinstance(audio_el, np.ndarray):
            audio_el = torch.from_numpy(audio_el)

        if audio_el.ndim > 1:
            # TODO: we would need to ensure somewhere audio is channel first
            if self.force_mono and audio_el.shape[0] > 1:
                audio_el = audio_el.mean(dim=0)
            elif audio_el.shape[0] == 1:
                audio_el = audio_el.squeeze(0)
            else:
                raise ValueError("Audio has more than one channel but force_mono is False")

        return audio_el

    def _pad_single(self, audio: "torch.Tensor", max_length: int) -> "torch.Tensor":
        """Pad a single audio tensor to a target length using torch.nn.functional.pad."""
        import torch.nn.functional as F

        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio

        if self.padding_value is None:
            raise ValueError(
                "Asking to pad but the audio processor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `audio_processor.padding_value = 0.0`."
            )

        if self.padding_side == "right":
            pad_args = (0, max_length - current_length)
        elif self.padding_side == "left":
            pad_args = (max_length - current_length, 0)
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")

        return F.pad(audio, pad_args, "constant", self.padding_value)

    def _extract_spectrogram(
        self,
        audio: list["torch.Tensor"],  # TODO: this can be either a audio or batch of audio and this should be documented
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list["torch.Tensor"]:
        """Compute the (power) spectrogram via STFT using the torch backend."""

        stft_cfg = spectrogram_config.stft_config
        computation_dtype = (
            getattr(torch, spectrogram_config.computation_dtype)
            if spectrogram_config.computation_dtype
            else None
        )

        magnitudes = _torch_spec._extract_spectrogram(
            audio,
            self.sample_rate,
            n_fft=stft_cfg.n_fft,
            win_length=stft_cfg.win_length,
            hop_length=stft_cfg.hop_length,
            window_fn=stft_cfg.window_fn,
            wkwargs=stft_cfg.wkwargs,
            power=stft_cfg.power,
            center=stft_cfg.center,
            pad_mode=stft_cfg.pad_mode,
            normalized=stft_cfg.normalized,
            pad=stft_cfg.pad,
            periodic=stft_cfg.periodic,
            preemphasis=spectrogram_config.preemphasis,
            remove_dc_offset=spectrogram_config.remove_dc_offset,
            computation_dtype=computation_dtype,
            left_align_fft=stft_cfg.left_align_fft,
        )

        return magnitudes

    def _apply_mel_scale(
        self,
        features: list["torch.Tensor"],
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list["torch.Tensor"]:
        """Apply mel filterbank to spectrogram features using the torch backend."""
        return _torch_spec._apply_mel_scale(features, self.mel_filters, mel_floor=spectrogram_config.mel_floor)

    def _normalize_magnitude(
        self,
        features: "torch.Tensor",
        *,
        spectrogram_config: SpectrogramConfig,
        reference: float = 1.0,
        min_value: float = 1e-10,
        db_range: float | None = None,
        dtype: "torch.dtype | None" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Apply magnitude normalization (log, log10, or dB scaling) to batched spectrogram features (torch.Tensor only)."""
        import torch

        log_mel = spectrogram_config.log_mode
        mel_floor = spectrogram_config.mel_floor
        power = spectrogram_config.stft_config.power

        if dtype is None:
            dtype = torch.float32

        if log_mel is None:
            return features

        # Clamp to mel_floor before taking log
        result = torch.clamp(features, min=mel_floor)

        if log_mel == "log":
            result = torch.log(result).to(dtype)
        elif log_mel == "log10":
            result = torch.log10(result).to(dtype)
        elif log_mel == "dB":
            if reference <= 0.0:
                raise ValueError("reference must be greater than zero")
            if min_value <= 0.0:
                raise ValueError("min_value must be greater than zero")
            reference = max(min_value, reference)
            multiplier = 10.0 if power == 2.0 else 20.0 if power == 1.0 else None
            if multiplier is None:
                raise ValueError(f"Cannot use log_mel option 'dB' with power {power}")
            log_ref = torch.log10(torch.tensor(reference, dtype=result.dtype, device=result.device))
            result = torch.clamp(result, min=min_value)
            result = multiplier * (torch.log10(result) - log_ref)
            if db_range is not None:
                if db_range <= 0.0:
                    raise ValueError("db_range must be greater than zero")
                # Clamp each sample so the minimum value is (max - db_range)
                max_vals = result.amax(dim=-2, keepdim=True) if result.ndim > 2 else result.max()
                result = torch.clamp(result, min=max_vals - db_range)
            result = result.to(dtype)
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        return result

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        computation_dtype = getattr(torch, mel_cfg.computation_dtype) if mel_cfg.computation_dtype else None
        mel_filters = _torch_spec.mel_filter_bank_torch(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
            triangularize_in_mel_space=mel_cfg.triangularize_in_mel_space,
            frequency_bin_mode=mel_cfg.frequency_bin_mode,
            computation_dtype=computation_dtype,
            bands_to_zero=mel_cfg.bands_to_zero,
        )
        # When computation_dtype is set only on the mel config (not on the
        # spectrogram config), the filters were computed in high precision for
        # accuracy but the spectrogram will be in the default dtype — cast back.
        if computation_dtype is not None and not spectrogram_config.computation_dtype:
            mel_filters = mel_filters.to(torch.get_default_dtype())
        return mel_filters

    def _to_batch(self, audio):
        return torch.stack(audio)

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        if do_extract_spectrogram:
            spec_cfg = spectrogram_config or self.spectrogram_config
            audio_lengths = torch.tensor([end - start for start, end in audio_ranges])
            features_lengths = self._get_features_lengths(audio_lengths, spec_cfg)
            n_features = self._get_features_lengths(padded_length, spec_cfg, include_center_frame=True)
            mask = (torch.arange(n_features)[None, :] < features_lengths[:, None]).to(torch.int32)
            return {"audio_features_mask": mask}
        else:
            mask = torch.zeros((len(audio_ranges), padded_length), dtype=torch.int32)
            for i, (start, end) in enumerate(audio_ranges):
                mask[i, start:end] = 1
            return {"audio_values_mask": mask}
