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

from .audio_processing_utils import BaseAudioProcessor
from .audio_utils import SpectrogramConfig, mel_filter_bank
from .feature_extraction_utils import BatchFeature
from .utils import PaddingStrategy, TensorType, is_torch_available, is_torch_tensor, logging, to_numpy


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


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
        from .audio_utils import spectrogram as compute_spectrogram, window_function

        stft_cfg = spectrogram_config.stft_config
        n_fft = stft_cfg.n_fft
        hop_length = stft_cfg.hop_length
        win_length = stft_cfg.win_length if stft_cfg.win_length is not None else n_fft

        # Build window — map torch names like "hann_window" to audio_utils names like "hann"
        window_name = stft_cfg.window_fn.replace("_window", "")
        window = window_function(win_length, window_name, periodic=stft_cfg.periodic)

        features = []
        for waveform in audio:
            w = waveform
            if spectrogram_config.waveform_scale is not None:
                w = np.squeeze(w) * spectrogram_config.waveform_scale
            spec = compute_spectrogram(
                w,
                window=window,
                frame_length=win_length,
                hop_length=hop_length,
                fft_length=n_fft,
                power=stft_cfg.power,
                center=stft_cfg.center,
                pad_mode=stft_cfg.pad_mode,
                preemphasis=spectrogram_config.preemphasis,
                remove_dc_offset=spectrogram_config.remove_dc_offset,
                mel_filters=None,
                mel_floor=spectrogram_config.mel_floor,
                log_mel=spectrogram_config.log_mode if spectrogram_config.log_mode != "log10" else "log10",
            )
            features.append(spec)

        return features

    def _apply_mel_scale(
        self,
        features: list[np.ndarray],
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list[np.ndarray]:
        """Apply mel filterbank to spectrogram features using the numpy backend."""
        if not hasattr(self, "mel_filters"):
            raise ValueError(
                f"{self.__class__.__name__} does not have `mel_filters`. "
                "Either set `mel_filters` or override `_apply_mel_scale`."
            )

        mel_filters = self.mel_filters
        return [mel_filters.T @ spec for spec in features]

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        return mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
        )

    def _preprocess(
        self,
        audio: list[np.ndarray],
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        spectrogram_config=None,
        do_extract_spectrogram=None,
        do_batch_spectrogram=True,
        **kwargs,
    ) -> BatchFeature:
        # pad and truncate
        audio = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)

        if do_extract_spectrogram:
            feature = self.extract_spectrogram(audio, spectrogram_config=spectrogram_config, do_batch_spectrogram=do_batch_spectrogram)
            output = BatchFeature({"audio_features": feature}, tensor_type=return_tensors)
        else:
            output = BatchFeature({"audio_values": audio}, tensor_type=return_tensors)

        return output


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
        audio: list["torch.Tensor"],
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list["torch.Tensor"]:
        """Compute the (power) spectrogram via STFT using the torch backend."""
        import torch

        stft_cfg = spectrogram_config.stft_config
        n_fft = stft_cfg.n_fft
        hop_length = stft_cfg.hop_length
        win_length = stft_cfg.win_length if stft_cfg.win_length is not None else n_fft

        # Stack list into batch for efficient batched STFT if not already batched
        if isinstance(audio, torch.Tensor) and audio.dim() == 2:
            waveform = audio
        else:
            waveform = torch.stack(audio)  # (batch, length)
        device = waveform.device

        if spectrogram_config.preemphasis is not None:
            audio_ranges = kwargs.get("audio_ranges", None)
            timemask = torch.arange(waveform.shape[1], device=device).unsqueeze(0)
            timemask = timemask < audio_ranges.unsqueeze(1)
            waveform = waveform.masked_fill(~timemask, 0.0)

        window_fn = getattr(torch, stft_cfg.window_fn, torch.hann_window)
        window = window_fn(win_length, periodic=stft_cfg.periodic, device=device)

        stft = torch.stft(
            waveform,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=stft_cfg.center,
            pad_mode=stft_cfg.pad_mode,
            normalized=stft_cfg.normalized,
            onesided=stft_cfg.onesided,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** stft_cfg.power

        return [magnitudes[i] for i in range(magnitudes.shape[0])]

    def _apply_mel_scale(
        self,
        features: list["torch.Tensor"],
        *,
        spectrogram_config: SpectrogramConfig,
        **kwargs,
    ) -> list["torch.Tensor"]:
        """Apply mel filterbank to spectrogram features using the torch backend."""
        import torch

        if not hasattr(self, "mel_filters"):
            raise ValueError(
                f"{self.__class__.__name__} does not have `mel_filters`. "
                "Either set `mel_filters` or override `_apply_mel_scale`."
            )

        device = features[0].device
        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        return [mel_filters.T @ spec for spec in features]

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        return mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
        )

    def _preprocess(
        self,
        audio: list["torch.Tensor"],
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        spectrogram_config=None,
        do_extract_spectrogram=None,
        do_batch_spectrogram=True,
        **kwargs,
    ) -> BatchFeature:
        import torch

        # pad and truncate
        audio = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)

        if do_extract_spectrogram:
            audio = torch.stack(audio) if do_batch_spectrogram else audio
            feature = self.extract_spectrogram(audio, spectrogram_config=spectrogram_config, do_batch_spectrogram=do_batch_spectrogram)
            output = BatchFeature({"audio_features": feature}, tensor_type=return_tensors)
        else:
            output = BatchFeature({"audio_values": audio}, tensor_type=return_tensors)

        return output