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
from .audio_utils import SpectrogramConfig, NormalizationConfig
from .feature_extraction_utils import BatchFeature
from .utils import logging, is_torch_available
from .utils.import_utils import requires


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


class NumpyAudioBackend(BaseAudioProcessor):
    """NumPy backend for portable CPU-only audio processing."""

    @property
    def backend(self) -> str:
        return "numpy"

    def process_audio(self, audio_el):
        """
        Process a single raw audio input into a np.ndarray.

        Handles mono conversion (averaging channels) and ensures numpy format.
        """
        if not isinstance(audio_el, np.ndarray):
            audio_el = np.asarray(audio_el)

        if self.force_mono:
            audio_el = audio_el.mean(axis=1) if audio_el.ndim > 1 else audio_el

        return audio_el

    def pad(self, audio: np.ndarray, max_length: int) -> np.ndarray:
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

    def pad_values(
        self,
        audio: list[np.ndarray],
        *,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
    ) -> list[np.ndarray]:
        """Truncate and/or pad raw audio values (stage 3)."""
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        if truncation:
            if max_length is None:
                raise ValueError("When setting `truncation=True`, make sure that `max_length` is defined.")
            audio = [a[..., :max_length] for a in audio]

        if max_length is None:
            max_length = max(a.shape[-1] for a in audio)

        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        audio = [self.pad(a, max_length) for a in audio]
        return audio

    def values_normalize(
        self,
        audio: list[np.ndarray],
        *,
        normalization_config: NormalizationConfig,
    ) -> list[np.ndarray]:
        """Normalize raw audio values (stage 4). Supports zero-mean-unit-var."""
        if normalization_config.method == "zero_mean_unit_var":
            return [
                (a - np.mean(a)) / (np.std(a) + 1e-7)
                for a in audio
            ]
        raise ValueError(f"Unknown normalization method: {normalization_config.method}")

    def extract_spectrogram(
        self,
        audio: list[np.ndarray],
        *,
        spectrogram_config: SpectrogramConfig,
    ) -> list[np.ndarray]:
        """Extract audio features (stage 5). Override in model-specific subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `extract_spectrogram`. "
            "Override this method in your model-specific audio processor."
        )

    def feature_normalize(
        self,
        features: list[np.ndarray],
        *,
        feature_normalization_config: NormalizationConfig,
    ) -> list[np.ndarray]:
        """Normalize extracted features (stage 6). Supports zero-mean-unit-var."""
        if feature_normalization_config.method == "zero_mean_unit_var":
            return [
                (f - np.mean(f)) / (np.std(f) + 1e-7)
                for f in features
            ]
        raise ValueError(f"Unknown normalization method: {feature_normalization_config.method}")

    def pad_features(
        self,
        features: list[np.ndarray],
        *,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> list[np.ndarray]:
        """Pad 2D features to a target length (stage 7)."""
        if max_length is None:
            max_length = max(f.shape[-1] for f in features)

        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        padded = []
        for f in features:
            current_length = f.shape[-1]
            if current_length >= max_length:
                padded.append(f[..., :max_length])
            else:
                pad_length = max_length - current_length
                if f.ndim == 2:
                    pad_width = [(0, 0), (0, pad_length)]
                else:
                    pad_width = [(0, 0)] * (f.ndim - 1) + [(0, pad_length)]
                padded.append(np.pad(f, pad_width, mode="constant", constant_values=0.0))
        return padded

    def _preprocess(
        self,
        audio: list[np.ndarray],
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        do_pad_values=None,
        do_values_normalize=None,
        normalization_config=None,
        spectrogram_config=None,
        do_feature_normalize=None,
        feature_normalization_config=None,
        do_pad_features=None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess using NumPy backend: 5-stage pipeline (stages 3-7)."""
        # Default do_values_normalize to True if a normalization config is provided
        if do_values_normalize is None:
            do_values_normalize = normalization_config is not None

        # Determine normalize_before_pad for values
        values_normalize_before_pad = (
            normalization_config.normalize_before_pad if normalization_config is not None else True
        )
        feature_normalize_before_pad = (
            feature_normalization_config.normalize_before_pad if feature_normalization_config is not None else True
        )

        # --- Stages 3 & 4: Values padding and normalization ---
        if values_normalize_before_pad:
            # Stage 4 before 3: normalize then pad
            if do_values_normalize and normalization_config is not None:
                audio = self.values_normalize(audio, normalization_config=normalization_config)
            if do_pad_values or padding:
                audio = self.pad_values(
                    audio, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of
                )
        else:
            # Stage 3 before 4: pad then normalize
            if do_pad_values or padding:
                audio = self.pad_values(
                    audio, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of
                )
            if do_values_normalize and normalization_config is not None:
                audio = self.values_normalize(audio, normalization_config=normalization_config)

        # --- Stage 5: Feature extraction ---
        if spectrogram_config is not None:
            features = self.extract_spectrogram(audio, spectrogram_config=spectrogram_config)
        else:
            features = audio

        # --- Stages 6 & 7: Feature normalization and padding ---
        if feature_normalize_before_pad:
            # Stage 6 before 7: normalize then pad
            if do_feature_normalize and feature_normalization_config is not None:
                features = self.feature_normalize(
                    features, feature_normalization_config=feature_normalization_config
                )
            if do_pad_features:
                features = self.pad_features(
                    features, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of
                )
        else:
            # Stage 7 before 6: pad then normalize
            if do_pad_features:
                features = self.pad_features(
                    features, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of
                )
            if do_feature_normalize and feature_normalization_config is not None:
                features = self.feature_normalize(
                    features, feature_normalization_config=feature_normalization_config
                )

        # Stack into batch
        output_key = self.model_input_names[0]
        stacked = np.stack(features, axis=0) if return_tensors else features
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


class TorchAudioBackend(BaseAudioProcessor):
    """Torch backend for audio processing."""

    @property
    def backend(self) -> str:
        return "torch"

    def process_audio(self, audio_el):
        """
        Process a single raw audio input into a torch.Tensor.

        Handles mono conversion (averaging channels) and numpy-to-torch conversion.
        """
        import torch

        if self.force_mono:
            audio_el = audio_el.mean(axis=1) if audio_el.ndim > 1 else audio_el

        if isinstance(audio_el, np.ndarray):
            audio_el = torch.from_numpy(audio_el)

        return audio_el

    def pad(self, audio: "torch.Tensor", max_length: int) -> "torch.Tensor":
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

    def pad_values(
        self,
        audio: list["torch.Tensor"],
        *,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
    ) -> list["torch.Tensor"]:
        """Truncate and/or pad raw audio values (stage 3)."""
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        if truncation:
            if max_length is None:
                raise ValueError("When setting `truncation=True`, make sure that `max_length` is defined.")
            audio = [a[..., :max_length] for a in audio]

        if max_length is None:
            max_length = max(a.shape[-1] for a in audio)

        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        audio = [self.pad(a, max_length) for a in audio]
        return audio

    def values_normalize(
        self,
        audio: list["torch.Tensor"],
        *,
        normalization_config: NormalizationConfig,
    ) -> list["torch.Tensor"]:
        """Normalize raw audio values (stage 4). Supports zero-mean-unit-var."""
        import torch

        if normalization_config.method == "zero_mean_unit_var":
            return [
                (a - torch.mean(a)) / (torch.std(a) + 1e-7)
                for a in audio
            ]
        raise ValueError(f"Unknown normalization method: {normalization_config.method}")

    def extract_spectrogram(
        self,
        audio: list["torch.Tensor"],
        *,
        spectrogram_config: SpectrogramConfig,
    ) -> list["torch.Tensor"]:
        """Extract log-mel spectrogram features using the provided config and mel_filters."""
        import torch

        if not hasattr(self, "mel_filters"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have `mel_filters`. "
                "Either set `mel_filters` or override `extract_spectrogram`."
            )

        stft_cfg = spectrogram_config.stft_config
        n_fft = stft_cfg.n_fft
        hop_length = stft_cfg.hop_length

        waveform = torch.stack(audio, dim=0)
        device = waveform.device
        window = torch.hann_window(n_fft, device=device)

        stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** stft_cfg.power

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return [log_spec[i] for i in range(log_spec.shape[0])]

    def feature_normalize(
        self,
        features: list["torch.Tensor"],
        *,
        feature_normalization_config: NormalizationConfig,
    ) -> list["torch.Tensor"]:
        """Normalize extracted features (stage 6). Supports zero-mean-unit-var."""
        import torch

        if feature_normalization_config.method == "zero_mean_unit_var":
            return [
                (f - torch.mean(f)) / (torch.std(f) + 1e-7)
                for f in features
            ]
        raise ValueError(f"Unknown normalization method: {feature_normalization_config.method}")

    def pad_features(
        self,
        features: list["torch.Tensor"],
        *,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> list["torch.Tensor"]:
        """Pad 2D features to a target length (stage 7)."""
        import torch.nn.functional as F

        if max_length is None:
            max_length = max(f.shape[-1] for f in features)

        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        padded = []
        for f in features:
            current_length = f.shape[-1]
            if current_length >= max_length:
                padded.append(f[..., :max_length])
            else:
                pad_length = max_length - current_length
                padded.append(F.pad(f, (0, pad_length), "constant", 0.0))
        return padded

    def _preprocess(
        self,
        audio: list["torch.Tensor"],
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        do_pad_values=None,
        do_values_normalize=None,
        normalization_config=None,
        spectrogram_config=None,
        do_feature_normalize=None,
        feature_normalization_config=None,
        do_pad_features=None,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess using Torch backend: 5-stage pipeline (stages 3-7)."""
        import torch

        # Default do_values_normalize to True if a normalization config is provided
        if do_values_normalize is None:
            do_values_normalize = normalization_config is not None

        # Determine normalize_before_pad for values
        values_normalize_before_pad = (
            normalization_config.normalize_before_pad if normalization_config is not None else True
        )
        feature_normalize_before_pad = (
            feature_normalization_config.normalize_before_pad if feature_normalization_config is not None else True
        )

        # --- Stages 3 & 4: Values padding and normalization ---
        if values_normalize_before_pad:
            # Stage 4 before 3: normalize then pad
            if do_values_normalize and normalization_config is not None:
                audio = self.values_normalize(audio, normalization_config=normalization_config)
            if do_pad_values or padding:
                audio = self.pad_values(
                    audio, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of
                )
        else:
            # Stage 3 before 4: pad then normalize
            if do_pad_values or padding:
                audio = self.pad_values(
                    audio, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of
                )
            if do_values_normalize and normalization_config is not None:
                audio = self.values_normalize(audio, normalization_config=normalization_config)

        # --- Stage 5: Feature extraction ---
        if spectrogram_config is not None:
            features = self.extract_spectrogram(audio, spectrogram_config=spectrogram_config)
        else:
            features = audio

        # --- Stages 6 & 7: Feature normalization and padding ---
        if feature_normalize_before_pad:
            # Stage 6 before 7: normalize then pad
            if do_feature_normalize and feature_normalization_config is not None:
                features = self.feature_normalize(
                    features, feature_normalization_config=feature_normalization_config
                )
            if do_pad_features:
                features = self.pad_features(
                    features, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of
                )
        else:
            # Stage 7 before 6: pad then normalize
            if do_pad_features:
                features = self.pad_features(
                    features, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of
                )
            if do_feature_normalize and feature_normalization_config is not None:
                features = self.feature_normalize(
                    features, feature_normalization_config=feature_normalization_config
                )

        # Stack into batch
        output_key = self.model_input_names[0]
        stacked = torch.stack(features, dim=0) if return_tensors else features
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)