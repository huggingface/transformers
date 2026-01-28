# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for S3Tokenizer."""

from typing import Optional, Union

import numpy as np
import torch

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, is_librosa_available, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


if is_librosa_available():
    import librosa


@requires(backends=("torch",))
class S3TokenizerFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a S3Tokenizer feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono audio.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        n_mels (`int`, *optional*, defaults to 128):
            Number of mel-frequency bins for the mel-spectrogram.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the FFT window for computing the mel-spectrogram.
        hop_length (`int`, *optional*, defaults to 160):
            Number of audio samples between adjacent STFT columns (10ms at 16kHz).
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        n_mels: int = 128,
        n_fft: int = 400,
        hop_length: int = 160,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._mel_filters = None
        self._mel_filters_torch = None
        self._window_torch = None

    def _get_mel_filters(self):
        if self._mel_filters is None:
            if not is_librosa_available():
                raise ImportError(
                    "librosa is required to compute mel filters in S3TokenizerFeatureExtractor. "
                    "Please install it with `pip install librosa`."
                )
            self._mel_filters = librosa.filters.mel(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.n_mels)
        return self._mel_filters

    def _get_mel_filters_torch(self) -> torch.Tensor:
        """
        Cached torch.Tensor version of mel filters for torch STFT path.
        """
        if self._mel_filters_torch is None:
            self._mel_filters_torch = torch.tensor(self._get_mel_filters(), dtype=torch.float32)
        return self._mel_filters_torch

    def _get_window_torch(self) -> torch.Tensor:
        """
        Cached torch.Tensor Hann window. Matches the model's `torch.hann_window` usage.
        """
        if self._window_torch is None:
            self._window_torch = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)
        return self._window_torch

    def _extract_mel_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of audio using torch STFT.

        This intentionally mirrors the model-side preprocessing (`S3Tokenizer.log_mel_spectrogram`):
        - `torch.stft(..., center=True)` with default `pad_mode="reflect"`
        - Hann window
        - power spectrogram + mel projection
        - log10 clamp + dynamic range compression + scaling

        Returns:
            np.ndarray of shape (time, n_mels) for Transformers padding convention.
        """
        audio_t = torch.as_tensor(audio, dtype=torch.float32)
        window = self._get_window_torch()
        stft = torch.stft(
            audio_t,
            self.n_fft,
            self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().pow(2)

        mel_filters = self._get_mel_filters_torch()
        mel_spec = mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Transpose to [time, n_mels] for padding convention, and return numpy.
        return log_spec.transpose(0, 1).cpu().numpy()

    def __call__(
        self,
        raw_audio: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_audio` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched = bool(
            isinstance(raw_audio, (list, tuple)) and (isinstance(raw_audio[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_audio = [np.asarray(audio, dtype=np.float32).squeeze() for audio in raw_audio]
        else:
            raw_audio = [np.asarray(raw_audio, dtype=np.float32).squeeze()]

        # Ensure all are 1D
        raw_audio = [audio if audio.ndim == 1 else audio.flatten() for audio in raw_audio]

        # Extract features
        input_features = [self._extract_mel_features(audio) for audio in raw_audio]

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_features": input_features})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=None,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

        return padded_inputs


__all__ = ["S3TokenizerFeatureExtractor"]
