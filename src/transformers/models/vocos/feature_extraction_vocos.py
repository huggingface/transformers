# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for Vocos"""

from typing import Optional, Union

import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor
from ...utils import PaddingStrategy, TensorType, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn.functional as F

logger = logging.get_logger(__name__)


class VocosFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Vocos feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the Short Time
    Fourier Transform (STFT).

    Args:
        feature_size (`int`, *optional*, defaults to 100):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 100):
            Number of Mel-frequency bins.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 256):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        padding (`str`, *optional*, defaults to `"center"`):
            Symmetric padding if 'same' and center padding if 'center'.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask. If left to the default, it will return the attention mask.

            [What are attention masks?](../glossary#attention-mask)

    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=100,
        sampling_rate=24000,
        num_mel_bins=100,
        n_fft=1024,
        hop_length=256,
        padding="center",
        padding_value=0.0,
        return_attention_mask=False,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.num_mel_bins = num_mel_bins
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be either `center` or `same`.")
        self.padding = padding
        self.window = window_function(
            self.n_fft,
            name="hann",
            center=(self.padding == "center"),
            frame_length=self.n_fft,
            periodic=True,
        )

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=(n_fft // 2) + 1,
            num_mel_filters=self.num_mel_bins,
            min_frequency=0.0,
            max_frequency=sampling_rate // 2,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="htk",
        )

    def _np_extract_fbank_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the input waveform using NumPy backend.
        Original Vocos pads the input when `padding=="same"` because spectrogram() only applies center padding internally.
        It computes the spectrogram using the filter bank, then clips values before log to avoid near-zero values, see
        https://github.com/gemelo-ai/vocos/blob/c859e3b7b534f3776a357983029d34170ddd6fc3/vocos/feature_extractors.py#L44
        """
        if self.padding == "same":
            pad = self.n_fft - self.hop_length
            waveform = np.pad(waveform, (pad // 2, pad // 2), mode="reflect")

        features = spectrogram(
            waveform,
            window=self.window,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            mel_filters=self.mel_filters,
            center=(self.padding == "center"),
            power=1,
        )

        features = features.astype(np.float32)
        features = np.log(np.clip(features, a_min=1e-7, a_max=None))
        return features

    def _torch_extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-mel spectrogram of the input waveform using torch backend via `torch.stft`.
        """
        if self.padding == "same":
            pad = self.win_length - self.hop_length
            waveform = F.pad(waveform, (pad // 2, pad // 2), mode="reflect")

        window = torch.hann_window(self.n_fft, periodic=True, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=(self.padding == "center"),
            pad_mode="reflect",
            return_complex=True,
        )
        # Vocos's original implementation uses torchaudio.Melspectrogram with the default power of 1
        magnitudes = stft.abs()
        mel_filters = torch.as_tensor(self.mel_filters, device=magnitudes.device, dtype=magnitudes.dtype)
        features = torch.matmul(mel_filters.T, magnitudes)
        features = torch.log(torch.clip(features, min=1e-7))
        return features

    def __call__(
        self,
        raw_speech: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
        sampling_rate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`,
            `list[list[float]]`, `list[list[list[float]]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            pad_to_multiple_of (`int`, *optional*, defaults to 2):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the tokenizer or the feature
                extractor.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")

        acceptable_types = (
            (torch.Tensor, np.ndarray, tuple, list) if is_torch_available() else (np.ndarray, tuple, list)
        )

        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], acceptable_types))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        batched = BatchFeature({"input_features": raw_speech})

        if is_torch_available():
            # pad the input audio to enable batch processing with torch.stft instead of processing each sample individually
            max_length = max(len(speech) for speech in raw_speech)
            padded_audio = torch.full(
                (len(raw_speech), max_length), fill_value=self.padding_value, dtype=torch.float32
            )
            for i, speech in enumerate(raw_speech):
                padded_audio[i, : len(speech)] = torch.tensor(speech, dtype=torch.float32)
            input_features = self._torch_extract_fbank_features(padded_audio)
            batched["input_features"] = input_features
        else:
            input_features = [self._np_extract_fbank_features(speech) for speech in raw_speech]
            batched["input_features"] = input_features

        padded = self.pad(
            batched,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )

        return padded


__all__ = ["VocosFeatureExtractor"]
