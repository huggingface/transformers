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

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


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

    model_input_names = ["input_values", "attention_mask"]

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
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(
        self,
        raw_audio: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        padding: Optional[Union[bool, str, PaddingStrategy]] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, *optional*, defaults to `False`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_values** -- Audio waveform ready for the model.
            - **attention_mask** -- Mask to avoid performing attention on padding token indices.
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
            isinstance(raw_audio, (list, tuple))
            and (isinstance(raw_audio[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_audio = [np.asarray(audio, dtype=np.float32) for audio in raw_audio]
        elif not is_batched and not isinstance(raw_audio, np.ndarray):
            raw_audio = np.asarray(raw_audio, dtype=np.float32)
        elif isinstance(raw_audio, np.ndarray) and raw_audio.dtype is np.dtype(np.float64):
            raw_audio = raw_audio.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_audio = [raw_audio]

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_values": raw_audio})

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

