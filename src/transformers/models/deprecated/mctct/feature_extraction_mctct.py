# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""
Feature extractor class for M-CTC-T
"""

from typing import Optional, Union

import numpy as np

from ....audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ....feature_extraction_sequence_utils import SequenceFeatureExtractor
from ....feature_extraction_utils import BatchFeature
from ....file_utils import PaddingStrategy, TensorType
from ....utils import logging


logger = logging.get_logger(__name__)


class MCTCTFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a M-CTC-T feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods. This
    code has been adapted from Flashlight's C++ code. For more information about the implementation, one can refer to
    this [notebook](https://colab.research.google.com/drive/1GLtINkkhzms-IsdcGy_-tVCkv0qNF-Gt#scrollTo=pMCRGMmUC_an)
    that takes the user step-by-step in the implementation.

    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features. This is the number of mel_frequency
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, defaults to 0.0):
            The value that is used to fill the padding values.
        hop_length (`int`, defaults to 10):
            Number of audio samples between windows. Otherwise referred to as "shift" in many papers.
        win_length (`int`, defaults to 25):
            Number of ms per window
        win_function (`str`, defaults to `"hamming_window"`):
            Name for the window function used for windowing, must be accessible via `torch.{win_function}`
        frame_signal_scale (`float`, defaults to 32768.0):
            Constant multiplied in creating the frames before applying DFT.
        preemphasis_coeff (`float`, defaults to 0.97):
            Constant multiplied in applying Pre-emphasis before DFT.
        mel_floor (`float` defaults to 1.0):
            Minimum value of mel frequency banks.
        normalize_means (`bool`, *optional*, defaults to `True`):
            Whether or not to zero-mean normalize the extracted features.
        normalize_vars (`bool`, *optional*, defaults to `True`):
            Whether or not to unit-variance normalize the extracted features.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        padding_value=0.0,
        hop_length=10,
        win_length=25,
        win_function="hamming_window",
        frame_signal_scale=32768.0,
        preemphasis_coeff=0.97,
        mel_floor=1.0,
        normalize_means=True,
        normalize_vars=True,
        return_attention_mask=False,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.hop_length = hop_length
        self.win_length = win_length
        self.frame_signal_scale = frame_signal_scale
        self.preemphasis_coeff = preemphasis_coeff
        self.mel_floor = mel_floor
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars
        self.win_function = win_function
        self.return_attention_mask = return_attention_mask

        self.sample_size = win_length * sampling_rate // 1000
        self.sample_stride = hop_length * sampling_rate // 1000

        self.n_fft = optimal_fft_length(self.sample_size)
        self.n_freqs = (self.n_fft // 2) + 1

    def _extract_mfsc_features(self, one_waveform: np.ndarray) -> np.ndarray:
        """
        Extracts MFSC Features for one waveform vector (unbatched). Adapted from Flashlight's C++ MFSC code.
        """
        if self.win_function == "hamming_window":
            window = window_function(window_length=self.sample_size, name=self.win_function, periodic=False)
        else:
            window = window_function(window_length=self.sample_size, name=self.win_function)

        fbanks = mel_filter_bank(
            num_frequency_bins=self.n_freqs,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=self.sampling_rate / 2.0,
            sampling_rate=self.sampling_rate,
        )

        msfc_features = spectrogram(
            one_waveform * self.frame_signal_scale,
            window=window,
            frame_length=self.sample_size,
            hop_length=self.sample_stride,
            fft_length=self.n_fft,
            center=False,
            preemphasis=self.preemphasis_coeff,
            mel_filters=fbanks,
            mel_floor=self.mel_floor,
            log_mel="log",
        )
        return msfc_features.T

    def _normalize_one(self, x, input_length, padding_value):
        # make sure we normalize float32 arrays
        if self.normalize_means:
            mean = x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        if self.normalize_vars:
            std = x[:input_length].std(axis=0)
            x = np.divide(x, std)

        if input_length < x.shape[0]:
            x[input_length:] = padding_value

        # make sure array is in float32
        x = x.astype(np.float32)

        return x

    def normalize(
        self, input_features: list[np.ndarray], attention_mask: Optional[np.ndarray] = None
    ) -> list[np.ndarray]:
        lengths = attention_mask.sum(-1) if attention_mask is not None else [x.shape[0] for x in input_features]
        return [self._normalize_one(x, n, self.padding_value) for x, n in zip(input_features, lengths)]

    def __call__(
        self,
        raw_speech: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). sequences. It returns the
        log-mel spectrogram of the input audio, as implemented in the original Flashlight MFSC feature extraction code.

        Args:
            raw_speech (`torch.Tensor`, `np.ndarray`, `list[float]`, `list[torch.Tensor]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a tensor, a numpy array, a list
                of float values, a list of tensors, a list of numpy arrays or a list of list of float values. Must be
                mono channel audio, not stereo, i.e. single float per timestep.
            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding_value (`float`, defaults to 0.0):
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
                "It is strongly recommended to pass the ``sampling_rate`` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
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

        # extract fbank features
        features = [self._extract_mfsc_features(one_waveform) for one_waveform in raw_speech]

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_features": features})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
            **kwargs,
        )
        # make sure list is in array format
        input_features = padded_inputs.get("input_features")
        if isinstance(input_features[0], list):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]

        attention_mask = padded_inputs.get("attention_mask")
        if attention_mask is not None:
            padded_inputs["attention_mask"] = [np.asarray(array, dtype=np.int32) for array in attention_mask]

        if self.normalize_means or self.normalize_vars:
            attention_mask = (
                np.array(attention_mask, dtype=np.int32)
                if self._get_padding_strategies(padding, max_length=max_length) is not PaddingStrategy.DO_NOT_PAD
                and padding
                else None
            )
            padded_inputs["input_features"] = self.normalize(
                padded_inputs["input_features"], attention_mask=attention_mask
            )

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["MCTCTFeatureExtractor"]
