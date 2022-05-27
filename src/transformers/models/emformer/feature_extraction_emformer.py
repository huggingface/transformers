# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Feature extractor class for Emformer
"""

import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)

DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)


class EmformerFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an Emformer feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:

        global_stats_file (`str`):
            Path to the global stats file containing the `mean` and `invstddev` statistics of the dataset.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in samples per second (Hz).
        n_fft (`int`, defaults to 400):
            Size of FFT, creates `n_fft // 2 + 1` bins.
        n_mels (`int`, defaults to 128):
            Number of mel filterbanks.
        hop_length (`int`, defaults to 200):
            Length of hop between STFT windows.
        padding_value (`float`, defaults to 0.0):
            The value that is used to fill the padding values.
        padding_value (`float`, defaults to 0.0):
            The value that is used to fill the padding values.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        sampling_rate=16000,
        n_fft=400,
        n_mels=80,
        hop_length=160,
        global_mean=None,
        global_invstddev=None,
        feature_size=80,
        padding_value=0.0,
        **kwargs
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sampling_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length
        )

        self.global_mean = torch.tensor(global_mean) if global_mean is not None else torch.zeros(80)
        self.global_invstddev = torch.tensor(global_invstddev) if global_invstddev is not None else torch.ones(80)

    @staticmethod
    def piecewise_linear_log(features: torch.FloatTensor):
        features[features > math.e] = torch.log(features[features > math.e])
        features[features <= math.e] = features[features <= math.e] / math.e
        return features

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract the mel spectrogram features and normalize them
        """
        waveform = torch.tensor(waveform)
        features = self.mel_transform(waveform)
        features = features.transpose(1, 0)
        features = self.piecewise_linear_log(features * GAIN)
        features = (features - self.global_mean) * self.global_invstddev
        features = features.numpy()

        return features

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). sequences.

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
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
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For Speech2TextTransoformer models, `attention_mask` should alwys be passed for batched inference, to
                avoid subtle bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding_value (`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
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
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
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

        # extract spectrogram features
        features = [self.extract_features(waveform) for waveform in raw_speech]

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_features": features})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        # make sure list is in array format
        input_features = padded_inputs.get("input_features")
        if isinstance(input_features[0], list):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]

        attention_mask = padded_inputs.get("attention_mask")
        if attention_mask is not None:
            padded_inputs["attention_mask"] = [np.asarray(array, dtype=np.int32) for array in attention_mask]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs

    def to_dict(self) -> Dict:
        output = super().to_dict()
        # avoid trying to serialize a json-incompatible MelSpectrogram
        del output["mel_transform"]
        output["global_mean"] = output["global_mean"].numpy()
        output["global_invstddev"] = output["global_invstddev"].numpy()
        return output
