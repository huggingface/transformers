# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Feature extractor class for Speech2Text
"""

from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...file_utils import PaddingStrategy, TensorType, is_torch_available, is_torchaudio_available
from ...utils import logging


if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio.compliance.kaldi as ta_kaldi

logger = logging.get_logger(__name__)


class Speech2TextFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Speech2Text feature extractor.

    This feature extractor inherits from :class:`~transformers.Speech2TextFeatureExtractor` which contains most of the
    main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using TorchAudio and applies utterance-level cepstral
    mean and variance normalization to the extracted features.

    Args:
        feature_size (:obj:`int`, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (:obj:`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        num_mel_bins (:obj:`int`, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (:obj:`float`, defaults to 0.0):
            The value that is used to fill the padding vectors.
        do_ceptral_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to apply utterance-level cepstral mean and variance normalization to extracted features.
        normalize_means (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to zero-mean normalize the extracted features.
        normalize_vars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to unit-variance normalize the extracted features.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        do_ceptral_normalize=True,
        normalize_means=True,
        normalize_vars=True,
        **kwargs
    ):
        if not is_torchaudio_available():
            raise ImportError("`Speech2TextFeatureExtractor` requires torchaudio: `pip install torchaudio`.")
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.do_ceptral_normalize = do_ceptral_normalize
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars
        self.return_attention_mask = True

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        waveform = waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        features = ta_kaldi.fbank(waveform, num_mel_bins=self.num_mel_bins, sample_frequency=self.sampling_rate)
        return features.numpy()

    @staticmethod
    def utterance_cmvn(
        x: np.ndarray, normalize_means: Optional[bool] = True, normalize_vars: Optional[bool] = True
    ) -> np.ndarray:
        mean = x.mean(axis=0)
        square_sums = (x ** 2).sum(axis=0)

        if normalize_means:
            x = np.subtract(x, mean)
        if normalize_vars:
            var = square_sums / x.shape[0] - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-10))
            x = np.divide(x, std)

        return x

    def normalize(self, input_values: List[np.ndarray]) -> List[np.ndarray]:
        return [self.utterance_cmvn(x, self.normalize_means, self.normalize_vars) for x in input_values]

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). sequences.

        Args:
            raw_speech (:obj:`np.ndarray`, :obj:`List[float]`, :obj:`List[np.ndarray]`, :obj:`List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                `What are attention masks? <../glossary.html#attention-mask>`__

                .. note::

                    For Speech2TextTransoformer models, :obj:`attention_mask` should alwys be passed for batched
                    inference, to avoid subtle bugs.

            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            sampling_rate (:obj:`int`, `optional`):
                The sampling rate at which the :obj:`raw_speech` input was sampled. It is strongly recommended to pass
                :obj:`sampling_rate` at the forward call to prevent silent errors.
            padding_value (:obj:`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of {self.sampling_rate}."
                    f"Please make sure that the provided `raw_speech` input was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function."
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        # make sure input is in list format
        if is_batched and not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(speech) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # extract fbank features
        features = [self._extract_fbank_features(waveform) for waveform in raw_speech]

        # Utterance-level cepstral mean and variance normalization
        if self.do_ceptral_normalize:
            features = self.normalize(features)

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_features": features})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            **kwargs,
        )

        return padded_inputs
