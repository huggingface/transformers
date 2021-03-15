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
Feature extractor class for Wav2Vec2
"""

from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...file_utils import PaddingStrategy, TensorType
from ...utils import logging


logger = logging.get_logger(__name__)


class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Wav2Vec2 feature extractor.

    This feature extractor inherits from
    :class:`~transformers.feature_extraction_sequence_utils.SequenceFeatureExtractor` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (:obj:`int`, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (:obj:`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        padding_value (:obj:`float`, defaults to 0.0):
            The value that is used to fill the padding values.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models, *e.g.*, `wav2vec2-lv60
            <https://huggingface.co/models?search=lv60>`__.
        return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not :meth:`~transformers.Wav2Vec2FeatureExtractor.__call__` should return :obj:`attention_mask`.

            .. note::

                Wav2Vec2 models that have set ``config.feat_extract_norm == "group"``, such as `wav2vec2-base
                <https://huggingface.co/facebook/wav2vec2-base-960h>`__, have **not** been trained using
                :obj:`attention_mask`. For such models, :obj:`input_values` should simply be padded with 0 and no
                :obj:`attention_mask` should be passed.

                For Wav2Vec2 models that have set ``config.feat_extract_norm == "layer"``, such as `wav2vec2-lv60
                <https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self>`__, :obj:`attention_mask` should be
                passed for batched inference.
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
        **kwargs
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    @staticmethod
    def zero_mean_unit_var_norm(input_values: List[np.ndarray]) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        return [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in input_values]

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). sequences.

        Args:
            raw_speech (:obj:`np.ndarray`, :obj:`List[float]`, :obj:`List[np.ndarray]`, :obj:`List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`False`):
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

                    Wav2Vec2 models that have set ``config.feat_extract_norm == "group"``, such as `wav2vec2-base
                    <https://huggingface.co/facebook/wav2vec2-base-960h>`__, have **not** been trained using
                    :obj:`attention_mask`. For such models, :obj:`input_values` should simply be padded with 0 and no
                    :obj:`attention_mask` should be passed.

                    For Wav2Vec2 models that have set ``config.feat_extract_norm == "layer"``, such as `wav2vec2-lv60
                    <https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self>`__, :obj:`attention_mask` should be
                    passed for batched inference.

            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            sampling_rate (:obj:`int`, `optional`):
                The sampling rate at which the ``raw_speech`` input was sampled. It is strongly recommended to pass
                ``sampling_rate`` at the forward call to prevent silent errors.
            padding_value (:obj:`float`, defaults to 0.0):
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of {self.sampling_rate}."
                    f"Please make sure that the provided `raw_speech` input was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the ``sampling_rate`` argument to this function."
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

        # zero-mean and unit-variance normalization
        if self.do_normalize:
            raw_speech = self.zero_mean_unit_var_norm(raw_speech)

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_values": raw_speech})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )

        return padded_inputs
