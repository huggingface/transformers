# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for EnCodec."""

from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class EncodecFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an EnCodec feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Instantiating a feature extractor with the defaults will yield a similar configuration to that of the
    [Matthijs/encodec_24khz](https://huggingface.co/Matthijs/encodec_24khz) architecture.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        chunk_length (`float`, *optional*):
            The length of each chunk of audio that will be processed sequentially . If `chunk_length = None` then
            chunking is disabled.
        chunk_stride(`float`, *optional*):
            The length of the right stride use to split the audio in smaller chunkgs. If `chunk_stride = None` then
            chunking is disabled.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~EncodecFeatureExtractor.__call__`] should return `attention_mask`.
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        chunk_length: int = 48000,
        chunk_stride: int = 47520,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.return_attention_mask = return_attention_mask
        self.chunk_stride = chunk_stride
        self.chunk_length = chunk_length

    def __call__(
        self,
        audios: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
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
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            audios (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`, *optional*):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. The numpy array must be of shape
                `(num_samples,)` for mono audio (`feature_size = 1`), or `(2, num_samples)` for stereo audio
                (`feature_size = 2`).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
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

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the ``sampling_rate`` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched = bool(
            isinstance(audios, (list, tuple))
            and (isinstance(audios[0], np.ndarray) or isinstance(audios[0], (tuple, list)))
        )

        if is_batched:
            audios = [np.asarray(audio, dtype=np.float32).T for audio in audios]
        elif not is_batched and not isinstance(audios, np.ndarray):
            audios = np.asarray(audios, dtype=np.float32)
        elif isinstance(audios, np.ndarray) and audios.dtype is np.dtype(np.float64):
            audios = audios.astype(np.float32)

        # always return batch
        if not is_batched:
            audios = [np.asarray(audios).T]

        # verify inputs are valid
        for idx, example in enumerate(audios):
            if example.ndim == 1:
                example = example[..., None]
                audios[idx] = example
            if example.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.T.shape}")
            if self.feature_size == 1 and example.shape[-1] != 1:
                raise ValueError(f"Expected mono audio but example has {example.shape[-1]} channels")
            if self.feature_size == 2 and example.shape[-1] != 2:
                raise ValueError(f"Expected stereo audio but example has {example.shape[-1]} channels")

        if self.chunk_stride is not None and self.chunk_length is not None:
            # Get nax length:
            max_length = max([array.shape[0] for array in audios])
            nb_step = int(np.ceil(max_length / self.chunk_stride))
            max_length = (nb_step - 1) * self.chunk_stride + self.chunk_length

            padded_audios = []
            padding_masks = []
            for sample in audios:
                padding_length = max_length - sample.shape[0]
                sample = np.pad(sample, pad_width=((0, padding_length), (0, 0)), mode="constant", constant_values=0)
                padded_audios.append(sample)
                if return_attention_mask:
                    padding_mask = np.ones(max_length)
                    padding_mask[..., -padding_length:] = 0
                    padding_masks.append(padding_mask)
            padded_inputs = BatchFeature({"input_values": padded_audios})
        else:
            padded_inputs = BatchFeature({"input_values": audios})

        # output shape is (batch, channels, num_samples)
        input_values = []
        for example in padded_inputs["input_values"]:
            if example.ndim == 1:
                example = example[..., None]  # add mono channel dimension
            input_values.append(example.T)

        # convert input values to correct format
        if not isinstance(input_values[0], np.ndarray):
            padded_inputs["input_values"] = [np.asarray(array, dtype=np.float32) for array in input_values]
        elif (
            not isinstance(input_values, np.ndarray)
            and isinstance(input_values[0], np.ndarray)
            and input_values[0].dtype is np.dtype(np.float64)
        ):
            padded_inputs["input_values"] = [array.astype(np.float32) for array in input_values]
        elif isinstance(input_values, np.ndarray) and input_values.dtype is np.dtype(np.float64):
            padded_inputs["input_values"] = input_values.astype(np.float32)
        else:
            padded_inputs["input_values"] = input_values

        # convert attention_mask to correct format
        if return_attention_mask and padding_masks is not None:
            padded_inputs["padding_mask"] = [np.asarray(array, dtype=np.int32) for array in padding_masks]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs
