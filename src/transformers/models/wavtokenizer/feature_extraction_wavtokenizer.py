# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for WavTokenizer"""

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class WavTokenizerFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a WavTokenizer feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This feature extractor does not resample or downmix audio. Input must already be mono and sampled at the configured
    `sampling_rate`; pass `sampling_rate` to [`__call__`] so it can validate the input rate.

    Note that [`WavTokenizerModel`] pads internally and accepts arbitrary lengths, so single inputs are returned
    unpadded, which keeps the produced audio codes bit-identical to the original WavTokenizer pipeline. Batches of
    different lengths are zero-padded to the longest sample (`padding_mask` marks the valid part); zero-padding can
    perturb the codes of shorter samples near the end, so encode each clip individually when bit-exact codes matter.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Only mono (1) is supported.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used for padding.
        hop_length (`int`, *optional*, defaults to 600):
            Number of audio samples represented by a single audio code.
    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        hop_length: int = 600,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.hop_length = hop_length

    def get_num_audio_codes(self, num_samples: int) -> int:
        """
        Number of discrete audio codes [`WavTokenizerModel`] produces for an input of `num_samples` samples,
        i.e. `ceil(num_samples / hop_length)`. The model pads internally, so this holds for arbitrary lengths.
        """
        return -(-num_samples // self.hop_length)

    def __call__(
        self,
        audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy | None = None,
        truncation: bool | None = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be processed. Each sequence must be a numpy array of shape
                `(num_samples,)` containing mono audio already sampled at this feature extractor's `sampling_rate`.
                Audio is not resampled or downmixed.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different
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
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if padding is None:
            # by default let's pad the inputs
            padding = True

        is_batched = bool(isinstance(audio, (list, tuple)) and (isinstance(audio[0], (np.ndarray, tuple, list))))

        if is_batched:
            audio = [np.asarray(example, dtype=np.float32) for example in audio]
        elif not is_batched and not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        elif isinstance(audio, np.ndarray) and audio.dtype is np.dtype(np.float64):
            audio = audio.astype(np.float32)

        # always return batch
        if not is_batched:
            audio = [np.asarray(audio)]

        # verify inputs are valid
        for idx, example in enumerate(audio):
            if example.ndim > 1:
                raise ValueError(f"Expected mono audio of shape (num_samples,) but got shape {example.shape}")
            if example.size == 0:
                raise ValueError(f"Input audio at index {idx} is empty")

        input_values = BatchFeature({"input_values": audio})

        # Truncate before padding. By default, batches pad to the longest sample while single inputs stay unpadded
        # because the model pads internally, which keeps codes bit-identical to the original WavTokenizer pipeline.
        padded_inputs = self.pad(
            input_values,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=padding,
        )
        if padding:
            padded_inputs["padding_mask"] = padded_inputs.pop("attention_mask")

        input_values = [example[np.newaxis, :] for example in padded_inputs.pop("input_values")]
        padded_inputs["input_values"] = input_values

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["WavTokenizerFeatureExtractor"]
