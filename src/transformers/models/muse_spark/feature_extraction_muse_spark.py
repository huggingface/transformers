# Copyright 2026 the HuggingFace Team. All rights reserved.
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
Feature extractor class for Muse Spark Open.
"""

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class MuseSparkFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Muse Spark Open feature extractor.

    Muse Spark Opens's audio tower consumes the raw waveform — there is no spectrogram front end — so this
    only batches, pads and (optionally) truncates. The one non-obvious job it does is pad every batch
    up to a multiple of `samples_per_token`: the encoder collapses exactly that many samples into one
    embedding and raises otherwise.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]
    which contains most of the main methods. Users should refer to this superclass for more information
    regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Muse Spark Open takes mono audio.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate, in hertz, the audio is expected at. Audio at any other rate must be
            resampled before it reaches this class.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to pad, on the right.
        samples_per_token (`int`, *optional*, defaults to 1920):
            Waveform samples the encoder collapses into one embedding — the product of the audio
            config's `patch_sizes` (16 * 6 * 5 * 4). Batches are padded up to a multiple of this.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether [`~MuseSparkFeatureExtractor.__call__`] should return `input_values_mask`. The audio
            tower is fully convolutional and does not consume one; it is off by default.
    """

    model_input_names = ["input_values", "input_values_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=24000,
        padding_value=0.0,
        samples_per_token=1920,
        return_attention_mask=True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.samples_per_token = samples_per_token

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str = "longest",
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = None,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare one or several sequence(s) for the model.

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to pad. Each sequence must be a mono 1-D float
                waveform sampled at `self.sampling_rate`.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `"longest"`):
                Padding strategy. Whatever is chosen, the result is still rounded up to a multiple of
                `pad_to_multiple_of`.
            max_length (`int`, *optional*):
                Maximum length of the returned list, and optionally padding length.
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to truncate sequences longer than `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                Pad to a multiple of this value. Defaults to `self.samples_per_token`; overriding it
                with a value that is not a multiple of `samples_per_token` will make the encoder raise.
            return_attention_mask (`bool`, *optional*):
                Whether to return `input_values_mask`. Defaults to `self.return_attention_mask`.
                Keep this on for batches: the model needs it to know how many tokens of each
                sample are real, since padding to the longest clip adds whole silent tokens.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, returns tensors instead of a list of python lists. `'pt'` for PyTorch.
            sampling_rate (`int`, *optional*):
                The sampling rate the `raw_speech` was sampled at. Passing it is strongly recommended:
                it turns a silent resampling mistake into an error.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that are hard to debug."
            )

        # add batch dimension if necessary
        if not isinstance(raw_speech, (list, tuple)):
            raw_speech = np.atleast_2d(raw_speech)
        elif not raw_speech or np.ndim(raw_speech[0]) == 0:
            raw_speech = [raw_speech]

        # convert to numpy arrays and downcast fp64 to fp32
        raw_speech = [
            np.asarray(array, dtype=np.float32)
            if not isinstance(array, np.ndarray) or array.dtype == np.dtype(np.float64)
            else array
            for array in raw_speech
        ]

        if any(array.ndim != 1 for array in raw_speech):
            raise ValueError(f"Only mono-channel audio is supported for input to {self.__class__.__name__}")

        # The encoder divides the waveform into `samples_per_token`-sample patches and raises on a
        # remainder, so this rounding is not an optimization - it is what makes the batch loadable.
        if pad_to_multiple_of is None:
            pad_to_multiple_of = self.samples_per_token

        padded_inputs = self.pad(
            BatchFeature({"input_values": raw_speech}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        input_values = padded_inputs["input_values"]
        if not isinstance(input_values[0], np.ndarray):
            padded_inputs["input_values"] = [np.asarray(array, dtype=np.float32) for array in input_values]
        elif input_values[0].dtype is np.dtype(np.float64):
            padded_inputs["input_values"] = [array.astype(np.float32) for array in input_values]

        # `pad` names the mask `attention_mask`; the model takes it as `input_values_mask` to keep it
        # distinct from the text attention mask, which lives on a different (token) axis.
        if "attention_mask" in padded_inputs:
            padded_inputs["input_values_mask"] = [
                np.asarray(array, dtype=np.int32) for array in padded_inputs.pop("attention_mask")
            ]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["MuseSparkFeatureExtractor"]
