# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for MossAudioTokenizer."""

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class MossAudioTokenizerFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a MossAudioTokenizer feature extractor.

    This feature extractor prepares mono waveform audio for [`MossAudioTokenizerModel`] by padding a batch of audio
    sequences and returning the corresponding padding mask.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension. MOSS Audio Tokenizer expects mono audio, so this should be 1.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which audio should be sampled.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value used for padding.
        hop_length (`int`, *optional*, defaults to 1920):
            The model downsampling factor. Inputs are padded to a multiple of this value by default.
    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        hop_length: int = 1920,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.hop_length = hop_length

    def __call__(
        self,
        raw_audio: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy | None = True,
        truncation: bool | None = False,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = True,
        return_tensors: str | TensorType | None = None,
        sampling_rate: int | None = None,
    ) -> BatchFeature:
        """
        Main method to prepare one or several waveform sequence(s) for the model.

        Args:
            raw_audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                A mono audio sequence or a batch of mono audio sequences.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Padding strategy passed to [`SequenceFeatureExtractor.pad`].
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to truncate to `max_length`.
            max_length (`int`, *optional*):
                Maximum sequence length when padding or truncating.
            pad_to_multiple_of (`int`, *optional*):
                If unset, inputs are padded to a multiple of `hop_length`.
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a padding mask.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, converts outputs to tensors.
            sampling_rate (`int`, *optional*):
                Sampling rate of `raw_audio`. Passing this is strongly recommended.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor was trained using a sampling rate of "
                    f"{self.sampling_rate}. Please make sure that the provided audio input was sampled with "
                    f"{self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if padding and truncation:
            raise ValueError("Both padding and truncation were set. Make sure you only set one.")

        is_batched = bool(
            isinstance(raw_audio, (list, tuple))
            and len(raw_audio) > 0
            and isinstance(raw_audio[0], (np.ndarray, list))
        )
        if is_batched:
            raw_audio = [np.asarray(audio, dtype=np.float32) for audio in raw_audio]
        else:
            raw_audio = [np.asarray(raw_audio, dtype=np.float32)]

        for example in raw_audio:
            if example.ndim != 1:
                raise ValueError(f"Expected mono audio with shape `(sequence_length,)`, got shape {example.shape}.")

        encoded_inputs = BatchFeature({"input_values": raw_audio})
        encoded_inputs = self.pad(
            encoded_inputs,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=return_attention_mask,
            pad_to_multiple_of=self.hop_length if pad_to_multiple_of is None else pad_to_multiple_of,
        )

        if return_attention_mask and "attention_mask" in encoded_inputs:
            encoded_inputs["padding_mask"] = encoded_inputs.pop("attention_mask")

        input_values = []
        for example in encoded_inputs.pop("input_values"):
            input_values.append(example[None, :])
        encoded_inputs["input_values"] = input_values

        if return_tensors is not None:
            encoded_inputs = encoded_inputs.convert_to_tensors(return_tensors)

        return encoded_inputs


__all__ = ["MossAudioTokenizerFeatureExtractor"]
