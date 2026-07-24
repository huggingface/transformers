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

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
from ...utils.import_utils import is_torch_available, requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


@requires(backends=("torch",))
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
        raw_audio: AudioInput | list[AudioInput],
        padding: bool | str | PaddingStrategy | None = True,
        truncation: bool | None = False,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = True,
        return_tensors: str | TensorType | None = "pt",
        sampling_rate: int | None = None,
    ) -> BatchFeature:
        """
        Main method to prepare one or several waveform sequence(s) for the model.

        Args:
            raw_audio (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`, `list[list[float]]`):
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
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `"pt"`):
                The tensor type to return. Only `"pt"` is supported.
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

        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        raw_audio = make_list_of_audio(raw_audio)
        for idx, example in enumerate(raw_audio):
            example = torch.as_tensor(example, dtype=torch.float32)
            if example.ndim != 1:
                raise ValueError(f"Expected mono audio with shape `(sequence_length,)`, got shape {example.shape}.")
            raw_audio[idx] = example

        encoded_inputs = BatchFeature({"input_values": raw_audio})
        encoded_inputs = self.pad(
            encoded_inputs,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_attention_mask=return_attention_mask,
            pad_to_multiple_of=self.hop_length if pad_to_multiple_of is None else pad_to_multiple_of,
            return_tensors="pt",
        )

        if return_attention_mask and "attention_mask" in encoded_inputs:
            encoded_inputs["padding_mask"] = encoded_inputs.pop("attention_mask")

        encoded_inputs["input_values"] = encoded_inputs["input_values"][:, None, :]

        return encoded_inputs


__all__ = ["MossAudioTokenizerFeatureExtractor"]
