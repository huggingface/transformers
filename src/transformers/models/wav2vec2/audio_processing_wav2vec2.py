# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Audio processor class for Wav2Vec2
"""

from typing import Optional, Union

import torch

from ...audio_processing_utils import BaseAudioProcessor
from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class Wav2Vec2AudioProcessor(BaseAudioProcessor):
    r"""
    Constructs a Wav2Vec2 audio processor.

    This audio processor inherits from [`~audio_processing_utils.BaseAudioProcessor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        sample_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models, *e.g.*,
            [wav2vec2-lv60](https://huggingface.co/models?search=lv60).
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        sample_rate: int = 16000,
        do_normalize: bool = True,
        force_mono: bool = True,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            force_mono=force_mono,
            **kwargs,
        )
        self.do_normalize = do_normalize

    def _preprocess(
        self,
        audio: list[torch.Tensor],
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        do_normalize: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        if do_normalize is None:
            do_normalize = self.do_normalize

        # Truncation and padding via base class
        result = super()._preprocess(
            audio,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,  # we handle conversion after normalization
        )

        audio_tensors = result["audio"]

        if do_normalize:
            audio_tensors = [self._zero_mean_unit_var_norm(t) for t in audio_tensors]

        input_values = torch.stack(audio_tensors, dim=0) if return_tensors else audio_tensors
        return BatchFeature(data={"input_values": input_values}, tensor_type=return_tensors)

    @staticmethod
    def _zero_mean_unit_var_norm(tensor: torch.Tensor) -> torch.Tensor:
        """Zero-mean unit-variance normalize a tensor."""
        return (tensor - tensor.mean()) / torch.sqrt(tensor.var() + 1e-7)


__all__ = ["Wav2Vec2AudioProcessor"]
