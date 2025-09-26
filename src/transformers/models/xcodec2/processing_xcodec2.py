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
Processor class for Xcodec2
"""

import torch.nn.functional as F

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput


class Xcodec2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


class Xcodec2Processor(ProcessorMixin):
    r"""
    Constructs an Xcodec2 processor which wraps an Xcodec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.
    Args:
        feature_extractor (`DacFeatureExtractor`):
            An instance of [`DacFeatureExtractor`]. The feature extractor is a required input.
        feature_extractor_semantic (`SeamlessM4TFeatureExtractor`):
            An instance of [`SeamlessM4TFeatureExtractor`]. The feature extractor is a required input.
    """

    attributes = ["feature_extractor", "semantic_feature_extractor"]

    feature_extractor_class = "DacFeatureExtractor"
    semantic_feature_extractor_class = "SeamlessM4TFeatureExtractor"

    def __init__(self, feature_extractor, semantic_feature_extractor):
        super().__init__(feature_extractor, semantic_feature_extractor)

    def __call__(
        self,
        audio: AudioInput,
        **kwargs: Unpack[Xcodec2ProcessorKwargs],
    ):
        """
        Main method to prepare inputs for the Xcodec2 model, namely processed audio for the acoustic model
        and semantic embeddings.
        """

        call_kwargs = self._merge_kwargs(
            Xcodec2ProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = call_kwargs["audio_kwargs"]
        common_kwargs = call_kwargs["common_kwargs"]
        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        inputs = self.feature_extractor(audio, **audio_kwargs)
        hop_length = self.feature_extractor.hop_length

        # Below is audio processing / preparation in the modeling of PyPI version (xcodec2==0.1.3)
        # 1) redundant padding which was probably accidental on their part, but needed to get same results since it
        # pads even if input is multiple of hop length
        seq_len = inputs["input_values"].shape[-1]
        padding = hop_length - (seq_len % hop_length)
        inputs["input_values"] = F.pad(inputs["input_values"], (0, padding))
        # -- pad mask as well
        inputs["padding_mask"] = F.pad(inputs["padding_mask"], (0, padding))

        # 2) pass through semantic model to get semantic embeddings
        inputs["semantic_input_values"] = self.semantic_feature_extractor(
            F.pad(inputs["input_values"], (hop_length // 2, hop_length // 2)).cpu().tolist(),
            return_attention_mask=False,
            **audio_kwargs,
        ).input_features

        return inputs


__all__ = ["Xcodec2Processor"]
