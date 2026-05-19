# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Processor for Granite Speech NAR."""

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import AudioInput
from ...utils import is_torch_available
from .feature_extraction_granite_speech_nar import GraniteSpeechNarFeatureExtractor


if is_torch_available():
    import torch


class GraniteSpeechNarProcessor(ProcessorMixin):
    """Processor combining audio feature extraction and tokenizer for GraniteSpeechNar."""

    feature_extractor_class = "GraniteSpeechNarFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor: GraniteSpeechNarFeatureExtractor, tokenizer=None, **kwargs):
        super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer, **kwargs)

    def __call__(
        self,
        audios: AudioInput,
        device: str | "torch.device" | None = None,
        **kwargs,
    ) -> dict:
        return self.feature_extractor(audios, device=device)

    def batch_decode(self, token_ids_list: list["torch.Tensor"], **kwargs) -> list[str]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Pass tokenizer to GraniteSpeechNarProcessor.")
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids_list]


__all__ = ["GraniteSpeechNarProcessor"]
