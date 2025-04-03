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
Processor class for Granite Speech.
"""

from typing import List, Union

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTokenizedInput, TextInput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GraniteSpeechProcessor(ProcessorMixin):
    attributes = ["audio_processor", "tokenizer"]
    valid_kwargs = ["audio_token"]

    audio_processor_class = "GraniteSpeechFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        audio_processor,
        tokenizer,
        audio_token="<|audio|>",
    ):
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        super().__init__(audio_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        audios: Union[torch.Tensor, List[torch.Tensor]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:

        text = self._get_validated_text(text)

        if audios is not None:
            # NOTE - we intentionally avoid throwing for potentially misaligned
            # text / audio inputs here because some inference engines will
            # trigger the conditions due to the way they call multimodal
            # processors, e.g., vLLM.
            audio_inputs = self.audio_processor(audios, device=device)
            audio_embed_sizes = audio_inputs.pop("audio_embed_sizes")
            # duplicate the audio placeholders to match the feature dims
            processed_text = self._expand_audio_placeholders(text, audio_embed_sizes)
        else:
            audio_inputs = {}

        text_inputs = self.tokenizer(processed_text, padding=True, **kwargs)
        return BatchFeature(data={**text_inputs, **audio_inputs})

    def _expand_audio_placeholders(self, text: list[str], num_audio_features: List[int]):
        """
        Expands audio placeholders in the formatted text to match the number of
        features of the corresponding embeddings; we can use the resulting text
        to conveniently mask the audio features into the text embeddings.
        """
        prompt_strings = []
        num_replaced = 0
        for sample in text:
            while self.audio_token in sample:
                sample = sample.replace(
                    self.audio_token,
                    "<placeholder>" * num_audio_features[num_replaced],
                    1,
                )
                num_replaced += 1
            prompt_strings.append(sample)

        prompt_strings = [sample.replace("<placeholder>", self.audio_token) for sample in prompt_strings]
        return prompt_strings

    def _get_validated_text(self, text: Union[str, list]) -> List[str]:
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list) and isinstance(text[0], str):
            return text
        raise TypeError("Invalid text provided! Text should be a string or list of strings.")



__all__ = ["GraniteSpeechProcessor"]
