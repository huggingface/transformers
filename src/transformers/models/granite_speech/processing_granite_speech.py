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
Processor class for Speech Granite.
"""
from typing import List, Union

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTokenizedInput, TextInput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GraniteSpeechProcessor(ProcessorMixin):

    attributes = ["feature_extractor", "tokenizer"]
    valid_kwargs = ["audio_token"]

    feature_extractor_class = "GraniteSpeechFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_token="<|audio|>",
    ):
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:

        if text is None and audios is None:
            raise ValueError("You have to provide audio or text")

        speech_inputs = {}
        text_inputs = {}

        text = self._get_validated_text(text)
        audios = self._get_validated_audios(audios)
        # TODO: assert that len(audios) == count(audio_token, text)
        
        if audios is not None:
            # Calculate Mel features & the number of placeholders we will need
            speech_inputs["input_features"] = self.feature_extractor(
                audios,
                device=device,
            )
            num_audio_features = self.feature_extractor._get_num_audio_features(
                speech_inputs["input_features"],
            )

            # duplicate the audio placeholders to match the feature dims
            text = self._expand_audio_placeholders(text, num_audio_features)

        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
        return BatchFeature(data={**text_inputs, **speech_inputs})

    def _expand_audio_placeholders(self, text: list[str], num_audio_features: int):
        """
        Expands audio placeholders in the formatted text to match the number of
        features of the corresponding embeddings; we can use the resulting text
        to conveniently mask the audio features into the text embeddings.
        """
        prompt_strings = []
        for sample in text:
            while self.audio_token in sample:
                # todo: (Avihu): this assumes all audios have the same length.
                sample = sample.replace(self.audio_token, "<placeholder>" * num_audio_features, 1)
                prompt_strings.append(sample)
            prompt_strings = [sample.replace("<placeholder>", self.audio_token) for sample in prompt_strings]
        return prompt_strings

    ##### Validation
    def _get_validated_text(self, text: Union[str, list]) -> List[str]:
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list) and isinstance(text[0], str):
            return text
        raise TypeError("Invalid text provided! Text should be a string or list of strings.")

    def _get_validated_audios(self, audios):
        # todo: if this is a list, collate and keep track of audio lengths
        if audios is not None and not isinstance(audios, torch.Tensor):
            raise TypeError("Invalid audios provided! Audio should be a torch tensor.")
        return audios


__all__ = ["GraniteSpeechProcessor"]