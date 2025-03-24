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

from collections.abc import Sequence
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
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        audios: Union[torch.Tensor, List[torch.Tensor]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:
        speech_inputs = {}
        text_inputs = {}

        text = self._get_validated_text(text)
        expected_num_audios = sum(t.count(self.audio_token) for t in text)

        if audios is not None:
            audios, audio_lengths = self._get_validated_audios(audios)
            if any(text.count(self.audio_token) != 1 for text in text):
                raise ValueError("Only one audio sample is currently supported per input")
            if len(audio_lengths) != expected_num_audios:
                raise ValueError("Text/Audio mismatch. The number of audios and audio tokens do not match")

            # Calculate Mel features & the number of placeholders we will need
            speech_inputs["input_features"] = self.feature_extractor(
                audios,
                device=device,
            )
            num_audio_features = self.feature_extractor._get_num_audio_features(audio_lengths)
            speech_inputs["input_features_mask"] = torch.arange(max(num_audio_features)).view(1, -1) <= torch.tensor(
                num_audio_features
            ).view(-1, 1)

            # duplicate the audio placeholders to match the feature dims
            text = self._expand_audio_placeholders(text, num_audio_features)
        else:
            assert expected_num_audios == 0, "No audio is provided, expecting no audio tokens"

        text_inputs = self.tokenizer(text, padding=True, **kwargs)
        return BatchFeature(data={**text_inputs, **speech_inputs})

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

    ##### Validation
    def _get_validated_text(self, text: Union[str, list]) -> List[str]:
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list) and isinstance(text[0], str):
            return text
        raise TypeError("Invalid text provided! Text should be a string or list of strings.")

    def _get_validated_audios(self, audios):
        # Coerce to PyTorch tensors if we have numpy arrays, since
        # currently we have a dependency on torch/torchaudio anyway
        if isinstance(audios, np.ndarray):
            audios = torch.from_numpy(audios)
        elif isinstance(audios, Sequence) and isinstance(audios[0], np.ndarray):
            audios = [torch.from_numpy(arr) for arr in audios]

        if isinstance(audios, torch.Tensor):
            if audios.ndim == 1:
                audios = audios.unsqueeze(0)
            if not torch.is_floating_point(audios):
                raise ValueError("Invalid audio provided. Audio should be a floating point between 0 and 1")

            if audios.shape[0] > 1:
                logger.warning("Audio samples are already collated; assuming they all have the same length")
            lengths = [audios.shape[-1]] * audios.shape[0]
            return audios, lengths

        elif isinstance(audios, Sequence) and isinstance(audios[0], torch.Tensor):
            if not torch.is_floating_point(audios[0]):
                raise ValueError("Invalid audio provided. Audio should be a floating point between 0 and 1")
            lengths = [audio.shape[-1] for audio in audios]
            padding = [max(lengths) - length for length in lengths]
            # ensure all audios have a batch dimension:
            audios = [audio.view(1, -1) for audio in audios]
            padded = [torch.nn.functional.pad(audio, (0, pad)) for audio, pad in zip(audios, padding)]
            audios = torch.cat(padded, dim=0)
            return audios, lengths

        raise TypeError("Invalid audio provided. Audio should be a one or more torch tensors or numpy arrays")


__all__ = ["GraniteSpeechProcessor"]
