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
"""Processor class for Granite Speech."""

from typing import List, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import PreTokenizedInput, TextInput
from ...utils import is_torch_available, logging
from ...utils.import_utils import requires_backends


if is_torch_available():
    import torch

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
        chat_template=None,
    ):
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        super().__init__(audio_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        audio: Union["torch.Tensor", List["torch.Tensor"]] = None,
        device: str = "cpu",
        images=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        requires_backends(self, ["torch"])

        text = self._get_validated_text(text)
        prompt_strings = text

        if audio is not None:
            # NOTE - we intentionally avoid throwing for potentially misaligned
            # text / audio inputs here because some inference engines will
            # trigger the conditions due to the way they call multimodal
            # processors, e.g., vLLM.
            audio_inputs = self.audio_processor(audio, device=device)

            # TODO (@alex-jw-brooks); we should add a util to get_num_audio_tokens
            # from feature lengths and call it here, rather than returning it
            # from the feature extractor.
            audio_embed_sizes = audio_inputs.pop("audio_embed_sizes")

            # Expand the audio placeholders to match the feature dims; this
            # is similar to how many VLMs handle image tokens, e.g., llava next
            prompt_strings = []
            num_replaced = 0
            for sample in text:
                while self.audio_token in sample:
                    sample = sample.replace(
                        self.audio_token,
                        "<placeholder>" * audio_embed_sizes[num_replaced],
                        1,
                    )
                    num_replaced += 1
                prompt_strings.append(sample)

            prompt_strings = [sample.replace("<placeholder>", self.audio_token) for sample in prompt_strings]
        else:
            audio_inputs = {}

        text_inputs = self.tokenizer(prompt_strings, padding=True, **kwargs)
        return BatchFeature(data={**text_inputs, **audio_inputs})

    def _get_validated_text(self, text: Union[str, list]) -> List[str]:
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list) and isinstance(text[0], str):
            return text
        raise TypeError("Invalid text provided! Text should be a string or list of strings.")


__all__ = ["GraniteSpeechProcessor"]
