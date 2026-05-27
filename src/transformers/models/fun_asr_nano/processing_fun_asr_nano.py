# Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Processor for Fun-ASR-Nano."""

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class FunAsrNanoProcessor(ProcessorMixin):
    r"""
    Constructs a Fun-ASR-Nano processor which wraps a feature extractor and a tokenizer.

    [`FunAsrNanoProcessor`] offers all the functionalities of [`FunAsrNanoFeatureExtractor`] and
    a tokenizer (typically Qwen3). See [`~FunAsrNanoProcessor.__call__`] and [`~FunAsrNanoProcessor.decode`]
    for more information.

    Args:
        feature_extractor (`FunAsrNanoFeatureExtractor`):
            The feature extractor for audio preprocessing.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer for text encoding/decoding.
        chat_template (`str`, *optional*):
            Jinja template for formatting chat messages with audio placeholders.
        audio_token (`str`, *optional*, defaults to `"<|AUDIO|>"`):
            Token used as placeholder for audio features.

    Example:

    ```python
    >>> from transformers import FunAsrNanoProcessor, FunAsrNanoFeatureExtractor, AutoTokenizer

    >>> feature_extractor = FunAsrNanoFeatureExtractor()
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>> processor = FunAsrNanoProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    ```
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "FunAsrNanoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|AUDIO|>",
        **kwargs,
    ):
        self.audio_token = audio_token
        if chat_template is None:
            chat_template = self.default_chat_template
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: np.ndarray | list[np.ndarray] | None = None,
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Process text and audio inputs for the Fun-ASR-Nano model.

        Args:
            text: Text input(s). Should contain `<|AUDIO|>` placeholder tokens where audio is inserted.
                Use `apply_chat_template` to generate proper text from conversation format.
            audio: Raw audio waveform(s) as numpy arrays.
            sampling_rate: Sampling rate of the audio (must be 16000).
            return_tensors: Type of tensors to return ("pt", "np", "tf").

        Returns:
            BatchFeature with input_ids, attention_mask, input_features, and feature_lengths.
        """
        if text is None:
            raise ValueError("You need to specify `text` input to process.")

        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=True,
            **kwargs,
        )

        if audio is not None:
            audio_features = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate or self.feature_extractor.sampling_rate,
                return_tensors=return_tensors,
            )
            return BatchFeature(data={**text_inputs, **audio_features})

        return BatchFeature(data=dict(text_inputs))

    def batch_decode(self, *args, **kwargs):
        """Forward to tokenizer's batch_decode."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Forward to tokenizer's decode."""
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def default_chat_template(self):
        """
        Default chat template for Fun-ASR-Nano. Formats messages with audio placeholders.

        Content can be a string or a list of dicts with "type" field ("audio" or "text").
        Audio elements are replaced with <|AUDIO|> token placeholder.

        Example:

        ```python
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "语音转写成中文："},
                {"type": "audio"},
            ]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ```
        """
        # fmt: off
        return (
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if content['type'] == 'audio' %}"
                            "<|AUDIO|>"
                        "{% elif content['type'] == 'text' %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(feature_extractor_input_names + tokenizer_input_names))


__all__ = ["FunAsrNanoProcessor"]
