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
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class FunAsrNanoProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "FunAsrNanoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|object_ref_start|>",
        audio_downsample_rate=1,
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|object_ref_start|>"`):
            The token to use for audio placeholders.
        audio_downsample_rate (`int`, *optional*, defaults to 1):
            Downsampling ratio applied by the audio adaptor, used to expand audio placeholder tokens.
        """
        if chat_template is None:
            chat_template = self.default_chat_template

        if tokenizer is not None and tokenizer.convert_tokens_to_ids(audio_token) is None:
            fallback_audio_token = "<|object_ref_start|>"
            if tokenizer.convert_tokens_to_ids(fallback_audio_token) is None:
                raise ValueError(f"Audio token {audio_token!r} is not present in the tokenizer vocabulary.")
            logger.warning_once(
                f"Audio token {audio_token!r} is not present in the tokenizer vocabulary. "
                f"Using {fallback_audio_token!r} instead."
            )
            chat_template = chat_template.replace(audio_token, fallback_audio_token)
            audio_token = fallback_audio_token

        self.audio_token = audio_token
        self.audio_downsample_rate = audio_downsample_rate
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: np.ndarray | list[np.ndarray] | None = None,
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        r"""
        sampling_rate (`int`, *optional*):
            Sampling rate of the input audio. Must be 16000 for Fun-ASR-Nano.
        """
        if text is None:
            raise ValueError("You need to specify `text` input to process.")

        is_batched_text = isinstance(text, list)
        text = list(text) if is_batched_text else [text]
        audio_features = None
        if audio is not None:
            audio_features = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate or self.feature_extractor.sampling_rate,
                return_tensors=return_tensors,
            )

            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = 1 if isinstance(audio, np.ndarray) and audio.ndim == 1 else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} "
                    f"in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}."
                )

            audio_lengths = audio_features["feature_lengths"].tolist()
            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    num_audio_tokens = (audio_length - 1) // self.audio_downsample_rate + 1
                    replace_str.append(self.audio_token * int(num_audio_tokens))
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        tokenizer_kwargs = kwargs.copy()
        tokenizer_kwargs.setdefault("padding", True)
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )

        if audio_features is not None:
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
        Audio elements are replaced with the <|object_ref_start|> token placeholder.

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
                            "<|object_ref_start|>"
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
