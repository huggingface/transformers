# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Processor for Qwen3-TTS."""

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class Qwen3TTSProcessorKwargs(ProcessingKwargs, total=False):
    """Kwargs for Qwen3TTS processor."""

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        }
    }


class Qwen3TTSProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen3TTS processor which combines a Qwen tokenizer and a feature extractor.

    [`Qwen3TTSProcessor`] offers all the functionalities of [`Qwen3TTSFeatureExtractor`] and [`Qwen2TokenizerFast`].
    See the [`~Qwen3TTSProcessor.__call__`] and [`~Qwen3TTSProcessor.decode`] for more information.

    Args:
        feature_extractor ([`Qwen3TTSFeatureExtractor`], *optional*):
            The feature extractor for extracting mel spectrogram features from audio.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer for encoding text inputs. Should be a Qwen2 tokenizer.
        chat_template (`str`, *optional*):
            The Jinja template to use for formatting conversations using the chat template.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "Qwen3TTSFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(self, text=None, audio=None, **kwargs) -> BatchFeature:
        """
        Prepare inputs for the Qwen3-TTS model.

        Args:
            text (`str` or `List[str]`, *optional*):
                The text string or batch of text strings to be encoded.
            audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, *optional*):
                The audio input to extract features from. Used for speaker embedding extraction.
            **kwargs:
                Additional keyword arguments passed to the tokenizer or feature extractor.

        Returns:
            BatchFeature: Dictionary containing tokenized text and/or audio features.
        """
        if text is None and audio is None:
            raise ValueError("You need to specify at least one of `text` or `audio` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3TTSProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if self.tokenizer is not None else {},
            **kwargs,
        )

        data = {}

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            data.update(text_inputs)

        if audio is not None:
            audio_inputs = self.feature_extractor(audio, **output_kwargs.get("audio_kwargs", {}))
            data["input_features"] = audio_inputs["input_features"]

        return BatchFeature(
            data=data,
            tensor_type=kwargs.get("return_tensors"),
        )

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        """
        Apply the chat template to a conversation.

        Args:
            conversations: Single conversation dict or list of conversation dicts.
            chat_template: Optional custom chat template.
            **kwargs: Additional arguments passed to the chat template.
        """
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))


__all__ = ["Qwen3TTSProcessor"]
