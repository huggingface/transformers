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
    Constructs a Qwen3TTS processor which combines a Qwen tokenizer with Qwen3-TTS-specific processing.

    Args:
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer for encoding text inputs. Should be a Qwen2 tokenizer.
        chat_template (`str`, *optional*):
            The Jinja template to use for formatting conversations using the chat template.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, tokenizer=None, chat_template=None):
        super().__init__(tokenizer, chat_template=chat_template)

    def __call__(self, text=None, **kwargs) -> BatchFeature:
        """
        Prepare text for the Qwen3-TTS model.

        Args:
            text (`str` or `List[str]`):
                The text string or batch of text strings to be encoded.
            **kwargs:
                Additional keyword arguments passed to the tokenizer.

        Returns:
            BatchFeature: Dictionary containing tokenized text inputs.
        """
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3TTSProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, list):
            text = [text]

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**text_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's batch_decode method.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's decode method.
        """
        return self.tokenizer.decode(*args, **kwargs)

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
        """Return the input names of the underlying tokenizer."""
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(tokenizer_input_names))


__all__ = ["Qwen3TTSProcessor"]
