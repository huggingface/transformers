# coding=utf-8
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Fast Tokenization class for Blenderbot."""

from typing import TYPE_CHECKING, List, Optional

from ...utils import logging
from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
from .tokenization_blenderbot import BlenderbotTokenizer


if TYPE_CHECKING:
    from transformers.pipelines.conversational import Conversation

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_config_file": "tokenizer_config.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/vocab.json"},
    "merges_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/merges.txt"},
    "tokenizer_config_file": {
        "facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/tokenizer_config.json"
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/blenderbot-3B": 128}


class BlenderbotTokenizerFast(RobertaTokenizerFast):
    r"""
    Construct a "fast" Blenderbot tokenizer (backed by HuggingFace's *tokenizers* library).

    [`BlenderbotFast`] is nearly identical to [`RobertaTokenizerFast`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece. The only difference is that it doesn't add BOS token to the beginning of sequences.

    Refer to superclass [`RobertaTokenizerFast`] for usage examples and documentation concerning parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = BlenderbotTokenizer

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Blenderbot sequence has the following format:

        - single sequence: ` X </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Will be ignored

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        return token_ids_0 + [self.eos_token_id]

    def _build_conversation_input_ids(self, conversation: "Conversation") -> List[int]:
        inputs = []
        for is_user, text in conversation.iter_texts():
            if is_user:
                # We need to space prefix as it's being done within blenderbot
                inputs.append(" " + text)
            else:
                # Generated responses should contain them already.
                inputs.append(text)

        full_string = "  ".join(inputs)
        input_ids = self.encode(full_string)
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
            logger.warning(f"Trimmed input from conversation as it was longer than {self.model_max_length} tokens.")
        return input_ids
