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
"""Tokenization class for Blenderbot."""

from typing import List

from ...utils import logging
from ..roberta.tokenization_roberta import RobertaTokenizer


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    # "tokenizer_config_file": "tokenizer_config.json",
}
CKPT_3B = "facebook/blenderbot-3B"


class BlenderbotTokenizer(RobertaTokenizer):
    r"""
    Construct a Blenderbot tokenizer.

    :class:`~transformers.Blenderbot` is nearly identical to :class:`~transformers.RobertaTokenizer` and runs
    end-to-end tokenization: punctuation splitting and wordpiece. The only difference is that it doesn't add BOS token
    to the beginning of sequences.

    Refer to superclass :class:`~transformers.RobertaTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "tokenizer_config_file": "tokenizer_config.json",
    }
    pretrained_vocab_files_map = {
        "vocab_file": {CKPT_3B: "https://huggingface.co/facebook/blenderbot-3B/resolve/main/vocab.json"},
        "merges_file": {CKPT_3B: "https://huggingface.co/facebook/blenderbot-3B/resolve/main/merges.txt"},
        "tokenizer_config_file": {
            CKPT_3B: "https://huggingface.co/facebook/blenderbot-3B/resolve/main/tokenizer_config.json"
        },
    }
    max_model_input_sizes = {"facebook/blenderbot-3B": 128}

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Blenderbot sequence has the following format:

        - single sequence: `` X </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Will be ignored

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        return token_ids_0 + [self.eos_token_id]


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs
