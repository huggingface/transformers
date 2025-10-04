# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Qwen2."""

import os
from typing import Optional

import regex as re
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from ...create_fast_tokenizer import generate_merges


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import Qwen2TokenizerFast

    >>> tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```
    This is expected.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*):
            Whether or not the tokenizer should automatically add a prefix space
        from_scratch (`bool`, *optional*, defaults to `False`):
            Whether to create an empty trainable tokenizer from scratch. When `True`, creates a minimal tokenizer
            with only basic special tokens that can be trained on new data.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=None,
        vocab=None,
        merges=None,
        **kwargs,
    ):
        # Set add_prefix_space attribute for use in override methods
        self.add_prefix_space = add_prefix_space if add_prefix_space is not None else False

        self._vocab = vocab if vocab is not None else self._vocab()
        self._merges = merges if merges is not None else generate_merges(self._vocab)

        # Prepare base-class construction helpers
        tokenizer_backend_config = None
        if tokenizer_file is None:
            tokenizer_backend_config = {
                "type": "bpe",
                "vocab": self._vocab,
                "merges": self._merges,
                "normalizer": self._normalizer,
                "pre_tokenizer": self._pre_tokenizer,
                "decoder": self._decoder,
                "tokenizer": self._tokenizer,
            }

        # Initialize the base class which will build the backend tokenizer
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            tokenizer_backend_config=tokenizer_backend_config,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # Add special tokens after tokenizer is initialized
        self.add_tokens([AddedToken(token, special=True) for token in self.all_special_tokens])

    def _tokenizer(self):
        """Tokenizer configuration for this tokenizer."""
        return Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )

    def _vocab(self):
        """Vocabulary handling for this tokenizer."""
        vocab = {
            "<|endoftext|>": 0,
        }
        return vocab

    def _decoder(self, replacement, add_prefix_space):
        """Decoder configuration for this tokenizer."""
        return decoders.ByteLevel()

    def _normalizer(self):
        """Normalizer configuration for this tokenizer."""
        return normalizers.NFC()

    def _pre_tokenizer(self, replacement, add_prefix_space):
        """Pre-tokenizer configuration for this tokenizer."""
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(PRETOKENIZE_REGEX),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=add_prefix_space,
                    use_regex=False,
                ),
            ]
        )


__all__ = ["Qwen2TokenizerFast"]
