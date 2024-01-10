# coding=utf-8
# Copyright 2024 Alibaba Cloud and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_qwen2 import Qwen2Tokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

# TODO: This is a mock. Update this when it is actually released.
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen2": "https://huggingface.co/qwen/qwen2/resolve/main/vocab.json"},
    "merges_file": {"qwen/qwen2": "https://huggingface.co/qwen/qwen2/resolve/main/merges.txt"},
    "tokenizer_file": {"qwen/qwen2": "https://huggingface.co/qwen/qwen2/resolve/main/tokenizer.json"},
}


class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Same with GPT2Tokenzier, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import Qwen2Tokenizer

    >>> tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2")
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
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = Qwen2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        **kwargs,
    ):
        # We need to at least pass vocab_file and merges_file to base class
        # in case a slow tokenizer needs to be initialized; other can be
        # configured through files
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            **kwargs,
        )

        # I think there are (de)serialization bugs in tokenizers: saving the
        # loaded tokenizer gets different tokenizer.json
        # (pre_tokenizer > pretokenizers > 1 > trim_offsets)
        # I don't find the difference in results though.

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
