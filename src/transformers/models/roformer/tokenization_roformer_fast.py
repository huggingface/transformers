# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for RoFormer."""
import json
from typing import List, Optional, Tuple

from tokenizers import normalizers
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_roformer import RoFormerTokenizer
from .tokenization_utils import JiebaPreTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_char_small": (
            "https://huggingface.co/junnyu/roformer_chinese_char_small/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_chinese_char_base": (
            "https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_discriminator": (
            "https://huggingface.co/junnyu/roformer_small_discriminator/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_generator": (
            "https://huggingface.co/junnyu/roformer_small_generator/resolve/main/vocab.txt"
        ),
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "junnyu/roformer_chinese_small": 1536,
    "junnyu/roformer_chinese_base": 1536,
    "junnyu/roformer_chinese_char_small": 512,
    "junnyu/roformer_chinese_char_base": 512,
    "junnyu/roformer_small_discriminator": 128,
    "junnyu/roformer_small_generator": 128,
}


PRETRAINED_INIT_CONFIGURATION = {
    "junnyu/roformer_chinese_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_base": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_base": {"do_lower_case": True},
    "junnyu/roformer_small_discriminator": {"do_lower_case": True},
    "junnyu/roformer_small_generator": {"do_lower_case": True},
}


class RoFormerTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" RoFormer tokenizer (backed by HuggingFace's *tokenizers* library).

    [`RoFormerTokenizerFast`] is almost identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece. There are some difference between them when tokenizing Chinese.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Example:

    ```python
    >>> from transformers import RoFormerTokenizerFast

    >>> tokenizer = RoFormerTokenizerFast.from_pretrained("junnyu/roformer_chinese_base")
    >>> tokenizer.tokenize("今天天气非常好。")
    ['今', '天', '天', '气', '非常', '好', '。']
    ```"""

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = RoFormerTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        self.do_lower_case = do_lower_case

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"].pre_tokenizer = BertPreTokenizer()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        vocab = self.__dict__["_tokenizer"].get_vocab()
        self.__dict__["_tokenizer"].pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer(vocab))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoFormer sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RoFormer
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def save_pretrained(
        self,
        save_directory,
        legacy_format=None,
        filename_prefix=None,
        push_to_hub=False,
        **kwargs,
    ):
        self.backend_tokenizer.pre_tokenizer = BertPreTokenizer()
        return super().save_pretrained(save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)
