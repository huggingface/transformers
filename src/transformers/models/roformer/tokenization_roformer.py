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
"""Tokenization class for RoFormer backed by 🤗 Tokenizers."""

from typing import Optional

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer

from ...tokenization_utils_tokenizers import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_utils import JiebaPreTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class RoFormerTokenizer(PreTrainedTokenizerFast):
    r"""
    Construct a RoFormer tokenizer. Based on [Rust Jieba](https://pypi.org/project/rjieba/).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Example:

    ```python
    >>> from transformers import RoFormerTokenizer

    >>> tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_base")
    >>> tokenizer.tokenize("今天天气非常好。")
    ['今', '天', '天', '气', '非常', '好', '。']
    ```
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab: Optional[dict[str, int]] = None,
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
        tokenizer = Tokenizer(models.WordPiece(vocab, unk_token=str(unk_token)))
        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(JiebaPreTokenizer(vocab))

        tokenizer.decoder = decoders.WordPiece(prefix="##")
        self._tokenizer = tokenizer
        super().__init__(
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
        cls_ = str(cls_token)
        sep_ = str(sep_token)
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_}:0 $A:0 {sep_}:0",
            pair=f"{cls_}:0 $A:0 {sep_}:0 $B:1 {sep_}:1",
            special_tokens=[
                (cls_, self.cls_token_id),
                (sep_, self.sep_token_id),
            ],
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        tokenizer_copy = Tokenizer.from_str(state["_tokenizer"].to_str())
        tokenizer_copy.pre_tokenizer = BertPreTokenizer()
        state["_tokenizer"] = tokenizer_copy
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

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Create token type IDs for RoFormer sequence pairs.

        The first sequence and associated special tokens are mapped to 0, while the second sequence (if provided) and
        its trailing separator are mapped to 1.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
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
        result = super().save_pretrained(save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)
        vocab = self.backend_tokenizer.get_vocab()
        self.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer(vocab))
        return result


RoFormerTokenizerFast = RoFormerTokenizer

__all__ = ["RoFormerTokenizer", "RoFormerTokenizerFast"]
