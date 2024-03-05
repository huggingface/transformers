# coding=utf-8
# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for GIT."""
from typing import TYPE_CHECKING, List

from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.utils.generic import PaddingStrategy, TensorType

from ...utils import is_torch_available, logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_git import GitTokenizer


if TYPE_CHECKING:
    if is_torch_available():
        import torch


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/git-base": "https://huggingface.co/microsoft/git-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "microsoft/git-base": "https://huggingface.co/microsoft/git-base/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/git-base": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/git-base": {"do_lower_case": False},
}


class GitTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" GIT tokenizer (backed by HuggingFace's *tokenizers* library).
    [`~GitTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization: punctuation splitting
    and wordpiece.
    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = GitTokenizer

    def __call__(
        self,
        text: str | List[str] | List[List[str]] = None,
        text_pair: str | List[str] | List[List[str]] | None = None,
        text_target: str | List[str] | List[List[str]] = None,
        text_pair_target: str | List[str] | List[List[str]] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        add_special_tokens = False
        encodings = super().__call__(
            text,
            text_pair,
            text_target,
            text_pair_target,
            add_special_tokens,
            padding,
            truncation,
            max_length,
            stride,
            is_split_into_words,
            pad_to_multiple_of,
            return_tensors,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_offsets_mapping,
            return_length,
            verbose,
            **kwargs,
        )
        input_ids = encodings["input_ids"]
        input_ids = torch.cat((torch.tensor([[self.cls_token_id]]), input_ids), dim=1)
        model_inputs = {"input_ids": input_ids, "max_length": 50}
        return model_inputs
