# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
""" Tokenization class for Funnel Transformer."""

from typing import List, Optional

from .tokenization_bert_fast import BertTokenizerFast
from .tokenization_funnel import FunnelTokenizer
from .utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

_model_names = [
    "small",
    "small-base",
    "medium",
    "medium-base",
    "intermediate",
    "intermediate-base",
    "large",
    "large-base",
    "xlarge",
    "xlarge-base",
]

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/vocab.txt",
        "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/vocab.txt",
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/vocab.txt",
        "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/vocab.txt",
        "funnel-transformer/intermediate": "https://huggingface.co/funnel-transformer/intermediate/resolve/main/vocab.txt",
        "funnel-transformer/intermediate-base": "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/vocab.txt",
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/vocab.txt",
        "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/vocab.txt",
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/vocab.txt",
        "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/tokenizer.json",
        "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/tokenizer.json",
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/tokenizer.json",
        "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/tokenizer.json",
        "funnel-transformer/intermediate": "https://huggingface.co/funnel-transformer/intermediate/resolve/main/tokenizer.json",
        "funnel-transformer/intermediate-base": "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/tokenizer.json",
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/tokenizer.json",
        "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/tokenizer.json",
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/tokenizer.json",
        "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/tokenizer.json",
    },
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {f"funnel-transformer/{name}": 512 for name in _model_names}
PRETRAINED_INIT_CONFIGURATION = {f"funnel-transformer/{name}": {"do_lower_case": True} for name in _model_names}


class FunnelTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" Funnel Transformer tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.FunnelTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs
    end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = FunnelTokenizer
    cls_token_type_id: int = 2

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
        clean_text=True,
        tokenize_chinese_chars=True,
        strip_accents=None,
        wordpieces_prefix="##",
        **kwargs
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
            bos_token=bos_token,
            eos_token=eos_token,
            clean_text=clean_text,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            wordpieces_prefix=wordpieces_prefix,
            **kwargs,
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Funnel
        Transformer sequence pair mask has the following format:

        ::

            2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0]
        return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
