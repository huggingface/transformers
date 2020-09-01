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

from .tokenization_bert import BertTokenizer, BertTokenizerFast
from .utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "funnel-small": "https://s3.amazonaws.com/models.huggingface.co/bert/huggingface/funnel-small/vocab.txt",
        "funnel-small-base": "https://s3.amazonaws.com/models.huggingface.co/bert/huggingface/funnel-small-base/vocab.txt",
        # "funnel-medium": "",
        # "funnel-medium-base": "",
        # "funnel": "",
        # "funnel-base": "",
        # "funnel-large": "",
        # "funnel-large-base": "",
        # "funnel-xlarge": "",
        # "funnel-xlarge-base": "",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "funnel-small": 512,
    "funnel-small-base": 512,
    # "funnel-medium": 512,
    # "funnel-medium-base": 512,
    # "funnel": 512,
    # "funnel-base": 512,
    # "funnel-large": 512,
    # "funnel-large-base": 512,
    # "funnel-large": 512,
    # "funnel-large-base": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "funnel-small": {"do_lower_case": True},
    "funnel-small-base": {"do_lower_case": True},
    # "funnel-medium": {"do_lower_case": True},
    # "funnel-medium-base": {"do_lower_case": True},
    # "funnel": {"do_lower_case": True},
    # "funnel-base": {"do_lower_case": True},
    # "funnel-large": {"do_lower_case": True},
    # "funnel-large-base": {"do_lower_case": True},
    # "funnel-xlarge": {"do_lower_case": True},
    # "funnel-xlarge-base": {"do_lower_case": True},
}


class FunnelTokenizer(BertTokenizer):
    r"""
    Tokenizer for the Funnel Transformer models.

    :class:`~transformers.FunnelTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )


class FunnelTokenizerFast(BertTokenizerFast):
    r"""
    "Fast" tokenizer for the Funnel Transformer models (backed by HuggingFace's :obj:`tokenizers` library).

    :class:`~transformers.FunnelTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs
    end-to-end tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_file,
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
