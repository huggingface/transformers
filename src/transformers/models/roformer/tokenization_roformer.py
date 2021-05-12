# coding=utf-8
# Copyright 2021 junnyu and The HuggingFace Inc. team. All rights reserved.
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

from ...utils import logging
from ..bert.tokenization_bert import BasicTokenizer, BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"junnyu/roformer_chinese_small": 512, "junnyu/roformer_chinese_base": 512}


PRETRAINED_INIT_CONFIGURATION = {
    "junnyu/roformer_chinese_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_base": {"do_lower_case": True},
}


class CustomBasicTokenizer(BasicTokenizer):
    def __init__(self, vocab, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        super().__init__(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )

        self.vocab = vocab

    def _tokenize_chinese_chars(self, text):
        try:
            import jieba
        except ImportError:
            raise ImportError("Please install jieba for Chinese pretokenize.")
        output = []
        for wholeword in jieba.cut(text, HMM=False):
            if wholeword in self.vocab:
                output.append(" ")
                output.append(wholeword)
                output.append(" ")
            else:
                for char in wholeword:
                    cp = ord(char)
                    if self._is_chinese_char(cp):
                        output.append(" ")
                        output.append(char)
                        output.append(" ")
                    else:
                        output.append(char)
        return "".join(output)


class RoFormerTokenizer(BertTokenizer):
    r"""
    Construct a RoFormer tokenizer.

    when use_jieba=False :class:`~transformers.RoFormerTokenizer` is identical to :class:`~transformers.BertTokenizer`
    and runs end-to-end tokenization: punctuation splitting and wordpiece. Refer to superclass
    :class:`~transformers.BertTokenizer` for usage examples and documentation concerning parameters.

    when use_jieba=True :class:`~transformers.RoFormerTokenizer` use jieba tokenizer to pretokenize the sentence. e.g.
    今天天气非常好。-> jieba pretokenize ['今天天气', '非常', '好', '。'] '今天天气' not in tokenizer.vocab but '非常' in tokenizer.vocab

    Example::

        >>> from transformers import RoFormerTokenizer
        >>> tokenizer = RoFormerTokenizer.from_pretrained('junnyu/roformer_chinese_base')
        >>> tokenizer.tokenize("今天天气非常好。")
        # ['今', '天', '天', '气', '非常', '好', '。']

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
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        use_jieba=True,
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
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        if use_jieba and self.do_basic_tokenize:
            self.basic_tokenizer = CustomBasicTokenizer(
                vocab=self.vocab,
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
