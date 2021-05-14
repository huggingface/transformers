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
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"junnyu/roformer_chinese_small": 1536, "junnyu/roformer_chinese_base": 1536}


PRETRAINED_INIT_CONFIGURATION = {
    "junnyu/roformer_chinese_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_base": {"do_lower_case": True},
}


class RoFormerTokenizer(BertTokenizer):
    r"""
    Construct a RoFormer tokenizer. Based on `Jieba <https://pypi.org/project/jieba/>` and BertTokenizer.

    This tokenizer inherits from :class:`~transformers.BertTokenizer` which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import jieba
        except ImportError:
            raise ImportError(
                "You need to install jieba to use RoFormerTokenizer."
                "See https://pypi.org/project/jieba/ for installation."
            )
        self.jieba = jieba

    def _tokenize(self, text, *args, **kwargs):
        # use jieba to pretokenize chinese char
        split_tokens = []
        for wholword in self.jieba.cut(text, HMM=False):
            if wholword in self.vocab:
                split_tokens.append(wholword)
            else:
                # use bert tokenizer to _tokenize
                char_list = super()._tokenize(wholword, *args, **kwargs)
                split_tokens.extend(char_list)
        return split_tokens
