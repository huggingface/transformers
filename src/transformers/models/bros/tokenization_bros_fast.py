# coding=utf-8
# Copyright 2022-present NAVER Corp, The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
"""Tokenization classes for Bros."""

from ...utils import logging
from ..bert.tokenization_bert_fast import BertTokenizerFast

from .tokenization_bros import BrosTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "naver-clova-ocr/bros-base-uncased": "https://huggingface.co/naver-clova-ocr/bros-base-uncased/resolve/main/vocab.txt",
        "naver-clova-ocr/bros-large-uncased": "https://huggingface.co/naver-clova-ocr/bros-large-uncased/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "naver-clova-ocr/bros-base-uncased": "https://huggingface.co/naver-clova-ocr/bros-base-uncased/resolve/main/tokenizer.json",
        "naver-clova-ocr/bros-large-uncased": "https://huggingface.co/naver-clova-ocr/bros-large-uncased/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "naver-clova-ocr/bros-base-uncased": 512,
    "naver-clova-ocr/bros-large-uncased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "naver-clova-ocr/bros-base-uncased": {"do_lower_case": True},
    "naver-clova-ocr/bros-large-uncased": {"do_lower_case": True},
}


class BrosTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" BERT tokenizer (backed by HuggingFace's `tokenizers` library). Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
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
        clean_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see `this
            issue <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
        wordpieces_prefix: (:obj:`str`, `optional`, defaults to :obj:`"##"`):
            The prefix for subwords.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = BrosTokenizer
