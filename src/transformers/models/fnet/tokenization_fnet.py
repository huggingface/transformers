# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Tokenization classes for FNet model."""

from ...utils import logging
from ..albert.tokenization_albert import AlbertTokenizer


logger = logging.get_logger(__name__)


class FNetTokenizer(AlbertTokenizer):
    """
    Construct an FNet tokenizer. Based on [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`AlbertTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether or not to keep accents when tokenizing.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    model_input_names = ["input_ids", "token_type_ids"]


# FNetTokenizerFast is an alias for FNetTokenizer (since AlbertTokenizer is already a fast tokenizer)
FNetTokenizerFast = FNetTokenizer

__all__ = ["FNetTokenizer", "FNetTokenizerFast"]
