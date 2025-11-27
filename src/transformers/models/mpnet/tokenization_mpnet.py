# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Tokenization classes for MPNet."""

from typing import Optional

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from ...tokenization_python import AddedToken
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class MPNetTokenizer(TokenizersBackend):
    r"""
    Construct a MPNet tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab (`dict`, *optional*):
            Dictionary mapping tokens to their IDs. If not provided, an empty vocab is initialized.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: Optional[dict] = None,
        do_lower_case=True,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="[UNK]",
        pad_token="<pad>",
        mask_token="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # Initialize vocab
        if vocab is not None:
            self._vocab = (
                {token: idx for idx, (token, _score) in enumerate(vocab)} if isinstance(vocab, list) else vocab
            )
        else:
            self._vocab = {}

        # Initialize the tokenizer with WordPiece model
        self._tokenizer = Tokenizer(WordPiece(self._vocab, unk_token=str(unk_token)))

        # Set normalizer based on MPNetConverter logic
        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )

        # Set pre-tokenizer
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # Set decoder
        self._tokenizer.decoder = decoders.WordPiece(prefix="##")

        # Store do_lower_case for later use
        self.do_lower_case = do_lower_case

        # Handle special token initialization
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # Store for later use
        tokenizer_object = self._tokenizer

        super().__init__(
            tokenizer_object=tokenizer_object,
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # Set post_processor after super().__init__ to ensure we have token IDs
        cls_str = str(self.cls_token)
        sep_str = str(self.sep_token)
        cls_token_id = self.cls_token_id if self.cls_token_id is not None else 0
        sep_token_id = self.sep_token_id if self.sep_token_id is not None else 2

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_str}:0 $A:0 {sep_str}:0",
            pair=f"{cls_str}:0 $A:0 {sep_str}:0 {sep_str}:0 $B:1 {sep_str}:1",  # MPNet uses two [SEP] tokens
            special_tokens=[
                (cls_str, cls_token_id),
                (sep_str, sep_token_id),
            ],
        )

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        MPNet tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *<mask>*.
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on MPNet.
        """
        # Mask token behave like a normal word, i.e. include the space before it
        # So we set lstrip to True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value


__all__ = ["MPNetTokenizer"]
