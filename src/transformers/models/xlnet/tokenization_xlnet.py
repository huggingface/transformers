# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
"""Tokenization classes for XLNet model."""

from typing import Optional, Union

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_base import _get_prepend_scheme
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

SPIECE_UNDERLINE = "▁"

# Segments (not really needed)
SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class XLNetTokenizer(TokenizersBackend):
    """
    Construct a XLNet tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab (`list of tuples`, *optional*):
            List of (token, score) tuples for Unigram model. If not provided, an empty list is used.
        unk_id (`int`, *optional*, defaults to 0):
            The ID of the unknown token in the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether to lowercase the input when tokenizing.
        remove_space (`bool`, *optional*, defaults to `True`):
            Whether to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether to keep accents when tokenizing.
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

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`list[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
            Additional special tokens used by the tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    padding_side = "left"
    model = Unigram

    def __init__(
        self,
        vocab: Optional[Union[str, list[tuple[str, float]]]] = None,
        unk_id: int = 0,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        additional_special_tokens=None,
        **kwargs,
    ):
        if additional_special_tokens is None:
            additional_special_tokens = ["<eop>", "<eod>"]

        if vocab is not None:
            self._vocab = vocab
        else:
            self._vocab = [(str(unk_token), 0.0)]

        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab,
                unk_id=unk_id,
                byte_fallback=False,
            )
        )

        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        #  if not keep_accents:
        list_normalizers.append(normalizers.NFKD())
        list_normalizers.append(normalizers.StripAccents())
        if do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        self._tokenizer.normalizer = normalizers.Sequence(list_normalizers)

        add_prefix_space = True
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme),
            ]
        )

        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        self._pad_token_type_id = 3
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        super().__init__(
            unk_id=unk_id,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A:0 {str(self.sep_token)}:0 {str(self.cls_token)}:2",
            pair=f"$A:0 {str(self.sep_token)}:0 $B:1 {str(self.sep_token)}:1 {str(self.cls_token)}:2",
            special_tokens=[
                (str(self.sep_token), self.sep_token_id),
                (str(self.cls_token), self.cls_token_id),
            ],
        )


__all__ = ["XLNetTokenizer"]
