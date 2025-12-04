# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""Tokenization classes for RemBert model."""

from typing import Optional, Union

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.model", "tokenizer_file": "tokenizer.json"}


class RemBertTokenizer(TokenizersBackend):
    """
    Construct a "fast" RemBert tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
    tokenizer inherits from [`AlbertTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        remove_space (`bool`, *optional*, defaults to `True`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether or not to keep accents when tokenizing.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the `sep_token`.
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

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: Optional[Union[str, list[tuple[str, float]]]] = None,
        do_lower_case: bool = False,
        keep_accents: bool = False,
        bos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
        unk_token: str = "<unk>",
        sep_token: str = "[SEP]",
        pad_token: str = "<pad>",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        add_prefix_space: bool = True,
        remove_space: bool = True,
        **kwargs,
    ):
        self.remove_space = remove_space
        self.do_lower_case = do_lower_case
        self.keep_accents = keep_accents

        if vocab is not None:
            self._vocab_scores = vocab
        else:
            self._vocab_scores = [
                (str(pad_token), 0.0),
                (str(unk_token), 0.0),
                (str(cls_token), 0.0),
                (str(sep_token), 0.0),
                (str(mask_token), 0.0),
            ]

        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab_scores,
                unk_id=2,
                byte_fallback=False,
            )
        )

        # Build normalizer matching RemBertConverter behavior
        # When loading from pretrained, this will be overridden by tokenizer.json config
        # When creating from extractor (vocab), this provides equivalent behavior
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
        if not self.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        # Add Precompiled equivalent (newline conversion + NFKC normalization)
        list_normalizers.extend(
            [
                normalizers.Replace(Regex(r"[\n\r\t]"), " "),  # Precompiled converts newlines/tabs to spaces
                normalizers.NFKC(),  # Precompiled does NFKC normalization
            ]
        )

        self._tokenizer.normalizer = normalizers.Sequence(list_normalizers)

        prepend_scheme = "always" if add_prefix_space else "never"
        # Remove WhitespaceSplit - should only have Metaspace (matches SpmConverter)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)

        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        super().__init__(
            add_prefix_space=add_prefix_space,
            do_lower_case=do_lower_case,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            remove_space=remove_space,
            **kwargs,
        )

        # Set post_processor after super().__init__() so we have token IDs available
        # This matches RemBertConverter.post_processor()
        cls_token_str = str(cls_token)
        sep_token_str = str(sep_token)
        cls_token_id = self.convert_tokens_to_ids(cls_token_str)
        sep_token_id = self.convert_tokens_to_ids(sep_token_str)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token_str}:0 $A:0 {sep_token_str}:0",
            pair=f"{cls_token_str}:0 $A:0 {sep_token_str}:0 $B:1 {sep_token_str}:1",
            special_tokens=[
                (cls_token_str, cls_token_id),
                (sep_token_str, sep_token_id),
            ],
        )

        super()._post_init()


__all__ = ["RemBertTokenizer"]
