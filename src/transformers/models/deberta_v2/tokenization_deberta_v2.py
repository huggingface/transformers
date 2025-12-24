# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
"""Tokenization class for model DeBERTa-v2."""

from typing import Optional, Union

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm.model", "tokenizer_file": "tokenizer.json"}


class DebertaV2Tokenizer(TokenizersBackend):
    """
    Construct a DeBERTa-v2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on Unigram tokenization.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file (SentencePiece model file). Not used directly but kept for compatibility.
        vocab (`str`, `dict` or `list`, *optional*):
            List of tuples (piece, score) for the vocabulary.
        precompiled_charsmap (`bytes`, *optional*):
            Precompiled character map for normalization.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        split_by_punct (`bool`, *optional*, defaults to `False`):
            Whether to split by punctuation.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        unk_id (`int`, *optional*, defaults to index of `unk_token` in vocab):
            The ID of the unknown token in the vocabulary.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    model = Unigram

    def __init__(
        self,
        vocab: Optional[Union[str, dict, list]] = None,
        do_lower_case=False,
        split_by_punct=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        add_prefix_space=True,
        unk_id=1,
        **kwargs,
    ):
        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self.add_prefix_space = add_prefix_space

        if vocab is None:
            vocab = [
                (str(pad_token), 0.0),
                (str(unk_token), 0.0),
                (str(bos_token), 0.0),
                (str(eos_token), 0.0),
                (str(sep_token), 0.0),
                (str(cls_token), 0.0),
                (str(mask_token), 0.0),
            ]
            unk_id = 1
        elif isinstance(vocab, list):
            unk_id = vocab.index((str(unk_token), 0.0)) if (str(unk_token), 0.0) in vocab else unk_id

        self._vocab = vocab
        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab,
                unk_id=unk_id,
                byte_fallback=False,
            )
        )

        list_normalizers = []
        if do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        list_normalizers.extend(
            [
                normalizers.Replace(Regex(r"\s{2,}|[\n\r\t]"), " "),
                normalizers.NFC(),
                normalizers.Strip(left=False, right=True),
            ]
        )
        self._tokenizer.normalizer = normalizers.Sequence(list_normalizers)

        list_pretokenizers = []
        if split_by_punct:
            list_pretokenizers.append(pre_tokenizers.Punctuation(behavior="isolated"))

        prepend_scheme = "always" if add_prefix_space else "first"
        list_pretokenizers.append(pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme))

        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(list_pretokenizers)
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_id=unk_id,
            do_lower_case=do_lower_case,
            split_by_punct=split_by_punct,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )


__all__ = ["DebertaV2Tokenizer"]
