# coding=utf-8
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
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
"""Tokenization class for model Reformer."""

from typing import Optional, Union

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)


SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class ReformerTokenizer(TokenizersBackend):
    """
    Construct a Reformer tokenizer (backed by HuggingFace's tokenizers library). Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=bpe#models).

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`list[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        vocab (`str` or `dict[str, int]`, *optional*):
            Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file`.
        merges (`str` or `list[str]`, *optional*):
            Custom merges list. If not provided, merges are loaded from `vocab_file`.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: Optional[Union[str, dict[str, int]]] = None,
        merges: Optional[Union[str, list[str]]] = None,
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        additional_special_tokens: Optional[list] = None,
        **kwargs,
    ):
        self._vocab = vocab or {}
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                unk_token=str(unk_token),
                fuse_unk=True,
                byte_fallback=False,
                dropout=None,
            )
        )

        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace(Regex(r"\s{2,}|[\n\r\t]"), " "),
                normalizers.NFC(),
                normalizers.Strip(left=False, right=True),
            ]
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens or [],
            **kwargs,
        )

        super()._post_init()


__all__ = ["ReformerTokenizer"]
