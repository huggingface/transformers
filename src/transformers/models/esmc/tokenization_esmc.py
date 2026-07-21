# Copyright 2026 BioHub and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ESMC."""

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

# Canonical amino acid vocabulary used by all ESMC checkpoints.
# Indices must be kept stable — they are hard-coded into model weights.
SEQUENCE_VOCAB = [
    "<cls>",  # 0
    "<pad>",  # 1
    "<eos>",  # 2
    "<unk>",  # 3
    "L",  # 4
    "A",  # 5
    "G",  # 6
    "V",  # 7
    "S",  # 8
    "E",  # 9
    "R",  # 10
    "T",  # 11
    "I",  # 12
    "D",  # 13
    "P",  # 14
    "K",  # 15
    "Q",  # 16
    "N",  # 17
    "F",  # 18
    "Y",  # 19
    "M",  # 20
    "H",  # 21
    "W",  # 22
    "C",  # 23
    "X",  # 24  ambiguous amino acid
    "B",  # 25  Asp/Asn ambiguity
    "U",  # 26  selenocysteine
    "Z",  # 27  Glu/Gln ambiguity
    "O",  # 28  pyrrolysine
    ".",  # 29  gap
    "-",  # 30  insertion
    "|",  # 31  chain-break
    "<mask>",  # 32
]


class EsmcTokenizer(TokenizersBackend):
    r"""
    Construct an ESMC tokenizer.

    This tokenizer is a character-level tokenizer backed by the HuggingFace `tokenizers` library.
    It wraps every sequence with ``<cls>`` and ``<eos>`` tokens and supports a ``|`` chain-break
    token for multi-chain inputs.

    Args:
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classification token (prepended to every sequence).
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The padding token.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The mask token, used for masked language modelling.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            The end-of-sequence token (appended to every sequence).
        bos_token (`str`, *optional*, defaults to `"<cls>"`):
            The beginning-of-sequence token (prepended to every sequence). When unset, uses cls_token.
        chain_break_token (`str`, *optional*, defaults to `"|"`):
            Token inserted between chains in multi-chain protein inputs.
        extra_special_tokens (`list[str]`, *optional*):
            Additional special tokens to register with the tokenizer (round-tripped from a saved config).

    Examples:

    ```python
    >>> from transformers import EsmcTokenizer

    >>> tokenizer = EsmcTokenizer()
    >>> tokenizer("ACDEFGHIKLMNPQRSTVWY")["input_ids"]
    [0, 5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19, 2]
    ```
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE
    # ESMC adds one model-specific named special token (the chain-break token) on top of the
    # standard set, which gives us the ``chain_break_token`` / ``chain_break_token_id`` attributes.
    SPECIAL_TOKENS_ATTRIBUTES = TokenizersBackend.SPECIAL_TOKENS_ATTRIBUTES + ["chain_break_token"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        bos_token=None,
        chain_break_token="|",
        extra_special_tokens=None,
        **kwargs,
    ):
        token_to_id = {tok: ind for ind, tok in enumerate(SEQUENCE_VOCAB)}

        # ESMC uses <cls> as the sequence-start token, so `bos_token` defaults to it.
        if bos_token is None:
            bos_token = cls_token
        cls_str, eos_str = self._ensure_str(cls_token), self._ensure_str(eos_token)

        self._tokenizer = Tokenizer(BPE(token_to_id, merges=[], unk_token=self._ensure_str(unk_token)))
        # Wrap every encoded sequence with <cls> … <eos>. The special tokens themselves (including the
        # chain-break token, passed via ``extra_special_tokens`` below) are registered into the backend
        # by TokenizersBackend from the special-token kwargs — no need to add them here.
        self._tokenizer.post_processor = TemplateProcessing(
            single=f"{cls_str} $A {eos_str}",
            special_tokens=[(cls_str, token_to_id[cls_str]), (eos_str, token_to_id[eos_str])],
        )

        # ``chain_break_token`` is a named special token (see ``SPECIAL_TOKENS_ATTRIBUTES``), so
        # TokenizersBackend registers it into the backend and exposes it as an attribute; a generic
        # ``extra_special_tokens`` list (round-tripped from a saved config) is passed through untouched.
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            chain_break_token=chain_break_token,
            extra_special_tokens=extra_special_tokens,
            **kwargs,
        )

    # ``chain_break_token`` / ``chain_break_token_id`` are provided automatically by the
    # ``extra_special_tokens`` named-token mechanism (SpecialTokensMixin).

    @staticmethod
    def _ensure_str(token) -> str:
        if isinstance(token, AddedToken):
            return token.content
        return str(token)


__all__ = ["EsmcTokenizer"]
