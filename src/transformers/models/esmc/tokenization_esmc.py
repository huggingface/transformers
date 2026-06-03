# Copyright 2026 Biohub. All rights reserved.
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

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # type: ignore[import]
from ...utils import logging  # type: ignore[import]


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


class ESMCTokenizer(PreTrainedTokenizerFast):
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
        chain_break_token (`str`, *optional*, defaults to `"|"`):
            Token inserted between chains in multi-chain protein inputs.

    Examples:

    ```python
    >>> from transformers import ESMCTokenizer

    >>> tokenizer = ESMCTokenizer()
    >>> tokenizer("ACDEFGHIKLMNPQRSTVWY")["input_ids"]
    [0, 5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19, 2]
    ```
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
        **kwargs,
    ):
        all_tokens = SEQUENCE_VOCAB
        token_to_id = {tok: ind for ind, tok in enumerate(all_tokens)}

        # Normalise: always work with plain strings
        unk_token = self._ensure_str(unk_token)
        cls_token = self._ensure_str(cls_token)
        pad_token = self._ensure_str(pad_token)
        mask_token = self._ensure_str(mask_token)
        eos_token = self._ensure_str(eos_token)
        chain_break_token = self._ensure_str(chain_break_token)

        # A character-level tokenizer is equivalent to BPE with no merges.
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)

        special_tokens = [
            cls_token,
            pad_token,
            mask_token,
            eos_token,
            chain_break_token,
        ]
        tokenizer.add_special_tokens(special_tokens)

        # Automatically wrap every encoded sequence with <cls> … <eos>.
        tokenizer.post_processor = TemplateProcessing(
            single=f"{cls_token} $A {eos_token}",
            special_tokens=[
                (cls_token, tokenizer.token_to_id(cls_token)),
                (eos_token, tokenizer.token_to_id(eos_token)),
            ],
        )

        # Expose chain-break token as an additional special token so it is
        # preserved during encode/decode and can be looked up easily.
        kwargs.setdefault("additional_special_tokens", [])
        if chain_break_token not in kwargs["additional_special_tokens"]:
            kwargs["additional_special_tokens"] = list(
                kwargs["additional_special_tokens"]
            ) + [chain_break_token]

        # Keep reference before super().__init__ so properties below work.
        self._chain_break_token = chain_break_token

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # bos / cls aliases
    # The model uses <cls> as the sequence-start token; the HF base class
    # expects `bos_token`.  We alias them to avoid confusion.
    # ------------------------------------------------------------------

    @property
    def bos_token(self):
        return self.cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id

    # ------------------------------------------------------------------
    # Chain-break token
    # ------------------------------------------------------------------

    @property
    def chain_break_token(self) -> str:
        return self._chain_break_token

    @property
    def chain_break_token_id(self) -> int:
        token_id = self.convert_tokens_to_ids(self._chain_break_token)
        assert isinstance(token_id, int)
        return token_id

    # ------------------------------------------------------------------
    # Convenience helpers used by downstream code
    # ------------------------------------------------------------------

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_str(token) -> str:
        if isinstance(token, AddedToken):
            return token.content
        return str(token)


__all__ = ["ESMCTokenizer"]
