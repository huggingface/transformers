# coding=utf-8
# Copyright 2025 the Facebook Research and HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for BLT."""

from typing import TYPE_CHECKING, Optional

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

# BLT tokenizer constants
SEP = " "
BOS_ID: int = 1
EOS_ID: int = 2
PAD_ID: int = 260  # Use valid ID after byte tokens (4-259)
BOE_ID: int = 0
BPE_ID: int = 3
OFFSET: int = 4
BYTE_UNITS: int = 256

VOCAB_FILES_NAMES = {}  # BLT doesn't require external vocab files


class BLTTokenizer(PreTrainedTokenizer):
    """
    Construct a BLT tokenizer. Based on byte-level tokenization where each byte is treated as a token.

    This tokenizer converts text to UTF-8 bytes and then maps each byte to a token ID with an offset.
    It supports special tokens for beginning of sequence (BOS), end of sequence (EOS),
    beginning of example (BOE), and padding (PAD).

    Args:
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<pad>"`):
            The padding token.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. Not used in BLT but kept for compatibility.
        boe_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<boe>"`):
            The beginning of example token, specific to BLT.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add a `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        boe_token="<boe>",
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
        **kwargs,
    ):
        # Store BLT-specific parameters first
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.vocab_size_unit_1 = BYTE_UNITS
        self.offsetting_special_char = OFFSET

        # BLT token IDs (exactly like original)
        self.boe_id = BOE_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.bpe_id = BPE_ID
        self.n_words = self.vocab_size_unit_1 + self.offsetting_special_char
        self.boe_token = boe_token

        # Build encoder (token -> id) and decoder (id -> token) mappings
        self.encoder = {}

        # Add special tokens to encoder
        self.encoder[str(bos_token)] = self.bos_id
        self.encoder[str(eos_token)] = self.eos_id
        self.encoder[str(pad_token)] = self.pad_id
        self.encoder[str(boe_token)] = self.boe_id

        # Add byte tokens (0-255) to encoder
        for i in range(self.vocab_size_unit_1):
            self.encoder[str(i)] = i + self.offsetting_special_char

        # Create decoder as reverse of encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            boe_token=boe_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns vocab size"""
        # Account for byte tokens (4-259) plus special tokens (0,1,2,3,260)
        return max(self.vocab_size_unit_1 + self.offsetting_special_char, PAD_ID + 1)

    def get_vocab(self):
        """Returns vocab as a dict"""
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        return self.encoder.get(token, self.added_tokens_encoder.get(token, self.unk_token_id))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        # Check added tokens first (they might override special token IDs)
        for token, token_id in self.added_tokens_encoder.items():
            if token_id == index:
                return token

        return self.decoder.get(index, str(self.unk_token))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens to a single string."""
        byte_values = []

        for token in tokens:
            # Skip special tokens by checking if they're in encoder but not byte tokens
            if token in self.encoder and token in {
                str(self.bos_token),
                str(self.eos_token),
                str(self.pad_token),
                str(self.boe_token),
            }:
                continue

            try:
                byte_val = int(token)
                if 0 <= byte_val <= 255:
                    byte_values.append(byte_val)
            except ValueError:
                continue

        return bytes(byte_values).decode("utf-8", errors="ignore")

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Converts a string to a list of tokens. For BLT, we work directly with byte values."""
        return [str(byte_val) for byte_val in text.encode("utf-8", errors="ignore")]

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A BLT sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        bos = [self.bos_id] if self.add_bos_token else []
        eos = [self.eos_id] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id + ([0] * len(token_ids_1)) + eos_token_id

    def get_vocab_size(self) -> int:
        """Get vocab size like the original tokenizer."""
        return self.vocab_size

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        # BLT doesn't require external vocabulary files since it uses byte-level tokenization
        return ()


__all__ = ["BLTTokenizer"]
