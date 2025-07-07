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

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput

logger = logging.get_logger(__name__)

# BLT tokenizer constants
SEP = " "
BOS_ID: int = 1
EOS_ID: int = 2
PAD_ID: int = -1
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
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.vocab_size_unit_1 = BYTE_UNITS
        self.offsetting_special_char = OFFSET
        
        self.boe_id = BOE_ID
        self.bos_id = BOS_ID  
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.bpe_id = BPE_ID
        self.n_words = self.vocab_size_unit_1 + self.offsetting_special_char

        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token
        self.boe_token = AddedToken(boe_token, normalized=False, special=True) if isinstance(boe_token, str) else boe_token

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.vocab_size_unit_1 + self.offsetting_special_char

    def get_vocab(self):
        """Returns vocab as a dict"""
        # Create a mapping for byte values + offset
        vocab = {}
        
        # Add special tokens (with defensive checks)
        if hasattr(self, 'bos_token'):
            vocab[str(self.bos_token)] = self.bos_id
        if hasattr(self, 'eos_token'):
            vocab[str(self.eos_token)] = self.eos_id  
        if hasattr(self, 'pad_token'):
            vocab[str(self.pad_token)] = self.pad_id
        if hasattr(self, 'boe_token'):
            vocab[str(self.boe_token)] = self.boe_id
        
        # Add byte tokens as string representations of byte values
        vocab_size_unit_1 = getattr(self, 'vocab_size_unit_1', BYTE_UNITS)
        offsetting_special_char = getattr(self, 'offsetting_special_char', OFFSET)
        for i in range(vocab_size_unit_1):
            vocab[str(i)] = i + offsetting_special_char
            
        # Add any additional tokens if available
        if hasattr(self, 'added_tokens_encoder'):
            vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. For BLT, we work directly with byte values.
        Returns a list of strings that represent the byte values.
        """
        # Convert text to UTF-8 bytes, just like the original
        try:
            bytes_data = text.encode("utf-8", errors="ignore")
        except UnicodeEncodeError:
            bytes_data = text.encode("utf-8", errors="ignore")
        
        # Return string representations of byte values for the tokenizer framework
        return [str(byte_val) for byte_val in bytes_data]

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        # Handle special tokens
        if token == str(self.bos_token):
            return self.bos_id
        elif token == str(self.eos_token):
            return self.eos_id
        elif token == str(self.pad_token):
            return self.pad_id
        elif token == str(self.boe_token):
            return self.boe_id
        else:
            try:
                # Convert byte value string to int and add offset 
                byte_val = int(token)
                if 0 <= byte_val <= 255:
                    return byte_val + self.offsetting_special_char
            except ValueError:
                pass
            
            return self.added_tokens_encoder.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        # Handle special tokens
        if index == self.bos_id:
            return str(self.bos_token)
        elif index == self.eos_id:
            return str(self.eos_token)
        elif index == self.pad_id:
            return str(self.pad_token)
        elif index == self.boe_id:
            return str(self.boe_token)
        elif index >= self.offsetting_special_char and index < self.vocab_size:
            # Convert back to byte value 
            byte_val = index - self.offsetting_special_char
            return str(byte_val)
        else:
            # Check added tokens
            for token, token_id in self.added_tokens_encoder.items():
                if token_id == index:
                    return token
            return str(self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens to a single string."""
        byte_values = []
        
        for token in tokens:
            # Skip special tokens
            if token in [str(self.bos_token), str(self.eos_token), str(self.pad_token), str(self.boe_token)]:
                continue
            
            try:
                # Convert token back to byte value 
                byte_val = int(token)
                if 0 <= byte_val <= 255:
                    byte_values.append(byte_val)
            except ValueError:
                continue
                    
        # Convert byte values back to string 
        try:
            return bytes(byte_values).decode("utf-8", errors="ignore")
        except (UnicodeDecodeError, ValueError):
            return ""

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs):
        add_bos = kwargs.get('add_bos', self.add_bos_token if add_special_tokens else False)
        add_eos = kwargs.get('add_eos', self.add_eos_token if add_special_tokens else False)

        # Since bpe_delim=False, we use the simple byte encoding
        tokens = bytes(text, encoding="utf-8", errors="ignore")

        # Offsetting 
        tokens = [int(unit) + self.offsetting_special_char for unit in tokens]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens, cut_at_eos: bool = False):
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        return bytes(
            [tok - self.offsetting_special_char for tok in tokens if tok - self.offsetting_special_char >= 0]
        ).decode("utf-8", errors="ignore")
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
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

    def get_vocab_size(self) -> int:
        """Get vocab size like the original tokenizer."""
        return self.vocab_size_unit_1 + self.offsetting_special_char

__all__ = ["BLTTokenizer"]