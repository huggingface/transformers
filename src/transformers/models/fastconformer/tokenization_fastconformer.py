# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""
FastConformer tokenizer for CTC models.
"""

import json
import re
from typing import Optional, Union

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class FastConformerTokenizer(PreTrainedTokenizer):
    """
    FastConformer tokenizer for CTC-based speech recognition models.

    This tokenizer is designed specifically for CTC (Connectionist Temporal Classification) models
    that use character-level or subword-level vocabularies. It handles CTC decoding which includes:
    - Removing blank tokens
    - Collapsing consecutive identical tokens
    - Converting token IDs to text with SentencePiece-style processing

    Args:
        vocab_file (`str`):
            Path to the vocabulary file in JSON format.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        blank_token_id (`int`, *optional*):
            The ID of the blank token used in CTC. If not provided, defaults to vocab_size.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        blank_token_id=None,
        do_lower_case=False,
        **kwargs,
    ):
        # Load vocabulary BEFORE calling super().__init__()
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Create reverse mapping (id -> token)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Set special token IDs
        self.unk_token_id = self.vocab.get(unk_token, 0)
        # CTC blank is typically vocab_size (one more than the highest vocab ID)
        self.blank_token_id = blank_token_id if blank_token_id is not None else len(self.vocab)

        self.do_lower_case = do_lower_case

        super().__init__(
            unk_token=unk_token,
            blank_token_id=blank_token_id,
            do_lower_case=do_lower_case,
            **kwargs,
        )

        logger.info(f"Loaded FastConformer vocabulary with {len(self.vocab)} tokens")
        logger.info(f"UNK token: '{unk_token}' (ID: {self.unk_token_id})")
        logger.info(f"Blank token ID: {self.blank_token_id}")

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        """Returns the vocabulary as a dictionary."""
        return self.vocab.copy()

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize a string into subword tokens.
        Note: This is primarily for compatibility. CTC models typically work directly with IDs.
        """
        if self.do_lower_case:
            text = text.lower()

        # Simple character-level tokenization
        # In practice, this should be replaced with the actual tokenization logic
        # that matches how the model was trained
        tokens = list(text)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        return self.id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Converts a sequence of tokens (string) into a single string.
        Handles SentencePiece-style word delimiters (▁).
        """
        # Join tokens
        text = "".join(tokens)

        # Handle SentencePiece-style word delimiters (▁ -> space)
        text = text.replace("▁", " ")

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def ctc_decode_ids(self, token_ids: list[int]) -> list[int]:
        """
        Apply CTC decoding to token IDs:
        1. Remove blank tokens
        2. Collapse consecutive identical tokens

        Args:
            token_ids: List of token IDs from CTC model output

        Returns:
            List of decoded token IDs with CTC rules applied
        """
        decoded_ids = []
        previous_id = None

        for token_id in token_ids:
            # Skip blank tokens
            if token_id == self.blank_token_id:
                previous_id = token_id
                continue

            # Skip repeated tokens (CTC collapse)
            if token_id != previous_id:
                decoded_ids.append(token_id)

            previous_id = token_id

        return decoded_ids

    def decode_ctc_tokens(self, token_ids: list[int]) -> str:
        """
        Complete CTC decoding: apply CTC collapse then convert to text.

        Args:
            token_ids: List of token IDs from CTC model output

        Returns:
            Decoded text string
        """
        # First apply CTC decoding (blank removal + consecutive collapse)
        ctc_decoded_ids = self.ctc_decode_ids(token_ids)

        # Filter out tokens outside vocabulary size
        filtered_ids = [t for t in ctc_decoded_ids if t < self.vocab_size]

        if not filtered_ids:
            return ""

        # Convert IDs to tokens
        tokens = [self._convert_id_to_token(token_id) for token_id in filtered_ids]

        # Convert tokens to text
        text = self.convert_tokens_to_string(tokens)

        return text

    def decode(
        self,
        token_ids: Union[int, list[int], list[list[int]]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        ctc_decode: bool = True,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids to a string.

        Args:
            token_ids: List of tokenized input ids.
            skip_special_tokens: Whether to remove special tokens in the decoding.
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces.
            ctc_decode: Whether to apply CTC decoding rules.

        Returns:
            The decoded string.
        """
        # Handle different input formats
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            # Batch decoding - take first sequence for now
            token_ids = token_ids[0]

        if ctc_decode:
            return self.decode_ctc_tokens(token_ids)
        else:
            # Standard decoding without CTC rules
            tokens = [self._convert_id_to_token(token_id) for token_id in token_ids if token_id < self.vocab_size]
            return self.convert_tokens_to_string(tokens)

    def batch_decode(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        ctc_decode: bool = True,
        **kwargs,
    ) -> list[str]:
        """
        Convert a list of lists of token ids into a list of strings.

        Args:
            sequences: List of tokenized input ids.
            skip_special_tokens: Whether to remove special tokens in the decoding.
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces.
            ctc_decode: Whether to apply CTC decoding rules.

        Returns:
            List of decoded strings.
        """
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                ctc_decode=ctc_decode,
                **kwargs,
            )
            for seq in sequences
        ]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """
        Save the vocabulary to a directory.

        Args:
            save_directory: Directory to save the vocabulary to.
            filename_prefix: Optional prefix for the vocabulary file.

        Returns:
            Tuple of saved file paths.
        """
        import os

        if filename_prefix is not None:
            vocab_file = os.path.join(save_directory, filename_prefix + "-" + VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def __len__(self) -> int:
        """Returns the vocabulary size."""
        return len(self.vocab)

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added.

        Args:
            token_ids_0: List of ids.
            token_ids_1: Optional second list of IDs for sequence pairs.
            already_has_special_tokens: Whether the token list is already formatted with special tokens.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # No special tokens in CTC models typically
        special_tokens_mask = [0] * len(token_ids_0)
        if token_ids_1 is not None:
            special_tokens_mask += [0] * len(token_ids_1)
        return special_tokens_mask


__all__ = ["FastConformerTokenizer"]
