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
Tokenization classes for Parakeet CTC.
"""

import json
import os
import re
from typing import Optional

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class ParakeetCTCTokenizer(PreTrainedTokenizer):
    """
    Parakeet CTC tokenizer for CTC-based speech recognition models.

    This tokenizer is designed to work with ParakeetCTC models that internally handle CTC decoding.
    The tokenizer expects token sequences that have already been CTC-decoded (with blanks removed
    and consecutive identical tokens collapsed) from the model's generate() method.

    The tokenizer handles:
    - Converting token IDs to text with SentencePiece-style processing
    - Proper handling of subword tokens and spacing can be done with `group_tokens=True` (see `ParakeetProcessor`).

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

    Example:
        ```python
        >>> from transformers import ParakeetForCTC, ParakeetCTCTokenizer, ParakeetFeatureExtractor
        >>>
        >>> # Load model, tokenizer, and feature extractor
        >>> repo_id = "bezzam/parakeet-ctc-1.1b-hf"
        >>> model = ParakeetForCTC.from_pretrained(repo_id)
        >>> tokenizer = ParakeetCTCTokenizer.from_pretrained(repo_id)
        >>> feature_extractor = ParakeetFeatureExtractor.from_pretrained(repo_id)
        >>>
        >>> # Process audio and generate token sequences (with CTC greedy decoding)
        >>> inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate)
        >>> token_sequences = model.generate(**inputs)
        >>>
        >>> # Decode to text
        >>> transcription = tokenizer.decode_ctc(token_sequences[0])
        ```
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: Optional[str] = "<unk>",
        blank_token_id=None,
        do_lower_case=False,
        **kwargs,
    ):
        # Load vocabulary BEFORE calling super().__init__()
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Create reverse mapping
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        # Set blank token ID
        if blank_token_id is None:
            self.blank_token_id = len(self.vocab)
        else:
            self.blank_token_id = blank_token_id
            # check no other token has this ID in the main vocab
            for k, v in self.vocab.items():
                if v == self.blank_token_id:
                    raise ValueError(
                        f"blank_token_id {self.blank_token_id} conflicts with existing token '{k}' in vocab"
                    )

        self.do_lower_case = do_lower_case

        # Now call super().__init__() with the vocab loaded
        super().__init__(
            unk_token=unk_token,
            blank_token_id=blank_token_id,
            do_lower_case=do_lower_case,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self.vocab_size + len(self.added_tokens_encoder)

    def get_vocab(self) -> dict[str, int]:
        """Returns the vocabulary as a dictionary."""
        vocab = self.vocab.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize a string into a list of tokens.

        For CTC models, this handles SentencePiece-style tokenization where spaces
        are converted to ▁ prefixes to indicate word boundaries.
        """
        if self.do_lower_case:
            text = text.lower()

        # Handle SentencePiece-style tokenization
        # Convert spaces to word boundary markers
        words = text.split(" ")
        tokens = []

        for i, word in enumerate(words):
            if word:  # Skip empty words from consecutive spaces
                # First word gets ▁ prefix, continuing characters are separate
                if word in self.vocab:
                    # If the whole word is in vocab as a subword
                    if i == 0 or f"▁{word}" in self.vocab:
                        tokens.append(f"▁{word}" if f"▁{word}" in self.vocab else word)
                    else:
                        tokens.append(f"▁{word}")
                else:
                    # Character-level fallback for words not in vocab
                    word_tokens = list(word)
                    if word_tokens:
                        # Add ▁ prefix to first character of the word
                        if i > 0:  # Not the first word
                            word_tokens[0] = f"▁{word_tokens[0]}"
                        tokens.extend(word_tokens)

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        # Check main vocabulary first
        if token in self.vocab:
            return self.vocab[token]

        # Check added tokens (managed by parent class)
        if hasattr(self, "added_tokens_encoder") and token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]

        # Return unknown token ID for unrecognized tokens
        return self.vocab.get(self.unk_token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        index = int(index)

        # Check main vocabulary first
        if index in self.ids_to_tokens:
            return self.ids_to_tokens[index]

        # Check added tokens (managed by parent class)
        if hasattr(self, "added_tokens_decoder") and index in self.added_tokens_decoder:
            return str(self.added_tokens_decoder[index])

        # For blank token, return a special marker that will be filtered later
        if index == self.blank_token_id:
            return "<BLANK>"  # Special marker for blank tokens

        # Return unknown token for unrecognized IDs
        return self.unk_token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """
        Converts a sequence of tokens (string) into a single string.

        Args:
            tokens: List of token strings to convert
        """

        # Filter None, blank tokens, pad_token, and unk_token
        filtered_tokens = list(
            filter(
                lambda token: (
                    token is not None
                    and token != "<BLANK>"  # Filter blank token markers
                    and token != "<blank>"  # Filter blank special tokens
                    and token != self.pad_token
                    and token != self.unk_token
                ),
                tokens,
            )
        )

        # Join tokens and handle SentencePiece-style subwords
        text = "".join(filtered_tokens)

        # Handle SentencePiece-style tokens (starting with ▁)
        text = text.replace("▁", " ")

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """
        Save the vocabulary to a directory.

        Args:
            save_directory: The directory in which to save the vocabulary.
            filename_prefix: An optional filename prefix.

        Returns:
            A tuple of the saved file paths.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0: List of IDs.
            token_ids_1: Optional second list of IDs for sequence pairs.
            already_has_special_tokens: Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # CTC models typically don't use special tokens in the traditional sense
        # The blank token is handled separately
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Return all zeros since CTC doesn't use special tokens like [CLS], [SEP]
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * len(token_ids_0) + [0] * len(token_ids_1)


__all__ = ["ParakeetCTCTokenizer"]
