# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for HindiCausalLM."""

import os
from typing import Dict, List, Optional, Tuple, Union

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging, is_sentencepiece_available


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

# Check if sentencepiece is available
if is_sentencepiece_available():
    import sentencepiece as spm
else:
    spm = None


class HindiCausalLMTokenizer(PreTrainedTokenizer):
    """
    Construct a HindiCausalLM tokenizer, using SentencePiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods.

    Args:
        vocab_file (`str`):
            Path to the SentencePiece model file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        sp_model_kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments to use when instantiating the SentencePiece tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sp_model_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        if not is_sentencepiece_available():
            raise ImportError(
                "You need to install SentencePiece to use HindiCausalLMTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.vocab_file = vocab_file

        # Load SentencePiece model
        if not os.path.exists(vocab_file):
            raise ValueError(f"SentencePiece model file {vocab_file} not found")

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # Set token IDs
        self.bos_token_id = 1  # As per provided config
        self.eos_token_id = 2  # As per provided config
        self.pad_token_id = 0  # As per provided config
        self.unk_token_id = 3  # As per provided config

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    def get_vocab(self):
        """Returns the vocabulary as a dict of {token: index}"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize a string."""
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """Converts a token to an id using SentencePiece."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index to a token using SentencePiece."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens to a single string."""
        return self.sp_model.DecodePieces(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence by appending eos_token_id.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of IDs for the second sequence.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        bos_token = [self.bos_token_id] if self.bos_token_id is not None else []
        eos_token = [self.eos_token_id] if self.eos_token_id is not None else []

        if token_ids_1 is None:
            return bos_token + token_ids_0 + eos_token

        # For pair tasks
        return bos_token + token_ids_0 + eos_token + token_ids_1 + eos_token

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of IDs for the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens.

        Returns:
            `List[int]`: A list of integers indicating if a token is a special token or not.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
        return [1] + [0] * len(token_ids_0) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs for sequence pairs.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of IDs for the second sequence.

        Returns:
            `List[int]`: List of token type IDs.
        """
        bos_token = [self.bos_token_id] if self.bos_token_id is not None else []
        eos_token = [self.eos_token_id] if self.eos_token_id is not None else []

        if token_ids_1 is None:
            return [0] * (len(bos_token + token_ids_0 + eos_token))

        return [0] * (len(bos_token + token_ids_0 + eos_token)) + [1] * (len(token_ids_1 + eos_token))

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary files to a directory.

        Args:
            save_directory (`str`):
                Directory to save the vocabulary to.
            filename_prefix (`str`, *optional*):
                Optional prefix for the vocabulary files.

        Returns:
            `Tuple[str]`: Paths to the saved files.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        if filename_prefix is not None:
            output_file = os.path.join(save_directory, filename_prefix + "-" + VOCAB_FILES_NAMES["vocab_file"])
        else:
            output_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        if os.path.abspath(self.vocab_file) != os.path.abspath(output_file):
            import shutil
            shutil.copyfile(self.vocab_file, output_file)

        return (output_file,)
