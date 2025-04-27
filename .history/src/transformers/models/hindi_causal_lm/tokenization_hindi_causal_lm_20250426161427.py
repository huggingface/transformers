# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the MIT License.
#

"""Tokenization classes for HindiCausalLM."""

import os
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "convaiinnovations/hindi-foundational-model-base": "https://huggingface.co/convaiinnovations/hindi-foundational-model-base/resolve/main/tokenizer.model",
    }
}


class HindiCausalLMTokenizer(PreTrainedTokenizer):
    """
    Construct a HindiCausalLM tokenizer based on SentencePiece.

    Args:
        vocab_file (`str`):
            SentencePiece model file.
        pad_token (`str`, *optional*, defaults to `""`):
            A special token used to make arrays of tokens the same size for batching purposes.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        pad_token="",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.sp_model = spm.SentencePieceProcessor(**(sp_model_kwargs or {}))
        self.sp_model.Load(vocab_file)

        # Store special token IDs
        self.vocab_size = self.sp_model.GetPieceSize()
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            sp_model_kwargs=sp_model_kwargs,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return self.sp_model.GetPieceSize()

    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string into subwords using the SentencePiece model."""
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.GetPieceSize():
            token = self.sp_model.IdToPiece(index)
        else:
            token = self.unk_token
        return token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens to a string."""
        return self.sp_model.DecodePieces(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to a directory.

        Args:
            save_directory (`str`):
                The directory to save the vocabulary files to.
            filename_prefix (`str`, *optional*):
                A prefix to add to the vocabulary filenames.

        Returns:
            `Tuple[str]`: The paths to the saved vocabulary files.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(out_vocab_file) != os.path.abspath(self.vocab_file):
            import shutil
            shutil.copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence by appending eos_token_id.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        bos_token_id = [self.bos_token_id] if self.bos_token_id is not None else []
        eos_token_id = [self.eos_token_id] if self.eos_token_id is not None else []

        if token_ids_1 is None:
            return bos_token_id + token_ids_0 + eos_token_id
        else:
            # Two sequences with special tokens
            return bos_token_id + token_ids_0 + eos_token_id + bos_token_id + token_ids_1 + eos_token_id

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added.

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

        bos_token_id = [1] if self.bos_token_id is not None else []
        eos_token_id = [1] if self.eos_token_id is not None else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        else:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id + bos_token_id + ([0] * len(token_ids_1)) + eos_token_id

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences for sequence classification tasks.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        bos_token_id = [self.bos_token_id] if self.bos_token_id is not None else []
        eos_token_id = [self.eos_token_id] if self.eos_token_id is not None else []

        if token_ids_1 is None:
            return [0] * (len(bos_token_id + token_ids_0 + eos_token_id))
        else:
            return [0] * (len(bos_token_id + token_ids_0 + eos_token_id + bos_token_id + token_ids_1 + eos_token_id))

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        return (text, kwargs)
