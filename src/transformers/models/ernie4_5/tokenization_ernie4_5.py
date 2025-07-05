# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All Rights Reserved.
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
"""Tokenization classes for Ernie 4.5"""

import os
from shutil import copyfile
from typing import Optional, Union

import numpy as np
import sentencepiece as spm
import torch

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    PaddingStrategy,
)
from ...utils import logging


logger = logging.get_logger(__name__)


class Ernie4_5Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "tokenizer.model",
    }
    # Model input names expected by the tokenizer
    model_input_names = ["input_ids", "position_ids", "attention_mask", "labels"]
    # Padding side (where to add padding tokens)
    padding_side = "left"

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        cls_token="<cls>",
        eos_token="</s>",
        mask_token="<mask:0>",
        pad_token="<pad>",
        sep_token="<sep>",
        unk_token="<unk>",
        additional_special_tokens=None,
        verbose=False,
        **kwargs,
    ):
        """
        Initialize the ERNIE tokenizer.

        Args:
            vocab_file (str): Path to the SentencePiece model file.
            bos_token (str, optional): Beginning of sentence token. Defaults to "<s>".
            cls_token (str, optional): Classification token. Defaults to "<cls>".
            eos_token (str, optional): End of sentence token. Defaults to "</s>".
            mask_token (str, optional): Mask token. Defaults to "<mask:0>".
            pad_token (str, optional): Padding token. Defaults to "<pad>".
            sep_token (str, optional): Separator token. Defaults to "<sep>".
            unk_token (str, optional): Unknown token. Defaults to "<unk>".
            additional_special_tokens (List[str], optional): Additional special tokens.
                Defaults to ["<mask:1>", "<mask:7>"].
            verbose (bool, optional): Whether to print detailed logs or progress information during execution.
            **kwargs: Additional keyword arguments passed to the parent class.
        """

        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        if additional_special_tokens is None:
            additional_special_tokens = ["<mask:1>", "<mask:7>"]
        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            verbose=verbose,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary.

        Returns:
            int: The number of tokens in the vocabulary.
        """
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """Get the vocabulary as a dictionary mapping tokens to their IDs.

        Returns:
            dict: A dictionary mapping tokens to their corresponding IDs.
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize text using SentencePiece.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: A list of tokens.
        """
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """Convert a token (str) to an ID using the vocabulary.

        Args:
            token (str): The token to convert.

        Returns:
            int: The corresponding token ID.
        """
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """Convert an ID to a token (str) using the vocabulary.

        Args:
            id (int): The token ID to convert.

        Returns:
            str: The corresponding token.
        """
        if id >= self.vocab_size:
            return self.unk_token
        else:
            return self.sp_model.id_to_piece(id)

    def convert_tokens_to_string(self, tokens):
        """Convert a sequence of tokens back to a single string.

        Args:
            tokens (List[str]): A list of tokens to convert.

        Returns:
            str: The reconstructed string.
        """
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def prepare_for_model(self, *args, **kwargs):
        if "add_special_tokens" in kwargs:
            kwargs.pop("add_special_tokens")
        return super().prepare_for_model(*args, **kwargs)

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (str): The directory in which to save the vocabulary.
            filename_prefix (Optional[str]): Optional prefix for the saved filename.

        Returns:
            Tuple[str]: Paths to the files saved.

        Raises:
            ValueError: If the save_directory is not a valid directory.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def _decode(self, *args, **kwargs):
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("spaces_between_special_tokens", None)
        return super()._decode(
            *args,
            **kwargs,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )

    def _pad(
        self,
        encoded_inputs: Union[dict],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs according to specified strategy.

        Args:
            encoded_inputs (Union[Dict]): Dictionary of encoded inputs.
            max_length (Optional[int]): Maximum length to pad to.
            padding_strategy (PaddingStrategy): Strategy for padding.
            pad_to_multiple_of (Optional[int]): Pad to a multiple of this value.
            return_attention_mask (Optional[bool]): Whether to return attention mask.

        Returns:
            dict: Dictionary with padded inputs and optional attention mask.

        Raises:
            ValueError: If attention_mask has unexpected type or invalid padding strategy.
        """
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask:
            required_input = encoded_inputs[self.model_input_names[0]]
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)
            if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

            if "attention_mask" in encoded_inputs and encoded_inputs["attention_mask"] is not None:
                attention_mask = encoded_inputs.pop("attention_mask")
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.numpy()
                elif isinstance(attention_mask, list):
                    attention_mask = np.array(attention_mask)
                elif not isinstance(attention_mask, np.ndarray):
                    raise ValueError(f"Unexpected type {type(attention_mask)} of attention_mask, ")
            else:
                # Create default attention mask if none provided
                attention_mask = np.tril(np.ones((len(required_input), len(required_input)), dtype=np.int64))
                attention_mask = np.expand_dims(attention_mask, axis=0)

            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if self.padding_side == "right":
                    if attention_mask.ndim == 1:
                        pad_width = [(0, difference)]
                    else:
                        pad_width = [(0, 0), (0, difference), (0, difference)]
                elif self.padding_side == "left":
                    if attention_mask.ndim == 1:
                        pad_width = [(difference, 0)]
                    else:
                        pad_width = [(0, 0), (difference, 0), (difference, 0)]
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))
                attention_mask = np.pad(
                    attention_mask,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )

        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask.tolist()
        return encoded_inputs


__all__ = ["Ernie4_5Tokenizer"]
