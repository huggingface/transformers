# coding=utf-8
# Copyright 2024 The Convai Innovations Authors and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for Hindi Causal LM."""

import os
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple

import sentencepiece as spm  # Direct dependency

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "convaiinnovations/hindi-foundational-model-base": "https://huggingface.co/convaiinnovations/hindi-foundational-model-base/resolve/main/tokenizer.model",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "convaiinnovations/hindi-foundational-model-base": 512,
}


class HindiCausalLMTokenizer(PreTrainedTokenizer):
    """
    Construct a HindiCausalLM tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the SentencePiece file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. Corresponds to ID 3 in the base model.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token. Corresponds to ID 1 in the base model.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token. Corresponds to ID 2 in the base model.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
             The padding token. Corresponds to ID 0 in the base model.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for subword regularization. Related to `enable_sampling`.
            - `alpha`: Smoothing parameter for subword regularization. Related to `enable_sampling`.
            - `dropout`: Probability related to subword regularization. Related to `enable_sampling`.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add the BOS token at the beginning of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
             Whether or not to add the EOS token at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding. Often set to `False` for SentencePiece based tokenizers.
        kwargs:
            Additional keyword arguments passed along to `PreTrainedTokenizer`.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>", # Default based on HF conventions, ID 3 from config
        bos_token="<s>", # Default based on HF conventions, ID 1 from config
        eos_token="</s>", # Default based on HF conventions, ID 2 from config
        pad_token="<pad>", # Default based on HF conventions, ID 0 from config
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True, # Often True for Causal LMs
        add_eos_token=False, # Often False for Causal LMs during training/prompting
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # Note: HF Tokenizer expects string tokens, but SentencePiece uses IDs directly.
        # We map the IDs provided in the config to conventional string representations.
        # The actual IDs used will come from the loaded .model file.

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self._tokenizer.Load(vocab_file)

        # Verify and set internal IDs based on loaded model vs expected config
        self._verify_special_ids()


    def _verify_special_ids(self):
        """Verify loaded SentencePiece model IDs match expectations."""
        # Expected IDs from config.json / tokenizer_config.json
        expected_pad_id = 0
        expected_bos_id = 1
        expected_eos_id = 2
        expected_unk_id = 3

        loaded_pad_id = self._tokenizer.pad_id()
        loaded_bos_id = self._tokenizer.bos_id()
        loaded_eos_id = self._tokenizer.eos_id()
        loaded_unk_id = self._tokenizer.unk_id()

        if loaded_pad_id != expected_pad_id:
             logger.warning(f"Loaded PAD ID {loaded_pad_id} != Expected {expected_pad_id}. Using loaded ID.")
             # Update the pad_token_id based on the loaded model if necessary
             # Note: self.pad_token_id is managed by the base class based on the pad_token string.
             # This check is primarily informational unless behavior needs overriding.

        if loaded_bos_id != expected_bos_id:
             logger.warning(f"Loaded BOS ID {loaded_bos_id} != Expected {expected_bos_id}. Using loaded ID.")

        if loaded_eos_id != expected_eos_id:
             logger.warning(f"Loaded EOS ID {loaded_eos_id} != Expected {expected_eos_id}. Using loaded ID.")

        if loaded_unk_id != expected_unk_id:
             logger.warning(f"Loaded UNK ID {loaded_unk_id} != Expected {expected_unk_id}. Using loaded ID.")


    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self._tokenizer.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self._tokenizer.id_to_piece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        # Uses EncodeAsPieces for string tokens, needed by base class logic
        return self._tokenizer.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._tokenizer.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self._tokenizer.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # Copied from LlamaTokenizer
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if current_sub_tokens:
                    out_string += self.sp_model.decode(current_sub_tokens)
                    current_sub_tokens = []
                out_string += token
                prev_is_special = True
            else:
                if prev_is_special:
                    out_string += " "
                current_sub_tokens.append(token)
                prev_is_special = False
        if current_sub_tokens:
            out_string += self.sp_model.decode(current_sub_tokens)
        return out_string


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the saved files.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
             copyfile(self.vocab_file, out_vocab_file)
             logger.info(f"Copying {self.vocab_file} to {out_vocab_file}")
        elif not os.path.isfile(self.vocab_file):
             logger.error(f"Cannot copy {self.vocab_file} to {out_vocab_file}: File not found.")


        return (out_vocab_file,)

    # Override build_inputs_with_special_tokens if needed (defaults might be ok)
    # def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None): ...

    # Override get_special_tokens_mask if needed (defaults might be ok)
    # def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False): ...

    # Override create_token_type_ids_from_sequences if needed (usually 0s for single sequence)
    # def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None): ...

    # --- Add direct access to underlying SentencePiece model if needed ---
    @property
    def sp_model(self):
        """Direct access to the SentencePieceProcessor instance."""
        return self._tokenizer

