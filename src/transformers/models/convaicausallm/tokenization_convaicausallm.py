# coding=utf-8
# Copyright 2024 Convai Innovations Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for ConvaiCausalLM."""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....utils import logging, requires


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPIECE_UNDERLINE = " " # Using SentencePiece default


@requires("sentencepiece")
class ConvaiCausalLMTokenizer(PreTrainedTokenizer):
    """
    Construct a ConvaiCausalLM tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str` or `AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            Note: The original `hindi-causal-lm` checkpoint used BOS=1.
        eos_token (`str` or `AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token. Note: The original `hindi-causal-lm` checkpoint used EOS=2.
        pad_token (`str` or `AddedToken`, *optional*):
            The padding token. Defaults to the `eos_token` if not set. Note: The original `hindi-causal-lm` checkpoint used PAD=0.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to clean up the tokenization spaces. For instance, convert ` Residence` to `Residence`.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
             Whether or not the default system prompt for ConvaiCausalLM should be used. Not applicable to the base model.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None # Set explicitly if this is the fast version

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>", # Default, adjust if your model has a different UNK
        bos_token="<s>",   # From your config.json (BOS=1)
        eos_token="</s>",   # From your config.json (EOS=2)
        pad_token=None,    # From your config.json (PAD=0 - handled below)
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False, # Usually False for base models
        spaces_between_special_tokens=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token

        # Handle pad token: Use "<pad>" if vocab has it, otherwise set to eos_token (or unk if eos not defined)
        # Your config specified pad_token_id = 0. Check if token 0 in your vocab is actually '<pad>'
        # If not, you might need to add it or map it correctly.
        # For now, let's assume ID 0 corresponds to a pad token conceptually, even if not named '<pad>'
        # We'll set pad_token = "<pad>" and rely on the ID mapping during init.
        if pad_token is None:
             # Check if the vocab file actually defines a pad token at index 0
             _sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
             _sp_model.Load(vocab_file)
             # If ID 0 is defined and not something else like <unk>, use a default pad token name
             if _sp_model.IdToPiece(0) is not None and _sp_model.IdToPiece(0) != "<unk>":
                  pad_token = AddedToken("<pad>", normalized=False, special=True) # Assume ID 0 is pad
             else: # Fallback if ID 0 is undefined or unk
                  pad_token = AddedToken("<unk>", normalized=False, special=True) # Or potentially eos_token if preferred fallback
        else:
             pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        self.use_default_system_prompt = use_default_system_prompt

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

        # Set pad_token_id based on config.json (which was 0)
        # This needs to happen *after* super().__init__ assigns default IDs
        self._pad_token_type_id = 0 # Default
        self._pad_token = "<pad>" # Default representation
        self.pad_token_id = 0 # Explicitly set based on your config

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string. Bias towards generating spaces BEFORE the word boundary
        """
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # Copied from LlamaTokenizer, should work for SentencePiece based models
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens_extended:
                if not prev_is_special and len(current_sub_tokens) > 0:
                     out_string += self.sp_model.decode(current_sub_tokens) + " "
                     current_sub_tokens = []
                out_string += token + " "
                prev_is_special = True
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        if len(current_sub_tokens) > 0:
             out_string += self.sp_model.decode(current_sub_tokens)

        return out_string.strip()

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs from a sequence or a pair of sequence for sequence classification tasks."""
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    # get_special_tokens_mask, create_token_type_ids_from_sequences
    # can often be inherited from PreTrainedTokenizer if the logic is standard
    # (e.g., sequence A is type 0, sequence B is type 1).
    # Let's rely on the parent implementation for now. If specific behavior is needed, override them here.