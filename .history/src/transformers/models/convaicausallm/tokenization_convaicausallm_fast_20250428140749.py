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
from typing import Optional, Tuple

from tokenizers import processors

from ....tokenization_utils_fast import PreTrainedTokenizerFast
from ....utils import is_sentencepiece_available, logging
from .tokenization_convaicausallm import ConvaiCausalLMTokenizer


if is_sentencepiece_available():
    from .tokenization_convaicausallm import ConvaiCausalLMTokenizer
else:
    ConvaiCausalLMTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}


class ConvaiCausalLMTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" ConvaiCausalLM tokenizer (backed by HuggingFace's *tokenizers* library). Based on BPE.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file. Contains the merges and the vocabulary. Can be converted from the SentencePiece
            vocabulary file using `convert_slow_tokenizer.py`
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
             The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The padding token. Defaults to the `eos_token`.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = ConvaiCausalLMTokenizer
    # Define the same default padding side as Llama/Gemma if applicable
    # padding_side = "left" # Or "right" depending on training
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,  # Will default to eos_token in super().__init__ if None
        add_bos_token=True,
        add_eos_token=False,
        **kwargs,
    ):
        # Set pad_token before super().__init__ if you want it different from eos
        # Based on your slow tokenizer, pad_token should be "<pad>" with ID 0
        _pad_token_id = 0 # From your config
        pad_token = "<pad>" if pad_token is None else pad_token

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token, # Pass the determined pad_token
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.vocab_file = vocab_file

        # Ensure pad_token_id is set correctly after initialization
        if self.pad_token_id is None:
            self.pad_token_id = _pad_token_id # Set ID 0 explicitly if not set by super

        # Set post processor based on Gemma's template processing logic
        # This assumes similar BOS/EOS behavior as Gemma
        self.update_post_processor()

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        # Adapting Gemma's TemplateProcessing logic
        single = f"{(bos + ':0 ') if self.add_bos_token else ''}$A:0{(' ' + eos + ':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' ' + bos + ':1 ') if self.add_bos_token else ''}$B:1{(' ' + eos + ':1') if self.add_eos_token else ''}"


        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))

        if special_tokens:
             self._tokenizer.post_processor = processors.TemplateProcessing(
                single=single, pair=pair, special_tokens=special_tokens
            )
        else:
            # If no special tokens are added, a simpler post-processor might be needed
            # or potentially no post-processor if the tokenizer handles it internally.
            # For now, let's assume TemplateProcessing with empty lists works or is handled.
            self._tokenizer.post_processor = processors.TemplateProcessing(single="$A:0", pair="$A:0 $B:1", special_tokens=[])


    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory. Copies the underlying SentencePiece model file.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                Unused.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if self.vocab_file and os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copying {self.vocab_file} to {out_vocab_file}")

        # Save the tokenizer.json file
        tokenizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["tokenizer_file"]
        )
        self.save(tokenizer_file)


        return (out_vocab_file, tokenizer_file)

    # build_inputs_with_special_tokens can often be inherited from PreTrainedTokenizerFast
    # if the logic matches TemplateProcessing. Let's rely on parent first.
