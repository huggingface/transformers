# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Inc. team. All rights reserved.
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
from typing import Dict, List, Optional, Tuple

from tokenizers import AddedToken
from tokenizers import processors

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_hindicausallm import HindiCausalLMTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}


# Copied from transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast with Llama->HindiCausalLM, LLAMA->HINDICAUSALLM
class HindiCausalLMTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a fast HindiCausalLM tokenizer (backed by HuggingFace's `tokenizers` library). Based on SentencePiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file. `tokenizer.model` for SentencePiece based tokenizers. `vocab.txt` for WordPiece based tokenizers.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer configuration file saved using the `save_pretrained()` method. This can be used to
            load a specific configuration file.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation. If None, will default to same as `eos_token`.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add the beginning of sequence token to the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add the end of sequence token to the end of sequences.
        add_prefix_space (`bool`, *optional*):
            Whether or not to add a space to the input before tokenization.
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether or not the post-processing step should trim offsets to avoid including whitespaces.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = HindiCausalLMTokenizer
    model_input_names = ["input_ids", "attention_mask"]
    # Override defaults inherited from slow_tokenizer_class
    padding_side = "left"
    # No need to pass added_tokens_decoder to the parent class, we handlle special tokens internally.
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None, # Default pad to None, will be set later
        add_bos_token=True,
        add_eos_token=False, # Typically false for Causal LM
        add_prefix_space=False, # Default to False
        legacy=False, # Keep legacy option for consistency
        **kwargs,
    ):
        # XXX (raghav): legacy is True by default for backward compatibility. We should default it to False for all new objects.
        # --> legacy=True means that the previous behavior stands where we prefix the inputs with a space for SentencePiece tokenizers.
        # --> legacy=False means we are removing this behavior.
        # We should check the saved vocab file to see if the normalizer is PrendPend. --> This is not needed as we instantiate from the json file?

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token

        # Default pad_token to eos_token if not specified
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        if pad_token is None:
            pad_token = eos_token # Default pad to eos if None

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            add_prefix_space=add_prefix_space,
            legacy=legacy, # Pass legacy to parent
            **kwargs,
        )
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.vocab_file = vocab_file

        # TODO(enetrom): The following logic should be maybe removed? Seems weird?
        # If add_prefix_space is not provided, we check the vocab file and see if the normalizer is Prepend(" ")
        # --> This requires loading the model outside the init, which is not ideal.
        # We can separate this into a different function? Or just keep it like this for now?
        # ---> Investigate if the slow tokenizer applies the same logic. If it does, then we are fine.
        # ----> The slow tokenizer does not have this logic. We should remove it? Or let the default value take place?
        # Add legacy=True/False logic here maybe?
        if add_prefix_space is None:
            if legacy:
                self.add_prefix_space = True
            else:
                self.add_prefix_space = False
        else:
            self.add_prefix_space = add_prefix_space

        self._build_processor()

    def _build_processor(self):
        """Builds the template processor using the provided BOS and EOS tokens"""
        # This logic is slightly different from Llama, adapted for typical causal LM use
        bos = self.bos_token
        eos = self.eos_token
        bos_token_id = self.bos_token_id
        eos_token_id = self.eos_token_id

        if bos is None and eos is None:
            self._tokenizer.post_processor = None
        elif bos is not None and eos is None:
             if self.add_bos_token:
                 self._tokenizer.post_processor = processors.TemplateProcessing(
                     single=f"{bos}:0 $A:0",
                     pair=f"{bos}:0 $A:0 {bos}:1 $B:1",
                     special_tokens=[
                         (bos, bos_token_id),
                     ],
                 )
             else:
                 self._tokenizer.post_processor = processors.TemplateProcessing(
                     single="$A:0",
                     pair="$A:0 $B:1",
                     special_tokens=[],
                 )
        elif bos is None and eos is not None:
             if self.add_eos_token:
                self._tokenizer.post_processor = processors.TemplateProcessing(
                    single="$A:0 {eos}:0",
                    pair="$A:0 {eos}:0 $B:1 {eos}:1",
                    special_tokens=[
                        (eos, eos_token_id),
                    ],
                )
             else:
                 self._tokenizer.post_processor = processors.TemplateProcessing(
                     single="$A:0",
                     pair="$A:0 $B:1",
                     special_tokens=[],
                 )
        elif bos is not None and eos is not None:
            if self.add_bos_token and self.add_eos_token:
                self._tokenizer.post_processor = processors.TemplateProcessing(
                    single=f"{bos}:0 $A:0 {eos}:0",
                    pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1", # Add BOS/EOS to both parts of pair
                    special_tokens=[
                        (bos, bos_token_id),
                        (eos, eos_token_id),
                    ],
                )
            elif self.add_bos_token and not self.add_eos_token:
                 self._tokenizer.post_processor = processors.TemplateProcessing(
                     single=f"{bos}:0 $A:0",
                     pair=f"{bos}:0 $A:0 {bos}:1 $B:1", # Add BOS to both parts of pair
                     special_tokens=[(bos, bos_token_id)],
                 )
            elif not self.add_bos_token and self.add_eos_token:
                 self._tokenizer.post_processor = processors.TemplateProcessing(
                     single=f"$A:0 {eos}:0",
                     pair=f"$A:0 {eos}:0 $B:1 {eos}:1", # Add EOS to both parts of pair
                     special_tokens=[(eos, eos_token_id)],
                 )
            else:
                 self._tokenizer.post_processor = processors.TemplateProcessing(
                    single="$A:0",
                    pair="$A:0 $B:1",
                    special_tokens=[],
                )
        else:
            self._tokenizer.post_processor = None

    # Copied from transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            # Determine if BOS/EOS should be added between sequences based on processor template
            # This is simplified; processor handles the complex cases.
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    # Copied from transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
             # Determine length based on processor template logic
            output = output + [0] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output


    # Copied from transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the SentencePiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named output files.

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

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copy vocab file to {out_vocab_file}")

        return (out_vocab_file,)