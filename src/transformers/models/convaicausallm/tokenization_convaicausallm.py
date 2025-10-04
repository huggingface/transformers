# coding=utf-8
# Copyright 2024 Convai Innovations and The HuggingFace Inc. team. All rights reserved.
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

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging, requires_backends


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "convaiinnovations/hindi-causal-lm": "https://huggingface.co/convaiinnovations/hindi-causal-lm/resolve/main/tokenizer.model",
    },
}

# Set max sequence length from your config
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "convaiinnovations/hindi-causal-lm": 512,
}


class ConvaiCausalLMTokenizer(PreTrainedTokenizer):
    """
    Construct a ConvaiCausalLM tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The padding token. Used for batching sequences. **Important:** The default behavior assumes this token corresponds to ID 0.
            Ensure your SentencePiece model was trained with this or set it explicitly if different. Check `self.pad_token_id`.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python)
            can be used, among other things, to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: Samples from the nbest_size results.
              - `nbest_size < 0`: Samples from the distribution associated with the segmentation probability.
            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merging operations for BPE-dropout.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add the BOS token at the beginning of sequences. Typically useful for generation tasks.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add the EOS token at the end of sequences. Typically useful for generation tasks.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
             Whether or not the tokenizer should clean up the tokenization spaces.
        from_slow (`bool`, *optional*, defaults to `False`):
            Whether or not the tokenizer is being initialized from a slow tokenizer.
         **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",  # Matches training default
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        from_slow=False,
        **kwargs,
    ):
        requires_backends(self, "sentencepiece")
        import sentencepiece as spm  # Import sentencepiece here

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.from_slow = from_slow

        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # Ensure special tokens are AddedTokens
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token

        # Validate pad_token and its ID
        sp_pad_id = self.sp_model.pad_id()
        if pad_token is None and sp_pad_id != -1:
            # Use pad token from SentencePiece model if specified and pad_token is None
            pad_token_str = self.sp_model.IdToPiece(sp_pad_id)
            logger.info(f"Using pad_token='{pad_token_str}' with ID {sp_pad_id} from SentencePiece model.")
            pad_token = AddedToken(pad_token_str, lstrip=False, rstrip=False)
        elif pad_token is None and sp_pad_id == -1:
            # Default to <pad> ID 0 if neither is set
            logger.warning(
                "The SentencePiece model does not define a pad token, and none was provided. "
                "Defaulting to pad_token='<pad>' (ID 0). Ensure this matches your model's training."
            )
            pad_token = AddedToken("<pad>", lstrip=False, rstrip=False)
        elif pad_token is not None:
            # Use user-provided pad_token
            pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
            pad_token_id_in_sp = self.sp_model.piece_to_id(str(pad_token))
            if pad_token_id_in_sp == -1:
                logger.warning(
                    f"The specified pad_token '{pad_token}' is not found in the SentencePiece vocabulary. "
                    "This might cause issues if padding is needed."
                )
            elif pad_token_id_in_sp != 0:
                logger.warning(
                    f"The pad_token='{pad_token}' provided has ID {pad_token_id_in_sp} in the SentencePiece vocabulary. "
                    "Hugging Face models often expect pad_token_id=0. Ensure this is intended."
                )
            # If pad_token_id_in_sp == 0, it matches the common HF expectation.
        else:
            # This case should not be reachable due to the logic above, but as a safeguard:
            logger.error("Unexpected condition while setting pad_token. Defaulting to '<pad>'.")
            pad_token = AddedToken("<pad>", lstrip=False, rstrip=False)

        # Call super().__init__ AFTER defining special tokens
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,  # Use the determined pad_token
            sp_model_kwargs=self.sp_model_kwargs,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            from_slow=from_slow,  # Pass from_slow
            **kwargs,
        )

        # Set pad_token_id explicitly AFTER super().__init__ to ensure consistency
        # self.pad_token_id should ideally be 0 based on the training script logic
        if self.pad_token is not None:
            self._pad_token_id = self.sp_model.piece_to_id(str(self.pad_token))
            if self._pad_token_id == self.sp_model.unk_id():
                logger.warning(
                    f"pad_token '{self.pad_token}' maps to the unknown token ID. This is likely unintended."
                )
            # Ensure the internal attribute matches the property
            # self.pad_token_id = self._pad_token_id # Property setter handles this

    # Define pad_token_id property explicitly if needed, otherwise base class handles it
    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        We explicitly set it to 0 by default or based on the SentencePiece model / user input.
        """
        # The base class property setter handles setting self._pad_token_id correctly during __init__
        # based on the pad_token argument. We just need to ensure the logic in __init__ is sound.
        return super().pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value):
        # Use the setter from the base class to handle AddedToken updates etc.
        super(ConvaiCausalLMTokenizer, type(self)).pad_token_id.__set__(self, value)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        """Returns vocab as Dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # bos/eos/unk properties rely on SentencePiece model's ids
    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp_model.eos_id()

    @property
    def unk_token_id(self) -> Optional[int]:
        return self.sp_model.unk_id()

    def _tokenize(self, text: str) -> List[str]:
        """Return a list of tokens"""
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
        # SentencePiece inherently handles space prefix markers (like ' ')
        # Using sp_model.decode should correctly reconstruct the string.
        return self.sp_model.decode(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named saved files.

        Returns:
            `Tuple(str)`: Paths to the saved files.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ()
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Vocabulary saved in {out_vocab_file}")
        elif not os.path.isfile(self.vocab_file):
            logger.error(f"Can't save vocabulary to {out_vocab_file}: source file '{self.vocab_file}' not found.")
            return ()
        else:
            logger.info(f"Vocabulary already exists in {out_vocab_file}. Skipping copy.")

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs from a sequence or a pair of sequence for sequence classification tasks."""
        bos_token_id = [self.bos_token_id] if self.add_bos_token and self.bos_token_id is not None else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token and self.eos_token_id is not None else []

        output = bos_token_id + token_ids_0

        if token_ids_1 is not None:
            # For causal LMs, typically don't add BOS between segments, but do add EOS at the end
            output = output + token_ids_1 + eos_token_id
        else:
            # Add EOS only if it's a single segment and add_eos_token is True
            output = output + eos_token_id

        return output

    # Usually no special handling needed for get_special_tokens_mask or create_token_type_ids_from_sequences
    # for single-segment causal LMs unless doing specific classification tasks.
