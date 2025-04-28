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
        # Add other pretrained model identifiers here if needed
    },
}

# Assuming max sequence length used during training was 512
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
            Ensure your SentencePiece model was trained with this or set it explicitly if different.
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
        pad_token="<pad>",  # Default pad token
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        from_slow=False,  # Added for consistency with newer tokenizer patterns
        **kwargs,
    ):
        requires_backends(self, "sentencepiece")
        # Import sentencepiece here so it's only required if the tokenizer is actually used
        import sentencepiece as spm

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.from_slow = from_slow  # Track if converting from slow

        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # Validate pad token ID assumption
        sp_pad_id = self.sp_model.pad_id()
        # Default <pad> to ID 0 if not defined in SP model or explicitly passed differently
        # Set the pad_token attribute using the logic below before calling super().__init__
        effective_pad_token = pad_token
        if pad_token is None and sp_pad_id != -1:
            logger.info(f"Using pad_token_id {sp_pad_id} from SentencePiece model.")
            effective_pad_token = self.sp_model.IdToPiece(sp_pad_id)
        elif pad_token is None and sp_pad_id == -1:
            logger.warning(
                "The SentencePiece model does not define a pad token, using default `<pad>`. "
                "Make sure `<pad>` is ID 0 in your model."
            )
            effective_pad_token = "<pad>"  # Keep default
        elif pad_token is not None and self.sp_model.piece_to_id(pad_token) == 0:
            pass  # User provided pad token matches ID 0 assumption, all good
        elif pad_token is not None and self.sp_model.piece_to_id(pad_token) != 0:
            logger.warning(
                f"You passed pad_token='{pad_token}' but the default uses ID 0. "
                "If your model expects pad ID 0, this might lead to unexpected behavior."
                f"The ID for '{pad_token}' in the vocab is {self.sp_model.piece_to_id(pad_token)}."
            )
            # Use the user-provided pad_token, but log warning.

        # Ensure special tokens are AddedTokens (important for internal handling)
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = (
            AddedToken(effective_pad_token, lstrip=False, rstrip=False)
            if isinstance(effective_pad_token, str)
            else effective_pad_token
        )

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,  # Use the determined pad_token
            sp_model_kwargs=self.sp_model_kwargs,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        """Returns vocab as Dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sequence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sequence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        return self.sp_model.eos_id()

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        return self.sp_model.unk_id()

    # pad_token_id property is handled by the base class using self.pad_token

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
        # Uses Huey's logic from https://github.com/huggingface/transformers/pull/19566
        if len(tokens) == 0:
            return ""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # Compatibility with BPE tokenizer's pattern to handle spaces properly
            # If the previous token was special, sequences of special tokens don't add spaces
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += self.sp_model.decode(current_sub_tokens)
                    current_sub_tokens = []
                out_string += token
                prev_is_special = True
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

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
            return ()  # Return empty tuple on error
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            try:
                copyfile(self.vocab_file, out_vocab_file)
            except OSError as e:
                logger.error(f"Error copying vocabulary file: {e}")
                return ()  # Return empty tuple on error

        elif not os.path.isfile(self.vocab_file):
            logger.warning(
                f"Can't copy source vocab file '{self.vocab_file}' to '{out_vocab_file}' as it doesn't exist. "
                "Check the path."
            )
            return ()  # Return empty tuple if source doesn't exist

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs from a sequence or a pair of sequence for sequence classification tasks."""
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            # Usually BOS is not added between segments for Causal LMs
            # output = output + bos_token_id + token_ids_1 + eos_token_id
            output = output + token_ids_1 + eos_token_id

        return output

    # No need for create_token_type_ids_from_sequences for a Causal LM typically
