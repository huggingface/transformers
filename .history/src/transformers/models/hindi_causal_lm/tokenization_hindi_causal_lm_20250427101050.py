# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Tokenization classes for Hindi Causal LM."""

import os
# Import Any from typing
from typing import Any, Dict, List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_sentencepiece_available, logging, requires_backends


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "convaiinnovations/hindi-foundational-model-base": "https://huggingface.co/convaiinnovations/hindi-foundational-model-base/resolve/main/tokenizer.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "convaiinnovations/hindi-foundational-model-base": 512,
}

# SentencePiece is optional
if is_sentencepiece_available():
    pass


class HindiCausalLMTokenizer(PreTrainedTokenizer):
    """
    Construct a Hindi Causal LM tokenizer based on SentencePiece. Adapted from the SentencePiece tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The mask token used in masked language modeling tasks.
        sp_model_kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the model initialization.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether to lowercase the input when tokenizing.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        # Ensure Any is imported from typing
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        do_lower_case=False,
        **kwargs,
    ):
        # Mask token behave like a normal word, i.e. include the space before it
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.do_lower_case = do_lower_case

        self.vocab_file = vocab_file
        self.sp_model = None
        if is_sentencepiece_available():
            self.sp_model = self.load_spm(vocab_file)
        else:
            requires_backends(self, ["sentencepiece"])

        # SentencePiece token maps - adjust based on actual model needs if different from standard SentencePiece
        # Check the actual trained tokenizer.model for special token IDs
        # Assuming standard SPM uses <s>=1, <pad>=0, </s>=2, <unk>=3 is a common convention but verify
        # Using provided special tokens for consistency
        self.fairseq_tokens_to_ids = {} # Start empty, will be populated by super().__init__
        self.fairseq_offset = 0 # Set offset based on how many special tokens are *not* handled by SPM

        # Set special tokens and initialize superclass
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )

        # Populate fairseq_tokens_to_ids with the actual token IDs assigned by super().__init__
        # or determined from the sp_model if needed.
        # This assumes super().__init__ handles adding special tokens correctly.
        # We might need to adjust vocab_size calculation based on how special tokens interact with sp_model.
        if is_sentencepiece_available() and self.sp_model:
            # Standard SentencePiece model vocab size
            spm_vocab_size = self.sp_model.GetPieceSize()

            # Calculate the effective vocab size considering added special tokens
            # `self.added_tokens_decoder` maps assigned IDs to token strings (added via super().__init__)
            num_added_tokens = len(self.added_tokens_decoder)

            # Check if special tokens are already part of the SentencePiece model vocab
            special_tokens_in_spm = 0
            for token_str in [bos_token, eos_token, unk_token, pad_token, mask_token]:
                 if token_str and self.sp_model.PieceToId(token_str) != self.sp_model.unk_id():
                     special_tokens_in_spm += 1
                 # Need to also check if the assigned ID (e.g., self.pad_token_id) matches sp_model's ID
                 # For simplicity, assume super().__init__ handles adding tokens if they aren't in spm

            # Base vocab size on sp_model + explicitly added tokens
            # This can be tricky if sp_model already contains some special tokens.
            # Using len(self) from PreTrainedTokenizer is usually the most reliable way
            # after initialization.
            self.vocab_size = len(self) # Let PreTrainedTokenizer determine the final size

            # Build token<->id maps ensuring consistency
            self.tokens_to_ids = {token: idx for idx, token in self.get_vocab().items()}
            self.ids_to_tokens = {idx: token for token, idx in self.tokens_to_ids.items()}

            # Verify if fairseq_offset logic is needed. Often it's simpler to rely on
            # PreTrainedTokenizer's handling unless there's a specific legacy reason.
            # If the base SPM model uses IDs 0, 1, 2, 3 for special tokens and we remap them,
            # then an offset might be needed for the rest of the vocab.
            # However, relying on get_vocab() derived from sp_model + added tokens is safer.
            # Let's remove the potentially confusing fairseq_offset logic unless proven necessary.
            # self.fairseq_offset = 0 # Resetting as it seems complex and possibly unnecessary
            # self.vocab_size = self.sp_model.GetPieceSize() + self.fairseq_offset # Old calculation, prefer len(self)

            # Rebuild maps based on final vocab from PreTrainedTokenizer
            # This ensures all added tokens are correctly mapped
            # self.tokens_to_ids = self.get_vocab() # get_vocab() already provides token -> id
            # self.ids_to_tokens = {v: k for k, v in self.tokens_to_ids.items()}

    def load_spm(self, vocab_file):
        """Load the SentencePiece model."""
        if is_sentencepiece_available():
            try:
                import sentencepiece as spm
            except ImportError:
                logger.warning(
                    "You need to install SentencePiece to use HindiCausalLMTokenizer: "
                    "https://github.com/google/sentencepiece\n"
                    "`pip install sentencepiece`"
                )
                raise

            sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
            sp_model.Load(vocab_file)
            return sp_model
        else:
            requires_backends(self, ["sentencepiece"])

    # Override methods from PreTrainedTokenizer if necessary
    # The following methods rely on SentencePiece and the special token handling defined above.

    @property
    def vocab_size(self):
        # Return the size calculated during __init__ which considers added tokens
        # Use len(self) for the most robust value after initialization.
        # Need to ensure self._vocab_size is set correctly in __init__ or rely on len(self)
        # Let's use the base property from PreTrainedTokenizer which relies on get_vocab()
        return len(self.get_vocab())

    def get_vocab(self) -> Dict[str, int]:
        """ Returns the vocabulary as a dictionary of token to index. """
        if not is_sentencepiece_available():
             requires_backends(self, ["sentencepiece"])
        if not hasattr(self, "_vocab"):
             # Build vocab from sp_model and added tokens
             vocab = {self.sp_model.IdToPiece(i): i for i in range(self.sp_model.GetPieceSize())}
             # Update with added tokens (special tokens), potentially overwriting sp_model IDs if needed
             # `self.added_tokens_encoder` stores the token -> id mapping for added tokens
             vocab.update(self.added_tokens_encoder)
             self._vocab = vocab # Cache it
        return self._vocab


    def _tokenize(self, text: str) -> List[str]:
        """ Tokenize a string using SentencePiece. """
        if not is_sentencepiece_available():
            requires_backends(self, ["sentencepiece"])

        if self.do_lower_case:
            text = text.lower()
        # Encode using SentencePiece model
        return self.sp_model.encode(text, out_type=str)


    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str) to an id using the vocabulary. """
        # Check added tokens first (includes special tokens)
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        # Fall back to SentencePiece model
        if is_sentencepiece_available() and self.sp_model:
             spm_id = self.sp_model.PieceToId(token)
             # SPM returns <unk> ID (often 0 or 3 depending on model) if token is unknown
             # Check if the returned ID corresponds to the *model's* unk piece ID
             if spm_id == self.sp_model.unk_id():
                  # If SPM considers it unknown, return the tokenizer's unk_token_id
                  return self.unk_token_id
             return spm_id
        else:
             # Should not happen if check at start passes, but fallback for safety
             return self.unk_token_id


    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocabulary."""
        # Check added tokens decoder first
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index]
        # Fall back to SentencePiece model
        if is_sentencepiece_available() and self.sp_model:
             # Check if index is within the valid range for the SPM model
             if 0 <= index < self.sp_model.GetPieceSize():
                 return self.sp_model.IdToPiece(index)
             else:
                 # Index out of range or corresponds to an added token already handled
                 return self.unk_token
        else:
             return self.unk_token


    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (list of str) back into a single string. """
        if not is_sentencepiece_available():
            requires_backends(self, ["sentencepiece"])

        # Filter out special tokens that shouldn't be decoded by SentencePiece
        # unless SentencePiece itself handles them (e.g., if <s> is part of its vocab)
        # A common approach is to decode piece by piece or use sp_model.decode if robust
        # Using sp_model.decode(tokens) might be simpler if tokens are guaranteed valid pieces
        try:
             # Attempt direct decoding
             current_sub_text = []
             out_string = ""
             for token in tokens:
                 # Handle special tokens explicitly if they were added and not part of spm vocab
                 if token in self.added_tokens_encoder:
                     # If we have accumulated sub-text, decode it first
                     if current_sub_text:
                         out_string += self.sp_model.decode(current_sub_text)
                         current_sub_text = []
                     # Append the special token string itself (or handle as needed)
                     # Usually, we just skip them for string reconstruction, or handle based on context
                     # E.g., self.sp_model.decode might handle some special tokens if trained with them
                     pass # Often skip special tokens in final string output
                 else:
                     current_sub_text.append(token)

             # Decode any remaining sub-text
             if current_sub_text:
                 out_string += self.sp_model.decode(current_sub_text)

             # Alternative simpler approach if sp_model handles its own special tokens:
             # text = self.sp_model.decode(tokens)

             # Using the more controlled piece-by-piece can be safer with added tokens
             # Let's stick to the provided implementation's likely intent: DecodePieces
             # Assuming DecodePieces handles underlying byte details correctly
             return self.sp_model.decode(tokens) # Revert to simpler if robust

        except Exception as e:
             logger.error(f"Error decoding tokens: {e}")
             # Fallback or simplistic join
             return " ".join(tokens) # Basic fallback


    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A HindiCausalLM sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>` (or other format depending on model training)

        Args:
            token_ids_0 (`List[int]`): List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            # Single sequence: <s> X </s>
            return bos + token_ids_0 + eos
        else:
            # Pair of sequences: <s> A </s> B </s>  (Confirm this is the correct format for the model)
            # Some models might use <s> A </s> </s> B </s> or other separators.
            # Assuming the format described in the docstring.
            return bos + token_ids_0 + eos + token_ids_1 + eos # Original LLaMA-like pair format

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            # If special tokens are already present, delegate to the superclass method
            # which should identify them based on their IDs.
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If special tokens haven't been added yet, construct the mask based on the structure
        # defined in `build_inputs_with_special_tokens`.
        bos_mask = [1]
        eos_mask = [1]

        if token_ids_1 is None:
            # Single sequence: <s> X </s> -> mask [1] + [0]*len(X) + [1]
            return bos_mask + ([0] * len(token_ids_0)) + eos_mask
        else:
            # Pair of sequences: <s> A </s> B </s> -> mask [1] + [0]*len(A) + [1] + [0]*len(B) + [1]
            return bos_mask + ([0] * len(token_ids_0)) + eos_mask + ([0] * len(token_ids_1)) + eos_mask


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the SentencePiece vocabulary model file to a directory.

        Args:
            save_directory (`str`): The directory where the vocabulary file will be saved.
            filename_prefix (`str`, *optional*): Optional prefix to add to the name of the saved file.

        Returns:
            `Tuple[str]`: Path to the saved vocabulary file.
        """
        if not is_sentencepiece_available():
            requires_backends(self, ["sentencepiece"])

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return () # Return empty tuple on error

        # Determine the output file path
        vocab_file_name = VOCAB_FILES_NAMES["vocab_file"]
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + vocab_file_name
        )

        # Check if the source vocab file exists and copy it
        if hasattr(self, 'vocab_file') and self.vocab_file and os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            import shutil
            shutil.copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Vocabulary file saved to {out_vocab_file}")
        # If source file doesn't exist or is the same, try saving from the sp_model object
        elif self.sp_model is not None:
             try:
                  # SentencePieceProcessor doesn't have a direct save method.
                  # We need to save the underlying model proto.
                  model_proto = self.sp_model.serialized_model_proto()
                  with open(out_vocab_file, "wb") as f:
                      f.write(model_proto)
                  logger.info(f"Vocabulary file saved to {out_vocab_file}")
             except AttributeError:
                  logger.error("Could not save vocabulary: sp_model does not have serialized_model_proto method.")
                  return ()
             except Exception as e:
                  logger.error(f"Could not save vocabulary: {e}")
                  return ()
        else:
             logger.error("Could not save vocabulary: No vocab_file path or sp_model available.")
             return ()

        # Return the path to the saved file in a tuple, as expected by the base class method.
        return (out_vocab_file,)