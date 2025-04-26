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
from typing import Dict, List, Optional, Tuple, Union

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

if is_sentencepiece_available():
    import sentencepiece as spm


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
            
        # SentencePiece token maps
        self.fairseq_tokens_to_ids = {"<s>": 1, "<pad>": 0, "</s>": 2, "<unk>": 3}
        self.fairseq_offset = 0
        
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
        
        # Make sure vocab size is set correctly
        if is_sentencepiece_available():
            self.vocab_size = self.sp_model.GetPieceSize() + self.fairseq_offset
            
            # Set up token conversions
            self.tokens_to_ids = self.fairseq_tokens_to_ids.copy()
            self.ids_to_tokens = {v: k for k, v in self.tokens_to_ids.items()}
            
            for i in range(self.sp_model.GetPieceSize()):
                piece = self.sp_model.IdToPiece(i)
                if piece not in self.tokens_to_ids:
                    self.tokens_to_ids[piece] = i + self.fairseq_offset
                    self.ids_to_tokens[i + self.fairseq_offset] = piece

    def load_spm(self, vocab_file):
        """Load the SentencePiece model."""
        if is_sentencepiece_available():
            try:
                import sentencepiece as spm
            except ImportError:
                logger.warning(
                    "You need to install SentencePiece to use HindiCausalLMTokenizer: https://github.com/google/sentencepiece"
                    "pip install sentencepiece"
                )
                raise

            sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
            sp_model.Load(vocab_file)
            return sp_model
        else:
            requires_backends(self, ["sentencepiece"])

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

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

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A HindiCausalLM sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize using SentencePiece.

        Args:
            text (`str`):
                The text to tokenize.

        Returns:
            `List[str]`: The tokenized tokens.
        """
        if not is_sentencepiece_available():
            requires_backends(self, ["sentencepiece"])
            
        if self.do_lower_case:
            text = text.lower()
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """
        Converts a token (str) to an id using the vocab.

        Args:
            token (`str`):
                The token to convert.

        Returns:
            `int`: The corresponding id.
        """
        if token in self.tokens_to_ids:
            return self.tokens_to_ids[token]
        # In case of a single character token, check if it's in the vocabulary
        if is_sentencepiece_available():
            spm_id = self.sp_model.PieceToId(token)
            if spm_id == self.sp_model.unk_id():
                # Unknown token, return unk_token_id
                return self.unk_token_id
            return spm_id + self.fairseq_offset
        else:
            requires_backends(self, ["sentencepiece"])

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) to a token (str) using the vocab.

        Args:
            index (`int`):
                The index to convert.

        Returns:
            `str`: The token string.
        """
        if index in self.ids_to_tokens:
            return self.ids_to_tokens[index]
        if is_sentencepiece_available():
            return self.sp_model.IdToPiece(index - self.fairseq_offset)
        else:
            requires_backends(self, ["sentencepiece"])

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (strings for sub-words) in a single string.

        Args:
            tokens (`List[str]`):
                The tokens to concatenate.

        Returns:
            `str`: The joined string.
        """
        if not is_sentencepiece_available():
            requires_backends(self, ["sentencepiece"])
        out_string = self.sp_model.DecodePieces(tokens)
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory where the vocabulary will be saved.
            filename_prefix (`str`, *optional*):
                Optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not is_sentencepiece_available():
            requires_backends(self, ["sentencepiece"])
            
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            import shutil
            shutil.copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)