# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
""" Tokenization class for SigLIP model."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils_base import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging


if is_sentencepiece_available():
    from .tokenization_siglip import SiglipTokenizer
else:
    SiglipTokenizer = None


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "nielsr/siglip-base-patch16-224": "https://huggingface.co/nielsr/siglip-base-patch16-224/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "nielsr/siglip-base-patch16-224": "https://huggingface.co/nielsr/siglip-base-patch16-224/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "nielsr/siglip-base-patch16-224": 512,
}


class SiglipTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" SigLIP tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            Tokenizer file.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = SiglipTokenizer

    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=None,
        **kwargs,
    ):
        pad_token = AddedToken(pad_token, rstrip=True, lstrip=True) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token, rstrip=True, lstrip=True) if isinstance(unk_token, str) else unk_token
        eos_token = AddedToken(eos_token, rstrip=True, lstrip=True) if isinstance(eos_token, str) else eos_token

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
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

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0
        else:
            token_ids_1 = token_ids_1 + [self.eos_token_id]
            return self.prefix_tokens + token_ids_0 + token_ids_1

    # Copied from transformers.models.siglip.tokenization_siglip.SiglipTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. Siglip does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]
