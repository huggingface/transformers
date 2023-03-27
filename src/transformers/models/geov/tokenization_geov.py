# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Modifications Copyright 2023 Better Planet Investments and Labml team. ALl rights reserved.
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
"""Tokenization classes for GeoV."""
from pathlib import Path
from typing import List, Optional, Tuple

import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import SPIECE_UNDERLINE, logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "GeoV/GeoV-9b": "https://huggingface.co/GeoV/GeoV-9b/resolve/main/spiece.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "GeoV-9b": 2049,
}


class GeoVTokenizer(PreTrainedTokenizer):
    """
    Construct an GeoV tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining.

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.

        new_line_token_id (`int`, *optional*, defaults to `65_499`):
            The token id of new line character.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
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
            new_line_token_id=65_499,
            **kwargs,
    ) -> None:
        super().__init__(
            vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            new_line_token_id=new_line_token_id,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.new_line_token_id = new_line_token_id

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        ret = []
        split_text = text.splitlines()
        for l in split_text:
            rl = self.sp_model.encode(l, out_type=str)
            ret.extend(rl)
            ret.append('\n')
        ret = ret[:-1]
        return ret

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token == '\n':
            return self.new_line_token_id
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index == self.new_line_token_id:
            return '\n'
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
            spaces_between_special_tokens: bool = True,
            **kwargs,
    ) -> str:
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        if skip_special_tokens:
            filtered_tokens = [t for t in filtered_tokens if t not in self.all_special_ids]

        text = self.convert_tokens_to_string(filtered_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        save_directory = Path(save_directory)
        if not save_directory.is_dir():
            raise ValueError(f"Vocabulary path ({save_directory}) should be a directory")
        vocab_fn = VOCAB_FILES_NAMES['vocab_file']
        filename_prefix = f'{filename_prefix}-' if filename_prefix else ''

        vocab_file = save_directory / f'{filename_prefix}{vocab_fn}'

        with open(str(vocab_file), "wb") as fi:
            content_spiece_model = self.sp_model.serialized_model_proto()
            fi.write(content_spiece_model)

        return (str(vocab_file),)
