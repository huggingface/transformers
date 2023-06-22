# coding=utf-8
# Copyright 2023 The MMS-TTS Authors and the HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for VITS."""


import json
import os
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "TODO": "https://huggingface.co/TODO/resolve/main/vocab.json",
    }
}

# TODO: We don't have this; still need to specify it?
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "TODO": 4096,
}


class VitsMmsTokenizer(PreTrainedTokenizer):
    """
    Construct a VITS tokenizer for MMS-TTS.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        language (`str`, *optional*):
            Language identifier.
        add_blank (`bool`, *optional*, defaults to `True`):
            Whether to insert token id 0 in between the other tokens.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        pad_token="<pad>",
        unk_token="<unk>",
        language=None,
        add_blank=True,
        **kwargs,
    ) -> None:
        super().__init__(pad_token=pad_token, unk_token=unk_token, language=language, add_blank=add_blank, **kwargs)

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        # The original model pads with token_id 0, but that doesn't work here
        # as token_id 0 is an actual character. So we add a fake padding token
        # and then filter that out in the model.
        if pad_token not in self.encoder:
            self.encoder[pad_token] = len(self.encoder)
        if unk_token not in self.encoder:
            self.encoder[unk_token] = len(self.encoder)

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.language = language
        self.add_blank = add_blank

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        text = self._preprocess_char(text.lower())
        valid_chars = self.encoder
        filtered_text = "".join(list(filter(lambda char: char in valid_chars, text)))
        tokens = list(filtered_text.strip())

        if self.add_blank:
            interspersed = [self._convert_id_to_token(0)] * (len(tokens) * 2 + 1)
            interspersed[1::2] = tokens
            tokens = interspersed

        return tokens

    def _preprocess_char(self, text):
        """Special treatement of characters in certain languages"""
        if self.language == "ron":
            text = text.replace("ț", "ţ")
        return text

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)
