# coding=utf-8
# Copyright Facebook and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ESM."""
from typing import List, Optional

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from ..gpt2.tokenization_gpt2 import GPT2Tokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/esm1b": "https://huggingface.co/facebook/esm1b/resolve/main/vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/esm1b": 1024,
}


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]


class ESMTokenizer(PreTrainedTokenizer):
    """
    Constructs a ESM tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = {ind: tok for ind, tok in enumerate(self.all_tokens)}
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}
        self.unk_token = "<unk>"
        self.cls_token = "<cls>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.eos_token = "<eos>"
        self.unique_no_split_tokens = self.all_tokens

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        return text.split()

    def get_vocab_size(self, with_added_tokens=False):
        return len(self._id_to_token)

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)
