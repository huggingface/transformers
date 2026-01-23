# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. Team and AIRAS Inc. Team. All rights reserved.
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
from typing import List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers import AutoTokenizer

logger = logging.get_logger(__name__)

SAPNOUS_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Sapnous-AI/Sapnous-VR-6B": "https://huggingface.co/Sapnous-AI/Sapnous-VR-6B/resolve/main/vocab.json",
    },
    "merges_file": {
        "Sapnous-AI/Sapnous-VR-6B": "https://huggingface.co/Sapnous-AI/Sapnous-VR-6B/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Sapnous-AI/Sapnous-VR-6B": 128000,
}

class SapnousT1Tokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    pretrained_vocab_files_map = SAPNOUS_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        **kwargs
    ):
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.add_prefix_space = add_prefix_space

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text: str) -> List[str]:
        """ Tokenize a string. """
        raise NotImplementedError("Implement in subclass")

    def _convert_token_to_id(self, token: str) -> int:
        """ Converts a token to an id using the vocab. """
        raise NotImplementedError("Implement in subclass")

    def _convert_id_to_token(self, index: int) -> str:
        """ Converts an index (integer) to a token. """
        raise NotImplementedError("Implement in subclass")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        """ Save the vocabulary and special tokens file to a directory. """
        raise NotImplementedError("Implement in subclass")

AutoTokenizer.register(SapnousT1Config, SapnousT1Tokenizer)