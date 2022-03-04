# coding=utf-8
# Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for LED."""

from typing import Dict, Optional, Union

from ...file_utils import PaddingStrategy
from ...tokenization_utils_base import BatchEncoding, EncodedInput
from ...utils import logging
from ..bart.tokenization_bart import BartTokenizer


logger = logging.get_logger(__name__)

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/vocab.json",
    },
    "merges_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/led-base-16384": 16384,
}


class LEDTokenizer(BartTokenizer):
    """
    Construct a LED tokenizer.

    [`LEDTokenizer`] is identical to [`BartTokenizer`] and runs end-to-end tokenization: punctuation splitting and
    wordpiece.

    Refer to superclass [`BartTokenizer`] for usage examples and documentation concerning parameters.
    """

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        return super()._pad(
            encoded_inputs=encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
