# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for SpeechToTextTransformer."""
from ...utils import logging
from ..bart.tokenization_bart import BartTokenizer


logger = logging.get_logger(__name__)

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "s2t_transformer_s": "https://huggingface.co/s2t_transformer_s/resolve/main/vocab.json",
    },
    "merges_file": {
        "s2t_transformer_s": "https://huggingface.co/s2t_transformer_s/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "s2t_transformer_s": "https://huggingface.co/s2t_transformer_s/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "s2t_transformer_s": 1024,
}


class SpeechToTextTransformerTokenizer(BartTokenizer):
    """
    Construct a SpeechToTextTransformer tokenizer.

    :class:`~transformers.SpeechToTextTransformerTokenizer` is identical to :class:`~transformers.BartTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BartTokenizer` for usage examples and documentation concerning
    parameters.
    """

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
