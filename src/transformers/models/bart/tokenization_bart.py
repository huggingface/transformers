# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

from ...utils import logging
from ..roberta.tokenization_roberta import RobertaTokenizer


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# See all BART models at https://huggingface.co/models?filter=bart
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/vocab.json",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/vocab.json",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/vocab.json",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/vocab.json",
    },
    "merges_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/merges.txt",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/merges.txt",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/merges.txt",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/merges.txt",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/bart-base": 1024,
    "facebook/bart-large": 1024,
    "facebook/bart-large-mnli": 1024,
    "facebook/bart-large-cnn": 1024,
    "facebook/bart-large-xsum": 1024,
    "yjernite/bart_eli5": 1024,
}


class BartTokenizer(RobertaTokenizer):
    r"""
    Construct a BART tokenizer.

    :class:`~transformers.BartTokenizer` is identical to :class:`~transformers.RobertaTokenizer`. Refer to superclass
    :class:`~transformers.RobertaTokenizer` for usage examples and documentation concerning the initialization
    parameters and other methods.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
