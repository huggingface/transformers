# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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

from .tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from .utils import logging


logger = logging.get_logger(__name__)


# vocab and merges same as roberta
vocab_url = "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json"
merges_url = "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt"
_all_longformer_models = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "allenai/longformer-base-4096-extra.pos.embd.only",
    "allenai/longformer-large-4096-extra.pos.embd.only",
]


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/longformer-base-4096": 4096,
    "allenai/longformer-large-4096": 4096,
    "allenai/longformer-large-4096-finetuned-triviaqa": 4096,
    "allenai/longformer-base-4096-extra.pos.embd.only": 4096,
    "allenai/longformer-large-4096-extra.pos.embd.only": 4096,
}


class LongformerTokenizer(RobertaTokenizer):
    r"""
    Construct a Longformer tokenizer.

    :class:`~transformers.LongformerTokenizer` is identical to :class:`~transformers.RobertaTokenizer`. Refer to
    the superclass for usage examples and documentation concerning parameters.
    """
    # merges and vocab same as Roberta
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = {
        "vocab_file": {m: vocab_url for m in _all_longformer_models},
        "merges_file": {m: merges_url for m in _all_longformer_models},
    }


class LongformerTokenizerFast(RobertaTokenizerFast):
    r"""
    Construct a "fast" Longformer tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.LongformerTokenizerFast` is identical to :class:`~transformers.RobertaTokenizerFast`. Refer
    to the superclass for usage examples and documentation concerning parameters.
    """
    # merges and vocab same as Roberta
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = {
        "vocab_file": {m: vocab_url for m in _all_longformer_models},
        "merges_file": {m: merges_url for m in _all_longformer_models},
    }
