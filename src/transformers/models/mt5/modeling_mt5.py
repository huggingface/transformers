# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch mT5 model. """

from ...utils import logging
from ..t5.modeling_t5 import T5ForConditionalGeneration, T5Model
from .configuration_mt5 import MT5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

####################################################
# This dict contains shortcut names and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/t5-small",
    "google/t5-base",
    "google/t5-large",
    "google/t5-3b",
    "google/t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


class MT5Model(T5Model):
    r"""
    This class overrides :class:`~transformers.T5Model`. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples::
        >>> from transformers import MT5Model, T5Tokenizer
        >>> model = MT5Model.from_pretrained("google/mt5-base")
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-base")
        >>> article = "UN Chief Says There Is No Military Solution in Syria"
        >>> batch = tokenizer.prepare_seq2seq_batch(src_texts=[article])
        >>> outputs = model(**batch)
        >>> last_hidden_states = outputs.last_hidden_states
    """
    model_type = "mt5"
    config_class = MT5Config
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]
    keys_to_never_save = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]


class MT5ForConditionalGeneration(T5ForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.T5ForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import MT5ForConditionalGeneration, T5Tokenizer
        >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-base")
        >>> article = "UN Chief Says There Is No Military Solution in Syria"
        >>> batch = tokenizer.prepare_seq2seq_batch(src_texts=[article])
        >>> generated_tokens = model.generate(**batch)
    """

    model_type = "mt5"
    config_class = MT5Config
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]
    keys_to_never_save = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
