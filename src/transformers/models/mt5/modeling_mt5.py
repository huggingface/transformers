# coding=utf-8
# Copyright 2020 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
from ..t5.modeling_t5 import T5EncoderModel, T5ForConditionalGeneration, T5Model
from .configuration_mt5 import MT5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"


class MT5Model(T5Model):
    r"""
    This class overrides [`T5Model`]. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples:

    ```python
    >>> from transformers import MT5Model, T5Tokenizer
    >>> model = MT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="pt")
    >>> with tokenizer.as_target_tokenizer():
    ...     labels = tokenizer(summary, return_tensors="pt")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]


class MT5ForConditionalGeneration(T5ForConditionalGeneration):
    r"""
    This class overrides [`T5ForConditionalGeneration`]. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples:

    ```python
    >>> from transformers import MT5ForConditionalGeneration, T5Tokenizer
    >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="pt")
    >>> with tokenizer.as_target_tokenizer():
    ...     labels = tokenizer(summary, return_tensors="pt")

    >>> outputs = model(**inputs,labels=labels["input_ids"])
    >>> loss = outputs.loss
    ```"""

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
    ]


class MT5EncoderModel(T5EncoderModel):
    r"""
    This class overrides [`T5EncoderModel`]. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples:

    ```python
    >>> from transformers import MT5EncoderModel, T5Tokenizer
    >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
    >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
    >>> outputs = model(input_ids)
    >>> hidden_state = outputs.last_hidden_state
    ```"""

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
    ]
