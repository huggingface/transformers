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
"""TF BART model, ported from the fairseq repo."""

import math
import random
import warnings
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Dense, LayerNormalization

from .activations_tf import ACT2FN
from .configuration_bart import BartConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPast, TFSeq2SeqLMOutput, TFSeq2SeqModelOutput
from .modeling_tf_bart import TFBartForConditionalGeneration
# Public API
from .modeling_tf_utils import (
    DUMMY_INPUTS,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    cast_bool_to_primitive,
    keras_serializable,
    shape_list,
)
from .tokenization_utils_base import BatchEncoding
from .utils import logging


_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

BART_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass.
    Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general
    usage and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the model weights.
"""


BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for translation and summarization training. By default, the model will create this tensor by
            shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.
        encoder_outputs (:obj:`tf.FloatTensor`, `optional`):
            hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            of shape :obj:`(batch_size, sequence_length, hidden_size)` is a sequence of
        past_key_values (:obj:`Tuple[Dict[str: tf.Tensor]]` of length :obj:`config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`). Set to :obj:`False` during training, :obj:`True` during generation
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.TFModelOutput` instead of a plain tuple.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""
LARGE_NEGATIVE = -1e8


logger = logging.get_logger(__name__)
@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.",
    BART_START_DOCSTRING,
)
class TFMarianMTModel(TFBartForConditionalGeneration):
    base_model_prefix = "model"
    authorized_missing_keys = [
        r"encoder\.version",
        r"decoder\.version",
        "model.encoder.embed_tokens.weight",
        "model.decoder.embed_tokens.weight",
    ]
    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        # THIS WILL BREAK IN TF!
        logits[:, self.config.pad_token_id] = float("-inf")  # never predict pad token.
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            logits = self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits
