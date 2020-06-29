# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
""" TF 2.0 Flaubert model.
"""

import logging
import random

import tensorflow as tf

from .configuration_flaubert import FlaubertConfig
from .file_utils import add_start_docstrings
from .modeling_tf_utils import keras_serializable, shape_list
from .modeling_tf_xlm import (
    TFXLMForMultipleChoice,
    TFXLMForQuestionAnsweringSimple,
    TFXLMForSequenceClassification,
    TFXLMForTokenClassification,
    TFXLMMainLayer,
    TFXLMModel,
    TFXLMWithLMHeadModel,
    get_masks,
)
from .tokenization_utils import BatchEncoding


logger = logging.getLogger(__name__)

TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all Flaubert models at https://huggingface.co/models?filter=flaubert
]

FLAUBERT_START_DOCSTRING = r"""

    This model is a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ sub-class.
    Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.FlaubertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

FLAUBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        langs (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            A parallel sequence of tokens to be used to indicate the language of each token in the input.
            Indices are languages ids which can be obtained from the language names by using two conversion mappings
            provided in the configuration of the model (only provided for multilingual models).
            More precisely, the `language name -> language id` mapping is in `model.config.lang2id` (dict str -> int) and
            the `language id -> language name` mapping is `model.config.id2lang` (dict int -> str).
            See usage examples detailed in the `multilingual documentation <https://huggingface.co/transformers/multilingual.html>`__.
        token_type_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        lengths (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Length of each sentence that can be used to avoid performing attention on padding token indices.
            You can also use `attention_mask` for the same result (see above), kept here for compatbility.
            Indices selected in ``[0, ..., input_ids.size(-1)]``:
        cache (:obj:`Dict[str, tf.Tensor]`, `optional`, defaults to :obj:`None`):
            dictionary with ``tf.Tensor`` that contains pre-computed
            hidden-states (key and values in the attention blocks) as computed by the model
            (see `cache` output below). Can be used to speed up sequential decoding.
            The dictionary object will be modified in-place during the forward pass to add newly computed hidden-states.
        head_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
"""


@add_start_docstrings(
    "The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertModel(TFXLMModel):
    config_class = FlaubertConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name="transformer")


@keras_serializable
class TFFlaubertMainLayer(TFXLMMainLayer):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.layerdrop = getattr(config, "layerdrop", 0.0)
        self.pre_norm = getattr(config, "pre_norm", False)

    def call(
        self,
        inputs,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        # removed: src_enc=None, src_len=None
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            langs = inputs[2] if len(inputs) > 2 else langs
            token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
            position_ids = inputs[4] if len(inputs) > 4 else position_ids
            lengths = inputs[5] if len(inputs) > 5 else lengths
            cache = inputs[6] if len(inputs) > 6 else cache
            head_mask = inputs[7] if len(inputs) > 7 else head_mask
            inputs_embeds = inputs[8] if len(inputs) > 8 else inputs_embeds
            assert len(inputs) <= 9, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            langs = inputs.get("langs", langs)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            lengths = inputs.get("lengths", lengths)
            cache = inputs.get("cache", cache)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 9, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            bs, slen = shape_list(input_ids)
        elif inputs_embeds is not None:
            bs, slen = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if lengths is None:
            if input_ids is not None:
                lengths = tf.reduce_sum(tf.cast(tf.not_equal(input_ids, self.pad_index), dtype=tf.int32), axis=1)
            else:
                lengths = tf.convert_to_tensor([slen] * bs, tf.int32)
        # mask = input_ids != self.pad_index

        # check inputs
        # assert shape_list(lengths)[0] == bs
        tf.debugging.assert_equal(shape_list(lengths)[0], bs)
        # assert lengths.max().item() <= slen
        # input_ids = input_ids.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        # if self.is_decoder and src_enc is not None:
        #     src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # position_ids
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(slen), axis=0)
        else:
            # assert shape_list(position_ids) == [bs, slen]  # (slen, bs)
            tf.debugging.assert_equal(shape_list(position_ids), [bs, slen])
            # position_ids = position_ids.transpose(0, 1)

        # langs
        if langs is not None:
            # assert shape_list(langs) == [bs, slen]  # (slen, bs)
            tf.debugging.assert_equal(shape_list(langs), [bs, slen])
            # langs = langs.transpose(0, 1)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x qlen x klen]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layers

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _slen = slen - cache["slen"]
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = self.dropout(tensor, training=training)
        tensor = tensor * mask[..., tf.newaxis]

        # transformer layers
        hidden_states = ()
        attentions = ()
        for i in range(self.n_layers):
            # LayerDrop
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):
                continue

            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            if not self.pre_norm:
                attn_outputs = self.attentions[i]([tensor, attn_mask, None, cache, head_mask[i]], training=training)
                attn = attn_outputs[0]
                attentions = attentions + (attn_outputs[1],)
                attn = self.dropout(attn, training=training)
                tensor = tensor + attn
                tensor = self.layer_norm1[i](tensor)
            else:
                tensor_normalized = self.layer_norm1[i](tensor)
                attn_outputs = self.attentions[i](
                    [tensor_normalized, attn_mask, None, cache, head_mask[i]], training=training
                )
                attn = attn_outputs[0]
                if output_attentions:
                    attentions = attentions + (attn_outputs[1],)
                attn = self.dropout(attn, training=training)
                tensor = tensor + attn

            # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            #     attn = F.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            if not self.pre_norm:
                tensor = tensor + self.ffns[i](tensor)
                tensor = self.layer_norm2[i](tensor)
            else:
                tensor_normalized = self.layer_norm2[i](tensor)
                tensor = tensor + self.ffns[i](tensor_normalized)

            tensor = tensor * mask[..., tf.newaxis]

        # Add last hidden state
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache["slen"] += tensor.size(1)

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        outputs = (tensor,)
        if output_hidden_states:
            outputs = outputs + (hidden_states,)
        if output_attentions:
            outputs = outputs + (attentions,)
        return outputs  # outputs, (hidden_states), (attentions)


@add_start_docstrings(
    """The Flaubert Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertWithLMHeadModel(TFXLMWithLMHeadModel):
    config_class = FlaubertConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name="transformer")


@add_start_docstrings(
    """Flaubert Model with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertForSequenceClassification(TFXLMForSequenceClassification):
    config_class = FlaubertConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name="transformer")


@add_start_docstrings(
    """Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertForQuestionAnsweringSimple(TFXLMForQuestionAnsweringSimple):
    config_class = FlaubertConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name="transformer")


@add_start_docstrings(
    """Flaubert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertForTokenClassification(TFXLMForTokenClassification):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name="transformer")


@add_start_docstrings(
    """Flaubert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    FLAUBERT_START_DOCSTRING,
)
class TFFlaubertForMultipleChoice(TFXLMForMultipleChoice):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name="transformer")
