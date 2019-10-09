# coding=utf-8
# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" TF 2.0 CTRL model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import sys
from io import open
import numpy as np
import tensorflow as tf

from .configuration_ctrl import CTRLConfig
from .modeling_tf_utils import TFPreTrainedModel, get_initializer, shape_list, TFSharedEmbeddings
from .file_utils import add_start_docstrings
from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model

logger = logging.getLogger(__name__)

TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP = {"ctrl": "https://s3.amazonaws.com/models.huggingface.co/bert/ctrl-tf_model.h5"}

def load_ctrl_pt_weights_in_tf2(tf_model, pytorch_checkpoint_path):
    # build the network
    inputs_list = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
    tf_inputs = tf.constant(inputs_list)
    tfo = tf_model(tf_inputs, training=False)
    return load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=tf_inputs)


def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model_size))
    return pos * angle_rates

def positional_encoding(position, d_model_size):
    # create the sinusoidal pattern for the positional encoding
    angle_rads = angle_defn(np.arange(position)[:, np.newaxis],
                            np.arange(d_model_size)[np.newaxis, :],
                            d_model_size)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    # pos_encoding = tf.cast(np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...], dtype=tf.float32)
    pos_encoding = tf.cast(np.concatenate([sines, cosines], axis=-1), dtype=tf.float32)
    return pos_encoding

def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    # calculate attention
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(shape_list(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e4)

    if attention_mask is not None:
        # Apply the attention mask
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 

    # Mask heads if we want to
    if head_mask is not None:
        attention_weights = attention_weights * head_mask

    output = tf.matmul(attention_weights, v) 

    return output, attention_weights


class TFMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, output_attentions=False, **kwargs):
        super(TFMultiHeadAttention, self).__init__(**kwargs)
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        self.depth = int(d_model_size / self.num_heads)

        self.Wq = tf.keras.layers.Dense(d_model_size, name='Wq')
        self.Wk = tf.keras.layers.Dense(d_model_size, name='Wk')
        self.Wv = tf.keras.layers.Dense(d_model_size, name='Wv')

        self.dense = tf.keras.layers.Dense(d_model_size, name='dense')

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        v, k, q, mask, layer_past, attention_mask, head_mask = inputs
        batch_size = q.shape[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=1)
            k = tf.concat((past_key, k), dim=-2)
            v = tf.concat((past_value, v), dim=-2)
        present = tf.stack((k, v), axis=1)

        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = tf.transpose(output[0], perm=[0, 2, 1, 3])
        attn = output[1]
        original_size_attention = tf.reshape(scaled_attention,  (batch_size, -1, self.d_model_size))
        output = self.dense(original_size_attention)

        outputs = (output, present)
        if self.output_attentions:
            outputs = outputs + (attn,)
        return outputs



def point_wise_feed_forward_network(d_model_size, dff, name=""):
    return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu', name="0"), 
            tf.keras.layers.Dense(d_model_size, name="2")
        ], name="ffn")


class TFEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, dff, rate=0.1, layer_norm_epsilon=1e-6, output_attentions=False, **kwargs):
        super(TFEncoderLayer, self).__init__(**kwargs)

        self.multi_head_attention = TFMultiHeadAttention(d_model_size,
                                                         num_heads,
                                                         output_attentions,
                                                         name="multi_head_attention")
        self.ffn = point_wise_feed_forward_network(d_model_size, dff, name="ffn")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layernorm1")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layernorm2")

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        x, mask, layer_past, attention_mask, head_mask = inputs
        normed = self.layernorm1(x)
        attn_outputs = self.multi_head_attention([normed, normed, normed, mask, layer_past,
                                                  attention_mask, head_mask], training=training)
        attn_output = attn_outputs[0]
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        outputs = (out2,) + attn_outputs[1:]
        return outputs


class TFCTRLMainLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFCTRLMainLayer, self).__init__(**kwargs)
        self.output_hidden_states = config.output_hidden_states
        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer

        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size)

        self.output_attentions = config.output_attentions

        self.w = TFSharedEmbeddings(config.vocab_size,
                                    config.n_embd,
                                    initializer_range=config.initializer_range,
                                    name="w")

        self.dropout = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFEncoderLayer(config.n_embd,
                                 config.n_head,
                                 config.dff,
                                 config.resid_pdrop,
                                 config.layer_norm_epsilon,
                                 config.output_attentions,
                                 name='h_._{}'.format(i)) for i in range(config.n_layer)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="layernorm")

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
                heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    def call(self, inputs, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            past = inputs[1] if len(inputs) > 1 else past
            attention_mask = inputs[2] if len(inputs) > 2 else attention_mask
            token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
            position_ids = inputs[4] if len(inputs) > 4 else position_ids
            head_mask = inputs[5] if len(inputs) > 5 else head_mask
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            past = inputs.get('past', past)
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        input_shape = shape_list(input_ids)
        input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = shape_list(past[0][0])[-2]
        if position_ids is None:
            position_ids = tf.range(past_length, shape_list(input_ids)[-1] + past_length, dtype=tf.int32)[tf.newaxis, :]
            position_ids = tf.tile(position_ids, [shape_list(input_ids)[0], 1])

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.

            attention_mask = tf.cast(attention_mask, tf.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_layers

        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.w(token_type_ids, mode='embedding')
            token_type_embeds *= tf.math.sqrt(tf.cast(self.d_model_size, tf.float32))
        else:
            token_type_embeds = 0
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        inputs_embeds = self.w(input_ids, mode='embedding')
        # x = embedded.unsqueeze(0) if len(input_ids.shape)<2 else embedded
        seq_len = input_shape[-1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        inputs_embeds *= tf.math.sqrt(tf.cast(self.d_model_size, tf.float32))

        pos_embeds = tf.gather(self.pos_encoding, position_ids)

        hidden_states = inputs_embeds + pos_embeds + token_type_embeds

        hidden_states = self.dropout(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]
        presents = ()
        all_hidden_states = ()
        all_attentions = []
        for i, (h, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)
            outputs = h([hidden_states, mask, layer_past, attention_mask, head_mask[i]], training=training)
            hidden_states, present = outputs[:2]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.layernorm(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs


class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = CTRLConfig
    pretrained_model_archive_map = TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "transformer"
    load_pt_weights = load_ctrl_pt_weights_in_tf2


CTRL_START_DOCSTRING = r"""    CTRL model was proposed in 
    `CTRL: A Conditional Transformer Language Model for Controllable Generation`_
    by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
    It's a causal (unidirectional) transformer pre-trained using language modeling on a very large
    corpus of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`CTRL: A Conditional Transformer Language Model for Controllable Generation`:
        https://www.github.com/salesforce/ctrl

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.CTRLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

CTRL_INPUTS_DOCSTRING = r"""    Inputs:
        **input_ids**: ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            CTRL is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.CTRLTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **past**:
            list of ``Numpy array`` or ``tf.Tensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding.
        **attention_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **position_ids**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``Numpy array`` or ``tf.Tensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

@add_start_docstrings("The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
                                            CTRL_START_DOCSTRING, CTRL_INPUTS_DOCSTRING)
class TFCTRLModel(TFCTRLPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``tf.Tensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import CTRLTokenizer, TFCTRLModel

        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = TFCTRLModel.from_pretrained('ctrl')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFCTRLModel, self).__init__(config, *inputs, **kwargs)
        self.transformer = TFCTRLMainLayer(config, name='transformer')

    def call(self, inputs, **kwargs):
        outputs = self.transformer(inputs, **kwargs)
        return outputs


class TFCTRLLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super(TFCTRLLMHead, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')
        super(TFCTRLLMHead, self).build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@add_start_docstrings("""The CTRL Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """, CTRL_START_DOCSTRING, CTRL_INPUTS_DOCSTRING)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``tf.Tensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``tf.Tensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import CTRLTokenizer, TFCTRLLMHeadModel

        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = TFCTRLLMHeadModel.from_pretrained('ctrl')

        input_ids = torch.tensor(tokenizer.encode("Links Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFCTRLLMHeadModel, self).__init__(config, *inputs, **kwargs)
        self.transformer = TFCTRLMainLayer(config, name='transformer')

        self.lm_head = TFCTRLLMHead(config, self.transformer.w, name="lm_head")

    def call(self, inputs, **kwargs):
        transformer_outputs = self.transformer(inputs, **kwargs)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]

        return outputs  # lm_logits, presents, (all hidden_states), (attentions)
