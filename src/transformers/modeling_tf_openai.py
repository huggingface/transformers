# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
""" TF 2.0 OpenAI GPT model."""


import logging

import numpy as np
import tensorflow as tf

from .configuration_openai import OpenAIGPTConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_tf_utils import (
    TFConv1D,
    TFPreTrainedModel,
    TFSequenceSummary,
    TFSharedEmbeddings,
    get_initializer,
    shape_list,
)
from .tokenization_utils import BatchEncoding


logger = logging.getLogger(__name__)

TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP = {"openai-gpt": "https://cdn.huggingface.co/openai-gpt-tf_model.h5"}


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    return x * tf.math.sigmoid(x)


ACT_FNS = {
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
}


class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.n_ctx = n_ctx
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name="c_attn")
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_proj")
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        pass

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, inputs, training=False):
        q, k, v, attention_mask, head_mask = inputs
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], tf.float32)  # scale attention_scores
            w = w / tf.math.sqrt(dk)

        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = tf.nn.softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

    def call(self, inputs, training=False):
        x, attention_mask, head_mask = inputs

        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attn_outputs = self._attn([query, key, value, attention_mask, head_mask], training=training)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)


class TFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_fc")
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name="c_proj")
        self.act = gelu
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class TFBlock(tf.keras.layers.Layer):
    def __init__(self, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.attn = TFAttention(nx, n_ctx, config, scale, name="attn")
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        self.mlp = TFMLP(4 * nx, config, name="mlp")
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")

    def call(self, inputs, training=False):
        x, attention_mask, head_mask = inputs

        output_attn = self.attn([x, attention_mask, head_mask], training=training)
        a = output_attn[0]  # output_attn: a, (attentions)

        n = self.ln_1(x + a)
        m = self.mlp(n, training=training)
        h = self.ln_2(n + m)

        outputs = [h] + output_attn[1:]
        return outputs  # x, (attentions)


class TFOpenAIGPTMainLayer(tf.keras.layers.Layer):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.num_hidden_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd

        self.tokens_embed = TFSharedEmbeddings(
            config.vocab_size, config.n_embd, initializer_range=config.initializer_range, name="tokens_embed"
        )
        self.positions_embed = tf.keras.layers.Embedding(
            config.n_positions,
            config.n_embd,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="positions_embed",
        )
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFBlock(config.n_ctx, config, scale=True, name="h_._{}".format(i)) for i in range(config.n_layer)]

    def get_input_embeddings(self):
        return self.tokens_embed

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = tf.range(input_shape[-1], dtype=tf.int32)[tf.newaxis, :]

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
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids, mode="embedding")
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.tokens_embed(token_type_ids, mode="embedding")
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        all_attentions = []
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)

            outputs = block([hidden_states, attention_mask, head_mask[i]], training=training)
            hidden_states = outputs[0]
            if self.output_attentions:
                all_attentions.append(outputs[1])

        hidden_states = tf.reshape(hidden_states, output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (all hidden_states), (attentions)


class TFOpenAIGPTPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = OpenAIGPTConfig
    pretrained_model_archive_map = TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "transformer"


OPENAI_GPT_START_DOCSTRING = r"""

    .. note::
        TF 2.0 models accepts two formats as inputs:

            - having all inputs as keyword arguments (like PyTorch models), or
            - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :obj:`tf.keras.Model.fit()` method which currently requires having
        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors
        in the first positional argument :

        - a single Tensor with input_ids only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({'input_ids': input_ids, 'token_type_ids': token_type_ids})`


    Parameters:
        config (:class:`~transformers.OpenAIGPTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

OPENAI_GPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.GPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        training (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to activate dropout modules (if set to :obj:`True`) during training or to de-activate them
            (if set to :obj:`False`) for evaluation.
"""


@add_start_docstrings(
    "The bare OpenAI GPT transformer model outputing raw hidden-states without any specific head on top.",
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")

    @add_start_docstrings_to_callable(OPENAI_GPT_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.OpenAIGPTConfig`) and inputs:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
        hidden_states (:obj:`tuple(tf.Tensor)` `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import tensorflow as tf
        from transformers import OpenAIGPTTokenizer, TFOpenAIGPTModel

        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = TFOpenAIGPTModel.from_pretrained('openai-gpt')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        outputs = self.transformer(inputs, **kwargs)
        return outputs


@add_start_docstrings(
    """OpenAI GPT Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTLMHeadModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")

    def get_output_embeddings(self):
        return self.transformer.tokens_embed

    @add_start_docstrings_to_callable(OPENAI_GPT_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.OpenAIGPTConfig`) and inputs:
        prediction_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import tensorflow as tf
        from transformers import OpenAIGPTTokenizer, TFOpenAIGPTLMHeadModel

        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = TFOpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]

        """
        transformer_outputs = self.transformer(inputs, **kwargs)
        hidden_states = transformer_outputs[0]

        lm_logits = self.transformer.tokens_embed(hidden_states, mode="linear")

        outputs = (lm_logits,) + transformer_outputs[1:]

        return outputs  # lm_logits, (all hidden_states), (attentions)


@add_start_docstrings(
    """OpenAI GPT Model transformer with a language modeling and a multiple-choice classification
    head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
    The language modeling head has its weights tied to the input embeddings,
    the classification head takes as input the input of a specified classification token index in the input sequence).
""",
    OPENAI_GPT_START_DOCSTRING,
)
class TFOpenAIGPTDoubleHeadsModel(TFOpenAIGPTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config.num_labels = 1
        self.transformer = TFOpenAIGPTMainLayer(config, name="transformer")
        self.multiple_choice_head = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="multiple_choice_head"
        )

    def get_output_embeddings(self):
        return self.transformer.tokens_embed

    @add_start_docstrings_to_callable(OPENAI_GPT_INPUTS_DOCSTRING)
    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        training=False,
    ):
        r"""
        mc_token_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.

    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.OpenAIGPTConfig`) and inputs:
        lm_prediction_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.


    Examples::

        # For example purposes. Not runnable.
        import tensorflow as tf
        from transformers import OpenAIGPTTokenizer, TFOpenAIGPTDoubleHeadsModel

        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = TFOpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')

        # Add a [CLS] to the vocabulary (we should train it also!)
        # This option is currently not implemented in TF 2.0
        raise NotImplementedError
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        input_ids = tf.constant([tokenizer.encode(s) for s in choices])[None, :]  # Batch size 1, 2 choices
        mc_token_ids = tf.constant([input_ids.size(-1), input_ids.size(-1)])[None, :]  # Batch size 1
        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """

        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            mc_token_ids = inputs[6] if len(inputs) > 6 else mc_token_ids
            assert len(inputs) <= 7, "Too many inputs."
        elif isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            mc_token_ids = inputs.get("mc_token_ids", mc_token_ids)
            assert len(inputs) <= 7, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None:
            input_shapes = shape_list(input_ids)
        else:
            input_shapes = shape_list(inputs_embeds)[:-1]

        seq_length = input_shapes[-1]

        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None

        flat_inputs = [
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            head_mask,
            inputs_embeds,
        ]

        transformer_outputs = self.transformer(flat_inputs, training=training)
        hidden_states = transformer_outputs[0]

        hidden_states = tf.reshape(hidden_states, input_shapes + shape_list(hidden_states)[-1:])

        lm_logits = self.transformer.tokens_embed(hidden_states, mode="linear")
        mc_logits = self.multiple_choice_head([hidden_states, mc_token_ids], training=training)

        mc_logits = tf.squeeze(mc_logits, axis=-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]

        return outputs  # lm logits, mc logits, (all hidden_states), (attentions)
