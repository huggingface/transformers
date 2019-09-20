# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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
""" TF 2.0 DistilBERT model
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import copy
import sys
from io import open

import itertools

import numpy as np
import tensorflow as tf

from .configuration_distilbert import DistilBertConfig
from .modeling_tf_utils import TFPreTrainedModel, TFSharedEmbeddings, shape_list
from .file_utils import add_start_docstrings
from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model

logger = logging.getLogger(__name__)


TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'distilbert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-tf_model.h5",
    'distilbert-base-uncased-distilled-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-tf_model.h5"
}


### UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE ###
def gelu(x):
    """ Gaussian Error Linear Unit.
    Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf

def gelu_new(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def load_distilbert_pt_weights_in_tf2(tf_model, pytorch_checkpoint_path):
    # build the network
    inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
    tf_inputs = [inputs_list, attns_list]
    tfo = tf_model(tf_inputs, training=False)
    return load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=tf_inputs)

class TFEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFEmbeddings, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.word_embeddings = TFSharedEmbeddings(, name='word_embeddings')  # padding_idx=0)
        self.position_embeddings = tf.keras.layers.Embedding(config.max_position_embeddings, config.dim, name='position_embeddings')
        if config.sinusoidal_embeddings:
            raise NotImplementedError

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self.hidden_size**-0.5))
        super(TFEmbeddings, self).build(input_shape)

    def call(self, inputs, mode="embedding", training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.
        
        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs, training=training)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs, training=False):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        input_ids, position_ids = inputs

        seq_length = tf.shape(input_ids)[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]

        words_embeddings = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings            # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)                       # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings, training=training)      # (bs, max_seq_length, dim)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


class TFMultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFMultiHeadSelfAttention, self).__init__(**kwargs)

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions

        assert self.dim % self.n_heads == 0

        self.q_lin = tf.keras.layers.Dense(config.dim, name="q_lin")
        self.k_lin = tf.keras.layers.Dense(config.dim, name="k_lin")
        self.v_lin = tf.keras.layers.Dense(config.dim, name="v_lin")
        self.out_lin = tf.keras.layers.Dense(config.dim, name="out_lin")

        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        query, key, value, mask, head_mask = inputs
        bs, q_length, dim = shape_list(query)
        k_length = shape_list(key)[1]
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        assert 2 <= len(tf.shape(mask)) <= 3
        causal = (len(tf.shape(mask)) == 3)
        mask_reshp = [bs, 1, 1, k_length]

        def shape(x):
            """ separate heads """
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            """ group heads """
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))

        q = shape(self.q_lin(query))           # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))             # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))           # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)                     # (bs, n_heads, q_length, dim_per_head)
        scores = tf.matmul(q, k, transpose_b=True)          # (bs, n_heads, q_length, k_length)
        mask = tf.reshape(mask, mask_reshape)                           # (bs, n_heads, qlen, klen)
        # scores.masked_fill_(mask, -float('inf'))            # (bs, n_heads, q_length, k_length)
        scores = scores - 1e30 * (1.0 - mask)

        weights = tf.nn.softmax(scores, axis=-1)                              # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights, training=training)                    # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = tf.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)             # (bs, q_length, dim)
        context = self.out_lin(context)        # (bs, q_length, dim)

        if self.output_attentions:
            return (context, weights)
        else:
            return (context,)

class TFFFN(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFFFN, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.lin1 = tf.keras.layers.Dense(config.hidden_dim, name="lin1")
        self.lin2 = tf.keras.layers.Dense(config.dim, name="lin2")
        assert config.activation in ['relu', 'gelu'], "activation ({}) must be in ['relu', 'gelu']".format(config.activation)
        self.activation = tf.keras.layers.Activation(gelu) if config.activation=='gelu' else tf.keras.activations.relu

    def call(self, input, training=False):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)
        return x


class TFTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFTransformerBlock, self).__init__(**kwargs)

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation = config.activation
        self.output_attentions = config.output_attentions

        assert config.dim % config.n_heads == 0

        self.attention = TFMultiHeadSelfAttention(config, name="attention")
        self.sa_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="sa_layer_norm")

        self.ffn = TFFFN(config, name="ffn")
        self.output_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="output_layer_norm")

    def call(self, inputs, training=False):  # removed: src_enc=None, src_len=None
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        x, attn_mask, head_mask = inputs

        # Self-Attention
        sa_output = self.attention([x, x, x, attn_mask, head_mask], training=training)
        if self.output_attentions:
            sa_output, sa_weights = sa_output                  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else: # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            # assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)          # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output, training=training)                             # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class TFTransformer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFTransformer, self).__init__(**kwargs)
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.layer = [TFTransformerBlock(config, name='layer_._{}'.format(i))
                      for i in range(config.n_layers)]

    def forward(self, inputs, training=False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        x, attn_mask, head_mask = inputs

        all_hidden_states = ()
        all_attentions = ()

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module([hidden_state, attn_mask, head_mask[i]], training=training)
            hidden_state = layer_outputs[-1]

            if self.output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class TFDistilBertMainLayer(tf.keras.layers.Layer):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config, **kwargs):
        super(TFDistilBertMainLayer, self).__init__(**kwargs)

        self.embeddings = TFEmbeddings(config, name="embeddings")   # Embeddings
        self.transformer = TFTransformer(config, name="transformer") # Encoder

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def call(self, inputs, training=False):
        if not isinstance(inputs, (dict, tuple, list)):
            input_ids = inputs
            (attention_mask, head_mask) = None, None
        elif isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else None
            head_mask = inputs[2] if len(inputs) > 2 else None
            assert len(inputs) <= 3, "Too many inputs."
        else:
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', None)
            head_mask = inputs.get('head_mask', None)
            assert len(inputs) <= 3, "Too many inputs."

        if attention_mask is None:
            attention_mask = tf.ones(shape_list(input_ids)) # (bs, seq_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids)   # (bs, seq_length, dim)
        tfmr_output = self.transformer([embedding_output, attention_mask, head_mask], training=training)

        hidden_state = tfmr_output[0]
        output = (hidden_state, ) + tfmr_output[1:]

        return output # last-layer hidden-state, (all hidden_states), (all attentions)


### INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL ###
class TFDistilBertPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = DistilBertConfig
    pretrained_model_archive_map = TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_pt_weights = load_distilbert_pt_weights_in_tf2
    base_model_prefix = "distilbert"


DISTILBERT_START_DOCSTRING = r"""
    DistilBERT is a small, fast, cheap and light Transformer model
    trained by distilling Bert base. It has 40% less parameters than
    `bert-base-uncased`, runs 60% faster while preserving over 95% of
    Bert's performances as measured on the GLUE language understanding benchmark.

    Here are the differences between the interface of Bert and DistilBert:

    - DistilBert doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. Just separate your segments with the separation token `tokenizer.sep_token` (or `[SEP]`)
    - DistilBert doesn't have options to select the input positions (`position_ids` input). This could be added if necessary though, just let's us know if you need this option.

    For more information on DistilBERT, please refer to our
    `detailed blog post`_
    
    .. _`detailed blog post`:
        https://medium.com/huggingface/distilbert-8cf3380435b5

    Parameters:
        config (:class:`~pytorch_transformers.DistilBertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

DISTILBERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids** ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The input sequences should start with `[CLS]` and end with `[SEP]` tokens.
            
            For now, ONLY BertTokenizer(`bert-base-uncased`) is supported and you should use this tokenizer when using DistilBERT.
        **attention_mask**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

@add_start_docstrings("The bare DistilBERT encoder/transformer outputing raw hidden-states without any specific head on top.",
                      DISTILBERT_START_DOCSTRING, DISTILBERT_INPUTS_DOCSTRING)
class DistilBertModel(DistilBertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config, *inputs, **kwargs):
        super(DistilBertModel, self).__init__(config, *inputs, **kwargs)
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")   # Embeddings

    def call(self, inputs, training=False):
        outputs = self.distilbert(inputs, training=training)
        return outputs


@add_start_docstrings("""DistilBert Model with a `masked language modeling` head on top. """,
                      DISTILBERT_START_DOCSTRING, DISTILBERT_INPUTS_DOCSTRING)
class TFDistilBertForMaskedLM(TFDistilBertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFDistilBertForMaskedLM, self).__init__(config, *inputs, **kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        self.vocab_transform = tf.keras.layers.Dense(config.dim, name="vocab_transform")
        self.act = tf.keras.layers.Activation(gelu)
        self.vocab_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="vocab_layer_norm")
        self.vocab_projector_weight = self.distilbert.embeddings

    def build(self, input_shape):
        self.vocab_projector_bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='vocab_projector_._bias')
        super(TFDistilBertForMaskedLM, self).build(input_shape)

    def call(self, inputs, training=False):
        dlbrt_output = self.distilbert(inputs, training=training)
        hidden_states = dlbrt_output[0]                              # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)      # (bs, seq_length, dim)
        prediction_logits = self.act(prediction_logits)                  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits) # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector_weight(prediction_logits, mode='linear') + self.vocab_projector_bias

        outputs = (prediction_logits, ) + dlbrt_output[1:]

        return outputs # prediction_logits, (all hidden_states), (all attentions)


@add_start_docstrings("""DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                         the pooled output) e.g. for GLUE tasks. """,
                      DISTILBERT_START_DOCSTRING, DISTILBERT_INPUTS_DOCSTRING)
class TFDistilBertForSequenceClassification(TFDistilBertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFDistilBertForSequenceClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.distilbert = TFDistilBertModel(config, name="distilbert")
        self.pre_classifier = tf.keras.layers.Dense(config.dim, activation='relu', name="pre_classifier")
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier")
        self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)

    def call(self, inputs, training=False):
        distilbert_output = self.distilbert(inputs, training=training)
        hidden_state = distilbert_output[0]                    # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]                    # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
        pooled_output = self.dropout(pooled_output, training=training)         # (bs, dim)
        logits = self.classifier(pooled_output)              # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        return outputs  # logits, (hidden_states), (attentions)


@add_start_docstrings("""DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
                         the hidden-states output to compute `span start logits` and `span end logits`). """,
                      DISTILBERT_START_DOCSTRING, DISTILBERT_INPUTS_DOCSTRING)
class TFDistilBertForQuestionAnswering(TFDistilBertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

    """
    def __init__(self, config, *inputs, **kwargs):
        super(TFDistilBertForQuestionAnswering, self).__init__(config, *inputs, **kwargs)

        self.distilbert = TFDistilBertModel(config, name="distilbert")
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, name='qa_output')
        assert config.num_labels == 2
        self.dropout = tf.keras.layers.Dropout(config.qa_dropout)

    def call(self, inputs, training=False):
        distilbert_output = self.distilbert(inputs, training=training)
        hidden_states = distilbert_output[0]                                 # (bs, max_query_len, dim)

        hidden_states = self.dropout(hidden_states, training=training)                       # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)                           # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        outputs = (start_logits, end_logits,) + distilbert_output[1:]
        return outputs  # start_logits, end_logits, (hidden_states), (attentions)
