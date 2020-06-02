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


import logging
import math

import numpy as np
import tensorflow as tf

from .configuration_distilbert import DistilBertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_tf_utils import TFPreTrainedModel, TFSharedEmbeddings, get_initializer, shape_list
from .tokenization_utils import BatchEncoding


logger = logging.getLogger(__name__)


TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    # See all DistilBERT models at https://huggingface.co/models?filter=distilbert
]


# UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE #
def gelu(x):
    """ Gaussian Error Linear Unit.
    Original Implementation of the gelu activation function in Google Bert repo when initially created.
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
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class TFEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.initializer_range = config.initializer_range
        self.word_embeddings = TFSharedEmbeddings(
            config.vocab_size, config.dim, initializer_range=config.initializer_range, name="word_embeddings"
        )  # padding_idx=0)
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.dim,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight", shape=[self.vocab_size, self.dim], initializer=get_initializer(self.initializer_range)
            )
        super().build(input_shape)

    def call(self, inputs, inputs_embeds=None, mode="embedding", training=False):
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
            return self._embedding(inputs, inputs_embeds=inputs_embeds, training=training)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs, inputs_embeds=None, training=False):
        """
        Parameters
        ----------
        input_ids: tf.Tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: tf.Tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        if not isinstance(inputs, (tuple, list)):
            input_ids = inputs
            position_ids = None
        else:
            input_ids, position_ids = inputs

        if input_ids is not None:
            seq_length = shape_list(input_ids)[1]
        else:
            seq_length = shape_list(inputs_embeds)[1]

        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings, training=training)  # (bs, max_seq_length, dim)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]

        x = tf.reshape(inputs, [-1, self.dim])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


class TFMultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
        self.output_attentions = config.output_attentions

        assert self.dim % self.n_heads == 0

        self.q_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="q_lin"
        )
        self.k_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="k_lin"
        )
        self.v_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="v_lin"
        )
        self.out_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="out_lin"
        )

        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        query: tf.Tensor(bs, seq_length, dim)
        key: tf.Tensor(bs, seq_length, dim)
        value: tf.Tensor(bs, seq_length, dim)
        mask: tf.Tensor(bs, seq_length)

        Outputs
        -------
        weights: tf.Tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: tf.Tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        query, key, value, mask, head_mask = inputs
        bs, q_length, dim = shape_list(query)
        k_length = shape_list(key)[1]
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshape = [bs, 1, 1, k_length]

        def shape(x):
            """ separate heads """
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            """ group heads """
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = tf.matmul(q, k, transpose_b=True)  # (bs, n_heads, q_length, k_length)
        mask = tf.reshape(mask, mask_reshape)  # (bs, n_heads, qlen, klen)
        # scores.masked_fill_(mask, -float('inf'))            # (bs, n_heads, q_length, k_length)
        scores = scores - 1e30 * (1.0 - mask)

        weights = tf.nn.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights, training=training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = tf.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if self.output_attentions:
            return (context, weights)
        else:
            return (context,)


class TFFFN(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.lin1 = tf.keras.layers.Dense(
            config.hidden_dim, kernel_initializer=get_initializer(config.initializer_range), name="lin1"
        )
        self.lin2 = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="lin2"
        )
        assert config.activation in ["relu", "gelu"], "activation ({}) must be in ['relu', 'gelu']".format(
            config.activation
        )
        self.activation = (
            tf.keras.layers.Activation(gelu) if config.activation == "gelu" else tf.keras.activations.relu
        )

    def call(self, input, training=False):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)
        return x


class TFTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

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
        x: tf.Tensor(bs, seq_length, dim)
        attn_mask: tf.Tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: tf.Tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: tf.Tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        x, attn_mask, head_mask = inputs

        # Self-Attention
        sa_output = self.attention([x, x, x, attn_mask, head_mask], training=training)
        if self.output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            # assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output, training=training)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class TFTransformer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.layer = [TFTransformerBlock(config, name="layer_._{}".format(i)) for i in range(config.n_layers)]

    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        x: tf.Tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: tf.Tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: tf.Tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[tf.Tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[tf.Tensor(bs, n_heads, seq_length, seq_length)]
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
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers

        self.embeddings = TFEmbeddings(config, name="embeddings")  # Embeddings
        self.transformer = TFTransformer(config, name="transformer")  # Encoder

    def get_input_embeddings(self):
        return self.embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def call(self, inputs, attention_mask=None, head_mask=None, inputs_embeds=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            head_mask = inputs[2] if len(inputs) > 2 else head_mask
            inputs_embeds = inputs[3] if len(inputs) > 3 else inputs_embeds
            assert len(inputs) <= 4, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 4, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.ones(input_shape)  # (bs, seq_length)
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(input_ids, inputs_embeds=inputs_embeds)  # (bs, seq_length, dim)
        tfmr_output = self.transformer([embedding_output, attention_mask, head_mask], training=training)

        return tfmr_output  # last-layer hidden-state, (all hidden_states), (all attentions)


# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #
class TFDistilBertPreTrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = DistilBertConfig
    base_model_prefix = "distilbert"


DISTILBERT_START_DOCSTRING = r"""
    This model is a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ sub-class.
    Use it as a regular TF 2.0 Keras Model and
    refer to the TF 2.0 documentation for all matter related to general usage and behavior.

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
        config (:class:`~transformers.DistilBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

DISTILBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, embedding_dim)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        training (:obj:`boolean`, `optional`, defaults to :obj:`False`):
            Whether to activate dropout modules (if set to :obj:`True`) during training or to de-activate them
            (if set to :obj:`False`) for evaluation.

"""


@add_start_docstrings(
    "The bare DistilBERT encoder/transformer outputing raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertModel(TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")  # Embeddings

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers,DistilBertConfig`) and inputs:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import DistilBertTokenizer, TFDistilBertModel

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = TFDistilBertModel.from_pretrained('distilbert-base-cased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """
        outputs = self.distilbert(inputs, **kwargs)
        return outputs


class TFDistilBertLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode="linear")
        hidden_states = hidden_states + self.bias
        return hidden_states


@add_start_docstrings(
    """DistilBert Model with a `masked language modeling` head on top. """, DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForMaskedLM(TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.vocab_size = config.vocab_size

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        self.vocab_transform = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="vocab_transform"
        )
        self.act = tf.keras.layers.Activation(gelu)
        self.vocab_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="vocab_layer_norm")
        self.vocab_projector = TFDistilBertLMHead(config, self.distilbert.embeddings, name="vocab_projector")

    def get_output_embeddings(self):
        return self.vocab_projector.input_embeddings

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""

    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers,DistilBertConfig`) and inputs:
        prediction_scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = TFDistilBertForMaskedLM.from_pretrained('distilbert-base-cased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        prediction_scores = outputs[0]

        """
        distilbert_output = self.distilbert(inputs, **kwargs)

        hidden_states = distilbert_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.act(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)

        outputs = (prediction_logits,) + distilbert_output[1:]
        return outputs  # logits, (hidden_states), (attentions)


@add_start_docstrings(
    """DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForSequenceClassification(TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        self.pre_classifier = tf.keras.layers.Dense(
            config.dim,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="relu",
            name="pre_classifier",
        )
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers,DistilBertConfig`) and inputs:
        logits (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]

        """
        distilbert_output = self.distilbert(inputs, **kwargs)

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        return outputs  # logits, (hidden_states), (attentions)


@add_start_docstrings(
    """DistilBert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForTokenClassification(TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers,DistilBertConfig`) and inputs:
        scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import DistilBertTokenizer, TFDistilBertForTokenClassification

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = TFDistilBertForTokenClassification.from_pretrained('distilbert-base-cased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        scores = outputs[0]
        """
        outputs = self.distilbert(inputs, **kwargs)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output, training=kwargs.get("training", False))
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # scores, (hidden_states), (attentions)


@add_start_docstrings(
    """DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForQuestionAnswering(TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        assert config.num_labels == 2
        self.dropout = tf.keras.layers.Dropout(config.qa_dropout)

    @add_start_docstrings_to_callable(DISTILBERT_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        r"""
    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers,DistilBertConfig`) and inputs:
        start_scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import tensorflow as tf
        from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased')
        input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        outputs = model(input_ids)
        start_scores, end_scores = outputs[:2]

        """
        distilbert_output = self.distilbert(inputs, **kwargs)

        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        hidden_states = self.dropout(hidden_states, training=kwargs.get("training", False))  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        outputs = (start_logits, end_logits,) + distilbert_output[1:]
        return outputs  # start_logits, end_logits, (hidden_states), (attentions)
