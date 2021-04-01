# coding=utf-8
# Copyright 2018 The Google Flax Team Authors and The HuggingFace Inc. team.
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

from typing import Callable, Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen import dot_product_attention
from jax import lax
from jax.random import PRNGKey

from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from ...utils import logging
from .configuration_bert import BertConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.FlaxPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading, saving and converting weights from
    PyTorch models)

    This model is also a Flax Linen `flax.nn.Module
    <https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html>`__ subclass. Use it as a regular Flax
    Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - `Just-In-Time (JIT) compilation <https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit>`__
    - `Automatic Differentiation <https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation>`__
    - `Vectorization <https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap>`__
    - `Parallelization <https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap>`__

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`numpy.ndarray` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class FlaxBertLayerNorm(nn.Module):
    """
    Layer normalization (https://arxiv.org/abs/1607.06450). Operates on the last axis of the input data.
    """

    hidden_size: int
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    scale: bool = True
    scale_init: Callable[..., np.ndarray] = jax.nn.initializers.ones
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.weight = self.param("weight", self.scale_init, (self.hidden_size,))
        self.bias = self.param("bias", self.scale_init, (self.hidden_size,))

    def __call__(self, x):
        """
        Applies layer normalization on the input. It normalizes the activations of the layer for each given example in
        a batch independently, rather than across a batch like Batch Normalization. i.e. applies a transformation that
        maintains the mean activation within each example close to 0 and the activation standard deviation close to 1

        Args:
          x: the inputs

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        var = mean2 - jax.lax.square(mean)
        mul = jax.lax.rsqrt(var + self.epsilon)

        if self.scale:
            mul = mul * jnp.asarray(self.weight)
        y = (x - mean) * mul

        if self.use_bias:
            y = y + jnp.asarray(self.bias)
        return y


class FlaxBertEmbedding(nn.Module):
    """
    Specify a new class for doing the embedding stuff as Flax's one use 'embedding' for the parameter name and PyTorch
    use 'weight'
    """

    vocab_size: int
    hidden_size: int
    initializer_range: float
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        init_fn: Callable[..., np.ndarray] = jax.nn.initializers.normal(stddev=self.initializer_range)
        self.embeddings = self.param("weight", init_fn, (self.vocab_size, self.hidden_size))

    def __call__(self, input_ids):
        return jnp.take(self.embeddings, input_ids, axis=0)


class FlaxBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = FlaxBertEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            initializer_range=self.config.initializer_range,
            dtype=self.dtype,
        )
        self.position_embeddings = FlaxBertEmbedding(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            initializer_range=self.config.initializer_range,
            dtype=self.dtype,
        )
        self.token_type_embeddings = FlaxBertEmbedding(
            self.config.type_vocab_size,
            self.config.hidden_size,
            initializer_range=self.config.initializer_range,
            dtype=self.dtype,
        )
        self.LayerNorm = FlaxBertLayerNorm(hidden_size=self.config.hidden_size, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(jnp.atleast_2d(input_ids.astype("i4")))
        position_embeds = self.position_embeddings(jnp.atleast_2d(position_ids.astype("i4")))
        token_type_embeddings = self.token_type_embeddings(jnp.atleast_2d(token_type_ids.astype("i4")))

        # Sum all embeddings
        hidden_states = inputs_embeds + jnp.broadcast_to(position_embeds, inputs_embeds.shape) + token_type_embeddings

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxBertSelfAttention(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`: {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )

    def __call__(self, hidden_states, attention_mask, deterministic=True):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_output = dot_product_attention(
            query_states,
            key_states,
            value_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        return attn_output.reshape(attn_output.shape[:2] + (-1,))


class FlaxBertSelfOutput(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.LayerNorm = FlaxBertLayerNorm(hidden_size=self.config.hidden_size)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FlaxBertAttention(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self = FlaxBertSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxBertSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, deterministic=True):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attn_output = self.self(hidden_states, attention_mask, deterministic=deterministic)
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxBertIntermediate(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxBertOutput(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = FlaxBertLayerNorm(hidden_size=self.config.hidden_size, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxBertLayer(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxBertAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxBertIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBertOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        attention_output = self.attention(hidden_states, attention_mask, deterministic=deterministic)
        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)
        return hidden_states


class FlaxBertLayerCollection(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxBertLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, deterministic=deterministic)
        return hidden_states


class FlaxBertEncoder(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxBertLayerCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, deterministic: bool = True):
        return self.layer(hidden_states, attention_mask, deterministic=deterministic)


class FlaxBertPooler(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


class FlaxBertPredictionHeadTransform(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = FlaxBertLayerNorm(hidden_size=self.config.hidden_size, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)


class FlaxBertLMPredictionHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transform = FlaxBertPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class FlaxBertOnlyMLMHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBertLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.predictions(hidden_states)
        return hidden_states


class FlaxBertOnlyNSPHead(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, pooled_output):
        return self.seq_relationship(pooled_output)


class FlaxBertPreTrainingHeads(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBertLMPredictionHead(self.config, dtype=self.dtype)
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, hidden_states, pooled_output):
        prediction_scores = self.predictions(hidden_states)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FlaxBertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"

    def _check_inputs(self, input_ids, attention_mask, token_type_ids, position_ids):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)

        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        return input_ids, attention_mask, token_type_ids, position_ids

    def init(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            jnp.zeros(input_shape, dtype="i4"), None, None, None
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids)["params"]


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class FlaxBertModel(FlaxBertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxBertModule(config=config, dtype=dtype, **kwargs)

        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxBertModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBertEncoder(self.config, dtype=self.dtype)
        self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = True):

        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        hidden_states = self.encoder(hidden_states, attention_mask, deterministic=deterministic)

        if not self.add_pooling_layer:
            return hidden_states

        pooled = self.pooler(hidden_states)
        return hidden_states, pooled


@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForPreTraining(FlaxBertPreTrainedModel):
    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxBertForPreTrainingModule(config, **kwargs)

        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxBertForPreTrainingModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config, dtype=self.dtype)
        self.cls = FlaxBertPreTrainingHeads(config=self.config, dtype=self.dtype)

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        # Model
        hidden_states, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic
        )
        prediction_scores, seq_relationship_score = self.cls(hidden_states, pooled_output)

        return (prediction_scores, seq_relationship_score)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class FlaxBertForMaskedLM(FlaxBertPreTrainedModel):
    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxBertForMaskedLMModule(config, **kwargs)

        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxBertForMaskedLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        self.cls = FlaxBertOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        # Model
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic)

        # Compute the prediction scores
        logits = self.cls(hidden_states)

        return (logits,)


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top. """,
    BERT_START_DOCSTRING,
)
class FlaxBertForNextSentencePrediction(FlaxBertPreTrainedModel):
    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxBertForNextSentencePredictionModule(config, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxBertForNextSentencePredictionModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config, dtype=self.dtype)
        self.cls = FlaxBertOnlyNSPHead(dtype=self.dtype)

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        # Model
        _, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic
        )

        seq_relationship_scores = self.cls(pooled_output)
        return (seq_relationship_scores,)


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForSequenceClassification(FlaxBertPreTrainedModel):
    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxBertForSequenceClassificationModule(config, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxBertForSequenceClassificationModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        # Model
        _, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic
        )

        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        return (logits,)


@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForMultipleChoice(FlaxBertPreTrainedModel):
    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxBertForMultipleChoiceModule(config, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxBertForMultipleChoiceModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # Model
        _, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic
        )

        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        return (reshaped_logits,)


@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForTokenClassification(FlaxBertPreTrainedModel):
    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxBertForTokenClassificationModule(config, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxBertForTokenClassificationModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        # Model
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic)

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        return (logits,)


@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class FlaxBertForQuestionAnswering(FlaxBertPreTrainedModel):
    def __init__(
        self, config: BertConfig, input_shape: Tuple = (1, 1), seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs
    ):
        module = FlaxBertForQuestionAnsweringModule(config, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
    ):
        input_ids, attention_mask, token_type_ids, position_ids = self._check_inputs(
            input_ids, attention_mask, token_type_ids, position_ids
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            rngs=rngs,
        )


class FlaxBertForQuestionAnsweringModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, deterministic: bool = True
    ):
        # Model
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic)

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return (start_logits, end_logits)
