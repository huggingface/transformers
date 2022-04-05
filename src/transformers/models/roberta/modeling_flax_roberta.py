# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
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
from typing import Callable, Optional, Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen.attention import dot_product_attention_weights
from jax import lax
from jax.random import PRNGKey

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roberta import RobertaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = (input_ids != padding_idx).astype("i4")

    if mask.ndim > 2:
        mask = mask.reshape((-1, mask.shape[-1]))
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask
        incremental_indices = incremental_indices.reshape(input_ids.shape)
    else:
        incremental_indices = jnp.cumsum(mask, axis=1).astype("i4") * mask

    return incremental_indices.astype("i4") + padding_idx


ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        head_mask (`numpy.ndarray` of shape `({0})`, `optional):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEmbeddings with Bert->Roberta
class FlaxRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfAttention with Bert->Roberta
class FlaxRobertaSelfAttention(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`\
                    : {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        deterministic=True,
        output_attentions: bool = False,
    ):
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
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertSelfOutput with Bert->Roberta
class FlaxRobertaSelfOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertAttention with Bert->Roberta
class FlaxRobertaAttention(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self = FlaxRobertaSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxRobertaSelfOutput(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        deterministic=True,
        output_attentions: bool = False,
    ):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attn_outputs = self.self(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertIntermediate with Bert->Roberta
class FlaxRobertaIntermediate(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertOutput with Bert->Roberta
class FlaxRobertaOutput(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayer with Bert->Roberta
class FlaxRobertaLayer(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxRobertaAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxRobertaIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxRobertaOutput(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertLayerCollection with Bert->Roberta
class FlaxRobertaLayerCollection(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxRobertaLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for \
                        {head_mask.shape[0]}."
                )

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                layer_head_mask=head_mask[i] if head_mask is not None else None,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertEncoder with Bert->Roberta
class FlaxRobertaEncoder(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxRobertaLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertPooler with Bert->Roberta
class FlaxRobertaPooler(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


class FlaxRobertaLMHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.decoder = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN["gelu"](hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states


class FlaxRobertaClassificationHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.out_proj = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class FlaxRobertaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    module_class: nn.Module = None

    def __init__(
        self,
        config: RobertaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.ones_like(input_ids)
        position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)
        attention_mask = jnp.ones_like(input_ids)
        head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(
            rngs, input_ids, attention_mask, token_type_ids, position_ids, head_mask, return_dict=False
        )["params"]

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.config.pad_token_id)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        if head_mask is None:
            head_mask = jnp.ones((self.config.num_hidden_layers, self.config.num_attention_heads))

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
            jnp.array(head_mask, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertModule with Bert->Roberta
class FlaxRobertaModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxRobertaEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxRobertaEncoder(self.config, dtype=self.dtype)
        self.pooler = FlaxRobertaPooler(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        head_mask: Optional[np.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        hidden_states = self.embeddings(
            input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic
        )
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaModel(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaModule


append_call_sample_docstring(
    FlaxRobertaModel, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutputWithPooling, _CONFIG_FOR_DOC
)


class FlaxRobertaForMaskedLMModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roberta = FlaxRobertaModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        self.lm_head = FlaxRobertaLMHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.roberta.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class FlaxRobertaForMaskedLM(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForMaskedLMModule


append_call_sample_docstring(
    FlaxRobertaForMaskedLM,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutputWithPooling,
    _CONFIG_FOR_DOC,
    mask="<mask>",
)


class FlaxRobertaForSequenceClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roberta = FlaxRobertaModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.classifier = FlaxRobertaClassificationHead(config=self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, deterministic=deterministic)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForSequenceClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForSequenceClassificationModule


append_call_sample_docstring(
    FlaxRobertaForSequenceClassification,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForMultipleChoiceModule with Bert->Roberta, with self.bert->self.roberta
class FlaxRobertaForMultipleChoiceModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roberta = FlaxRobertaModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForMultipleChoice(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForMultipleChoiceModule


overwrite_call_docstring(
    FlaxRobertaForMultipleChoice, ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
append_call_sample_docstring(
    FlaxRobertaForMultipleChoice,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForTokenClassificationModule with Bert->Roberta, with self.bert->self.roberta
class FlaxRobertaForTokenClassificationModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roberta = FlaxRobertaModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForTokenClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForTokenClassificationModule


append_call_sample_docstring(
    FlaxRobertaForTokenClassification,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)


# Copied from transformers.models.bert.modeling_flax_bert.FlaxBertForQuestionAnsweringModule with Bert->Roberta, with self.bert->self.roberta
class FlaxRobertaForQuestionAnsweringModule(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.roberta = FlaxRobertaModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class FlaxRobertaForQuestionAnswering(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForQuestionAnsweringModule


append_call_sample_docstring(
    FlaxRobertaForQuestionAnswering,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)
