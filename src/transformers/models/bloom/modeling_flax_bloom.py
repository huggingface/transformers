# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and Bigscience Workshop. All rights reserved.
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
""" Flax BLOOM model. """
# TODO: see todos throughout this file
# TODO: check correctness against pytorch implementation
# TODO: add unit tests
# TODO: add documentation / check that documentation is correct
# TODO: BLOOM_INPUTS_DOCSTRING might be wrong still (position_ids)
# TODO: check that code is jit-able

import math
from functools import partial
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.activation import tanh
from flax.linen.partitioning import scan_with_axes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutput,
)
from ...modeling_flax_utils import FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_bloom import BloomConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigscience/bloom"
_CONFIG_FOR_DOC = "BloomConfig"
_TOKENIZER_FOR_DOC = "BloomTokenizer"


def masked_fill(mask, a, fill):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))


def attention_mask_func(attention_scores, attention_mask, causal_mask):
    attention_mask_bool = ~(attention_mask)

    query_length, key_length, n_heads = attention_scores.shape[2], attention_scores.shape[3], attention_scores.shape[1]
    padded_causal_mask = jnp.logical_or(
        attention_mask_bool[:, None, key_length - query_length : key_length, None],
        ~(causal_mask[:, :, key_length - query_length : key_length, :key_length] == 1),
    )
    padded_causal_mask = jnp.logical_or(padded_causal_mask, attention_mask_bool[:, None, None, :key_length])
    # Make use of floats
    return (
        masked_fill(
            jnp.broadcast_to(
                ~padded_causal_mask,
                (padded_causal_mask.shape[0], n_heads, padded_causal_mask.shape[2], padded_causal_mask.shape[3]),
            ),
            attention_scores,
            -1e4,
        ),
        ~padded_causal_mask,
    )


BLOOM_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

BLOOM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BloomTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def flax_unsqueeze(x, axis):
    return jnp.expand_dims(x, axis)


def build_alibi_tensor_flax(max_seq_len, n_head, dtype):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = jnp.array(get_slopes(n_head))
    slopes = flax_unsqueeze(flax_unsqueeze(slopes, 1), 1)
    arange_tensor = flax_unsqueeze(flax_unsqueeze(jnp.arange(max_seq_len, dtype=dtype), 0), 0)

    alibi = slopes * jnp.broadcast_to(arange_tensor, (n_head, 1, arange_tensor.shape[-1]))
    return alibi


class FlaxBloomScaledSoftmax(nn.Module):
    config: BloomConfig
    mask_func: Callable
    softmax_in_fp32: bool
    """
    Scaled Softmax module. Also performs masking.
    Args:
        mask_func (`function`, *required*):
            mask function to be applied.
        softmax_in_fp32 (`bool`, *required*):
            if true, softmax in performed at fp32 precision.
        scale (`float`, *optional*):
            scaling factor used in input tensor scaling.
    """

    def setup(self):
        pass

    def __call__(self, input, mask, causal_mask, scale):
        input_dtype = input.dtype
        input_in_16bit = input_dtype in [jnp.float16, jnp.bfloat16]
        softmax_dtype = jnp.float32 if self.softmax_in_fp32 else input_dtype

        if scale is not None:
            input = input * scale

        if mask is None:
            mask = jnp.ones((input.shape[0], input.shape[1]), dtype=bool)

        mask_output, padded_causal_mask = self.mask_func(input, mask, causal_mask)
        mask_output.astype(softmax_dtype)
        probs = nn.softmax(mask_output, axis=-1) * (padded_causal_mask)

        if input_dtype != softmax_dtype:
            probs = probs.astype(input_dtype)

        return probs


class FlaxBloomAttention(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # TODO: make sure these affect behavior correctly
        self.pretraining_tp = self.config.pretraining_tp
        self.slow_but_exact = self.config.slow_but_exact

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        # TODO: deal with softmax
        self.attention_softmax_in_fp32 = self.config.attention_softmax_in_fp32
        # TODO: deal with hidden dropout
        self.hidden_dropout = self.config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by `num_heads` (got `hidden_size`: {self.hidden_size} and "
                f"`num_heads`: {self.num_heads})."
            )

        self.attn_dropout = nn.Dropout(self.config.attention_dropout)

        self.scale_mask_softmax = FlaxBloomScaledSoftmax(
            self.config,
            attention_mask_func,
            self.attention_softmax_in_fp32,
        )

        dense = partial(
            nn.Dense,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.query_key_value = dense(self.hidden_size * 3)
        self.dense = dense(self.hidden_size)
        self.attention_dropout = nn.Dropout(self.config.attention_dropout)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        residual,
        alibi,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        layer_number: int = None,
    ):
        # TODO: this module __call__ needs checking for correctness of implementation.
        fused_qkv = self.query_key_value(hidden_states)

        new_tensor_shape = fused_qkv.shape[:-1] + (self.num_heads, 3 * self.head_dim)
        fused_qkv = fused_qkv.reshape(new_tensor_shape)

        query, key, value = jnp.split(fused_qkv, 3, axis=-1)

        query_length, key_length = query.shape[1], key.shape[1]

        # TODO: check size of hidden_states to confirm this is the right dim for causal mask to use
        causal_mask = make_causal_mask(jnp.ones((1, hidden_states.shape[1]), dtype="bool"), dtype="bool")

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        if attention_mask is not None:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        attention_bias = None
        # transform boolean mask into float mask
        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e9).astype(self.dtype),
            )

        # Reshape input tensors
        output_size = (query.shape[0], query.shape[2], query.shape[1], key.shape[1])

        # [batch_size, q_length, num_heads, head_dim] -> [q_length, batch_size * num_heads, head_dim]
        query = jnp.transpose(query, (1, 0, 2, 3)).reshape(output_size[2], output_size[0] * output_size[1], -1)

        # [batch_size, k_length, num_heads, head_dim] -> [k_length, batch_size * num_heads, head_dim]
        key = jnp.transpose(key, (1, 0, 2, 3)).reshape(output_size[3], output_size[0] * output_size[1], -1)

        # Reshape according to batch size
        query = jnp.transpose(query, (1, 0, 2))
        key = jnp.transpose(key, (1, 2, 0))

        # Layer-wise attention scaling
        # layer_number matters for attn scaling and should not be 0. see `FlaxBloomAttention` for its use.
        layer_number = jnp.where(layer_number < 1, 1, layer_number)
        norm_factor = jnp.sqrt(self.head_dim).astype(self.dtype) * layer_number
        alpha = 1.0 / norm_factor
        beta = 1.0 / layer_number

        # usual dot product attention
        attn_weights = beta * alibi + alpha * jnp.matmul(query, key)
        attn_weights = attn_weights.reshape(output_size)

        # TODO: apply softmax to attention weights
        att_probs = self.scale_mask_softmax(attn_weights, attention_mask, causal_mask, layer_number)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        output_size = (value.shape[0], value.shape[2], query.shape[1], value.shape[3])

        value = jnp.transpose(value, (1, 0, 2, 3)).reshape(value.shape[1], output_size[0] * output_size[1], -1)
        attention_probs_reshaped = jnp.reshape(att_probs, (output_size[0] * output_size[1], output_size[2], -1))

        context = jnp.matmul(attention_probs_reshaped, jnp.transpose(value, (1, 0, 2)))
        context = context.reshape(output_size)
        context = jnp.transpose(context, (2, 0, 1, 3))

        # [q_length, batch_size, num_heads, head_dim] --> [q_length, batch_size, hidden_size]
        new_context_layer_shape = context.shape[:-2] + (self.hidden_size,)
        context = context.reshape(new_context_layer_shape)

        attn_output = self.dense(context)
        attn_output = jnp.transpose(attn_output, (1, 0, 2)) + residual

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class BloomGELU(nn.Module):
    def setup(self):
        self.dtype = jnp.float32

    def __call__(self, x):
        return x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


class FlaxBloomMLP(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden_size = self.config.hidden_size

        self.pretraining_tp = self.config.pretraining_tp
        self.slow_but_exact = self.config.slow_but_exact

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        self.dense_h_to_4h = nn.Dense(4 * hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        self.dense_4h_to_h = nn.Dense(hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        self.hidden_dropout = nn.Dropout(self.config.hidden_dropout)
        self.act = BloomGELU()

    def __call__(self, hidden_states, residual, deterministic: bool = True):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)

        # TODO: this code block is from the pytorch implementation. needs changing to work.
        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = jnp.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + nn.functional.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        intermediate_output = intermediate_output + residual
        hidden_states = self.hidden_dropout(intermediate_output, deterministic=deterministic)

        return hidden_states


class FlaxBloomBlock(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    def setup(self):
        self.input_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        self.self_attention = FlaxBloomAttention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        self.mlp = FlaxBloomMLP(self.config, dtype=self.dtype)

        self.apply_residual_connection_post_layernorm = self.config.apply_residual_connection_post_layernorm
        self.hidden_dropout = self.config.hidden_dropout

    def __call__(
        self,
        hidden_states,
        alibi,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        use_cache: bool = False,
        layer_number: int = None,
    ):
        if self.use_scan:
            hidden_states = hidden_states[0]

        layernorm_output = self.input_layernorm(hidden_states)
        # layer norm before saving residual if config calls for it
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # self-attention
        attn_outputs = self.self_attention(
            layernorm_output,
            residual=residual,
            alibi=alibi,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            layer_number=layer_number,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        post_layernorm = self.post_attention_layernorm(attention_output)

        # set residual based on config
        if self.apply_residual_connection_post_layernorm:
            residual = post_layernorm
        else:
            residual = attention_output

        output = self.mlp(post_layernorm, residual, deterministic=deterministic)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        if self.use_scan:
            outputs = (outputs, None)

        return outputs


# TODO: does this still require position_ids?
# TODO: gradient checkpointing
# TODO: _no_split_modules?
# TODO: check initialization
class FlaxBloomPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BloomConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: BloomConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        use_scan: bool = False,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, use_scan=use_scan, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    # TODO: check whether this is correct (position ids might not be required)
    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        past_key_values: dict = None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, sequence_length = input_ids.shape

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxBloomAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False
        # TODO: build alibi here?
        # TODO: check the inputs and their order to this
        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxBloomBlockCollection(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    @nn.compact
    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.use_scan:
            # since all decoder layers are the same, we use nn.scan directly
            # assert not output_attentions, "cannot use `scan` with `output_attentions` set to `True`"
            # assert not output_hidden_states, "cannot use `scan` with `output_hidden_states` set to `True`"
            hidden_states = (hidden_states,)

            hidden_states, _ = scan_with_axes(
                FlaxBloomBlock,
                variable_axes={"params": 0, "cache": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, 0),
                length=self.config.num_hidden_layers,
            )(self.config, dtype=self.dtype, use_scan=True, name="FlaxBloomBlockLayers")(
                hidden_states,
                alibi,
                None,  # kwargs not supported by scan
                None,
                None,
                deterministic,
                init_cache,
                output_attentions,
                False,
                layer_number=jnp.arange(self.config.num_hidden_layers),
            )
            hidden_states = hidden_states[0]

        else:
            for layer_number in range(self.config.num_hidden_layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = FlaxBloomBlock(self.config, name=str(layer_number), dtype=self.dtype, use_scan=False)(
                    hidden_states,
                    alibi,
                    attention_mask,
                    deterministic=deterministic,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    layer_number=layer_number,
                )
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxBloomModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxBloomModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    def setup(self):
        # TODO: check initialization correctness
        self.embed_dim = self.config.hidden_size

        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)

        # word embeddings (no positional embedding layer) TODO: confirm this statement correct
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=embedding_init,
        )
        # post-embedding layernorm
        self.word_embeddings_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)

        # transformer layers
        self.h = FlaxBloomBlockCollection(self.config, dtype=self.dtype, use_scan=self.use_scan)

        # final layernorm
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        # TODO: change how gradient checkpointing is done
        self.gradient_checkpointing = False

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # do post-embedding layernorm
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        curr_seq_len = hidden_states.shape[1]
        alibi = build_alibi_tensor_flax(curr_seq_len, self.config.n_head, hidden_states.dtype)

        past_key_values = () if use_cache else None  # TODO: come back to this line
        # TODO: how to handle alibi? build alibi tensor here?

        # TODO: fix inputs to this (and args to submodules in general)
        # TODO: gradient checkpointing
        outputs = self.h(
            hidden_states,
            alibi=alibi,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            # TODO: don't think this return value / ordering is correct
            return tuple(v for v in [outputs[0], past_key_values, outputs[-1]] if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


@add_start_docstrings(
    "The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.",
    BLOOM_START_DOCSTRING,
)
# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoModel with GPTNeo->Bloom
class FlaxBloomModel(FlaxBloomPreTrainedModel):
    module_class = FlaxBloomModule


append_call_sample_docstring(
    FlaxBloomModel, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC
)


# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoForCausalLMModule with GPTNeo->Bloom
class FlaxBloomForCausalLMModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    def setup(self):
        self.transformer = FlaxBloomModule(self.config, dtype=self.dtype, use_scan=self.use_scan)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["word_embeddings"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    BLOOM_START_DOCSTRING,
)
# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoForCausalLM with GPTNeo->Bloom
class FlaxBloomForCausalLM(FlaxBloomPreTrainedModel):
    module_class = FlaxBloomForCausalLMModule
    # TODO: check if this class is correct / take out position ids
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Bloom uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


append_call_sample_docstring(
    FlaxBloomForCausalLM, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC
)
