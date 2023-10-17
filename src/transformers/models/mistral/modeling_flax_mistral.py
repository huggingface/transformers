# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" Flax Mistral model."""
import math
from typing import List, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import torch
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.initializers import ones
from flax.traverse_util import flatten_dict, unflatten_dict

from ...modeling_flax_outputs import (FlaxBaseModelOutputWithPast,
                                      FlaxCausalLMOutputWithCrossAttentions,
                                      FlaxSequenceClassifierOutput)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, logging
from .configuration_mistral import MistralConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), dim=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlaxMistralRMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("weight", ones, self.hidden_size, self.dtype)
        variance = (hidden_states**2).mean(-1, keepdims=True)
        hidden_states = hidden_states * 1 / jnp.sqrt(variance + self.eps)
        return weight * hidden_states


class FlaxMistralRotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int = 2048
    base: int = 10000
    max_seq_len: int = 4096
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.inv_freq = 1 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        t = jnp.arange(0, self.max_seq_len, dtype=self.dtype)
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=1)
        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    @nn.compact
    def __call__(self, x: jnp.ndarray, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (self.cos_cached[:seq_len], self.sin_cached[:seq_len])


class FlaxMistralMLP(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=False, dtype=self.dtype)
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, x):
        if self.config.pretraining_tp > 1:
            gate_proj_slices = jnp.split(
                self.gate_proj.variables["params"]["kernel"].transpose(1, 0), self.config.pretraining_tp
            )
            up_proj_slices = jnp.split(
                self.up_proj.variables["params"]["kernel"].transpose(1, 0), self.config.pretraining_tp
            )
            down_proj_slices = jnp.split(
                self.down_proj.variables["params"]["kernel"].transpose(1, 0), self.config.pretraining_tp, axis=1
            )

            gate_proj = jnp.concatenate(
                [x @ gate_proj_slices[i].transpose(1, 0) for i in range(self.config.pretraining_tp)], axis=-1
            )
            up_proj = jnp.concatenate(
                [x @ up_proj_slices[i].transpose(1, 0) for i in range(self.config.pretraining_tp)], axis=-1
            )

            intermediate_states = jnp.split(self.act_fn(gate_proj) * up_proj, self.config.pretraining_tp, axis=2)
            down_proj = [
                intermediate_states[i] @ down_proj_slices[i].transpose(1, 0) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def flax_rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.apply_rotary_pos_emb
def flax_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = jnp.expand_dims(cos[position_ids], 1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    sin = jnp.expand_dims(sin[position_ids], 1)
    q_embed = (q * cos) + (flax_rotate_half(q) * sin)
    k_embed = (k * cos) + (flax_rotate_half(k) * sin)
    return q_embed, k_embed


def flax_repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.repeat(hidden_states[:, :, None, :, :], n_rep, axis=2)
    new_size = (batch, num_key_value_heads * n_rep, slen, head_dim)
    return jax.lax.reshape(hidden_states, new_size)


class FlaxMistralAttention(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=self.config.attention_bias, dtype=self.dtype)
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=self.config.attention_bias, dtype=self.dtype
        )
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=self.config.attention_bias, dtype=self.dtype
        )
        self.o_proj = nn.Dense(self.hidden_size, use_bias=self.config.attention_bias, dtype=self.dtype)
        self.rotary_emb = FlaxMistralRotaryEmbedding(
            self.head_dim, self.max_position_embeddings, base=self.rope_theta, dtype=self.dtype
        )

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        bsz, q_len, _ = hidden_states.shape
        if self.config.pretraining_tp > 1:
            # query_slicing = (self.num_heads * self.head_dim) // self.config.pretraining_tp
            # key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = jnp.split(
                self.q_proj.variables["params"]["kernel"].transpose(1, 0), self.config.pretraining_tp
            )
            key_slices = jnp.split(
                self.k_proj.variables["params"]["kernel"].transpose(1, 0), self.config.pretraining_tp
            )
            value_slices = jnp.split(
                self.v_proj.variables["params"]["kernel"].transpose(1, 0), self.config.pretraining_tp
            )

            query_states = [hidden_states @ query_slices[i].transpose(1, 0) for i in range(self.config.pretraining_tp)]
            query_states = jnp.concatenate(query_states, axis=-1)

            key_states = [hidden_states @ key_slices[i].transpose(1, 0) for i in range(self.config.pretraining_tp)]
            key_states = jnp.concatenate(key_states, axis=-1)

            value_states = [hidden_states @ value_slices[i].transpose(1, 0) for i in range(self.config.pretraining_tp)]
            value_states = jnp.concatenate(value_states, axis=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_shape = (bsz, q_len, self.num_heads, self.head_dim)
        kv_shape = (bsz, q_len, self.num_key_value_heads, self.head_dim)
        query_states = jax.lax.reshape(query_states, query_shape).transpose(0, 2, 1, 3)
        key_states = jax.lax.reshape(key_states, kv_shape).transpose(0, 2, 1, 3)
        value_states = jax.lax.reshape(value_states, kv_shape).transpose(0, 2, 1, 3)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = flax_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = jnp.concatenate([past_key_value[0], key_states], axis=2)
            value_states = jnp.concatenate([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = flax_repeat_kv(key_states, self.num_key_value_groups)
        value_states = flax_repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = query_states @ key_states.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ value_states

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = jax.lax.reshape(attn_output, (bsz, q_len, self.hidden_size))

        if self.config.pretraining_tp > 1:
            attn_output = jnp.split(attn_output, self.config.pretraining_tp, axis=2)
            o_proj_slices = jnp.split(self.o_proj.variables["params"]["kernel"], self.config.pretraining_tp)
            attn_output = sum([attn_output[i] @ o_proj_slices[i] for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FlaxMistralDecoderLayer(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.self_attn = FlaxMistralAttention(config=self.config, dtype=self.dtype)
        self.mlp = FlaxMistralMLP(self.config, dtype=self.dtype)
        self.input_layernorm = FlaxMistralRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype
        )
        self.post_attention_layernorm = FlaxMistralRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class FlaxMistralPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MistralConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: MistralConfig,
        input_shape: Tuple = (1, 5),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            **kwargs,
        )
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(rngs, input_ids, attention_mask, return_dict=True)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

        # Copied from transformers.models.bart.modeling_flax_bart.FlaxBartDecoderPreTrainedModel.init_cache

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
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=True, use_cache=True
        )
        return unfreeze(init_variables["past_key_values"])

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache: bool = False,
        params: dict = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: dict = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        inputs = {"params": params or self.params}

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            position_ids=jnp.array(position_ids, dtype="i4"),
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
        )

        return outputs


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _flax_make_causal_mask(input_ids_shape: Tuple[int], dtype: jnp.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = jnp.full((tgt_len, tgt_len), jnp.finfo(dtype).min)
    causal_mask = jnp.triu(jnp.ones((tgt_len, tgt_len)), k=1)
    mask = mask * causal_mask
    mask = mask[None, None, :, :]
    mask = jnp.repeat(mask, bsz, axis=0)

    if past_key_values_length > 0:  # TODO: Update when past_key_values_length is not 0
        mask = jnp.concatenate([jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
        mask = jnp.repeat(mask, tgt_len + past_key_values_length, axis=1)
    return mask


def _flax_expand_mask(mask: jnp.ndarray, dtype: jnp.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :]
    expanded_mask = jnp.repeat(expanded_mask, tgt_len, axis=2)

    inverted_mask = 1.0 - expanded_mask
    inverted_mask = inverted_mask * jnp.full(inverted_mask.shape, jnp.finfo(dtype).min)

    return inverted_mask


class FlaxMistralModule(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size)
        self.layers = [
            FlaxMistralDecoderLayer(self.config, dtype=self.dtype) for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = FlaxMistralRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _flax_make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _flax_expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def __call__(
        self,
        input_ids: jnp.ndarray = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[jnp.ndarray]] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = jnp.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=jnp.int64,
            )
            position_ids = jnp.expand_dims(position_ids, axis=0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        padding_mask = None
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length_with_past), dtype=jnp.bool_)
        #     padding_mask = None
        # else:
        #     if (0 == attention_mask).any():
        #         padding_mask = attention_mask
        #     else:
        #         padding_mask = None

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return FlaxBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class FlaxMistralModel(FlaxMistralPreTrainedModel):
    module_class = FlaxMistralModule


class FlaxMistralForCausalLMModule(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.model = FlaxMistralModule(self.config, dtype=self.dtype)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.dtype)

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxCausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = jnp.split(
                self.lm_head.variables["params"]["kernel"].transpose(1, 0), self.config.pretraining_tp
            )
            logits = [hidden_states @ lm_head_slices[i].transpose(1, 0) for i in range(self.config.pretraining_tp)]
            logits = jnp.concatenate(logits, axis=-1)
        else:
            logits = self.lm_head(hidden_states)

        return FlaxCausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs[2],
            attentions=outputs.attentions if return_dict else outputs[3],
        )


class FlaxMistralForCausalLM(FlaxMistralPreTrainedModel):
    module_class = FlaxMistralForCausalLMModule

    def prepare_inputs_for_generation(
        self, input_ids, max_length, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids = position_ids * attention_mask + (1 - attention_mask)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = jnp.concatenate(
            [attention_mask, jnp.ones((attention_mask.shape[0], 1))], axis=1
        )
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


class FlaxMistralForSequenceClassificationModule(nn.Module):

    config: MistralConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.num_labels = self.config.num_labels
        self.model = FlaxMistralModule(self.config, dtype=self.dtype)
        self.score = nn.Dense(self.num_labels, use_bias=False, dtype=self.dtype)

        # Initialize weights and apply final processing

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = jax.lax.eq(input_ids, self.config.pad_token_id).argmax(-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return output

        return FlaxSequenceClassifierOutput(
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class FlaxMistralForSequenceClassification(FlaxMistralPreTrainedModel):
    module_class = FlaxMistralForSequenceClassificationModule
