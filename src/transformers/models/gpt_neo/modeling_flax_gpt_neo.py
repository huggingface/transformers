import os
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.linen import combine_masks, dot_product_attention, make_causal_mask
from jax import lax

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPast, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import logging
from .configuration_gpt_neo import GPTNeoConfig
from .modeling_gpt_neo import GPT_NEO_INPUTS_DOCSTRING


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPTNeoConfig"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neo-1.3B",
    # See all GPTNeo models at https://huggingface.co/models?filter=gpt_neo
]

_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neo-1.3B"

GPT_NEO_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.FlaxPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a Flax Linen `flax.nn.Module
    <https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html>`__ subclass. Use it as a regular Flax
    Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - `Just-In-Time (JIT) compilation <https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit>`__
    - `Automatic Differentiation <https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation>`__
    - `Vectorization <https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap>`__
    - `Parallelization <https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap>`__

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.FlaxPreTrainedModel.from_pretrained` method to load the
            model weights.
"""


class FlaxGPTNeoAttentionMixin:
    """
    A few attention related utilities for attention modules in GPT Neo, to be used as a mixin.
    """

    @staticmethod
    def _get_block_length_and_num_blocks(seq_length, window_size):
        """
        Computes ``block_length`` and ``num_blocks`` such that ``seq_length`` becomes evenly divisible by
        ``block_length``.
        """
        block_length = window_size
        while seq_length % block_length != 0:
            block_length -= 1
        num_blocks = seq_length // block_length
        return block_length, num_blocks

    @staticmethod
    def _look_back(tensor, block_length, window_size, pad_value=0, is_key_value=True):
        """
        Used to implement attention between consecutive blocks. This method assumes that dim 1 of :obj:`tensor`
        represents the :obj:`seq_length` dimension. It splits :obj:`seq_length` dimension into :obj:`num_blocks` and
        :obj:`window_size` + :obj:`block_length`. It pads the :obj:`seq_length` dimension if necessary.

        Example::

            tensor: torch.tensor([[[ 0.4983], [ 2.6918], [-0.0071], [ 1.0492], [-1.8348], [ 0.7672], [ 0.2986], [ 0.0285]]])
            with shape (1, 8, 1)
            block_length = window_size = 4
            _look_back =>
            torch.tensor([[[[ 0.0000], [ 0.0000], [ 0.0000], [ 0.0000], [ 0.4983], [ 2.6918], [-0.0071], [ 1.0492]],
                           [[ 0.4983], [ 2.6918], [-0.0071], [ 1.0492], [-1.8348], [ 0.7672], [ 0.2986], [ 0.0285]]]])

        Args:
            tensor (:obj:`torch.Tensor`): tensor of shape :obj:`[batch_size, seq_length, hidden_dim]` or :obj:`[batch_size, seq_length]`
            block_length (:obj:`int`): An integer specifying the length of each block, used as a step size when creating the blocks.
            window_size (:obj:`int`): An integer specifying the size of attention window, used to calculate the final block size when creating the block.
            pad_value (obj:`int`): An integer specifying the value to use when padding the :obj:`tensor`.
            is_key_value (:obj:`bool`): A boolean indicating if the :obj:`tensor` is a key/value tensor.

        Returns:
            tensor of shape :obj:`[batch_size, num_blocks, window_size + block_length, ...]` if :obj:`is_key_value` is
            :obj:`True` else a tensor of shape :obj:`[batch_size, window_size + block_length, num_blocks, ...]`
        """
        if len(tensor.shape) == 3:
            padding_side = (0, 0, window_size, 0)
        elif len(tensor.shape) == 2:
            padding_side = (window_size, 0)
        else:
            raise ValueError(f"Input tensor rank should be one of [2, 3], but is: {len(tensor.shape)}")

        padded_tensor = jnp.pad(tensor, padding_side, value=pad_value)
        # how can we replace unfold here?
        padded_tensor = padded_tensor.unfold(dimension=1, size=window_size + block_length, step=block_length)

        if is_key_value:
            padded_tensor = padded_tensor.transpose(-2, -1)
        return padded_tensor

    @staticmethod
    def _split_seq_length_dim_to(tensors, dim_factor_1, dim_factor_2):
        """
        Splits sequence length dim of tensors into `dim_factor_1` and `dim_factor_2` dims
        """
        batch_size = tensors.shape[0]
        split_dim_shape = (batch_size, dim_factor_1, dim_factor_2)

        if len(tensors.shape) == 3:
            return jnp.reshape(tensors, split_dim_shape + (-1,))
        elif len(tensors.shape) == 2:
            return jnp.reshape(tensors, split_dim_shape)
        else:
            raise ValueError(f"Input vector rank should be one of [2, 3], but is: {len(tensors.shape)}")

    @staticmethod
    def create_local_attention_mask(batch_size, seq_length, window_size, device, attention_mask=None):
        block_length, num_blocks = FlaxGPTNeoAttentionMixin._get_block_length_and_num_blocks(seq_length, window_size)
        indices = jnp.arange(seq_length, dtype=jnp.int64, device=device).repeat(batch_size, 1)

        query_indices = FlaxGPTNeoAttentionMixin._split_seq_length_dim_to(indices, num_blocks, block_length)
        key_indices = FlaxGPTNeoAttentionMixin._look_back(indices, block_length, window_size, is_key_value=False)

        # create mask tensor such that each block contains a causal_mask for that block
        causal_mask = lax.ge(jnp.expand_dims(query_indices, -1), jnp.expand_dims(key_indices, -2))

        if attention_mask is None:
            attention_mask = jnp.ones(batch_size, seq_length, dtype=jnp.int64, device=device)

        # A block can also be padded because of the _look_back operation
        # look back into the attention_block such that it will also get padded the same way
        # and have 0s in the padded position
        attention_mask = FlaxGPTNeoAttentionMixin._look_back(
            attention_mask, block_length, window_size, is_key_value=False
        )
        attention_mask = jnp.expand_dims(attention_mask, -2)  # Add an extra dimension to account for hidden_dim

        # Multiply the causal_mask with attention_mask so the padded positions (by _look_back operation)
        # will contain 0s.
        # This also makes sure that other positions ignored by the attention_mask will also be ignored
        # in the causal_mask.
        causal_mask = causal_mask * attention_mask

        # In GPT Neo's local attention each window can attend to at most window_size tokens
        # rest of the tokens should be ignored.
        relative_position = jnp.expand_dims(key_indices, -2) - jnp.expand_dims(query_indices, -1)
        visible = lax.gt(relative_position, -window_size)

        causal_mask = causal_mask * visible
        causal_mask = jnp.expand_dims(causal_mask, -3).bool()  # Add an extra dimension to account for num_heads

        return causal_mask

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        return tensor.reshape(
            tensor.shape[:2]
            + (
                num_heads,
                attn_head_size,
            )
        )

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        return tensor.reshape(tensor.shape[:2] + (num_heads * attn_head_size,))


class FlaxGPTNeoSelfAttention(nn.Module, FlaxGPTNeoAttentionMixin):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        max_positions = self.config.max_position_embeddings

        self.resid_dropout = nn.Dropout(self.config.resid_dropout)

        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.v_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.q_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.out_proj = nn.Dense(self.embed_dim, use_bias=True)
        self.causal_mask = make_causal_mask(jnp.ones((1, max_positions), dtype="bool"), dtype="bool")

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
        attention_mask=None,
        layer_past=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = jnp.cat((past_key, key), dim=-2)
            value = jnp.cat((past_value, value), dim=-2)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, -1e4).astype(self.dtype),
        )

        attn_output = dot_product_attention(
            query,
            key,
            value,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

        # TODO: at the moment it's not possible to retrieve attn_weights from
        # dot_product_attention, but should be in the future -> add functionality then

        return (attn_output,)


class FlaxGPTNeoLocalSelfAttention(nn.Module, FlaxGPTNeoAttentionMixin):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        max_positions = self.config.max_position_embeddings

        self.resid_dropout = nn.Dropout(self.config.resid_dropout)

        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.v_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.q_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.out_proj = nn.Dense(self.embed_dim, use_bias=True)
        self.window_size = self.config.window_size

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)

        if layer_past is not None:
            past = layer_past[0]
            key_value_hidden_states = jnp.cat([past, hidden_states], dim=1)
            past_length = past.shape[1]
        else:
            key_value_hidden_states = hidden_states
            past_length = 0

        key = self.k_proj(key_value_hidden_states)
        value = self.v_proj(key_value_hidden_states)

        # compute block length and num_blocks
        batch_size, seq_length = hidden_states.shape[:2]
        full_seq_length = seq_length + past_length
        block_length, num_blocks = self._get_block_length_and_num_blocks(full_seq_length, self.window_size)

        # create buckets
        if layer_past is not None:
            # we just need 1 block with block_length 1 when caching is enabled
            query = self._split_seq_length_dim_to(query, 1, 1)
        else:
            query = self._split_seq_length_dim_to(query, num_blocks, block_length)

        key = self._look_back(key, block_length, self.window_size)
        value = self._look_back(value, block_length, self.window_size)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # select key/value vectors only for the last block
        if layer_past is not None:
            key = key[:, -1:, ...]
            value = value[:, -1:, ...]

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            # only take the mask for the last block
            attention_mask = attention_mask[:, -1:, :, -1:, :]

        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # attn
        attn_output = dot_product_attention(
            query,
            key,
            value,
            bias=attention_mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # TODO: at the moment it's not possible to retrieve attn_weights from
        # dot_product_attention, but should be in the future -> add functionality then

        return (attn_output,)


class FlaxGPTNeoAttention(nn.Module):
    config: GPTNeoConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer_id = self.layer_id
        self.attention_layers = self.config.attention_layers
        self.attention_type = self.attention_layers[self.layer_id]

        if self.attention_type == "global":
            self.attention = FlaxGPTNeoSelfAttention(self.config)
        elif self.attention_type == "local":
            self.attention = FlaxGPTNeoLocalSelfAttention(self.config)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{self.config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def __call__(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # cache the hidden_states instead of key_value_states
        # for local attention layer
        if self.attention_type == "local":
            if layer_past is None:
                past = hidden_states
            else:
                past = jnp.cat([layer_past[0], hidden_states], dim=1)
            outputs = (outputs[0], (past,)) + outputs[1:]
        return outputs


class FlaxGPTNeoMLP(nn.Module):
    intermediate_size: int
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):  # in MLP: intermediate_size= 4 * hidden_size
        embed_dim = self.config.hidden_size
        self.c_fc = nn.Dense(self.intermediate_size)
        self.c_proj = nn.Dense(embed_dim)
        self.act = ACT2FN[self.config.activation_function]
        self.dropout = nn.Dropout(self.config.resid_dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class FlaxGPTNeoBlock(nn.Module):
    config: GPTNeoConfig
    layer_id: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden_size = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, epsilon=self.config.layer_norm_epsilon)
        self.attn = FlaxGPTNeoAttention(self.config, self.layer_id)
        self.ln_2 = nn.LayerNorm(hidden_size, epsilon=self.config.layer_norm_epsilon)
        self.mlp = FlaxGPTNeoMLP(inner_dim, self.config)

    def __call__(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class FlaxGPTNeoBlockCollection(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks = [
            FlaxGPTNeoBlock(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(hidden_states, attention_mask, deterministic=deterministic, init_cache=init_cache)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FlaxGPTNeoPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: GPTNeoConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (:obj:`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (:obj:`int`):
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
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
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
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPT2Attention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
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


class FlaxGPTNeoModule(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxGPTNeoBlockCollection(self.config, dtype=self.dtype)
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.reshape(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].reshape[-2]

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = jnp.arange(past_length, input_shape[-1] + past_length, dtype=jnp.int64, device=device)
            position_ids = jnp.expand_dims(position_ids, 0).reshape(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            global_attention_mask = attention_mask.reshape(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            global_attention_mask = global_attention_mask[:, None, None, :]

            # Since global_attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            global_attention_mask = global_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            global_attention_mask = (1.0 - global_attention_mask) * -10000.0
        else:
            global_attention_mask = None

        # Local causal attention mask
        batch_size, seq_length = input_shape
        full_seq_length = seq_length + past_length
        local_attention_mask = FlaxGPTNeoAttentionMixin.create_local_attention_mask(
            batch_size, full_seq_length, self.config.window_size, device, attention_mask
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            attn_type = self.config.attention_layers[i]
            attn_mask = global_attention_mask if attn_type == "global" else local_attention_mask

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attn_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.reshape(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return FlaxBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    "The bare Flax GPT Neo Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEO_START_DOCSTRING,
)
class FlaxGPTNeoModel(FlaxGPTNeoPreTrainedModel):
    module_class = FlaxGPTNeoModule


append_call_sample_docstring(
    FlaxGPTNeoModel, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxBaseModelOutput, _CONFIG_FOR_DOC
)


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_NEO_START_DOCSTRING,
)
class FlaxGPTNeoLMHeadModule(nn.Module):
    config: GPTNeoConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.transformer = FlaxGPTNeoModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range, dtype=self.dtype),
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
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


append_call_sample_docstring(
    FlaxGPTNeoLMHeadModule, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC
)
