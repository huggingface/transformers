from functools import partial
import math
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from transformers.models.mpt.configuration_mpt import MptConfig

class FlaxMptAttention(nn.module):
    config: MptConfig
    attention_type: str
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.max_seq_length = config.max_seq_len
        self.head_dim = self.hidden_size // self.n_heads
        self.softmax_scale = config.attn_config.softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.n_heads)

        self.attn_dropout_p = config.attn_config.attn_pdrop
        
        self.Wqkv = nn.Dense(
            3*self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
        self.out_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
    
    def __call__(
        self,
        hidden_states,
        position_bias: jnp.ndarray,
        attention_mask: Optional[Tuple[jnp.ndarray]] = None,
        past_key_value: Optional[Tuple[jnp.ndarray]] = None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        mixed_qkv = self.Wqkv(hidden_states)
        query_states, key_states, value_states = jnp.split(mixed_qkv, 3, axis=-1)
        query_states = query_states.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose((0, 2, 1, 3))
        key_states = key_states.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose((0, 2, 1, 3))
        value_states = value_states.reshape((batch_size, seq_length, self.n_heads, self.head_dim)).transpose((0, 2, 1, 3))

        if past_key_value is not None:
            if past_key_value[0].shape[2] > 0:
                key_states = jnp.concatenate([past_key_value[0], key_states], axis=2)
                value_states = jnp.concatenate([past_key_value[1], value_states], axis=2)
            past_key_value = (key_states, value_states)
        else:
            past_key_value = (key_states, value_states)

        attention_scores = jnp.matmul(query_states, key_states.transpose((0, 1, 3, 2))) * self.softmax_scale

        query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

        if position_bias is not None:
            if position_bias.ndim != 3:
                raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {position_bias.ndim}")
            key_length = key_states.shape[2]

            position_bias_query_index = max(0, position_bias.shape[1] - query_length)
            position_bias_key_index = max(0, position_bias.shape[2] - key_length)

            position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

            attention_scores = attention_scores + position_bias

        if attention_mask is not None:
            attention_scores = jax.lax.select(attention_mask, jnp.full_like(attention_scores, -jnp.inf), attention_scores)

        attn_weights = nn.softmax(attention_scores, axis=-1).astype(value_states.dtype)
        attn_weights = nn.dropout(attn_weights, rate=self.attn_dropout_p, deterministic=not self.training)

        context_states = jnp.matmul(attn_weights, value_states)
        context_states = context_states.transpose((0, 2, 1, 3)).reshape((batch_size, seq_length, -1))
        attn_output = nn.Dense(features=self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(context_states)

        return attn_output, attn_weights, past_key_value

class FlaxMptMLP(nn.Module):
    hidden_size: int
    attn_pdrop: float

    def setup(self):
        self.up_proj = nn.Dense(features=4 * self.hidden_size, kernel_init=nn.initializers.xavier_uniform(), use_bias=False)
        self.act = nn.gelu
        self.down_proj = nn.Dense(features=self.hidden_size, kernel_init=nn.initializers.xavier_uniform(), use_bias=False)

    def forward(self, hidden_states, residual):
        hidden_states = self.act(self.up_proj(hidden_states))

        intermediate_output = self.down_proj(hidden_states)

        output = nn.dropout(intermediate_output, rate=self.attn_pdrop, deterministic=not self.training)
        output = output + residual

        return output
    
class FlaxMptBlock(nn.Module):
    hidden_size: int
    n_heads: int
    attn_pdrop: float
    layer_norm_epsilon: float

    def setup(self):
        self.norm_1 = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
        self.attn = FlaxMptAttention(self.hidden_size, self.n_heads, self.attn_pdrop)
        self.norm_2 = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
        self.ffn = FlaxMptMLP(self.hidden_size, self.attn_pdrop)
        self.resid_attn_dropout = nn.Dropout(rate=self.attn_pdrop)

    def forward(
        self,
        hidden_states,
        position_bias,
        attention_mask,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        layernorm_output = self.norm_1(hidden_states)
        residual = hidden_states

        attn_outputs, attn_weights, past_key_value = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            layer_past=layer_past,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs) + residual

        layernorm_output = self.norm_2(hidden_states)
        residual = hidden_states

        output = self.ffn(layernorm_output, residual)
        outputs = (output,)

        if use_cache:
            outputs += (past_key_value,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
class MptPreTrainedModel(FlaxPreTrainedModel):
    config_class = MptConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MptBlock"]
    _keys_to_ignore_on_load_missing = [r"lm_head.*."]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Dense):
            module.param = module.param.init(jax.random.PRNGKey(0), jnp.ones((1, module.param.shape[-1])))
            if module.bias is not None:
                module.bias = module.bias.init(jax.random.PRNGKey(0), jnp.zeros((1, module.bias.shape[-1])))
        elif isinstance(module, nn.Embed):
            module.weight = module.weight.init(jax.random.PRNGKey(0), jnp.ones((1, module.weight.shape[-1])))
            if module.padding_idx is not None:
                module.weight = module.weight.at[:, module.padding_idx].set(0.0)
        elif isinstance(module, nn.LayerNorm):
            module.param = module.param.init(jax.random.PRNGKey(0), jnp.ones((1, module.param.shape[-1])))
            if module.bias is not None:
                module.bias = module.bias.init(jax.random.PRNGKey(0), jnp.zeros((1, module.bias.shape[-1])))

    @staticmethod
    def _convert_to_mpt_cache(
        past_key_value: Tuple[Tuple[jnp.array, jnp.array]],
    ) -> Tuple[Tuple[jnp.array, jnp.array]]:
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        return tuple(
            (
                layer_past[0].reshape(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].reshape(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )
