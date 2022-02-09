from cmath import sin
from typing import Optional, Tuple

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import get_initializer

from .configuration_gptj import GPTJConfig


def fixed_pos_embedding(x: tf.Tensor, seq_dim: int = 1, seq_len: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (tf.range(0, dim, 2) / dim))
    sinusoid_inp = tf.cast(tf.einsum("i , j -> i j", tf.range(seq_len), inv_freq), tf.float32)
    return tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)


def rotate_every_two(x: tf.Tensor) -> tf.Tensor:
    rotate_half_tensor = tf.stack((x[:, :, :, 1::2], x[:, :, :, ::2]), axis=-1)
    rotate_half_tensor = tf.reshape(rotate_half_tensor, rotate_half_tensor.shape[:-2] + (-1,))
    return rotate_half_tensor


def apply_rotary_pos_emb(x: tf.Tensor, sincos: tf.Tensor, offset: int = 0) -> tf.Tensor:
    sin_pos, cos_pos = sincos
    sin_pos = tf.repeat(sin_pos[None, offset : x.shape[1] + offset, None, :], 2, 3)
    cos_pos = tf.repeat(cos_pos[None, offset : x.shape[1] + offset, None, :], 2, 3)
    return (x * cos_pos) + (rotate_every_two(x) * sin_pos)


class TFGPTJAttention(tf.keras.layers.Layer):
    def __init__(self, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = tf.Variable(tf.sqrt(self.head_dim), dtype=tf.float32, trainable=False, name="scale_attn")

        self.rotary_dim = config.rotary_dim

        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)

        self.q_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="q_proj",
        )
        self.k_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="k_proj",
        )
        self.v_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="v_proj",
        )

        max_positions = config.max_position_embeddings
        self.bias = tf.Variable(
            tf.reshape(
                tf.cast(tf.experimental.numpy.tril(tf.ones((max_positions, max_positions))), tf.int8),
                (1, 1, max_positions, max_positions),
            ),
            trainable=False,
            name="bias",
        )
        self.masked_bias = tf.Variable(-1e9, trainable=False, name="masked_bias")

    def _split_heads(self, hidden_states: tf.Tensor) -> tf.Tensor:
        return tf.reshape(hidden_states, hidden_states.shape[:2] + (self.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states: tf.Tensor) -> tf.Tensor:
        return tf.reshape(hidden_states, hidden_states.shape[:2] + (self.embed_dim,))

    def _attn(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # compute causal mask from causal mask buffer
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = tf.cast(self.bias[:, :, key_length - query_length : key_length, :key_length], tf.bool)

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)

        attn_weights = tf.matmul(query, key, transpose_b=True)
        attn_weights = tf.where(causal_mask, attn_weights, tf.cast(self.masked_bias, attn_weights.dtype))

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = tf.matmul(attn_weights, value)

        return attn_output, attn_weights

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        layer_past: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        sincos = tf.experimental.numpy.take(self.embed_positions, )
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = tf.concat((k_rot, k_pass), axis=-1)
            query = tf.concat((q_rot, q_pass), axis=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = tf.transpose(key, [0, 2, 1, 3])
        query = tf.transpose(query, [0, 2, 1, 3])

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = tf.concat((past_key, key), axis=-2)
            value = tf.concat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class TFGPTJMLP(tf.keras.layers.Layer):
    def __init__(self, intermediate_size: int, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        embed_dim = config.n_embd

        self.fc_in = tf.keras.layers.Dense(
            intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="fc_in"
        )
        self.fc_out = tf.keras.layers.Dense(
            embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="fc_out"
        )

        self.act = get_tf_activation(config.activation_function)
        self.dropout = tf.keras.layers.Layer(config.resid_pdrop)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.fc_dropout(hidden_states)


class TFGPTJBlock(tf.keras.layers.Layer):
    def __init__(self, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=self.config.layer_norm_epsilon)
        self.attn = TFGPTJAttention(config)
        self.mlp = TFGPTJMLP(inner_dim, config)

    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
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
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)
