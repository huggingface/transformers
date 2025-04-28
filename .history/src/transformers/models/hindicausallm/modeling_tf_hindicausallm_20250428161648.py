# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Inc. team. All rights reserved.
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
"""TensorFlow HindiCausalLM model."""

from typing import List, Optional, Tuple, Union  # Added List

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,  # Changed import
    TFCausalLMOutputWithPast,  # Changed import
    TFSequenceClassifierOutput,  # Changed Output name
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (  # Added replace_return_docstrings
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_hindicausallm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HindiCausalLMConfig"
_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-causal-lm"  # Added checkpoint
_REAL_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-causal-lm"  # Added real checkpoint



class TFHindiCausalLMLayerNorm(tf.keras.layers.Layer):
    """Layer normalization component."""

    def __init__(self, hidden_size, eps=1e-5, **kwargs):  # Updated eps to match config
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.hidden_size,), initializer=tf.keras.initializers.Ones(), trainable=True, name="weight"
        )
        # Removed bias to align with common RMSNorm implementations like Llama
        # self.bias = self.add_weight(
        #     shape=(self.hidden_size,), initializer=tf.keras.initializers.Zeros(), trainable=True, name="bias"
        # )
        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = tf.cast(hidden_states, dtype=tf.float32)
        variance = tf.math.reduce_mean(tf.math.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.eps)

        # Removed bias addition
        # hidden_states = hidden_states + self.bias
        hidden_states = self.weight * hidden_states

        return tf.cast(hidden_states, dtype=self.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "eps": self.eps})
        return config


# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaRotaryEmbedding with Llama->HindiCausalLM
class TFHindiCausalLMRotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (
            self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / tf.cast(self.dim, dtype=tf.float32))
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = tf.range(self.max_seq_len_cached, dtype=tf.float32)
        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        emb = tf.concat([freqs, freqs], axis=-1)
        self.cos_cached = tf.cos(emb)[None, None, :, :]
        self.sin_cached = tf.sin(emb)[None, None, :, :]

    def call(self, value, seq_len=None):
        # value: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len)

        return (
            tf.cast(self.cos_cached[:, :, :seq_len, ...], dtype=value.dtype),
            tf.cast(self.sin_cached[:, :, :seq_len, ...], dtype=value.dtype),
        )


# Copied from transformers.models.llama.modeling_tf_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : tf.shape(x)[-1] // 2]
    x2 = x[..., tf.shape(x)[-1] // 2 :]
    return tf.concat((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_tf_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = tf.expand_dims(position_ids, axis=-1)  # Shape: [batch_size, seq_len, 1]
    gather_indices = tf.tile(gather_indices, [1, 1, tf.shape(cos)[-1]])  # Shape: [batch_size, seq_len, dim]
    cos = tf.gather(tf.squeeze(tf.squeeze(cos, axis=0), axis=0), gather_indices, batch_dims=1)
    sin = tf.gather(tf.squeeze(tf.squeeze(sin, axis=0), axis=0), gather_indices, batch_dims=1)
    cos = tf.expand_dims(cos, axis=1)  # Shape: [batch_size, 1, seq_len, dim]
    sin = tf.expand_dims(sin, axis=1)  # Shape: [batch_size, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_tf_llama.repeat_kv
def repeat_kv(hidden_states: tf.Tensor, n_rep: int) -> tf.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = shape_list(hidden_states)
    if n_rep == 1:
        return hidden_states
    hidden_states = tf.expand_dims(hidden_states, 2)
    hidden_states = tf.tile(hidden_states, [1, 1, n_rep, 1, 1])

    return tf.reshape(hidden_states, [batch, num_key_value_heads * n_rep, slen, head_dim])


# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaAttention with Llama->HindiCausalLM
@keras_serializable
class TFHindiCausalLMAttention(tf.keras.layers.Layer):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = tf.keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="q_proj",
        )
        self.k_proj = tf.keras.layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="k_proj",
        )
        self.v_proj = tf.keras.layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="v_proj",
        )
        self.o_proj = tf.keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="o_proj",
        )

        self.rotary_emb = TFHindiCausalLMRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            name="rotary_emb",
        )

    def build(self, input_shape: tf.TensorShape):
        super().build(input_shape)
        if getattr(self, "layer_idx", None) is None:
            raise ValueError(f"The layer index should be specified for {self.__class__.__name__}")

    def _reshape(self, states: tf.Tensor, sequence_length: int, num_heads: int) -> tf.Tensor:
        """Reshape the hidden states."""
        batch_size = shape_list(states)[0]
        return tf.reshape(states, (batch_size, sequence_length, num_heads, self.head_dim))

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: bool = False,
        training: bool = False,
        cache_position: Optional[tf.Tensor] = None,  # Added cache_position
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        batch_size, seq_length, _ = shape_list(hidden_states)

        # Project hidden states to query, key, and value states
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape query, key, and value states
        query_states = self._reshape(query_states, seq_length, self.num_heads)
        key_states = self._reshape(key_states, seq_length, self.num_key_value_heads)
        value_states = self._reshape(value_states, seq_length, self.num_key_value_heads)

        # Transpose query, key, and value states
        query_states = tf.transpose(query_states, (0, 2, 1, 3))  # (batch_size, num_heads, seq_length, head_dim)
        key_states = tf.transpose(key_states, (0, 2, 1, 3))  # (batch_size, num_key_value_heads, seq_length, head_dim)
        value_states = tf.transpose(
            value_states, (0, 2, 1, 3)
        )  # (batch_size, num_key_value_heads, seq_length, head_dim)

        # Apply rotary position embeddings
        kv_seq_len = shape_list(key_states)[-2]
        if past_key_value is not None:
            kv_seq_len += shape_list(past_key_value[0])[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Reuse k, v, self_attention
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)

        # Update the cache
        past_key_value = (key_states, value_states)

        # Repeat key and value states for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Calculate attention scores
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)  # [bsz, num_heads, q_len, kv_len]

        attn_weights = attn_weights / tf.math.sqrt(tf.cast(self.head_dim, query_states.dtype))

        # Check attention weights shape
        if attn_weights.shape != (batch_size, self.num_heads, seq_length, shape_list(key_states)[2]):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_length, shape_list(key_states)[2])}, but is"
                f" {attn_weights.shape}"
            )

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, 1, seq_length, shape_list(key_states)[2]):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_length, shape_list(key_states)[2])}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # Apply softmax and dropout
        attn_weights = stable_softmax(attn_weights, axis=-1)
        attn_weights = tf.nn.dropout(attn_weights, rate=self.attention_dropout)

        # Calculate attention output
        attn_output = tf.matmul(attn_weights, value_states)  # [bsz, num_heads, q_len, head_dim]

        # Check attention output shape
        if attn_output.shape != (batch_size, self.num_heads, seq_length, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_length, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        # Reshape and transpose attention output
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (batch_size, seq_length, self.hidden_size))

        # Project attention output
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaMLP with Llama->HindiCausalLM
@keras_serializable
class TFHindiCausalLMMLP(tf.keras.layers.Layer):
    def __init__(self, config: HindiCausalLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = tf.keras.layers.Dense(
            self.intermediate_size,
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="gate_proj",
        )
        self.up_proj = tf.keras.layers.Dense(
            self.intermediate_size,
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="up_proj",
        )
        self.down_proj = tf.keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="down_proj",
        )
        self.act_fn = get_tf_activation(config.hidden_act)

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaDecoderLayer with Llama->HindiCausalLM
@keras_serializable
class TFHindiCausalLMDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: HindiCausalLMConfig, layer_idx: int, **kwargs):
        super().__init__(name=f"layers.{layer_idx}", **kwargs)
        self.hidden_size = config.hidden_size
        self.self_attn = TFHindiCausalLMAttention(config, layer_idx=layer_idx, name="self_attn")
        self.mlp = TFHindiCausalLMMLP(config, name="mlp")
        self.input_layernorm = TFHindiCausalLMLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, name="input_layernorm"
        )
        self.post_attention_layernorm = TFHindiCausalLMLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, name="post_attention_layernorm"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = False,
        cache_position: Optional[tf.Tensor] = None,  # Added cache_position
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
            cache_position=cache_position,  # Pass cache_position
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

        # `present_key_value` is always returned, regardless of `training`
        outputs += (present_key_value,)

        return outputs


HINDICAUSALLM_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the instance
    methods like `__call__` or `call()` you need to be explicit and pass your inputs either as a list/tuple (e.g.
    `model(inputs_list, training=True)` or as a dict (e.g. `model(inputs_dict, training=True)`).

    </Tip>

    Args:
        config ([`HindiCausalLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

HINDICAUSALLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read specifications of the chosen attention mechanism.
            See the documentation of the chosen attention mechanism for more information regarding the expected padding
            strategy and corresponding attributes (`padding_side`, `padding_strategy`). For more information on how
            `attention_mask` interacts with the chosen attention mechanism, please refer to the documentation of the
            chosen attention mechanism.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
        position_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(tf.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` Decoder cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        training (`bool`, *optional*, defaults to `False`):
             Whether or not to run the model in training mode. If `True` the model will perform dropout, etc.
        cache_position (`tf.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
                this tensor is not affected by padding. It is used to update the cache in the correct position and calculate
                the cached mask. Non-mutable.
"""


# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaMainLayer with Llama->HindiCausalLM
@keras_serializable
class TFHindiCausalLMMainLayer(tf.keras.layers.Layer):
    config_class = HindiCausalLMConfig

    def __init__(self, config: HindiCausalLMConfig, **kwargs):
        super().__init__(name="model", **kwargs)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="embed_tokens",
        )
        self.layers = [
            TFHindiCausalLMDecoderLayer(config, layer_idx=i, name=f"layers.{i}")
            for i in range(config.num_hidden_layers)
        ]
        self.norm = TFHindiCausalLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps, name="norm")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens.weight = value
        self.embed_tokens.vocab_size = shape_list(value)[0]

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        batch_size, seq_length = input_shape

        if seq_length > 1:
            mask_values = tf.cast(
                tf.fill((batch_size, seq_length, seq_length), tf.float32.min), dtype=inputs_embeds.dtype
            )
            mask_cond = tf.range(seq_length) < tf.reshape(tf.range(seq_length), (-1, 1))
            causal_mask = tf.where(mask_cond, 0.0, mask_values)
            if past_key_values_length > 0:
                causal_mask = tf.concat(
                    [tf.zeros((batch_size, seq_length, past_key_values_length), dtype=causal_mask.dtype), causal_mask],
                    axis=-1,
                )
            combined_attention_mask = tf.expand_dims(causal_mask, axis=1)  # [bsz, 1, seq_len, kv_seq_len]

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = tf.expand_dims(tf.expand_dims(attention_mask, axis=1), axis=1)
            # Correct size for broadcasting
            kv_seq_len = past_key_values_length + seq_length
            expanded_attn_mask = expanded_attn_mask[:, :, :, -kv_seq_len:]  # Ensure correct kv_seq_len dim

            mask_value = tf.cast(tf.float32.min, dtype=inputs_embeds.dtype)
            expanded_attn_mask = tf.where(
                expanded_attn_mask > 0, 0.0, mask_value
            )  # Invert mask: 0 for keep, min_float for mask

            if combined_attention_mask is not None:
                combined_attention_mask = (
                    expanded_attn_mask + combined_attention_mask
                )  # Add masks: 0 keeps, min_float masks
            else:
                combined_attention_mask = expanded_attn_mask

        return combined_attention_mask

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        cache_position: Optional[tf.Tensor] = None,  # Added cache_position
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
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
            batch_size, seq_length = shape_list(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = shape_list(inputs_embeds)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is not None and cache_position is None:
            # If static cache, cache_position is not required
            past_key_values_length = shape_list(past_key_values[0][0])[
                2
            ]  # K tensor shape: [batch, heads, seq_len, dim]
        elif cache_position is not None:
            # If dynamic cache, cache_position is required to indicate the length of the cache
            past_key_values_length = shape_list(cache_position)[0]
        else:
            past_key_values_length = 0

        if position_ids is None:
            position_ids = tf.range(past_key_values_length, seq_length + past_key_values_length, dtype=tf.int64)
            position_ids = tf.expand_dims(position_ids, axis=0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = [] if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
                cache_position=cache_position,  # Pass cache_position
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache.append(layer_outputs[2 if output_attentions else 1])

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaPreTrainedModel with Llama->HindiCausalLM
class TFHindiCausalLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HindiCausalLMConfig
    base_model_prefix = "model"
    # names with deeper structure matching are still supported
    _keep_in_fp32_modules = ["rotary_emb.inv_freq"]  # Keep inv_freq in fp32
    _keys_to_ignore_on_load_unexpected = [r"decoder\.block\.\d+\.attn\.bias", r"mask_token"]


@add_start_docstrings(
    "The bare HindiCausalLM Model outputting raw hidden-states without any specific head on top.",
    HINDICAUSALLM_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaModel with Llama->HindiCausalLM, HINDI -> HINDICAUSALLM
class TFHindiCausalLMModel(TFHindiCausalLMPreTrainedModel):
    # The call method is inherited from the main layer class
    def __init__(self, config: HindiCausalLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFHindiCausalLMMainLayer(config, name="model")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(HINDICAUSALLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        cache_position: Optional[tf.Tensor] = None,  # Added cache_position
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, TFHindiCausalLMModel

        >>> tokenizer = AutoTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")
        >>> model = TFHindiCausalLMModel.from_pretrained("convaiinnovations/hindi-causal-lm")

        >>> inputs = tokenizer("नमस्ते दुनिया", return_tensors="tf") # "Hello World" in Hindi

        >>> # Forward pass through the model
        >>> outputs = model(inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
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
            training=training,
            cache_position=cache_position,  # Pass cache_position
        )
        return outputs

    def serving_output(self, output: TFBaseModelOutputWithPast) -> TFBaseModelOutputWithPast:
        # Convert Keras Tensors to TF Tensors
        past_key_values = tf.nest.map_structure(tf.identity, output.past_key_values)
        hidden_states = tf.nest.map_structure(tf.identity, output.hidden_states)
        attentions = tf.nest.map_structure(tf.identity, output.attentions)

        return TFBaseModelOutputWithPast(
            last_hidden_state=output.last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )


@add_start_docstrings(
    """
    The HindiCausalLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    HINDICAUSALLM_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaForCausalLM with Llama->HindiCausalLM, HINDI->HINDICAUSALLM
class TFHindiCausalLMForCausalLM(TFHindiCausalLMPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config: HindiCausalLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFHindiCausalLMModel(config, name="model")
        self.vocab_size = config.vocab_size
        # The LM head weights are automatically tied to the input embeddings - see `_tie_weights`
        self.lm_head = tf.keras.layers.Dense(
            config.vocab_size,
            use_bias=False,
            name="lm_head",
            kernel_initializer=get_initializer(config.initializer_range),
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # Assuming new_embeddings is a TF Variable or Tensor
        self.lm_head.kernel = new_embeddings  # Set kernel directly
        self.lm_head.units = shape_list(new_embeddings)[1]
        self.vocab_size = shape_list(new_embeddings)[1]

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(HINDICAUSALLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        cache_position: Optional[tf.Tensor] = None,  # Added cache_position
    ) -> Union[Tuple, TFCausalLMOutputWithPast]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the causal language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size - 1]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size - 1]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, TFHindiCausalLMForCausalLM

        >>> model = TFHindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-causal-lm")
        >>> tokenizer = AutoTokenizer.from_pretrained("convaiinnovations/hindi-causal-lm")

        >>> prompt = "भारत एक विशाल देश है" # "India is a vast country"
        >>> inputs = tokenizer(prompt, return_tensors="tf")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'भारत एक विशाल देश है जो दुनिया के सबसे बड़े लोकतंत्रों में से एक है।'
        ```"""
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
            training=training,
            cache_position=cache_position,  # Pass cache_position
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = tf.cast(logits, tf.float32)  # Cast logits to float32

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]

            # Calculate loss
            loss = self.hf_compute_loss(shift_labels, shift_logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFCausalLMOutputWithPast) -> TFCausalLMOutputWithPast:
        # Convert Keras Tensors to TF Tensors
        past_key_values = tf.nest.map_structure(tf.identity, output.past_key_values)
        hidden_states = tf.nest.map_structure(tf.identity, output.hidden_states)
        attentions = tf.nest.map_structure(tf.identity, output.attentions)

        return TFCausalLMOutputWithPast(
            logits=output.logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = shape_list(past_key_values[0][0])[2]  # K tensor shape: [batch, heads, seq_len, dim]

            # Some generation methods already pass only the last input ID
            if shape_list(input_ids)[1] > 1:
                input_ids = input_ids[:, past_length:]
        elif attention_mask is not None and shape_list(input_ids)[1] != shape_list(attention_mask)[1]:
            # This implementation assumes that input_ids are padded on the left Use the last token in the provided
            # input_ids
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = tf.cast(tf.cumsum(attention_mask, axis=-1) - 1, dtype=tf.int64)
            position_ids = tf.where(attention_mask == 0, 1, position_ids)
            if past_key_values:
                position_ids = position_ids[:, -1:]

        inputs = {"input_ids": input_ids}

        inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append([tf.gather(past_state, beam_idx, axis=0) for past_state in layer_past])
        return reordered_past


@add_start_docstrings(
    """
    The HindiCausalLM Model transformer with a sequence classification head on top (linear layer).

    [`TFHindiCausalLMForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    HINDICAUSALLM_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_tf_llama.TFLlamaForSequenceClassification with Llama->HindiCausalLM, HINDI->HINDICAUSALLM
class TFHindiCausalLMForSequenceClassification(TFHindiCausalLMPreTrainedModel):
    def __init__(self, config: HindiCausalLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.model = TFHindiCausalLMModel(config, name="model")
        self.score = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="score",
            use_bias=False,
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(HINDICAUSALLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        cache_position: Optional[tf.Tensor] = None,  # Added cache_position
    ) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            training=training,
            cache_position=cache_position,  # Pass cache_position
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states)  # [batch_size, seq_len, num_labels]

        if input_ids is not None:
            batch_size, sequence_length = shape_list(input_ids)[:2]
        elif inputs_embeds is not None:
            batch_size, sequence_length = shape_list(inputs_embeds)[:2]
        else:
            batch_size = 1  # Assume batch_size = 1 if no input is provided

        if self.config.pad_token_id is None:
            if batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            else:
                sequence_lengths = -1
        else:
            if input_ids is not None:
                # Assuming pad_token_id is not None
                # Find the last non-padding token index
                sequence_lengths = (
                    tf.reduce_sum(tf.cast(tf.not_equal(input_ids, self.config.pad_token_id), tf.int32), axis=-1) - 1
                )
            else:
                sequence_lengths = -1  # Cannot determine length from embeds

        # Gather logits for the last token of each sequence
        # Ensure sequence_lengths are valid indices (>= 0)
        valid_sequence_lengths = tf.maximum(sequence_lengths, 0)
        pooled_logits = tf.gather(logits, valid_sequence_lengths, batch_dims=1, axis=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and tf.dtypes.as_dtype(labels.dtype).is_integer:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
                if self.num_labels == 1:
                    loss = loss_fn(tf.squeeze(labels, axis=-1), tf.squeeze(pooled_logits, axis=-1))
                else:
                    loss = loss_fn(labels, pooled_logits)
            elif self.config.problem_type == "single_label_classification":
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
                )
                loss = loss_fn(tf.squeeze(labels, axis=-1), pooled_logits)
            elif self.config.problem_type == "multi_label_classification":
                loss_fn = tf.keras.losses.BinaryCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
                )
                loss = loss_fn(labels, pooled_logits)

        # Calculate the average loss over the batch if loss is calculated
        if loss is not None:
            loss = tf.reduce_mean(loss)

        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        # Convert Keras Tensors to TF Tensors
        # past_key_values = tf.nest.map_structure(tf.identity, output.past_key_values) # Not needed for classification
        hidden_states = tf.nest.map_structure(tf.identity, output.hidden_states)
        attentions = tf.nest.map_structure(tf.identity, output.attentions)

        return TFSequenceClassifierOutput(
            logits=output.logits,
            # past_key_values=past_key_values, # Not needed for classification
            hidden_states=hidden_states,
            attentions=attentions,
        )
