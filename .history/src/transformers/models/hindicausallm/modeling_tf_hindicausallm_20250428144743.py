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

from typing import Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFSequenceClassifierOutput,
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
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_hindicausallm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HindiCausalLMConfig"


class TFHindiCausalLMLayerNorm(tf.keras.layers.Layer):
    """Layer normalization for HindiCausalLM."""

    def __init__(self, hidden_size, eps=1e-5, **kwargs):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight("weight", shape=[self.hidden_size], initializer="ones")
        self.bias = self.add_weight("bias", shape=[self.hidden_size], initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * tf.math.rsqrt(variance + self.eps)
        return x * self.weight + self.bias


class TFHindiCausalLMAttention(tf.keras.layers.Layer):
    """Multi-headed attention for HindiCausalLM."""

    def __init__(self, config: HindiCausalLMConfig, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        initializer = get_initializer(config.initializer_range)
        self.q_proj = tf.keras.layers.Dense(
            self.num_heads * self.head_dim, kernel_initializer=initializer, name="q_proj", use_bias=False
        )
        self.k_proj = tf.keras.layers.Dense(
            self.num_key_value_heads * self.head_dim, kernel_initializer=initializer, name="k_proj", use_bias=False
        )
        self.v_proj = tf.keras.layers.Dense(
            self.num_key_value_heads * self.head_dim, kernel_initializer=initializer, name="v_proj", use_bias=False
        )
        self.o_proj = tf.keras.layers.Dense(
            self.hidden_size, kernel_initializer=initializer, name="o_proj", use_bias=False
        )
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout)

        self.rotary_dim = self.head_dim
        self._init_rope()

    def _init_rope(self):
        # Initialize rotary embeddings
        inv_freq = 1.0 / (10000 ** (tf.range(0, self.rotary_dim, 2, dtype=tf.float32) / self.rotary_dim))
        self.inv_freq = tf.Variable(inv_freq, trainable=False, name="inv_freq")
        # Build the positions for the past so that we don't need to create it during inference
        positions = tf.range(self.max_position_embeddings, dtype=tf.float32)
        self.cache_positions = positions

    def _compute_sin_cos(self, positions):
        # positions: [seq_len]
        # inv_freq: [rotary_dim / 2]
        freqs = tf.einsum("i,j->ij", positions, self.inv_freq)  # [seq_len, rotary_dim / 2]
        emb = tf.concat([freqs, freqs], axis=-1)  # [seq_len, rotary_dim]
        return tf.sin(emb), tf.cos(emb)  # [seq_len, rotary_dim]

    def _apply_rotary_emb(self, query, key, cos, sin, position_ids):
        # query, key: [batch, heads, seq_len, head_dim]
        # cos, sin: [seq_len, head_dim]
        # position_ids: [batch, seq_len]
        cos = tf.gather(cos, position_ids)  # [batch, seq_len, head_dim]
        sin = tf.gather(sin, position_ids)  # [batch, seq_len, head_dim]

        cos = tf.expand_dims(cos, 1)  # [batch, 1, seq_len, head_dim]
        sin = tf.expand_dims(sin, 1)  # [batch, 1, seq_len, head_dim]

        # Apply rotary embeddings
        def _rotate_half(x):
            x1, x2 = tf.split(x, 2, axis=-1)
            return tf.concat([-x2, x1], axis=-1)

        query_rot = _rotate_half(query)
        key_rot = _rotate_half(key)

        query = query * cos + query_rot * sin
        key = key * cos + key_rot * sin

        return query, key

    def _repeat_kv(self, hidden_states, n_rep):
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = shape_list(hidden_states)
        if n_rep == 1:
            return hidden_states

        hidden_states = tf.expand_dims(hidden_states, 2)  # [batch, num_kv_heads, 1, seq_len, head_dim]
        hidden_states = tf.broadcast_to(
            hidden_states, [batch, num_key_value_heads, n_rep, slen, head_dim]
        )
        hidden_states = tf.reshape(hidden_states, [batch, num_key_value_heads * n_rep, slen, head_dim])
        return hidden_states

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        batch_size, seq_length = shape_list(hidden_states)[:2]

        # Project to query, key, and value
        query_states = self.q_proj(hidden_states)  # [batch, seq_len, num_heads * head_dim]
        key_states = self.k_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]

        # Reshape
        query_states = tf.reshape(query_states, [batch_size, seq_length, self.num_heads, self.head_dim])
        query_states = tf.transpose(query_states, [0, 2, 1, 3])  # [batch, num_heads, seq_len, head_dim]

        key_states = tf.reshape(key_states, [batch_size, seq_length, self.num_key_value_heads, self.head_dim])
        key_states = tf.transpose(key_states, [0, 2, 1, 3])  # [batch, num_kv_heads, seq_len, head_dim]

        value_states = tf.reshape(value_states, [batch_size, seq_length, self.num_key_value_heads, self.head_dim])
        value_states = tf.transpose(value_states, [0, 2, 1, 3])  # [batch, num_kv_heads, seq_len, head_dim]

        kv_seq_len = seq_length
        if past_key_value is not None:
            kv_seq_len += shape_list(past_key_value[0])[2]  # Adding sequence length of past keys

        # Apply RoPE to query and key states
        cos, sin = self._compute_sin_cos(self.cache_positions[:kv_seq_len])
        query_states, key_states = self._apply_rotary_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # Reuse k, v with past_key_value
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if past_key_value is not None else None

        # Repeat kv states for multi-query attention
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        # Scale query
        query_states = query_states / tf.math.sqrt(tf.cast(self.head_dim, query_states.dtype))

        # [batch, num_heads, seq_len, kv_seq_len]
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = stable_softmax(attn_weights, axis=-1)
        attn_weights = self.attention_dropout(attn_weights, training=training)

        # [batch, num_heads, seq_len, head_dim]
        attn_output = tf.matmul(attn_weights, value_states)

        # Reshape output
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])  # [batch, seq_len, num_heads, head_dim]
        attn_output = tf.reshape(attn_output, [batch_size, seq_length, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class TFHindiCausalLMMLP(tf.keras.layers.Layer):
    def __init__(self, config: HindiCausalLMConfig, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        initializer = get_initializer(config.initializer_range)
        self.gate_proj = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=initializer, name="gate_proj", use_bias=False
        )
        self.up_proj = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=initializer, name="up_proj", use_bias=False
        )
        self.down_proj = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=initializer, name="down_proj", use_bias=False
        )
        self.act_fn = get_tf_activation(config.hidden_act)

    def call(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TFHindiCausalLMDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: HindiCausalLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size

        self.self_attn = TFHindiCausalLMAttention(config, name="self_attn")
        self.mlp = TFHindiCausalLMMLP(config, name="mlp")
        self.input_layernorm = TFHindiCausalLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps, name="input_layernorm")
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
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:

        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
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

        if present_key_value is not None:
            outputs += (present_key_value,)

        return outputs


@keras_serializable
class TFHindiCausalLMMainLayer(tf.keras.layers.Layer):
    config_class = HindiCausalLMConfig

    def __init__(self, config: HindiCausalLMConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = tf.keras.layers.Embedding(
            config.vocab_size, config.hidden_size, name="embed_tokens"
        )
        self.layers = [
            TFHindiCausalLMDecoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]
        self.norm = TFHindiCausalLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps, name="norm")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            if attention_mask is None:
                batch_size, seq_length = input_shape
                # Create a causal attention mask
                mask = tf.ones((batch_size, seq_length, seq_length))
                mask_cond = tf.range(seq_length)[:, None] < tf.range(seq_length)[None, :]
                mask = tf.where(mask_cond, 0.0, mask) * -1e4
                mask = tf.expand_dims(mask, axis=1)  # [batch, 1, seq_len, seq_len]
                combined_attention_mask = mask
            else:
                # Convert attention_mask from [batch, seq_len] to [batch, 1, 1, seq_len]
                expanded_attn_mask = tf.expand_dims(tf.expand_dims(attention_mask, axis=1), axis=2)
                # Create causal mask
                batch_size, seq_length = input_shape
                mask_cond = tf.range(seq_length)[:, None] < tf.range(seq_length)[None, :]
                mask = tf.cast(mask_cond, dtype=tf.float32) * -1e4
                mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)  # [1, 1, seq_len, seq_len]
                mask = tf.broadcast_to(mask, [batch_size, 1, seq_length, seq_length])
                # Combine the masks
                combined_attention_mask = mask + (1.0 - tf.cast(expanded_attn_mask, dtype=tf.float32)) * -1e4

        return combined_attention_mask

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = shape_list(input_ids)
            input_shape = (batch_size, seq_length)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = shape_list(past_key_values[0][0])[2]  # past_key_values[0][0] shape: [batch, heads, past_seq_len, head_dim]

        if position_ids is None:
            position_ids = tf.range(past_key_values_length, seq_length + past_key_values_length, dtype=tf.int32)[tf.newaxis, :]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # Prepare for decoder
        hidden_states = inputs_embeds

        # Initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * self.num_hidden_layers

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Decoder layers
        for idx, (decoder_layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
        )


class TFHindiCausalLMPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HindiCausalLMConfig
    base_model_prefix = "model"


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
    methods like `call()` you will need to be explicit and pass your inputs either as a list/tuple (e.g.,
    `model(inputs, training=True)` or as keyword arguments (e.g., `model(input_ids=inputs, training=True)`).

    </Tip>

    Args:
        config ([`HindiCausalLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare HindiCausalLM Model transformer outputting raw hidden-states without any specific head on top.",
    HINDICAUSALLM_START_DOCSTRING,
)
class TFHindiCausalLMModel(TFHindiCausalLMPreTrainedModel):
    def __init__(self, config: HindiCausalLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFHindiCausalLMMainLayer(config, name="model")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(HINDICAUSALLM_START_DOCSTRING)
    def call(
        self,
        input_ids: TFModelInputType = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
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
        )
        return outputs

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            hidden_states=hs,
            attentions=attns,
        )


@add_start_docstrings(
    """
    The HindiCausalLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    HINDICAUSALLM_START_DOCSTRING,
)
class TFHindiCausalLMForCausalLM(TFHindiCausalLMPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config: HindiCausalLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFHindiCausalLMModel(config, name="model")

        # Create the LM head
        self.lm_head = tf.keras.layers.Dense(
            config.vocab_size, use_bias=False, name="lm_head", kernel_initializer=get_initializer(config.initializer_range)
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.weight = new_embeddings

    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # If past_key_values are used, only the last token should be passed
        if past_key_values is not None:
            inputs = tf.expand_dims(inputs[:, -1], -1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            batch_size, seq_length = shape_list(attention_mask)
            position_ids = tf.range(0, seq_length, dtype=tf.int32)[tf.newaxis, :]

            # If past_key_values is used, calculate the position_ids differently
            if past_key_values is not None:
                position_ids = position_ids[:, -1:]

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @unpack_inputs
    @add_start_docstrings_to_model_forward(HINDICAUSALLM_START_DOCSTRING)
    def call(
        self,
        input_ids: TFModelInputType = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFCausalLMOutputWithCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
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
        )

        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        logits = self.lm_head(sequence_output)  # [batch_size, seq_len, vocab_size]

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

        return TFCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        cross_attns = (
            tf.convert_to_tensor(output.cross_attentions)
            if self.config.output_attentions and output.cross_attentions is not None
            else None
        )

        return TFCausalLMOutputWithCrossAttentions(
            logits=output.logits,
            past_key_values=pkv,
            hidden_states=hs,
            attentions=attns,
            cross_attentions=cross_attns,
        )


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
    @add_start_docstrings_to_model_forward(HINDICAUSALLM_START_DOCSTRING)
    def call(
        self,
        input_ids: TFModelInputType = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
        )
        hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size]
        logits = self.score(hidden_states)  # [batch_size, seq_len, num_labels]

        if input_ids is not None:
            batch_size = shape_list(input_ids)[0]
        else:
            batch_size = shape_list(inputs_embeds)[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        # Find the last non-padding token
        if self.config.pad_token_id is not None:
            if input_ids is not None:
                sequence_lengths = tf.reduce_sum(tf.cast(tf.not_equal(input_ids, self.config.pad_token_id), tf.int32), axis=-1) - 1
            else:
                sequence_lengths = -1
        else:
            sequence_lengths = -1

        # Get the logits for the last token
        pooled_logits = tf.gather_nd(
            logits,
            tf.stack([tf.range(batch_size), tf.cast(tf.math.maximum(sequence_lengths, 0), tf.int32)], axis=1)
        )

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
                    loss = loss_fn(tf.reshape(labels, [-1]), tf.reshape(pooled_logits, [-1]))
                else:
                    loss = loss_fn(tf.reshape(labels, [-1, self.num_labels]), tf.reshape(pooled_logits, [-1, self.num_labels]))
                loss = tf.reshape(loss, [-1, 1])
            elif self.config.problem_type == "single_label_classification":
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
                loss = loss_fn(labels, pooled_logits)
            elif self.config.problem_type == "multi_label_classification":
                loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
                loss = loss_fn(labels, pooled_logits)

        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)
