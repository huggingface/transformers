# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
""" TensorFlow Gemma model."""
import math
import warnings
from typing import Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFPreTrainedModel,
    get_initializer,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_gemma import GemmaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GemmaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = tf.reduce_sum(tf.cast(attention_mask, tf.int32), axis=-1)
    indices = tf.where(tf.reshape(attention_mask, [-1]))
    max_seqlen_in_batch = tf.reduce_max(seqlens_in_batch)
    cu_seqlens = tf.pad(tf.cumsum(seqlens_in_batch), [[1, 0]])
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class TFGemmaRMSNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.hidden_size = hidden_size

    def _norm(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)

    def build(self, input_shape=None):
        self.weight = self.add_weight(shape=(self.hidden_size,), initializer="zeros", trainable=True, name="weight")

    def call(self, x):
        output = self._norm(tf.cast(x, tf.float32))
        return output * (1 + self.weight)


class TFGemmaRotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

    def build(self, input_shape):
        self.inv_freq = 1.0 / (self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))
        # self.inv_freq = tf.expand_dims(tf.expand_dims(inv_freq, 0), 0)
        super().build(input_shape)

    def call(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = tf.expand_dims(tf.expand_dims(self.inv_freq, 0), -1)
        inv_freq_expanded = tf.cast(inv_freq_expanded, tf.float32)
        tile_multiples = [position_ids.shape[0], 1, 1]  # Define how many times to replicate along each axis
        inv_freq_expanded = tf.tile(inv_freq_expanded, tile_multiples)

        position_ids_expanded = tf.cast(tf.expand_dims(position_ids, axis=1), dtype=x.dtype)
        freqs = tf.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = tf.transpose(freqs, perm=[0, 2, 1])
        emb = tf.concat([freqs, freqs], axis=-1)
        return tf.cos(emb), tf.sin(emb)


# Claude: Translated from PyTorch to TensorFlow
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : tf.shape(x)[-1] // 2]
    x2 = x[..., tf.shape(x)[-1] // 2 :]
    return tf.concat([-x2, x1], axis=-1)


# Claude: Translated from PyTorch to TensorFlow
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`tf.Tensor`): The query tensor.
        k (`tf.Tensor`): The key tensor.
        cos (`tf.Tensor`): The cosine part of the rotary embedding.
        sin (`tf.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(tf.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = tf.expand_dims(cos, unsqueeze_dim)
    sin = tf.expand_dims(sin, unsqueeze_dim)
    q_embed = tf.math.multiply(q, cos) + tf.math.multiply(rotate_half(q), sin)
    k_embed = tf.math.multiply(k, cos) + tf.math.multiply(rotate_half(k), sin)
    return q_embed, k_embed


# Claude: Translated from PyTorch to TensorFlow
class TFGemmaMLP(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.gate_proj = tf.keras.layers.Dense(config.intermediate_size, use_bias=False, name="gate_proj")
        self.up_proj = tf.keras.layers.Dense(config.intermediate_size, use_bias=False, name="up_proj")
        self.down_proj = tf.keras.layers.Dense(config.hidden_size, use_bias=False, name="down_proj")
        self.act_fn = get_tf_activation(config.hidden_act)

    def call(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Claude: Translated from PyTorch to TensorFlow
def repeat_kv(hidden_states, n_rep):
    """
    This is the equivalent of tf.repeat(x, repeats=n_rep, axis=1). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = shape_list(hidden_states)
    if n_rep == 1:
        return hidden_states
    hidden_states = tf.expand_dims(hidden_states, 2)
    hidden_states = tf.repeat(hidden_states, n_rep, axis=2)
    return tf.reshape(hidden_states, [batch, num_key_value_heads * n_rep, slen, head_dim])


# Claude: Translated from PyTorch to TensorFlow
class TFGemmaAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = tf.keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=config.attention_bias, name="q_proj"
        )
        self.k_proj = tf.keras.layers.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=config.attention_bias, name="k_proj"
        )
        self.v_proj = tf.keras.layers.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=config.attention_bias, name="v_proj"
        )
        self.o_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=config.attention_bias, name="o_proj")
        self.rotary_emb = TFGemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            name="rotary_emb",
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: bool = None,
        **kwargs,
    ):
        bsz, q_len = shape_list(hidden_states)[:2]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = tf.reshape(query_states, [bsz, q_len, self.num_heads, self.head_dim])
        query_states = tf.transpose(query_states, [0, 2, 1, 3])
        key_states = tf.reshape(key_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        key_states = tf.transpose(key_states, [0, 2, 1, 3])
        value_states = tf.reshape(value_states, [bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states = tf.transpose(value_states, [0, 2, 1, 3])

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # Claude: The above code is commented out since the Cache class is not defined in this translation.
            # It would need to be implemented separately in TensorFlow. For now, just using the key and value states directly.
            pass

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = tf.matmul(query_states, key_states, transpose_b=True) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                start_position = cache_position[0]
                end_position = cache_position[1]  # Assuming you want a slice until this position
                # Generate a range of indices based on start and end positions
                indices = tf.range(start=start_position, limit=end_position)
                causal_mask = tf.gather(attention_mask, indices, axis=2)
            else:
                causal_mask = attention_mask
            attn_weights = attn_weights + tf.cast(causal_mask, attn_weights.dtype)

        # upcast attention to fp32
        attn_weights = stable_softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, query_states.dtype)
        # TODO: use `training` to figure out value for rate
        attn_weights = tf.nn.dropout(attn_weights, rate=0)
        attn_output = tf.matmul(attn_weights, value_states)

        attn_output_shape = shape_list(attn_output)
        expected_shape = [bsz, self.num_heads, q_len, self.head_dim]
        tf.debugging.assert_equal(
          attn_output_shape,
          expected_shape,
          message=f"`attn_output` should be of size {expected_shape}, but is {attn_output_shape}"
        )
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Claude: Translated from PyTorch to TensorFlow
class TFGemmaDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_idx, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size

        self.self_attn = TFGemmaAttention(config, layer_idx=layer_idx, name="self_attn")

        self.mlp = TFGemmaMLP(config, name="mlp")
        self.input_layernorm = TFGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="input_layernorm")
        self.post_attention_layernorm = TFGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, name="post_attention_layernorm"
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
            cache_position=cache_position,
            **kwargs,
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


GEMMA_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`GemmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Gemma Model outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
class TFGemmaPreTrainedModel(TFPreTrainedModel):
    config_class = GemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keep_in_fp32_modules = ["inv_freq", "rotary_emb", "cos_cached", "sin_cached"]
    _no_split_modules = ["TFGemmaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, tf.keras.layers.Dense):
            module.kernel.assign(tf.keras.initializers.TruncatedNormal(stddev=std)(shape_list(module.kernel)))
            if module.bias is not None:
                module.bias.assign(tf.keras.initializers.Zeros()(shape_list(module.bias)))
        elif isinstance(module, tf.keras.layers.Embedding):
            module.embeddings.assign(tf.keras.initializers.TruncatedNormal(stddev=std)(shape_list(module.embeddings)))
            if module.padding_idx is not None:
                module.embeddings[module.padding_idx].assign(tf.zeros_like(module.embeddings[module.padding_idx]))

    def _setup_cache(self, max_batch_size, max_cache_len=None):
        # Claude: The Cache class is not defined in this translation, so this method is left unimplemented for now.
        pass

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


GEMMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_tf_utils._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(tf.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class TFGemmaModel(TFGemmaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFGemmaMainLayer(config, name="model")

    @unpack_inputs
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
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

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)


@add_start_docstrings(
    "The bare Gemma Model outputting raw hidden-states without any specific head on top.",
    GEMMA_START_DOCSTRING,
)
# Claude: Translated from PyTorch to TensorFlow
class TFGemmaMainLayer(tf.keras.layers.Layer):
    config_class = GemmaConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            mask_zero=True if self.padding_idx == 0 else False,
            # embeddings_initializer=get_initializer(config.initializer_range),
            name="embed_tokens",
        )

        self.decoder_layers = [
            TFGemmaDecoderLayer(config, layer_idx=i, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]
        self.norm = TFGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="norm")

        self.causal_mask = 1 - tf.linalg.band_part(
            tf.ones((config.max_position_embeddings, config.max_position_embeddings)), -1, 0
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens.embeddings = value
        self.embed_tokens.vocab_size = shape_list(value)[0]

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[tf.Tensor] = None,
        training: bool = False,
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
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            # Claude: The Cache class is not defined in this translation, so this code path is not implemented.
            pass

        if cache_position is None:
            cache_position = tf.range(past_seen_tokens, past_seen_tokens + input_shape[1])

        if position_ids is None:
            position_ids = tf.expand_dims(cache_position, 0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        hidden_states = hidden_states * (self.config.hidden_size**0.5)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.decoder_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = None
        if use_cache:
            # Claude: The Cache class is not defined in this translation, so this code path is not implemented.
            pass

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        batch_size, seq_length = shape_list(input_tensor)[:2]
        dtype = input_tensor.dtype

        # support going beyond cached `max_position_embedding`
        if seq_length > self.config.max_position_embeddings:
            causal_mask = tf.ones(
                (2 * self.config.max_position_embeddings, 2 * self.config.max_position_embeddings), dtype=tf.bool
            )
            causal_mask = tf.linalg.band_part(causal_mask, 0, -1)
            causal_mask = tf.cast(causal_mask, dtype)
        else:
            causal_mask = tf.cast(self.causal_mask[:seq_length, :seq_length], dtype)

        causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 0)
        causal_mask = tf.repeat(causal_mask, batch_size, axis=0)

        if attention_mask is not None and tf.rank(attention_mask) == 2:
            mask_length = shape_list(attention_mask)[-1]
            padding_mask = tf.equal(causal_mask[..., :mask_length], 0) & tf.equal(
                tf.expand_dims(tf.expand_dims(attention_mask, 1), 1), 0
            )

            causal_mask = tf.where(padding_mask, tf.cast(tf.float32.min, dtype), causal_mask[..., :mask_length])

        return causal_mask


# Claude: Translated from PyTorch to TensorFlow
class TFGemmaForCausalLM(TFGemmaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config)
        self.model = TFGemmaMainLayer(config, name="model")
        self.vocab_size = config.vocab_size
        self.lm_head = tf.keras.layers.Dense(
            config.vocab_size,
            use_bias=False,
            name="lm_head",
            kernel_initializer=get_initializer(config.initializer_range),
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens.embeddings = value
        self.model.embed_tokens.vocab_size = shape_list(value)[0]

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[tf.Tensor] = None,
        training: bool = False,
    ):
        r"""
        Args:
            labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, TFGemmaForCausalLM

        >>> model = TFGemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="tf")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
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
            cache_position=cache_position,
            training=training,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                shift_labels, shift_logits, from_logits=True, axis=-1
            )

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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        past_length = 0

        if past_key_values is not None:
            # Claude: The Cache class is not defined in this translation, so this code path is not fully implemented.
            # It would need to be adapted to work with the TensorFlow cache format.
            past_length = shape_list(past_key_values[0][0])[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and shape_list(attention_mask)[1] > shape_list(input_ids)[1]:
                input_ids = input_ids[:, -(shape_list(attention_mask)[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < shape_list(input_ids)[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = tf.cumsum(attention_mask, axis=-1, exclusive=True)
            position_ids = tf.where(tf.equal(attention_mask, 0), 1, position_ids)
            if past_key_values:
                position_ids = position_ids[:, -shape_list(input_ids)[1] :]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = tf.range(past_length, past_length + shape_list(position_ids)[-1])

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # Claude: The Cache class is not defined in this translation, so this method is left unimplemented for now.
        # It would need to be adapted to work with the TensorFlow cache format.
        pass


@add_start_docstrings(
    """
    The Gemma Model transformer with a sequence classification head on top (linear layer).

    [`TFGemmaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GEMMA_START_DOCSTRING,
)
# Claude: Translated from PyTorch to TensorFlow
class TFGemmaForSequenceClassification(TFGemmaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = TFGemmaMainLayer(config, name="model")
        self.score = tf.keras.layers.Dense(self.num_labels, use_bias=False, name="score")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens.embeddings = value
        self.model.embed_tokens.vocab_size = shape_list(value)[0]

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GEMMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=TFSequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ):
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
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
            training=training,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = shape_list(input_ids)[0]
        else:
            batch_size = shape_list(inputs_embeds)[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    tf.reduce_sum(tf.cast(tf.not_equal(input_ids, self.config.pad_token_id), tf.int32), axis=-1) - 1
                )
            else:
                sequence_lengths = tf.constant([inputs_embeds.shape[1] - 1] * inputs_embeds.shape[0])

        indicies = tf.stack([tf.range(batch_size), sequence_lengths], axis=1)
        pooled_logits = tf.gather_nd(logits, indicies)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == tf.int32 or labels.dtype == tf.int64):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = tf.keras.losses.MeanSquaredError()
                if self.num_labels == 1:
                    loss = loss_fct(labels, pooled_logits)
                else:
                    loss = loss_fct(labels, pooled_logits)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                loss = loss_fct(labels, pooled_logits)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                loss = loss_fct(labels, pooled_logits)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# Claude: Some key points about the translation:
# - The PyTorch modules and functions have been replaced with their TensorFlow/Keras equivalents where possible. This includes using `tf.keras.layers` for the model layers, `tf.keras.losses` for the loss functions, etc.
# - The `torch.Tensor` inputs and outputs have been replaced with `tf.Tensor`.
# - PyTorch operations like `torch.matmul`, `torch.arange`, `torch.where`, etc. have been replaced with the TensorFlow equivalents.
# - The `forward` methods have been renamed to `call` to match the Keras layer API.
# - The `past_key_values` cache format seems to be custom in the PyTorch implementation with a `Cache` class. I've left comments indicating that this would need to be implemented separately for TensorFlow, as the cache format is different. For now, the cache-related code paths are mostly unimplemented or simplified.
# - There are a few PyTorch-specific functions like `apply_rotary_pos_emb` that I've directly translated to TensorFlow, but they may need further testing and optimization.
# - I've added some comments prefixed with "Claude:" to explain certain choices or highlight areas that may need further work.
# Let me know if you have any other questions! Translating a complex model like this requires careful testing and iteration to ensure correctness and performance.g
