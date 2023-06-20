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
""" TensorFlow LLaMA model."""
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from ...activations import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutputWithPast
from ...modeling_tf_utils import TFPreTrainedModel, get_initializer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ...tf_utils import shape_list
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

LARGE_NEGATIVE = -1e8


# Copied from transformers.models.bart.modeling_tf_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    mask_cond = tf.range(shape_list(mask)[-1])

    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE
class TFLlamaRMSNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(**kwargs)
        self.weight = self.add_weight(name="weight", shape=(hidden_size,), initializer="ones")
        self.variance_epsilon = eps

    def call(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = tf.math.reduce_mean(tf.math.square(tf.cast(hidden_states, tf.float32)), axis=-1, keepdims=True)
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).cast(input_dtype)
class TFLlamaRotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super().__init__(**kwargs)
        inv_freq = 1.0 / (base ** (tf.range(0, dim, 2, dtype=tf.float32) / dim))
        self.inv_freq = tf.constant(inv_freq, dtype=tf.float32)

        # Build here to make `tf.function` work.
        self.max_seq_len_cached = max_position_embeddings
        t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = tf.concat([freqs, freqs], axis=-1)
        self.cos_cached = tf.math.cos(emb)[None, None, :, :]
        self.sin_cached = tf.math.sin(emb)[None, None, :, :]

    def call(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = tf.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = tf.concat([freqs, freqs], axis=-1)
            self.cos_cached = tf.math.cos(emb)[None, None, :, :]
            self.sin_cached = tf.math.sin(emb)[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )
def rotate_half(self, x):
    """Rotates half the hidden dims of the input."""
    x_shape = shape_list(x)
    x1 = x[..., : x_shape[-1] // 2]
    x2 = x[..., x_shape[-1] // 2 :]
    return tf.concat((-x2, x1), axis=-1)
def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = tf.squeeze(cos, axis=[0, 1])  # [seq_len, dim]
    sin = tf.squeeze(sin, axis=[0, 1])  # [seq_len, dim]
    cos = tf.gather(cos, position_ids, axis=0)
    cos = tf.expand_dims(cos, axis=1)  # [bs, 1, seq_len, dim]
    sin = tf.gather(sin, position_ids, axis=0)
    sin = tf.expand_dims(sin, axis=1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (self.rotate_half(q) * sin)
    k_embed = (k * cos) + (self.rotate_half(k) * sin)
    return q_embed, k_embed
class TFLlamaMLP(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gate_proj = tf.keras.layers.Dense(intermediate_size, use_bias=False, name="gate_proj")
        self.down_proj = tf.keras.layers.Dense(hidden_size, use_bias=False, name="down_proj")
        self.up_proj = tf.keras.layers.Dense(intermediate_size, use_bias=False, name="up_proj")
        self.act_fn = ACT2FN[hidden_act]

    def call(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
class TFLlamaAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="v_proj")
        self.o_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name="o_proj")
        self.rotary_emb = TFLlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, name="rotary_emb")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        bsz, q_len, _ = shape_list(hidden_states)

        query_states = self._shape(self.q_proj(hidden_states), q_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), q_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), q_len, bsz)

        kv_seq_len = shape_list(key_states)[-2]
        if past_key_value is not None:
            kv_seq_len += shape_list(past_key_value[0])[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = tf.matmul(query_states, tf.transpose(key_states, perm=[0, 1, 3, 2])) / tf.math.sqrt(float(self.head_dim))

        if shape_list(attn_weights) != [bsz, self.num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of size {[bsz, self.num_heads, q_len, kv_seq_len]}, but is"
                f" {shape_list(attn_weights)}"
            )

        if attention_mask is not None:
            if shape_list(attention_mask) != [bsz, 1, q_len, kv_seq_len]:
                raise ValueError(
                    f"Attention mask should be of size {[bsz, 1, q_len, kv_seq_len]}, but is {shape_list(attention_mask)}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = tf.math.maximum(
                attn_weights, tf.constant(float(tf.keras.backend.min_value()), dtype=attn_weights.dtype)
            )

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_output = tf.matmul(attn_weights, value_states)

        if shape_list(attn_output) != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {[bsz, self.num_heads, q_len, self.head_dim]}, but is"
                f" {shape_list(attn_output)}"
            )

        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
class TFLlamaDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.self_attn = TFLlamaAttention(config=config, name="self_attn")
        self.mlp = TFLlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            name="mlp",
        )
        self.input_layernorm = TFLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="input_layernorm")
        self.post_attention_layernorm = TFLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="post_attention_layernorm")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training=False,
    ) -> Tuple[tf.Tensor, Optional[Tuple[tf.Tensor, tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(tf.Tensor)`, *optional*): cached past key and value projection states
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

        if use_cache:
            outputs += (present_key_value,)

        return outputs


TF_LLAMA_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a TensorFlow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) subclass. Use it as a
    regular TensorFlow Layer and refer to the TensorFlow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    TF_LLAMA_START_DOCSTRING,
)
class TFLlamaPreTrainedModel(TFPreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TFLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, tf.keras.layers.Dense):
            module.kernel = tf.random.normal(shape=module.kernel.shape, mean=0.0, stddev=std)
            if module.bias is not None:
                module.bias = tf.zeros_like(module.bias)
        elif isinstance(module, tf.keras.layers.Embedding):
            module.embeddings = tf.random.normal(shape=module.embeddings.shape, mean=0.0, stddev=std)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TFLlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
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

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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
@add_start_docstrings(
    "The bare TFLLaMA Model outputting raw hidden-states without any specific head on top.",
    TF_LLAMA_START_DOCSTRING,
)
class TFLlamaModel(TFLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TFLlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = tf.keras.layers.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TFLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = TFLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = shape_list(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = shape_list(inputs_embeds)
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = shape_list(past_key_values[0][0])[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = tf.range(
                past_key_values_length, seq_length + past_key_values_length, dtype=tf.int32
            )
            position_ids = tf.expand_dims(position_ids, 0)
        else:
            position_ids = tf.cast(position_ids, dtype=tf.int32)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = tf.ones(
                (batch_size, seq_length_with_past), dtype=tf.bool
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

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
                training=training,
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
        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
class TFLlamaForCausalLM(TFLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = TFLlamaModel(config)

        self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False, name="lm_head")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training=False,
    ) -> Union[Tuple, TFCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, TFLlamaForCausalLM

        >>> model = TFLlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="tf")

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
            loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            shift_logits = tf.reshape(shift_logits, (-1, self.config.vocab_size))
            shift_labels = tf.reshape(shift_labels, (-1,))
            loss = loss_fct(shift_labels, shift_logits)

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
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = tf.math.cumsum(tf.cast(attention_mask, tf.int32), axis=-1) - 1
            position_ids = tf.where(attention_mask == 0, 1, position_ids)
            if past_key_values:
                position_ids = position_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(tf.gather(past_state, beam_idx) for past_state in layer_past),
            )
        return reordered_past
@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`TFLlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    TF_LLAMA_START_DOCSTRING,
)
class TFLlamaForSequenceClassification(TFLlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.model = TFLlamaModel(config)
        self.score = tf.keras.layers.Dense(self.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="score")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def call(
        self,
        input_ids: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[tf.Tensor]] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFSequenceClassifierOutputWithPast]:
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
                sequence_lengths = tf.reduce_sum(tf.cast(tf.not_equal(input_ids, self.config.pad_token_id), dtype=tf.int32), axis=-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = tf.gather(logits, sequence_lengths, batch_dims=1)

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
                    loss = loss_fct(pooled_logits[:, 0], labels)
                else:
                    loss = loss_fct(pooled_logits, labels)
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