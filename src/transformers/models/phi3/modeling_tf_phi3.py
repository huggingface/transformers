

import math
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

# from ...cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFSequenceClassifierOutputWithPast,
)
from transformers.modeling_tf_utils import (
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from transformers.tf_utils import shape_list
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

_CHECKPOINT_FOR_DOC = "microsoft/Phi-3-mini-4k-instruct"
_CONFIG_FOR_DOC = "Phi3Config"

logger = logging.get_logger(__name__)
class TFPhi3RMSNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super(TFPhi3RMSNorm, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = self.add_weight(name='weight',
                                      shape=[hidden_size],
                                      initializer='ones',
                                      trainable=True)

    def call(self, inputs):
        input_dtype = inputs.dtype
        hidden_states = tf.cast(inputs, tf.float32)
        variance = tf.reduce_mean(tf.square(hidden_states), axis=-1, keepdims=True)
        normalized_states = hidden_states * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * tf.cast(normalized_states, input_dtype)

def _get_unpad_data(attention_mask):
    # Calculate sequence lengths in the batch
    seqlens_in_batch = tf.reduce_sum(attention_mask, axis=-1)

    # Get the indices of non-zero elements
    indices = tf.where(tf.reshape(attention_mask, [-1]))
    indices = tf.reshape(indices, [-1])

    # Find the maximum sequence length in the batch
    max_seqlen_in_batch = tf.reduce_max(seqlens_in_batch).numpy()

    # Compute cumulative sequence lengths
    cu_seqlens = tf.concat([[0], tf.cumsum(seqlens_in_batch)], axis=0)

    return indices, cu_seqlens, max_seqlen_in_batch

class TFPhi3RotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super(TFPhi3RotaryEmbedding, self).__init__(**kwargs)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim))
        self.inv_freq = self.add_weight(name="inv_freq",
                                        shape=inv_freq.shape,
                                        initializer=tf.constant_initializer(inv_freq.numpy()),
                                        trainable=False)

    def build(self, input_shape):
        self.inv_freq = 1.0 / (
            self.base ** (tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim)
        )
        super().build(input_shape)

    def call(self, x, position_ids):
        inv_freq_expanded = tf.expand_dims(self.inv_freq, axis=0)
        inv_freq_expanded = tf.expand_dims(inv_freq_expanded, axis=-1)
        inv_freq_expanded = tf.tile(inv_freq_expanded, [position_ids.shape[0], 1, 1])

        position_ids_expanded = tf.expand_dims(position_ids, axis=1)
        position_ids_expanded = tf.cast(position_ids_expanded, tf.float32)

        freqs = tf.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = tf.transpose(freqs, perm=[0, 2, 1])
        emb = tf.concat([freqs, freqs], axis=-1)

        cos = tf.math.cos(emb)
        sin = tf.math.sin(emb)

        return tf.cast(cos, x.dtype), tf.cast(sin, x.dtype)

class TFPhi3SuScaledRotaryEmbedding(TFPhi3RotaryEmbedding):
    def __init__(self, dim, config, **kwargs):
        super(TFPhi3SuScaledRotaryEmbedding, self).__init__(dim, config.max_position_embeddings, config.rope_theta, **kwargs)
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    def call(self, x, position_ids, seq_len=None):
        seq_len = tf.reduce_max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = tf.constant(self.long_factor, dtype=tf.float32)
        else:
            ext_factors = tf.constant(self.short_factor, dtype=tf.float32)

        inv_freq_shape = tf.range(0, self.dim, 2, dtype=tf.float32) / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = tf.expand_dims(self.inv_freq, axis=0)
        inv_freq_expanded = tf.expand_dims(inv_freq_expanded, axis=-1)
        inv_freq_expanded = tf.tile(inv_freq_expanded, [position_ids.shape[0], 1, 1])

        position_ids_expanded = tf.expand_dims(position_ids, axis=1)
        position_ids_expanded = tf.cast(position_ids_expanded, tf.float32)

        freqs = tf.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = tf.transpose(freqs, perm=[0, 2, 1])
        emb = tf.concat([freqs, freqs], axis=-1)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

        cos = tf.math.cos(emb) * scaling_factor
        sin = tf.math.sin(emb) * scaling_factor

        return tf.cast(cos, x.dtype), tf.cast(sin, x.dtype)

class TFPhi3YarnScaledRotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, config, **kwargs):
        super(TFPhi3YarnScaledRotaryEmbedding, self).__init__(**kwargs)
        self.dim = dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, position_ids, seq_len=None):
        seq_len = tf.reduce_max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = tf.convert_to_tensor(self.long_factor, dtype=tf.float32)
        else:
            ext_factors = tf.convert_to_tensor(self.short_factor, dtype=tf.float32)

        inv_freq_shape = tf.cast(tf.range(0, self.dim, 2), tf.float32) / self.dim
        inv_freq = 1.0 / (ext_factors * (self.base ** inv_freq_shape))

        inv_freq_expanded = tf.expand_dims(inv_freq, axis=0)
        inv_freq_expanded = tf.expand_dims(inv_freq_expanded, axis=2)
        inv_freq_expanded = tf.tile(inv_freq_expanded, [tf.shape(position_ids)[0], 1, 1])

        position_ids_expanded = tf.cast(tf.expand_dims(position_ids, axis=1), tf.float32)
        position_ids_expanded = tf.transpose(position_ids_expanded, perm=[0, 2, 1])


        # Perform the rotary embedding calculations
        freqs = tf.matmul(inv_freq_expanded, position_ids_expanded, transpose_b=True)
        emb = tf.concat([freqs, freqs], axis=-1)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = 0.1 * math.log(scale) + 1.0

        cos = tf.math.cos(emb) * scaling_factor
        sin = tf.math.sin(emb) * scaling_factor

        return tf.cast(cos, dtype=x.dtype), tf.cast(sin, dtype=x.dtype)

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



class TFPhi3MLP(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(TFPhi3MLP, self).__init__(**kwargs)
        self.config = config
        self.gate_up_proj = tf.keras.layers.Dense(2 * config.intermediate_size, use_bias=False)
        self.down_proj = tf.keras.layers.Dense(config.hidden_size, use_bias=False)
        self.activation_fn = self.get_activation_function(config.hidden_act)

    def get_activation_function(self, activation_name):
        if activation_name == "relu":
            return tf.nn.relu
        elif activation_name == "gelu":
            return tf.nn.gelu
        elif activation_name == "tanh":
            return tf.nn.tanh
        elif activation_name == "sigmoid":
            return tf.nn.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def call(self, hidden_states):
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = tf.split(up_states, 2, axis=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)

    def build(self, input_shape=None):
        self.gate_up_proj.build(input_shape)
        self.down_proj.build([input_shape[0], self.config.intermediate_size])
        super().build(input_shape)


def repeat_kv(hidden_states: tf.Tensor, n_rep: int) -> tf.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = tf.shape(hidden_states)
    if n_rep == 1:
        return hidden_states
    hidden_states = tf.expand_dims(hidden_states, axis=2)  # Shape: (batch, num_key_value_heads, 1, seqlen, head_dim)
    hidden_states = tf.tile(hidden_states, [1, 1, n_rep, 1, 1])  # Shape: (batch, num_key_value_heads, n_rep, seqlen, head_dim)
    return tf.reshape(hidden_states, [batch, num_key_value_heads * n_rep, slen, head_dim])

class TFPhi3Attention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            tf.get_logger().warning(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = Dense(self.hidden_size, use_bias=False, name="o_proj")
        self.qkv_proj = Dense(op_size, use_bias=False, name="qkv_proj")
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = TFPhi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "su":
                self.rotary_emb = TFPhi3SuScaledRotaryEmbedding(self.head_dim, self.config)
            elif scaling_type == "yarn":
                self.rotary_emb = TFPhi3YarnScaledRotaryEmbedding(self.head_dim, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        training=False,
        **kwargs,
    ):
        tf.get_logger().warning("You are not running the flash-attention implementation, expect numerical differences.")

        bsz, q_len, _ = shape_list(hidden_states)

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = tf.transpose(tf.reshape(query_states, [bsz, q_len, self.num_heads, self.head_dim]), perm=[0, 2, 1, 3])
        key_states = tf.transpose(tf.reshape(key_states, [bsz, q_len, self.num_key_value_heads, self.head_dim]), perm=[0, 2, 1, 3])
        value_states = tf.transpose(tf.reshape(value_states, [bsz, q_len, self.num_key_value_heads, self.head_dim]), perm=[0, 2, 1, 3])

        kv_seq_len = shape_list(key_states)[-2]
        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # Claude: The above code is commented out since the Cache class is not defined in this translation.
            # It would need to be implemented separately in TensorFlow. For now, just using the key and value states directly.
            pass
        # kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = tf.matmul(query_states, key_states, transpose_b=True) / tf.math.sqrt(float(self.head_dim))

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

        # upcast attention to fp32
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, dtype=value_states.dtype)
        attn_weights = Dropout(self.attention_dropout)(attn_weights)

        attn_output = tf.matmul(attn_weights, value_states)

        if shape_list(attn_output) != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {[bsz, self.num_heads, q_len, self.head_dim]}, but is"
                f" {shape_list(attn_output)}"
            )

        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [bsz, q_len, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def build(self, input_shape=None):
        self.qkv_proj.build(input_shape)
        self.o_proj.build([input_shape[0], input_shape[1], self.hidden_size])
        super().build(input_shape)


class TFPhi3DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = TFPhi3Attention(config, layer_idx=layer_idx, name="self_attn")
        self.mlp = TFPhi3MLP(config, name="mlp")
        self.input_layernorm = TFPhi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="input_layernorm")
        self.post_attention_layernorm = TFPhi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="post_attention_layernorm")
        self.resid_attn_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.resid_mlp_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        training=False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`tf.Tensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`tf.Tensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. 
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            past_key_value (`Tuple(tf.Tensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            training=training,
            **kwargs,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs, training=training)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states, training=training)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def build(self, input_shape=None):
        self.self_attn.build(input_shape)
        self.mlp.build(input_shape)
        self.input_layernorm.build(input_shape)
        self.post_attention_layernorm.build(input_shape)
        super().build(input_shape)


PHI3_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a TensorFlow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass.
    Use it as a regular TensorFlow Model and refer to the TensorFlow documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Phi3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare Phi-3 model outputting raw hidden-states without any specific head on top.",
    PHI3_START_DOCSTRING,
)
class TFPhi3PreTrainedModel(TFPreTrainedModel):
    config_class = Phi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TFPhi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False  # FlashAttention is not supported in Keras
    _supports_sdpa = False
    _supports_cache_class = True

    _version = "0.0.5"

    def _init_weights(self, layer):
        std = self.config.initializer_range
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_initializer = get_initializer(std)
            if layer.bias is not None:
                layer.bias_initializer = tf.zeros_initializer()
        elif isinstance(layer, tf.keras.layers.Embedding):
            layer.embeddings_initializer = get_initializer(std)
            if hasattr(layer, 'padding_idx') and layer.padding_idx is not None:
                embeddings = layer.embeddings
                embeddings[layer.padding_idx].assign(tf.zeros_like(embeddings[layer.padding_idx]))



PHI3_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
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
        past_key_values (`Cache` or `tuple(tuple(tf.Tensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(tf.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

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

@add_start_docstrings(
    "The bare Phi-3 model outputting raw hidden-states without any specific head on top.",
    PHI3_START_DOCSTRING,
)



class TFPhi3Model(TFPhi3PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFPhi3MainLayer(config, name="model")

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
    "The bare Phi-3 model outputting raw hidden-states without any specific head on top.",
    PHI3_START_DOCSTRING,
)

# Claude: Translated from PyTorch to TensorFlow
@tf.keras.utils.register_keras_serializable()
class TFPhi3MainLayer(keras.layers.Layer):
    config_class = Phi3Config

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
            TFPhi3DecoderLayer(config, layer_idx=i, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]
        self.norm = TFPhi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, name="norm")

        self.causal_mask = 1 - tf.linalg.band_part(
            tf.ones((config.max_position_embeddings, config.max_position_embeddings)), -1, 0
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @unpack_inputs
    @add_start_docstrings_to_model_forward(PHI3_INPUTS_DOCSTRING)
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
            input_shape = tf.shape(input_ids)
        elif inputs_embeds is not None:
            input_shape = tf.shape(inputs_embeds)[:-1]
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
    @tf.function
    def _update_causal_mask(self, attention_mask, input_tensor):
        batch_size = tf.shape(input_tensor)[0]
        seq_length = tf.shape(input_tensor)[1]
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
            mask_length = tf.shape(attention_mask)[-1]
            padding_mask = tf.equal(causal_mask[..., :mask_length], 0) & tf.equal(
                tf.expand_dims(tf.expand_dims(attention_mask, 1), 1), 0
            )

            causal_mask = tf.where( padding_mask, tf.cast(tf.float32.min, dtype), causal_mask[..., :mask_length],)

        return causal_mask

    def build(self, input_shape=None):
        if self.built:
            return

        if input_shape is None:
            input_shape = [None, self.config.max_position_embeddings]
        # Check if input_shape is a dictionary and extract shapes
        if isinstance(input_shape, dict):
            if "input_ids" in input_shape.keys():
                input_shape = input_shape["input_ids"]
            else:
                input_shape = input_shape["inputs_embeds"]
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(input_shape)
        if getattr(self, "norm", None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build(input_shape)
        if getattr(self, "decoder_layers", None) is not None:
            for layer in self.decoder_layers:
                with tf.name_scope(layer.name):
                    layer.build([None, input_shape[1], self.config.hidden_size])
        self.built = True

class TFPhi3ForCausalLM(TFPhi3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = TFPhi3MainLayer(config, name="model")
        self.vocab_size = config.vocab_size
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False, name="lm_head", kernel_initializer=get_initializer(config.initializer_range))


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        if getattr(self.config, "use_output_embedding", False):
            return self.lm_head
        return None

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def call(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
             inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        cache_position=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                             return_dict=return_dict, cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = tf.reshape(logits[..., :-1, :], [-1, self.config.vocab_size])
            shift_labels = tf.reshape(labels[..., 1:], [-1])
            loss = tf.keras.losses.sparse_categorical_crossentropy(shift_labels, shift_logits, from_logits=True)

        if not return_dict:
            return (logits,) + outputs[1:] if loss is None else (loss, logits) + outputs[1:]

        return TFCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                        hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0].shape[2]
            input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = tf.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {"input_ids": input_ids}
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds

        model_inputs.update({"position_ids": position_ids, "past_key_values": past_key_values,
                             "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask})
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(tf.gather(layer_past, beam_idx, axis=0) for layer_past in past_key_values)

    def build(self, input_shape=None):
        if self.built:
            return

        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(input_shape)
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, self.config.hidden_size])


@add_start_docstrings(
    """
    The `TFPhi3Model` with a sequence classification head on top (linear layer).

    `TFPhi3ForSequenceClassification` uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.
    """,
    PHI3_START_DOCSTRING,
)
class TFPhi3ForSequenceClassification(TFPhi3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = TFPhi3MainLayer(config, name="model")
        self.score = tf.keras.layers.Dense(config.num_labels, use_bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def build(self, input_shape):
        # Build the underlying model layers
        if isinstance(input_shape, dict):
            if "input_ids" in input_shape.keys():
                input_shape = input_shape["input_ids"]
            else:
                input_shape = input_shape["inputs_embeds"]

        self.model.build(input_shape)
        self.score.build((input_shape[0], self.config.hidden_size))
        super().build(input_shape)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(PHI3_INPUTS_DOCSTRING)
    def call(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
             inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
             return_dict=None, training=False):
        if isinstance(input_ids, dict):
            input_ids = input_ids.get("input_ids")
            attention_mask = input_ids.get("attention_mask")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                                   output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                   return_dict=return_dict, training=training)
        hidden_states = model_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = tf.shape(input_ids)[0]
            sequence_lengths = tf.reduce_max(tf.cast(tf.not_equal(input_ids, self.config.pad_token_id), tf.int32), axis=-1) - 1
        else:
            batch_size = tf.shape(inputs_embeds)[0]
            sequence_lengths = tf.reduce_max(tf.cast(tf.not_equal(inputs_embeds, self.config.pad_token_id), tf.int32), axis=-1) - 1

        if self.config.pad_token_id is None:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        pooled_logits = tf.gather(logits, sequence_lengths, batch_dims=1)

        loss = None
        if labels is not None:
            labels = tf.cast(labels, tf.float32)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss = tf.keras.losses.mean_squared_error(labels, pooled_logits)
            elif self.config.problem_type == "single_label_classification":
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, pooled_logits, from_logits=True)

        if not return_dict:
            return (pooled_logits,) + model_outputs[1:] if loss is None else (loss, pooled_logits) + model_outputs[1:]

        return TFSequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, past_key_values=model_outputs.past_key_values,
                                                  hidden_states=model_outputs.hidden_states, attentions=model_outputs.attentions)

@add_start_docstrings(
    """
    [`TFPhi3Model`] with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    PHI3_START_DOCSTRING,
)
class TFPhi3ForTokenClassification(TFPhi3PreTrainedModel):
    def __init__(self, config: Phi3Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.model = TFPhi3MainLayer(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Dense(config.num_labels, activation='softmax')

    @add_start_docstrings_to_model_forward(PHI3_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor, tf.Tensor], ...]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[tf.Tensor], TokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, seq_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = model_outputs[0]
        hidden_states = self.dropout(hidden_states, training=kwargs.get('training', False))
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fn = SparseCategoricalCrossentropy(from_logits=False)
            loss = loss_fn(labels, logits)

        if not return_dict:
            output = (logits,) + model_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
