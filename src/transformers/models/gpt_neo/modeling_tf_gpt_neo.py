# coding=utf-8
# Copyright 2023 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
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
""" TensorFlow GPT Neo model."""


from typing import Optional, Tuple, Union

import tensorflow as tf

from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFCausalLMOutputWithPast,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import TFPreTrainedModel, get_tf_activation
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gpt_neo import GPTNeoConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPTNeoConfig"

TF_GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neo-1.3B",
    # See all GPTNeo models at https://huggingface.co/models?filter=gpt_neo
]

_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neo-1.3B"


class TFGPTNeoSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, attention_type, **kwargs):
        super().__init__(**kwargs)
        max_positions = config.max_position_embeddings
        bias = tf.linalg.band_part(tf.ones((max_positions, max_positions), dtype=tf.bool), -1, 0)
        bias = tf.reshape(bias, (1, 1, max_positions, max_positions))
        if attention_type == "local":
            bias = tf.math.logical_xor(bias, tf.linalg.band_part(bias, -1, -config.window_size))
        self.bias = bias
        self.masked_bias = tf.constant(-1e9)
        self.attn_dropout = tf.keras.layers.Dropout(rate=config.attention_dropout)
        self.resid_dropout = tf.keras.layers.Dropout(rate=config.resid_dropout)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim: {self.embed_dim} and num_heads:"
                f" {self.num_heads})."
            )
        self.k_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=False, name="v_proj")
        self.q_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=False, name="q_proj")
        self.out_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True, name="out_proj")

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tf.reshape(tensor, new_shape)
        return tf.transpose(tensor, perm=[0, 2, 1, 3])  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tf.reshape(tensor, new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        query = tf.cast(query, dtype=tf.float32)
        key = tf.cast(key, dtype=tf.float32)
        attn_weights = tf.matmul(query, key, transpose_b=True)
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = tf.keras.backend.min(attn_weights)
        mask_value = tf.broadcast_to(mask_value, attn_weights.shape)
        attn_weights = tf.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, dtype=value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = tf.matmul(attn_weights, value)
        return attn_output, attn_weights

    def call(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
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
            key = tf.concat((past_key, key), axis=-2)
            value = tf.concat((past_value, value), axis=-2)
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs  # a, present, (attentions)


class TFGPTNeoAttention(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = TFGPTNeoSelfAttention(
                config, self.attention_type, name=f"{self.attention_type}_attention"
            )
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def call(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )


class TFGPTNeoMLP(tf.keras.layers.Layer):
    def __init__(self, intermediate_size, config, **kwargs):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__(**kwargs)
        embed_dim = config.hidden_size
        self.c_fc = tf.keras.layers.Dense(embed_dim, intermediate_size, name="c_fc")
        self.c_proj = tf.keras.layers.Dense(intermediate_size, embed_dim, name="c_proj")
        self.act = get_tf_activation(config.activation_function)
        self.dropout = tf.keras.layers.Dropout(float(config.resid_dropout))

    def call(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TFGPTNeoBlock(tf.keras.layers.Layer):
    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = tf.keras.layers.LayerNormalization(hidden_size, epsilon=config.layer_norm_epsilon, name="ln_1")
        self.attn = TFGPTNeoAttention(config, layer_id, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(hidden_size, epsilon=config.layer_norm_epsilon, name="ln_2")
        self.mlp = TFGPTNeoMLP(inner_dim, config, name="mlp")

    def call(
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


GPT_NEO_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`GPTNeoConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_NEO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.num_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`tf.Tensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
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


class TFGPTNeoPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


@add_start_docstrings(
    "The bare GPT Neo Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEO_START_DOCSTRING,
)
class TFGPTNeoModel(TFGPTNeoPreTrainedModel, tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.embed_dim = config.hidden_size
        self.wte = tf.keras.layers.Embedding(config.vocab_size, self.embed_dim, name="wte")
        self.wpe = tf.keras.layers.Embedding(config.max_position_embeddings, self.embed_dim, name="wpe")
        self.drop = tf.keras.layers.Dropout(float(config.embed_dropout))
        self.h = [TFGPTNeoBlock(config, layer_id=i, name=f"h_{i}") for i in range(config.num_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[tf.Tensor]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = tf.shape(input_ids)[:-1]
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = tf.shape(inputs_embeds)[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, (-1, input_shape[-1]))
        if position_ids is not None:
            position_ids = tf.reshape(position_ids, (-1, input_shape[-1]))
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]
        if position_ids is None:
            position_ids = tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, 0)
            position_ids = tf.reshape(position_ids, (-1, input_shape[-1]))
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = tf.reshape(attention_mask, (batch_size, -1))
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = tf.cast(attention_mask, self.dtype)
            attention_mask = (1.0 - attention_mask) * tf.reduce_min(self.dtype)
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
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
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
        hidden_states = tf.reshape(hidden_states, output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """
    The GPT Neo Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_NEO_START_DOCSTRING,
)
class TFGPTNeoForCausalLM(TFGPTNeoPreTrainedModel, tf.keras.layers.Layer):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"lm_head.weight",
        r"h\.\d+\.attn\.attention\.bias",
    ]
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.transformer = TFGPTNeoModel(config)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(tf.cast(attention_mask, tf.int64), axis=-1) - 1
            position_ids = tf.where(attention_mask == 0, 1, position_ids)
            if past_key_values:
                position_ids = position_ids[:, -1:]
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[tf.Tensor]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[tf.Tensor], TFCausalLMOutputWithCrossAttentions]:
        r"""
        labels (tf.Tensor of shape (batch_size, sequence_length), optional):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            labels = input_ids Indices are selected in [-100, 0, ..., config.vocab_size] All labels set to -100 are
            ignored (masked), the loss is only computed for labels in [0, ..., config.vocab_size]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            lm_logits = tf.cast(lm_logits, tf.float32)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
            loss = loss_fct(
                y_true=tf.reshape(shift_labels, [-1]), y_pred=tf.reshape(shift_logits, [-1, shift_logits.shape[-1]])
            )
            lm_logits = tf.cast(lm_logits, hidden_states.dtype)
            loss = tf.cast(loss, hidden_states.dtype)
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[tf.Tensor]], beam_idx: tf.Tensor) -> Tuple[Tuple[tf.Tensor]]:
        """
        This function is used to re-order the past_key_values cache if [~PretrainedModel.beam_search] or
        [~PretrainedModel.beam_sample] is called. This is required to match past_key_values with the correct beam_idx
        at every generation step.
        """
        return tuple(
            tuple(tf.gather(past_state, beam_idx, axis=0) for past_state in layer_past)
            for layer_past in past_key_values
        )


@add_start_docstrings(
    """
    The GPTNeo Model transformer with a sequence classification head on top (linear layer).
    [TFGPTNeoForSequenceClassification] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do. Since it does classification on the last token, it requires to know the position of the last
    token. If a pad_token_id is defined in the configuration, it finds the last token that is not a padding token in
    each row. If no pad_token_id is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when inputs_embeds are passed instead of input_ids, it does the same (take the last value
    in each row of the batch).
    """,
    GPT_NEO_START_DOCSTRING,
)
class TFGPTNeoForSequenceClassification(TFGPTNeoPreTrainedModel, tf.keras.layers.Layer):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFGPTNeoModel(config, name="transformer")
        self.score = tf.keras.layers.Dense(config.hidden_size, self.num_labels, use_bias=False, name="score")
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[Tuple[tf.Tensor]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[tf.Tensor], TFSequenceClassifierOutputWithPast]:
        r"""
        labels (tf.Tensor of shape (batch_size,), optional):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If config.num_labels == 1 a regression loss is computed (Mean-Square loss), If
            config.num_labels > 1 a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (tf.math.not_equal(input_ids, self.config.pad_token_id).sum(-1) - 1).numpy()
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in inputs_embeds. Results may be "
                    "unexpected if using padding tokens in conjunction with inputs_embeds."
                )
            pooled_logits = logits[tf.range(batch_size), sequence_lengths]
            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == tf.int64 or labels.dtype == tf.int32):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"
                if self.config.problem_type == "regression":
                    loss_fct = tf.keras.losses.MeanSquaredError()
                    if self.num_labels == 1:
                        loss = loss_fct(pooled_logits[:, tf.newaxis], labels[:, tf.newaxis])
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
