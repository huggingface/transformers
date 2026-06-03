# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
"""TF IdeficsVision model: a copy of CLIPVisionModel using a simpler config object"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFPreTrainedModel, shape_list
from ...tf_utils import flatten
from ...utils import ModelOutput, logging
from .configuration_idefics import IdeficsVisionConfig


logger = logging.get_logger(__name__)


@dataclass
class TFIdeficsVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[tf.Tensor] = None
    last_hidden_state: Optional[tf.Tensor] = None
    hidden_states: Optional[tuple[tf.Tensor]] = None
    attentions: Optional[tuple[tf.Tensor]] = None


class TFIdeficsVisionEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            padding="valid",
            data_format="channels_last",
            name="patch_embedding",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = tf.keras.layers.Embedding(
            self.num_positions, self.embed_dim, name="position_embedding"
        )
        # self.position_ids = tf.range(self.num_positions)[tf.newaxis, :]

    def interpolate_pos_encoding(self, embeddings: tf.Tensor, height: int, width: int) -> tf.Tensor:
        num_patches = shape_list(embeddings)[1] - 1
        pos_embed = self.position_embedding(self.position_ids)
        num_positions = shape_list(pos_embed)[1] - 1
        if num_patches == num_positions and height == width:
            return pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        embed_dim = shape_list(embeddings)[-1]
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size
        num_h_patches, num_w_patches = num_h_patches + 0.1, num_w_patches + 0.1
        sqrt_num_positions = math.sqrt(float(num_positions))
        patch_pos_embed = tf.reshape(patch_pos_embed, (1, int(sqrt_num_positions), int(sqrt_num_positions), embed_dim))

        scale_height = num_h_patches / sqrt_num_positions
        scale_width = num_w_patches / sqrt_num_positions
        original_height = tf.cast(tf.shape(patch_pos_embed)[1], tf.float32)
        original_width = tf.cast(tf.shape(patch_pos_embed)[2], tf.float32)
        # Apply scaling
        new_height = tf.cast(original_height * scale_height, tf.int32)
        new_width = tf.cast(original_width * scale_width, tf.int32)

        patch_pos_embed = tf.image.resize(
            patch_pos_embed, size=[new_height, new_width], method=tf.image.ResizeMethod.BICUBIC
        )

        if (
            int(num_h_patches) != shape_list(patch_pos_embed)[-3]
            or int(num_w_patches) != shape_list(patch_pos_embed)[-2]
        ):
            raise ValueError(
                f"Number of patches for images ({int(num_h_patches), int(num_w_patches)}) don't match the "
                f"shape of position embedding ({shape_list(patch_pos_embed)[-2], shape_list(patch_pos_embed)[-1]})"
            )
        patch_pos_embed = tf.reshape(patch_pos_embed, (1, -1, embed_dim))
        return tf.concat((class_pos_embed[tf.newaxis, :], patch_pos_embed), axis=1)

    def call(self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False) -> tf.Tensor:
        # Input `pixel_values` is NCHW format which doesn't run on CPU so first thing we do is
        # transpose it to change it to NHWC. We don't care to transpose it back because
        # the Conv2D layer is only hit once for each query

        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        batch_size, height, width, num_channels = shape_list(pixel_values)
        if not interpolate_pos_encoding:
            if height != self.image_size or width != self.image_size:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size}*{self.image_size}). You should try to set `interpolate_pos_encoding=True`"
                )

        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        patch_embeds = flatten(patch_embeds, 1, 2)

        class_embeds = tf.broadcast_to(
            self.class_embedding[tf.newaxis, tf.newaxis, :], [batch_size, 1, self.embed_dim]
        )
        embeddings = tf.concat([class_embeds, patch_embeds], axis=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        self.position_ids = tf.range(self.num_positions, name="self.position_ids")[tf.newaxis, :]
        self.class_embedding = self.add_weight(shape=(self.embed_dim,), name="class_embedding")
        if getattr(self, "patch_embedding", None) is not None:
            with tf.name_scope(self.patch_embedding.name):
                self.patch_embedding.build([None, None, None, self.config.num_channels])
        if getattr(self, "position_embedding", None) is not None:
            with tf.name_scope(self.position_embedding.name):
                self.position_embedding.build(None)


class TFIdeficsVisionAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = tf.keras.layers.Dense(self.embed_dim, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.embed_dim, name="v_proj")
        self.q_proj = tf.keras.layers.Dense(self.embed_dim, name="q_proj")
        self.out_proj = tf.keras.layers.Dense(self.embed_dim, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        causal_attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[tf.Tensor, Optional[tf.Tensor], Optional[tuple[tf.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        src_len = shape_list(key_states)[1]
        attn_weights = tf.linalg.matmul(query_states, key_states, transpose_b=True)

        tf.debugging.assert_equal(
            tf.shape(attn_weights),
            [bsz * self.num_heads, tgt_len, src_len],
            message=f"Attention weights should be of size {[bsz * self.num_heads, tgt_len, src_len]}, but is {tf.shape(attn_weights)}",
        )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if shape_list(causal_attention_mask) != [bsz, 1, tgt_len, src_len]:
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {shape_list(causal_attention_mask)}"
                )
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + causal_attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        if attention_mask is not None:
            if shape_list(attention_mask) != [bsz, 1, tgt_len, src_len]:
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {shape_list(attention_mask)}"
                )
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tf.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = tf.reshape(attn_weights_reshaped, (bsz * self.num_heads, tgt_len, src_len))
        else:
            attn_weights_reshaped = None

        attn_probs = tf.nn.dropout(attn_weights, rate=self.dropout)

        attn_output = tf.linalg.matmul(attn_probs, value_states)

        tf.debugging.assert_equal(
            tf.shape(attn_output),
            [bsz * self.num_heads, tgt_len, self.head_dim],
            message=f"Attention weights should be of size {[bsz * self.num_heads, tgt_len, self.head_dim]}, but is {tf.shape(attn_output)}",
        )

        attn_output = tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build((self.embed_dim, self.embed_dim))
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build((self.embed_dim, self.embed_dim))
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build((self.embed_dim, self.embed_dim))
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build((self.embed_dim, self.embed_dim))


class TFIdeficsVisionMLP(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.activation_fn = get_tf_activation(config.hidden_act)
        self.fc1 = tf.keras.layers.Dense(config.intermediate_size, name="fc1")
        self.fc2 = tf.keras.layers.Dense(config.hidden_size, name="fc2")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build(self.config.hidden_size)
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build(self.config.intermediate_size)


class TFIdeficsVisionEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: IdeficsVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.self_attn = TFIdeficsVisionAttention(config, name="self_attn")
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        self.mlp = TFIdeficsVisionMLP(config, name="mlp")
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])


class TFIdeficsVisionEncoder(tf.keras.layers.Layer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`TFIdeficsVisionEncoderLayer`].

    Args:
        config: IdeficsVisionConfig
    """

    def __init__(self, config: IdeficsVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layers = [
            TFIdeficsVisionEncoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)
        ]
        self.gradient_checkpointing = False

    def call(
        self,
        inputs_embeds,
        attention_mask: Optional[tf.Tensor] = None,
        causal_attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[tuple, TFBaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = tf.recompute_grad(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


class TFIdeficsVisionTransformer(TFPreTrainedModel):
    def __init__(self, config: IdeficsVisionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.embed_dim = config.hidden_size

        self.embeddings = TFIdeficsVisionEmbeddings(config, name="embeddings")
        self.pre_layrnorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="pre_layrnorm")
        self.encoder = TFIdeficsVisionEncoder(config, name="encoder")
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")

    # Adapted from transformers.models.clip.modeling_clip.CLIPVisionTransformer.forward
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[tuple, TFBaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "pre_layrnorm", None) is not None:
            with tf.name_scope(self.pre_layrnorm.name):
                self.pre_layrnorm.build([None, None, self.embed_dim])
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "post_layernorm", None) is not None:
            with tf.name_scope(self.post_layernorm.name):
                self.post_layernorm.build([None, self.embed_dim])
