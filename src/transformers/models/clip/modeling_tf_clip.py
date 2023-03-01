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
""" TF 2.0 CLIP model."""


import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling

# Public API
from ...modeling_tf_utils import (
    DUMMY_INPUTS,
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"

TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",
    # See all CLIP models at https://huggingface.co/models?filter=clip
]


LARGE_NEGATIVE = -1e8


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


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


def clip_loss(similarity: tf.Tensor) -> tf.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0


@dataclass
class TFCLIPOutput(ModelOutput):
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`TFCLIPTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`TFCLIPVisionModel`].
        text_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFCLIPTextModel`].
        vision_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFCLIPVisionModel`].
    """

    loss: Optional[tf.Tensor] = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class TFCLIPVisionEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: CLIPVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.config = config

        self.patch_embedding = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range * self.config.initializer_factor),
            name="patch_embedding",
        )

    def build(self, input_shape: tf.TensorShape):
        factor = self.config.initializer_factor

        self.class_embedding = self.add_weight(
            shape=(self.embed_dim,),
            initializer=get_initializer(self.embed_dim**-0.5 * factor),
            trainable=True,
            name="class_embedding",
        )

        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.num_positions, self.embed_dim),
                initializer=get_initializer(self.config.initializer_range * factor),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        """`pixel_values` is expected to be of NCHW format."""

        batch_size, num_channels, height, width = shape_list(pixel_values)

        # When running on CPU, `tf.nn.conv2d` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        patch_embeds = self.patch_embedding(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        patch_embeds = tf.reshape(tensor=patch_embeds, shape=(batch_size, self.num_patches, -1))

        # add the [CLS] token to the embedded patch tokens
        class_embeds = tf.broadcast_to(self.class_embedding, shape=(batch_size, 1, self.embed_dim))
        embeddings = tf.concat((class_embeds, patch_embeds), axis=1)

        embeddings = embeddings + self.position_embedding

        return embeddings


class TFCLIPTextEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size

        self.config = config

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("token_embedding"):
            self.weight = self.add_weight(
                shape=(self.config.vocab_size, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="weight",
            )

        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.config.max_position_embeddings, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            # Note: tf.gather, on which the embedding layer is based, won't check positive out of bound
            # indices on GPU, returning zeros instead. This is a dangerous silent behavior.
            tf.debugging.assert_less(
                input_ids,
                tf.cast(self.config.vocab_size, dtype=input_ids.dtype),
                message=(
                    "input_ids must be smaller than the embedding layer's input dimension (got"
                    f" {tf.math.reduce_max(input_ids)} >= {self.config.vocab_size})"
                ),
            )
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        final_embeddings = inputs_embeds + position_embeds

        return final_embeddings


class TFCLIPAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.embed_dim // self.num_attention_heads
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_attention_heads})."
            )

        factor = config.initializer_factor
        in_proj_std = (self.embed_dim**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        out_proj_std = (self.embed_dim**-0.5) * factor

        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.q_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="q_proj"
        )
        self.k_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="k_proj"
        )
        self.v_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="v_proj"
        )

        self.dropout = tf.keras.layers.Dropout(rate=config.attention_dropout)

        self.out_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name="out_proj"
        )

    # copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """Input shape: Batch x Time x Channel"""

        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.q_proj(inputs=hidden_states)
        mixed_key_layer = self.k_proj(inputs=hidden_states)
        mixed_value_layer = self.v_proj(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            # Apply the causal attention mask (precomputed for all layers in TFCLIPModel call() function)
            attention_scores = tf.add(attention_scores, causal_attention_mask)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in TFCLIPModel call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=_attention_probs, training=training)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, embed_dim)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))

        attention_output = self.out_proj(attention_output, training=training)
        # In TFBert, attention weights are returned after dropout.
        # However, in CLIP, they are returned before dropout.
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)

        return outputs


class TFCLIPMLP(tf.keras.layers.Layer):
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        self.activation_fn = get_tf_activation(config.hidden_act)

        factor = config.initializer_factor
        in_proj_std = (config.hidden_size**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        fc_std = (2 * config.hidden_size) ** -0.5 * factor

        self.fc1 = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name="fc1"
        )
        self.fc2 = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name="fc2"
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.fc1(inputs=hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(inputs=hidden_states)
        return hidden_states


class TFCLIPEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.self_attn = TFCLIPAttention(config, name="self_attn")
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        self.mlp = TFCLIPMLP(config, name="mlp")
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            causal_attention_mask (`tf.Tensor`): causal attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`):
                Whether or not to return the attentions tensors of all attention layers. See `outputs` under returned
                tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(inputs=hidden_states)
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = attention_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(inputs=hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + attention_outputs[1:]  # add attentions if we output them

        return outputs


class TFCLIPEncoder(tf.keras.layers.Layer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`TFCLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        self.layers = [TFCLIPEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class TFCLIPTextTransformer(tf.keras.layers.Layer):
    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFCLIPTextEmbeddings(config, name="embeddings")
        self.encoder = TFCLIPEncoder(config, name="encoder")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

    def call(
        self,
        input_ids: TFModelInputType,
        attention_mask: tf.Tensor,
        position_ids: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        input_shape = shape_list(input_ids)

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        batch_size, seq_length = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)

        # check attention mask and invert
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.final_layer_norm(inputs=sequence_output)

        # text_embeds.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        pooled_output = tf.gather_nd(
            params=sequence_output,
            indices=tf.stack(
                values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(input_ids, axis=-1)), axis=1
            ),
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
        # It is possible with an unspecified sequence length for seq_length to be
        # a runtime value, which is unsupported by tf.constant. Per the TensorFlow
        # docs, tf.fill can handle runtime dynamic shapes:
        # https://www.tensorflow.org/api_docs/python/tf/fill
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)

        # set an additive 2D attention mask with all places being masked
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)

        # set diagonal & lower triangular parts to 0 (i.e. the places not to be masked)
        # TIP: think the 2D matrix as the space of (query_seq, key_seq)
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        # to_mask = tf.linalg.band_part(to_mask, -1, 0)
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)

        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))


@keras_serializable
class TFCLIPTextMainLayer(tf.keras.layers.Layer):
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.text_model = TFCLIPTextTransformer(config, name="text_model")

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = shape_list(input_ids)

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        text_model_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return text_model_outputs


class TFCLIPVisionTransformer(tf.keras.layers.Layer):
    def __init__(self, config: CLIPVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFCLIPVisionEmbeddings(config, name="embeddings")
        self.pre_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="pre_layrnorm")
        self.encoder = TFCLIPEncoder(config, name="encoder")
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")

    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        embedding_output = self.embeddings(pixel_values=pixel_values)
        embedding_output = self.pre_layernorm(inputs=embedding_output)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.post_layernorm(inputs=pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@keras_serializable
class TFCLIPVisionMainLayer(tf.keras.layers.Layer):
    config_class = CLIPVisionConfig

    def __init__(self, config: CLIPVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vision_model = TFCLIPVisionTransformer(config, name="vision_model")

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.vision_model.embeddings

    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        vision_model_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return vision_model_outputs


@keras_serializable
class TFCLIPMainLayer(tf.keras.layers.Layer):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        self.config = config

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim

        self.text_model = TFCLIPTextTransformer(text_config, name="text_model")
        self.vision_model = TFCLIPVisionTransformer(vision_config, name="vision_model")

        self.visual_projection = tf.keras.layers.Dense(
            units=self.projection_dim,
            kernel_initializer=get_initializer(vision_config.hidden_size**-0.5 * self.config.initializer_factor),
            use_bias=False,
            name="visual_projection",
        )

        self.text_projection = tf.keras.layers.Dense(
            units=self.projection_dim,
            kernel_initializer=get_initializer(text_config.hidden_size**-0.5 * self.config.initializer_factor),
            use_bias=False,
            name="text_projection",
        )

    def build(self, input_shape: tf.TensorShape):
        self.logit_scale = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
            name="logit_scale",
        )

        super().build(input_shape)

    @unpack_inputs
    def get_text_features(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = shape_list(input_ids)

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(inputs=pooled_output)

        return text_features

    @unpack_inputs
    def get_image_features(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(inputs=pooled_output)

        return image_features

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        pixel_values: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        input_shape = shape_list(input_ids)

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(inputs=image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(inputs=text_embeds)

        # normalized features
        image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output

        return TFCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class TFCLIPPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPConfig
    base_model_prefix = "clip"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]


CLIP_START_DOCSTRING = r"""

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
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details. output_attentions (`bool`, *optional*): Whether or not to
            return the attentions tensors of all attention layers. See `attentions` under returned tensors for more
            detail. This argument can be used only in eager mode, in graph mode the value in the config will be used
            instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

CLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


class TFCLIPTextModel(TFCLIPPreTrainedModel):
    config_class = CLIPTextConfig

    def __init__(self, config: CLIPTextConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.clip = TFCLIPTextMainLayer(config, name="clip")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFCLIPTextModel

        >>> model = TFCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs: Dict[str, tf.Tensor]) -> TFBaseModelOutputWithPooling:
        output = self.call(inputs)
        return self.serving_output(output)

    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=hs,
            attentions=attns,
        )


class TFCLIPVisionModel(TFCLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.clip = TFCLIPVisionMainLayer(config, name="clip")

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(len(DUMMY_INPUTS), 3, self.config.image_size, self.config.image_size), dtype=tf.float32
        )
        return {"pixel_values": VISION_DUMMY_INPUTS}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
            }
        ]
    )
    def serving(self, inputs: Dict[str, tf.Tensor]) -> TFBaseModelOutputWithPooling:
        """
        Method used for serving the model.

        Args:
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFCLIPVisionModel

        >>> model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        outputs = self.clip(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=hs,
            attentions=attns,
        )


@add_start_docstrings(CLIP_START_DOCSTRING)
class TFCLIPModel(TFCLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.clip = TFCLIPMainLayer(config, name="clip")

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(len(DUMMY_INPUTS), 3, self.config.vision_config.image_size, self.config.vision_config.image_size),
            dtype=tf.float32,
        )
        return {
            "input_ids": tf.constant(DUMMY_INPUTS, dtype=tf.int32),
            "pixel_values": VISION_DUMMY_INPUTS,
        }

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs: Dict[str, tf.Tensor]) -> TFCLIPOutput:
        """
        Method used for serving the model.

        Args:
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def get_text_features(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFCLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFCLIPModel

        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```"""

        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return text_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFCLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFCLIPModel

        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```"""

        image_features = self.clip.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFCLIPOutput, config_class=CLIPConfig)
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        pixel_values: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFCLIPModel

        >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```"""

        outputs = self.clip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    def serving_output(self, output: TFCLIPOutput) -> TFCLIPOutput:
        # TODO: As is this currently fails with saved_model=True, because
        # TensorFlow cannot trace through nested dataclasses. Reference:
        # https://github.com/huggingface/transformers/pull/16886
        return output
