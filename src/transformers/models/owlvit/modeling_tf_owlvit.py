# coding=utf-8
# Copyright 2022 Google AI and The HuggingFace Team. All rights reserved.
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
""" TF 2.0 OWL-ViT model."""


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
from ...tf_utils import stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_owlvit import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/owlvit-base-patch32"

# See all OwlViT models at https://huggingface.co/models?filter=owlvit
TF_OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/owlvit-base-patch32",
    "google/owlvit-base-patch16",
    "google/owlvit-large-patch14",
]


LARGE_NEGATIVE = -1e8


def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = tf.shape(mask)[1]
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
            y_true=tf.range(tf.shape(logits)[0]), y_pred=logits, from_logits=True
        )
    )


def owlvit_loss(similarity: tf.Tensor) -> tf.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0


def center_to_corners_format(boxes: tf.Tensor) -> tf.Tensor:
    """
    Converts a TensorFlow tensor of bounding boxes of center format (center_x, center_y, width, height) to corners
    format (left, top, right, bottom).
    """
    x_center, y_center, width, height = tf.squeeze(boxes, axis=-1)
    boxes = [(x_center - 0.5 * width), (y_center - 0.5 * height), (x_center + 0.5 * width), (y_center + 0.5 * height)]
    return tf.stack(boxes, dim=-1)


def box_area(boxes: tf.Tensor) -> tf.Tensor:
    """
    Args:
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.
        boxes (`tf.Tensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.
    Returns:
        `tf.FloatTensor`: a tensor containing the area for each box.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = tf.reduce_max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = tf.reduce_max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = right_bottom - left_top  # [N,M,2]
    width_height = tf.clip_by_value(width_height, clip_value_min=0)
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format. Returns:
        `tf.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = tf.reduce_min(boxes1[:, None, :2], boxes2[:, :2])
    rb = tf.reduce_max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = tf.clip_by_value(rb - lt, clip_value_min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


@dataclass
class TFOwlViTOutput(ModelOutput):
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
        text_embeds(`tf.Tensor` of shape `(batch_size * num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`TFOwlViTTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`TFOwlViTVisionModel`].
        text_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFOwlViTTextModel`].
        vision_model_output([`~modeling_tf_utils.TFBaseModelOutputWithPooling`]):
            The output of the [`TFOwlViTVisionModel`].
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


@dataclass
class TFOwlViTObjectDetectionOutput(ModelOutput):
    """
    Output type of [`TFOwlViTForObjectDetection`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`tf.Tensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`tf.Tensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~OwlViTFeatureExtractor.post_process`] to retrieve the unnormalized
            bounding boxes.
        text_embeds (`tf.Tensor`` of shape `(batch_size, num_max_text_queries, output_dim)`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`TFOwlViTTextModel`].
        image_embeds (`tf.Tensor` of shape `(batch_size, patch_size, patch_size, output_dim)`):
            Pooled output of [`TFOwlViTVisionModel`]. OWL-ViT represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`tf.Tensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_last_hidden_states (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states extracted from the [`TFOwlViTTextModel`].
        vision_model_last_hidden_states (`tf.Tensor` of shape `(batch_size, num_patches + 1, hidden_size)`):
            Last hidden states extracted from the [`TFOwlViTVisionModel`]. OWL-ViT represents images as a set of image
            patches where the total number of patches is (image_size / patch_size)**2.
    """

    loss: Optional[tf.Tensor] = None
    loss_dict: Optional[Dict] = None
    logits: tf.Tensor = None
    pred_boxes: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    class_embeds: tf.Tensor = None
    text_model_last_hidden_states: Optional[tf.Tensor] = None
    vision_model_last_hidden_states: Optional[tf.Tensor] = None


class TFOwlViTVisionEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: OwlViTVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.patch_embedding = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=config.patch_size,
            strides=config.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=get_initializer(self.config.initializer_range * self.config.initializer_factor),
            name="patch_embedding",
        )

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_positions,
            output_dim=self.embed_dim,
            embeddings_initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
            name="position_embedding",
        )

    def build(self, input_shape: tf.TensorShape):

        factor = self.config.initializer_factor

        self.class_embedding = self.add_weight(
            shape=(self.embed_dim,),
            initializer=get_initializer(self.embed_dim**-0.5 * factor),
            trainable=True,
            name="class_embedding",
        )

        super().build(input_shape)

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        """`pixel_values` is expected to be of NCHW format."""

        batch_size, num_channels, height, width = tf.shape(pixel_values)

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


class TFOwlViTTextEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: OwlViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
            name="token_embedding",
        )

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=config.max_position_embeddings,
            output_dim=config.hidden_size,
            embeddings_initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
            name="position_embedding",
        )

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor. Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        batch_size, seq_len = tf.shape(inputs_embeds)[:-1]

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=seq_len), axis=0)

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(batch_size, 1, 1))
        final_embeddings = inputs_embeds + position_embeds

        return final_embeddings


class TFOwlViTAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OwlViTConfig, **kwargs):
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

        batch_size = tf.shape(hidden_states)[0]
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
            # Apply the causal attention mask (precomputed for all layers in TFOwlViTModel call() function)
            attention_scores = tf.add(attention_scores, causal_attention_mask)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in TFOwlViTModel call() function)
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
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)

        return outputs


# Copied from transformers.models.clip.modeling_tf_clip.TFCLIPMLP with CLIP->OwlViT
class TFOwlViTMLP(tf.keras.layers.Layer):
    def __init__(self, config: OwlViTConfig, **kwargs):
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


class TFOwlViTEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: OwlViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.self_attn = TFOwlViTAttention(config, name="self_attn")
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        self.mlp = TFOwlViTMLP(config, name="mlp")
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


class TFOwlViTEncoder(tf.keras.layers.Layer):
    """
    Args:
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`TFOwlViTEncoderLayer`].
        config: OwlViTConfig
    """

    def __init__(self, config: OwlViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.layers = [TFOwlViTEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

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


class TFOwlViTTextTransformer(tf.keras.layers.Layer):
    def __init__(self, config: OwlViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFOwlViTTextEmbeddings(config, name="embeddings")
        self.encoder = TFOwlViTEncoder(config, name="encoder")
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
        input_shape = tf.shape(input_ids)

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        num_samples, seq_length = input_shape  # num_samples = batch_size * num_max_text_queries
        # OwlViT's (alias for CLIP) text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(
            num_samples, seq_length, dtype=embedding_output.dtype
        )

        # check attention mask and invert
        # [num_samples, seq_len] -> [num_samples, 1, tgt_seq_len, src_seq_len]
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
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)

        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))


@keras_serializable
class TFOwlViTTextMainLayer(tf.keras.layers.Layer):
    config_class = OwlViTTextConfig

    def __init__(self, config: OwlViTTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.text_model = TFOwlViTTextTransformer(config, name="text_model")

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = tf.shape(value)[0]

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

        input_shape = tf.shape(input_ids)

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


class TFOwlViTVisionTransformer(tf.keras.layers.Layer):
    def __init__(self, config: OwlViTVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFOwlViTVisionEmbeddings(config, name="embeddings")
        self.pre_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="pre_layernorm")
        self.encoder = TFOwlViTEncoder(config, name="encoder")
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")

    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
        use_hidden_state: bool = True,
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

        if use_hidden_state:
            pooled_output = self.post_layernorm(inputs=sequence_output)
        else:
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
class TFOwlViTVisionMainLayer(tf.keras.layers.Layer):
    config_class = OwlViTVisionConfig

    def __init__(self, config: OwlViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vision_model = TFOwlViTVisionTransformer(config, name="vision_model")

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
class TFOwlViTMainLayer(tf.keras.layers.Layer):
    config_class = OwlViTConfig

    def __init__(self, config: OwlViTConfig, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(config.text_config, OwlViTTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type OwlViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, OwlViTVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type OwlViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        self.config = config

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim

        self.text_model = TFOwlViTTextTransformer(text_config, name="text_model")
        self.vision_model = TFOwlViTVisionTransformer(vision_config, name="vision_model")

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

        input_shape = tf.shape(input_ids)

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
        return_base_image_embeds: Optional[bool] = None,
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

        # Apply post_layernorm to last_hidden_state, return non-projected output
        if return_base_image_embeds:
            last_hidden_state = vision_outputs[0]
            image_features = self.vision_model.post_layernorm(inputs=last_hidden_state)

        # Unmodified OwlViTModel / CLIP embeddings
        else:
            pooled_output = vision_outputs[1]
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
        return_base_image_embeds: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFOwlViTOutput, Tuple[tf.Tensor]]:

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        input_shape = tf.shape(input_ids)

        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            use_hidden_state=False,
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
            loss = owlvit_loss(logits_per_text)

        if return_base_image_embeds:
            last_hidden_state = vision_outputs[0]
            image_embeds = self.vision_model.post_layernorm(inputs=last_hidden_state)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output

        return TFOwlViTOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class TFOwlViTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OwlViTConfig
    base_model_prefix = "owlvit"


OWLVIT_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass.
    Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general
    usage and behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:
    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.
    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`. If you choose this second option, there
    are three possibilities you can use to gather all the input Tensors in the first positional argument :
    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
      `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
      `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    </Tip>

    Args:
        config ([`OwlViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

OWLVIT_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.__call__`] and [`PreTrainedTokenizer.encode`] for details. [What are input
            IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`. [What are position IDs?](../glossary#position-ids)
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

OWLVIT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`OwlViTFeatureExtractor`]. See
            [`OwlViTFeatureExtractor.__call__`] for details. output_attentions (`bool`, *optional*): Whether or not to
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

OWLVIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.__call__`] and [`PreTrainedTokenizer.encode`] for details. [What are input
            IDs?](../glossary#input-ids)
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`OwlViTFeatureExtractor`]. See
            [`OwlViTFeatureExtractor.__call__`] for details.
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`. [What are position IDs?](../glossary#position-ids)
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
        return_base_image_embeds (`bool`, *optional*):
            Whether or not to return unprojected image embeddings. Set to True when `TFOwlViTModel` is called within
            `TFOwlViTForObjectDetection`.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.
        input_ids (`tf.Tensor` of shape `(batch_size * num_max_text_queries, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`CLIPTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, num_max_text_queries, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
"""


class TFOwlViTTextModel(TFOwlViTPreTrainedModel):
    config_class = OwlViTTextConfig

    def __init__(self, config: OwlViTTextConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.owlvit = TFOwlViTTextMainLayer(config, name="owlvit")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=OwlViTTextConfig)
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
        >>> from transformers import OwlViTProcessor, TFOwlViTTextModel

        >>> model = TFOwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="tf"
        ... )
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        outputs = self.owlvit(
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


class TFOwlViTVisionModel(TFOwlViTPreTrainedModel):
    config_class = OwlViTVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: OwlViTVisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.owlvit = TFOwlViTVisionMainLayer(config, name="owlvit")

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network. Returns:
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
        Args:
        Method used for serving the model.
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
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
        >>> from transformers import OwlViTProcessor, TFOwlViTVisionModel

        >>> model = TFOwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        outputs = self.owlvit(
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


@add_start_docstrings(OWLVIT_START_DOCSTRING)
class TFOwlViTModel(TFOwlViTPreTrainedModel):
    config_class = OwlViTConfig

    def __init__(self, config: OwlViTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.owlvit = TFOwlViTMainLayer(config, name="owlvit")

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network. Returns:
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
    def serving(self, inputs: Dict[str, tf.Tensor]) -> TFOwlViTOutput:
        """
        Args:
        Method used for serving the model.
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
            the projection layer to the pooled output of [`TFOwlViTTextModel`].

        Examples:
        ```python
        >>> from transformers import OwlViTProcessor, TFOwlViTModel

        >>> model = TFOwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="tf"
        ... )
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        text_features = self.owlvit.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return text_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_projected: Optional[bool] = True,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        Returns:
            image_features (`tf.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`TFOwlViTVisionModel`].

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import OwlViTProcessor, TFOwlViTModel

        >>> model = TFOwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="tf")
        >>> image_features = model.get_image_features(**inputs)
        ```"""

        image_features = self.owlvit.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_projected=return_projected,
        )

        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFOwlViTOutput, config_class=OwlViTConfig)
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        pixel_values: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_base_image_embeds: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFOwlViTOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import OwlViTProcessor, TFOwlViTPModel

        >>> model = TFOwlViTModel.from_pretrained("google/owlvit-base-patch32")
        >>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```"""

        outputs = self.owlvit(
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

    def serving_output(self, output: TFOwlViTOutput) -> TFOwlViTOutput:
        # TODO: As is this currently fails with saved_model=True, because
        # TensorFlow cannot trace through nested dataclasses. Reference:
        # https://github.com/huggingface/transformers/pull/16886
        return output


class TFOwlViTBoxPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: OwlViTConfig, **kwargs):
        super().__init__(**kwargs)

        width = config.vision_config.hidden_size
        self.dense0 = tf.keras.layers.Dense(units=width, name="dense0")
        self.dense1 = tf.keras.layers.Dense(units=width, name="dense1")
        self.gelu = get_tf_activation("quick_gelu")
        self.dense2 = tf.keras.layers.Dense(units=4, name="dense2")

    def call(self, image_features: tf.Tensor) -> tf.Tensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output


class TFOwlViTClassPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: OwlViTConfig, **kwargs):
        super().__init__(**kwargs)

        out_dim = config.text_config.hidden_size

        self.dense0 = tf.keras.layers.Dense(units=out_dim, name="dense0")
        self.logit_shift = tf.keras.layers.Dense(units=1, name="logit_shift")
        self.logit_scale = tf.keras.layers.Dense(units=1, name="logit_scale")
        self.elu = get_tf_activation("elu")

    def call(
        self,
        image_embeds: tf.Tensor,
        query_embeds: tf.Tensor,
        query_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor]:

        image_class_embeds = self.dense0(image_embeds)

        if query_embeds is None:
            return (None, image_class_embeds)

        # Normalize image and text features
        image_class_embeds = (
            image_class_embeds / tf.norm(tensor=image_class_embeds, ord="euclidean", axis=-1, keepdims=True) + 1e-6
        )
        query_embeds = query_embeds / tf.norm(tensor=query_embeds, ord="euclidean", axis=-1, keepdims=True) + 1e-6

        # Get class predictions
        pred_logits = tf.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if len(query_mask.shape) > 1:
                query_mask = tf.expand_dims(query_mask, axis=-2)

            pred_logits = tf.where(query_mask == 0, -1e6, pred_logits)

        return (pred_logits, image_class_embeds)


class TFOwlViTForObjectDetection(TFOwlViTPreTrainedModel):
    config_class = OwlViTConfig

    def __init__(self, config: OwlViTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.owlvit = TFOwlViTMainLayer(config, name="owlvit")
        self.class_head = TFOwlViTClassPredictionHead(config, name="class_head")
        self.box_head = TFOwlViTBoxPredictionHead(config, name="box_head")
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.vision_config.layer_norm_eps, name="layernorm"
        )

    def normalize_grid_corner_coordinates(self, feature_map: Optional[TFModelInputType] = None):
        # Computes normalized xy corner coordinates from feature_map.
        if not len(feature_map.shape) == 4:
            raise ValueError("Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]")

        num_patches = feature_map.shape[1]

        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # Flatten (num_patches, num_patches, 2) -> (num_patches**2, 2)
        box_coordinates = box_coordinates.reshape(num_patches**2, box_coordinates.shape[2])
        box_coordinates = tf.convert_to_tensor(box_coordinates)

        return box_coordinates

    def compute_box_bias(self, feature_map: tf.Tensor) -> tf.Tensor:
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        box_coordinates = tf.clip_by_value(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = tf.math.log(box_coordinates + 1e-4) - tf.math.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = tf.ones_like(box_coord_bias) * 1.0 / feature_map.shape[-2]
        box_size_bias = tf.math.log(box_size + 1e-4) - tf.math.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = tf.concat([box_coord_bias, box_size_bias], axis=-1)
        return box_bias

    def box_predictor(
        self,
        image_feats: Optional[TFModelInputType] = None,
        feature_map: Optional[TFModelInputType] = None,
    ) -> tf.Tensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = tf.keras.activations.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(
        self,
        image_feats: Optional[TFModelInputType] = None,
        query_embeds: Optional[TFModelInputType] = None,
        query_mask: Optional[TFModelInputType] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)

    def image_text_embedder(
        self,
        input_ids: Optional[TFModelInputType] = None,
        pixel_values: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[tf.Tensor]:

        # Encode text and image
        outputs = self.owlvit(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_base_image_embeds=True,
        )

        # Resize class token
        image_embeds = outputs.image_embeds
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = tf.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(inputs=image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        batch_size, all_patches, hidden_size = tf.shape(image_embeds)
        num_patches = int(np.sqrt(all_patches))
        new_size = (batch_size, num_patches, num_patches, hidden_size)

        image_embeds = tf.reshape(image_embeds, new_size)
        text_embeds = outputs.text_embeds

        # Last hidden states from text and vision transformers
        text_model_last_hidden_states = outputs.text_model_output.last_hidden_state
        vision_model_last_hidden_states = outputs.vision_model_output.last_hidden_state

        return (text_embeds, image_embeds, text_model_last_hidden_states, vision_model_last_hidden_states)

    def embed_image_query(
        self, query_pixel_values: tf.Tensor, query_feature_map: tf.Tensor, iou_threshold: float = 0.65
    ) -> tf.Tensor:

        (_, class_embeds) = self.class_predictor(query_pixel_values)
        pred_boxes = self.box_predictor(query_pixel_values, query_feature_map)
        pred_boxes_as_corners = center_to_corners_format(pred_boxes)

        filtered_embeds = []
        for i in range(query_pixel_values.shape[0]):
            each_query_box = tf.Tensor([[0, 0, 1, 1]])
            each_query_pred_boxes = pred_boxes_as_corners[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            if tf.math.reduce_all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            selected_inds = ious[0] >= iou_threshold
            non_zero_mask = tf.greater(selected_inds, 0)
            selected_inds = tf.boolean_mask(selected_inds, non_zero_mask)
            selected_inds = tf.squeeze(selected_inds, axis=0)

            if tf.size(selected_inds) > 0:
                selected_embeddings = class_embeds[i][selected_inds]
                mean_embeds = tf.math.reduce_mean(class_embeds[i], axis=0)
                mean_sim = tf.einsum("d,id->i", mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[tf.math.argmin(mean_sim)]
                filtered_embeds.append(class_embeds[i][best_box_ind])

        if filtered_embeds:
            query_embeds = tf.stack(filtered_embeds)
        else:
            query_embeds = None

        return query_embeds

    def image_image_embedder(
        self,
        query_pixel_values: tf.Tensor,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[tf.Tensor]:

        query_image_embeds = self.owlvit.get_image_features(
            pixel_values=query_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_base_image_embeds=True,
        )
        image_embeds = self.owlvit.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_base_image_embeds=True,
        )

        # Normalize query image, image embeddings
        image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True)
        query_image_embeds = query_image_embeds / tf.norm(
            tensor=query_image_embeds, ord="euclidean", axis=-1, keepdims=True
        )

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = tf.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(inputs=image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = tf.reshape(image_embeds, new_size)

        # Resize class token
        new_size = tuple(np.array(query_image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = tf.broadcast_to(query_image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        query_image_embeds = query_image_embeds[:, 1:, :] * class_token_out
        query_image_embeds = self.layer_norm(inputs=query_image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            query_image_embeds.shape[0],
            int(np.sqrt(query_image_embeds.shape[1])),
            int(np.sqrt(query_image_embeds.shape[1])),
            query_image_embeds.shape[-1],
        )
        query_image_embeds = tf.reshape(query_image_embeds, new_size)

        return (query_image_embeds, image_embeds)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(OWLVIT_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFOwlViTObjectDetectionOutput, config_class=OwlViTConfig)
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        pixel_values: Optional[TFModelInputType] = None,
        query_pixel_values: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TFOwlViTObjectDetectionOutput:
        r"""
        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import tensorflow as tf
        >>> from transformers import OwlViTProcessor, TFOwlViTForObjectDetection

        >>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> model = TFOwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = [["a photo of a cat", "a photo of a dog"]]
        >>> inputs = processor(text=texts, images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        query_embeds, query_mask = None, None
        assert (
            input_ids is None or query_pixel_values is None
        ), "Both input_ids and query_pixel_values cannot be passed"

        # Embed images and image queries
        if query_pixel_values is not None:
            outputs = self.image_image_embedder(
                query_pixel_values=query_pixel_values,
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            query_feature_map = outputs[0]
        else:
            # Embed images and text queries
            outputs = self.image_text_embedder(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            # Last hidden states of text and vision transformers
            text_model_last_hidden_states = outputs[2]
            vision_model_last_hidden_states = outputs[3]

            query_embeds = outputs[0]

        feature_map = outputs[1]

        batch_size, num_patches, num_patches, hidden_dim = tf.shape(feature_map)
        image_feats = tf.reshape(feature_map, (batch_size, num_patches**2, hidden_dim))

        if input_ids is not None:
            # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
            max_text_queries = tf.shape(input_ids)[0] // batch_size
            query_embeds = tf.reshape(query_embeds, (batch_size, max_text_queries, tf.shape(query_embeds)[-1]))

            # If first token is 0, then this is a padded query [batch_size, num_queries].
            input_ids = tf.reshape(input_ids, (batch_size, max_text_queries, tf.shape(input_ids)[-1]))
            query_mask = input_ids[..., 0] > 0
        else:
            batch_size, num_patches, num_patches, hidden_dim = tf.shape(query_feature_map)
            query_image_feats = tf.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
            query_embeds = self.embed_image_query(query_image_feats, query_feature_map)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats, feature_map)

        # Text and vision model last hidden states have different dimensions and are returned separately
        if not return_dict:
            output = (
                pred_logits,
                pred_boxes,
                query_embeds,
                feature_map,
                class_embeds,
                text_model_last_hidden_states,
                vision_model_last_hidden_states,
            )
            output = tuple(x for x in output if x is not None)
            return output

        return TFOwlViTObjectDetectionOutput(
            image_embeds=feature_map,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            text_model_last_hidden_states=text_model_last_hidden_states,
            vision_model_last_hidden_states=vision_model_last_hidden_states,
        )

    def serving_output(self, output: TFOwlViTObjectDetectionOutput) -> TFOwlViTObjectDetectionOutput:
        return TFOwlViTObjectDetectionOutput(
            image_embeds=output.feature_map,
            text_embeds=output.query_embeds,
            pred_boxes=output.pred_boxes,
            logits=output.pred_logits,
            class_embeds=output.class_embeds,
            text_model_last_hidden_states=output.text_model_last_hidden_states,
            vision_model_last_hidden_states=output.vision_model_last_hidden_states,
        )
