# coding=utf-8
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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
""" TensorFlow BLIP model."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf

from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    DUMMY_INPUTS,
    TFPreTrainedModel,
    get_initializer,
    get_tf_activation,
    keras_serializable,
    shape_list,
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
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/blip-vqa-base"

TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip-vqa-base",
    "Salesforce/blip-vqa-capfilt-large",
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip-itm-large-coco",
    "Salesforce/blip-itm-base-flickr",
    "Salesforce/blip-itm-large-flickr",
    # See all BLIP models at https://huggingface.co/models?filter=blip
]


# Copied from transformers.models.clip.modeling_tf_clip.contrastive_loss
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


# Copied from transformers.models.clip.modeling_tf_clip.clip_loss with clip->blip
def blip_loss(similarity: tf.Tensor) -> tf.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0


@dataclass
class TFBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`tf.Tensor`, *optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Languge modeling loss from the text decoder.
        decoder_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head of the text decoder model.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained after applying the Vision Transformer model to the input image.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.`
    """

    loss: Optional[Tuple[tf.Tensor]] = None
    decoder_logits: Optional[Tuple[tf.Tensor]] = None
    image_embeds: Optional[tf.Tensor] = None
    last_hidden_state: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFBlipTextVisionModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[tf.Tensor] = None
    image_embeds: Optional[tf.Tensor] = None
    last_hidden_state: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFBlipImageTextMatchingModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        itm_score (`tf.Tensor`):
            The image-text similarity scores.
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        vision_pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the vision of the vision-only branch of the model.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        question_embeds (`tf.Tensor`):
            The question embeddings obtained by the text projection layer.
    """

    itm_score: Optional[tf.Tensor] = None
    loss: Optional[tf.Tensor] = None
    image_embeds: Optional[tf.Tensor] = None
    last_hidden_state: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    vision_pooler_output: Optional[tf.Tensor] = None
    attentions: Optional[Tuple[tf.Tensor]] = None
    question_embeds: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFBlipOutput(ModelOutput):
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
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
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


class TFBlipVisionEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: BlipVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            kernel_initializer=get_initializer(self.config.initializer_range),
            data_format="channels_last",
            name="patch_embedding",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="class_embedding",
        )

        self.position_embedding = self.add_weight(
            shape=(1, self.num_positions, self.embed_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="position_embedding",
        )

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        # Input is channels-first, we transpose. PyTorch transposes after the conv because PyTorch
        # likes channels-first convs.
        batch_size = tf.shape(pixel_values)[0]
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = tf.reshape(patch_embeds, (batch_size, self.num_patches, -1))

        class_embeds = tf.broadcast_to(self.class_embedding, (batch_size, 1, self.embed_dim))
        embeddings = tf.concat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding[:, : tf.shape(embeddings)[1], :]
        return embeddings


# Copied from transformers.models.clip.modeling_tf_clip.TFCLIPTextEmbeddings with CLIP->Blip
class TFBlipTextEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs):
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


class TFBlipAttention(tf.keras.layers.Layer):
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
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout, name="dropout")

        self.qkv = tf.keras.layers.Dense(
            3 * self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="qkv"
        )

        self.projection = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="projection"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        mixed_qkv = self.qkv(hidden_states)
        mixed_qkv = tf.reshape(mixed_qkv, (bsz, tgt_len, 3, self.num_heads, self.head_dim))
        mixed_qkv = tf.transpose(mixed_qkv, perm=(2, 0, 3, 1, 4))

        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query_states @ tf.transpose(key_states, (0, 1, 3, 2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.transpose(attention_probs @ value_states, perm=(0, 2, 1, 3))

        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.embed_dim]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


class TFBlipMLP(tf.keras.layers.Layer):
    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)

        self.activation_fn = get_tf_activation(config.hidden_act)

        in_proj_std = (config.hidden_size**-0.5) * ((2 * config.num_hidden_layers) ** -0.5)
        fc_std = (2 * config.hidden_size) ** -0.5

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


class TFBlipEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.self_attn = TFBlipAttention(config, name="self_attn")
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        self.mlp = TFBlipMLP(config, name="mlp")
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor]:
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
            head_mask=attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFBlipPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BlipConfig
    base_model_prefix = "blip"
    _keys_to_ignore_on_load_missing = [r"position_ids"]


BLIP_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BlipConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

BLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`BlipImageProcessor`]. See [`BlipImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

BLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`BlipImageProcessor`]. See [`BlipImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@keras_serializable
class TFBlipEncoder(tf.keras.layers.Layer):
    config_class = BlipConfig
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`BlipEncoderLayer`].

    Args:
        config (`BlipConfig`):
            The corresponding vision configuration for the `BlipEncoder`.
    """

    def __init__(self, config: BlipConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layers = [TFBlipEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

    @unpack_inputs
    def call(
        self,
        inputs_embeds,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

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
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                training=training,
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


class TFBlipVisionModel(TFBlipPreTrainedModel):
    main_input_name = "pixel_values"
    config_class = BlipVisionConfig

    def __init__(self, config: BlipVisionConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        self.embeddings = TFBlipVisionEmbeddings(config, name="embeddings")
        self.encoder = TFBlipEncoder(config, name="encoder")
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")

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

    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=hs,
            attentions=attns,
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=BlipVisionConfig)
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:
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

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        # TF gets confused if we call the layer with inputs of different ranks, so insert a singleton dimension
        pooled_output = self.post_layernorm(tf.expand_dims(pooled_output, 1))
        pooled_output = tf.squeeze(pooled_output, 1)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


class TFBlipMainLayer(tf.keras.layers.Layer):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type BlipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type BlipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = TFBlipTextModel(text_config, name="text_model")
        self.vision_model = TFBlipVisionModel(vision_config, name="vision_model")

        self.visual_projection = tf.keras.layers.Dense(
            self.projection_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="visual_projection",
        )
        self.text_projection = tf.keras.layers.Dense(
            self.projection_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="text_projection",
        )

        self.config = config

    def build(self, input_shape):
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=[],
            initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
        )

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        pixel_values: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBlipOutput]:
        # Use BLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / tf.norm(image_embeds, ord=2, axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(text_embeds, ord=2, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = tf.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)

        loss = None
        if return_loss:
            loss = blip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return TFBlipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class TFBlipModel(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = [r"text_decoder.cls.predictions.decoder.bias"]
    main_input_name = "input_ids"

    def __init__(self, config: BlipConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.blip = TFBlipMainLayer(config, name="blip")

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
    def serving(self, inputs: Dict[str, tf.Tensor]) -> TFBlipOutput:
        """
        Method used for serving the model.

        Args:
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)

    def serving_output(self, output: TFBlipOutput) -> TFBlipOutput:
        return TFBlipOutput(
            logits_per_image=output.logits_per_image,
            logits_per_text=output.logits_per_text,
            text_embeds=output.text_embeds,
            image_embeds=output.image_embeds,
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipOutput, config_class=BlipConfig)
    def call(
        self,
        input_ids: Optional[tf.Tensor] = None,
        pixel_values: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBlipOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```"""
        outputs = self.blip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> tf.Tensor:
        r"""
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFBlipTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.blip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> tf.Tensor:
        r"""
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFBlipVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.blip.vision_model(pixel_values=pixel_values, return_dict=return_dict)

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features


@add_start_docstrings(
    """
    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass
    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,
    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption
    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.
    """,
    BLIP_START_DOCSTRING,
)
class TFBlipForConditionalGeneration(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = [r"text_decoder.cls.predictions.decoder.bias"]
    main_input_name = "pixel_values"

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")

        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name="text_decoder")

        self.decoder_input_ids = config.text_config.bos_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.vision_model.embeddings.patch_embedding

    @property
    def dummy_inputs(self):
        input_ids = tf.constant(DUMMY_INPUTS, dtype=tf.int32)
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(len(DUMMY_INPUTS), 3, self.config.vision_config.image_size, self.config.vision_config.image_size),
            dtype=tf.float32,
        )
        return {"input_ids": input_ids, "pixel_values": VISION_DUMMY_INPUTS}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
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

    def serving_output(
        self, output: TFBlipForConditionalGenerationModelOutput
    ) -> TFBlipForConditionalGenerationModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBlipForConditionalGenerationModelOutput(
            last_hidden_state=output.last_hidden_state,
            image_embeds=output.image_embeds,
            hidden_states=hs,
            attentions=attns,
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipForConditionalGenerationModelOutput, config_class=BlipConfig)
    def call(
        self,
        pixel_values: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBlipForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A picture of"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")

        >>> outputs = model(**inputs)
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        image_embeds = vision_outputs[0]

        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        if outputs.loss is not None and outputs.loss.shape.rank == 0:
            outputs.loss = tf.reshape(outputs.loss, (1,))

        return TFBlipForConditionalGenerationModelOutput(
            loss=outputs.loss,
            decoder_logits=outputs.logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def generate(
        self,
        pixel_values: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        **generate_kwargs,
    ) -> tf.Tensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:
                Input image to be processed
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        two cats are laying on a couch
        ```
        """

        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)

        if isinstance(input_ids, list):
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        elif input_ids is None:
            input_ids = tf.convert_to_tensor(
                [[self.decoder_input_ids, self.config.text_config.eos_token_id]], dtype=tf.int32
            )

            input_ids = tf.tile(input_ids, (batch_size, 1))

        # PyTorch: input_ids[:, 0] = self.config.text_config.bos_token_id
        input_ids = tf.concat(
            [tf.ones((batch_size, 1), dtype=tf.int32) * self.config.text_config.bos_token_id, input_ids[:, 1:]], axis=1
        )
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs


@add_start_docstrings(
    """
    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text
    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together
    with the encoding of the image, and the text decoder will output the answer to the question.
    """,
    BLIP_START_DOCSTRING,
)
class TFBlipForQuestionAnswering(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = [r"text_decoder.cls.predictions.decoder.bias"]

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")

        self.text_encoder = TFBlipTextModel(config.text_config, name="text_encoder", add_pooling_layer=False)

        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name="text_decoder")

        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.vision_model.embeddings.patch_embedding

    @property
    def dummy_inputs(self):
        input_ids = tf.constant(DUMMY_INPUTS, dtype=tf.int32)
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(len(DUMMY_INPUTS), 3, self.config.vision_config.image_size, self.config.vision_config.image_size),
            dtype=tf.float32,
        )
        return {"input_ids": input_ids, "pixel_values": VISION_DUMMY_INPUTS, "decoder_input_ids": input_ids}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
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

    def serving_output(self, output: TFBlipTextVisionModelOutput) -> TFBlipTextVisionModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBlipTextVisionModelOutput(
            image_embeds=output.image_embeds,
            last_hidden_state=output.last_hidden_state,
            hidden_states=hs,
            attentions=attns,
        )

    # Adapted from transformers.models.t5.modeling_tf_t5.TFT5PreTrainedModel._shift_right
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.decoder_start_token_id
        pad_token_id = self.decoder_pad_token_id

        if decoder_start_token_id is None or pad_token_id is None:
            raise ValueError("decoder_start_token_id and pad_token_id must be defined!")

        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)  # Ensure compatible dtypes for concatenation
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = tf.where(
            shifted_input_ids == -100,
            tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
            shifted_input_ids,
        )

        # "Verify that `labels` has only positive values and -100"
        tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype))

        return shifted_input_ids

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def call(
        self,
        input_ids: tf.Tensor,
        pixel_values: tf.Tensor,
        decoder_input_ids: Optional[tf.Tensor] = None,
        decoder_attention_mask: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        foutput_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[tf.Tensor] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBlipTextVisionModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering

        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # training
        >>> text = "How many cats are in the picture?"
        >>> label = "2"
        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> labels = processor(text=label, return_tensors="tf").input_ids

        >>> inputs["labels"] = labels
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> loss.backward()

        >>> # inference
        >>> text = "How many cats are in the picture?"
        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```"""
        if labels is None and decoder_input_ids is None:
            raise ValueError(
                "Either `decoder_input_ids` or `labels` should be passed when calling `forward` with"
                " `TFBlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you"
                " are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`"
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int64)

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
            training=training,
        )

        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right - this is used in training mode
            decoder_input_ids = self._shift_right(labels)
            # replace possible -100 values in labels by `pad_token_id`
            labels = tf.where(labels == self.decoder_pad_token_id, -100, labels)

        answer_output = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            training=training,
        )

        if labels is not None:
            decoder_loss = tf.reduce_mean(answer_output.loss) if return_dict else tf.reduce_mean(answer_output[0])
        else:
            decoder_loss = None

        if not return_dict:
            outputs = (decoder_loss, image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return TFBlipTextVisionModelOutput(
            loss=decoder_loss,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def generate(
        self,
        input_ids: tf.Tensor,
        pixel_values: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        **generate_kwargs,
    ) -> tf.Tensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:
                Input image to be processed
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            generate_kwargs (dict, *optional*):
                Additional arguments passed to the `generate` function of the decoder


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering

        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are in the picture?"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)

        if isinstance(input_ids, list):
            input_ids = tf.Tensor(input_ids)

        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )

        question_embeds = question_outputs[0]

        question_attention_mask = tf.ones(shape_list(question_embeds)[:-1], dtype=tf.int32)

        bos_ids = tf.fill(
            (tf.shape(question_embeds)[0], 1), value=tf.cast(self.decoder_start_token_id, input_ids.dtype)
        )

        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            **generate_kwargs,
        )

        return outputs


@add_start_docstrings(
    """
    BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of
    image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to
    the image.
    """,
    BLIP_START_DOCSTRING,
)
class TFBlipForImageTextRetrieval(TFBlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.vision_model = TFBlipVisionModel(config.vision_config, name="vision_model")

        self.text_encoder = TFBlipTextModel(config.text_config, name="text_encoder", add_pooling_layer=False)

        # vision projection layer
        self.vision_proj = tf.keras.layers.Dense(
            config.image_text_hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="vision_proj",
        )

        # text projection layer
        self.text_proj = tf.keras.layers.Dense(
            config.image_text_hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="text_proj",
        )

        # image text matching head
        self.itm_head = tf.keras.layers.Dense(
            2, kernel_initializer=get_initializer(config.initializer_range), name="itm_head"
        )

        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.vision_model.embeddings.patch_embedding

    @property
    def dummy_inputs(self):
        input_ids = tf.constant(DUMMY_INPUTS, dtype=tf.int32)
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(len(DUMMY_INPUTS), 3, self.config.vision_config.image_size, self.config.vision_config.image_size),
            dtype=tf.float32,
        )
        return {"input_ids": input_ids, "pixel_values": VISION_DUMMY_INPUTS}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
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

    def serving_output(self, output: TFBlipImageTextMatchingModelOutput) -> TFBlipImageTextMatchingModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBlipImageTextMatchingModelOutput(
            itm_score=output.itm_score,
            last_hidden_state=hs,
            hidden_states=output.hidden_states,
            attentions=attns,
            question_embeds=output.question_embeds,
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipImageTextMatchingModelOutput, config_class=BlipVisionConfig)
    def call(
        self,
        input_ids: tf.Tensor,
        pixel_values: Optional[tf.Tensor] = None,
        use_itm_head: Optional[bool] = True,
        attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = None,
    ) -> Union[Tuple, TFBlipImageTextMatchingModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForImageTextRetrieval

        >>> model = TFBlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> outputs = model(**inputs)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        image_embeds = vision_outputs[0]
        image_atts = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int64)

        # Matt: In PyTorch, only one path (itm/non-itm) is taken. However, in TensorFlow this can result in
        # some layers not being built! To avoid this, we always call both paths, then use an if statement to select
        # which output to pass to the final output. The unnecessary nodes will be pruned from the final graph, but
        # not before the layers have all been built correctly.
        itm_question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=return_dict,
            training=training,
        )
        itm_question_embeds = itm_question_embeds[0] if not return_dict else itm_question_embeds.last_hidden_state

        itm_output = self.itm_head(itm_question_embeds[:, 0, :])

        no_itm_question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            training=training,
        )
        no_itm_question_embeds = (
            no_itm_question_embeds[0] if not return_dict else no_itm_question_embeds.last_hidden_state
        )

        image_feat, _ = tf.linalg.normalize(self.vision_proj(image_embeds[:, 0, :]), ord=2, axis=-1)
        text_feat, _ = tf.linalg.normalize(self.text_proj(no_itm_question_embeds[:, 0, :]), ord=2, axis=-1)

        no_itm_output = tf.matmul(image_feat, text_feat, transpose_b=True)

        if use_itm_head:
            output = itm_output
            question_embeds = itm_question_embeds
        else:
            output = no_itm_output
            question_embeds = no_itm_question_embeds

        if not return_dict:
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple(output for output in outputs if output is not None)

        return TFBlipImageTextMatchingModelOutput(
            itm_score=output,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            question_embeds=question_embeds,
        )
