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
""" TF 2.0 CLIP model. """


import math
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling

# Public API
from ...modeling_tf_utils import (
    DUMMY_INPUTS,
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"

TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",
    # See all CLIP models at https://huggingface.co/models?filter=clip
]


LARGE_NEGATIVE = -1e8


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None, past_key_values_length: int = 0):
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
    return tf.keras.metrics.sparse_categorical_crossentropy(
        y_true=tf.range(shape_list(logits)[0]),
        y_pred=logits,
        from_logits=True
    )


def clip_loss(similarity: tf.Tensor) -> tf.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class TFCLIPOutput(ModelOutput):
    """
    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`return_loss` is :obj:`True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(:obj:`tf.Tensor` of shape :obj:`(image_batch_size, text_batch_size)`):
            The scaled dot product scores between :obj:`image_embeds` and :obj:`text_embeds`. This represents the
            image-text similarity scores.
        logits_per_text:(:obj:`tf.Tensor` of shape :obj:`(text_batch_size, image_batch_size)`):
            The scaled dot product scores between :obj:`text_embeds` and :obj:`image_embeds`. This represents the
            text-image similarity scores.
        text_embeds(:obj:`tf.Tensor` of shape :obj:`(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.TFCLIPTextModel`.
        image_embeds(:obj:`tf.Tensor` of shape :obj:`(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.TFCLIPVisionModel`.
        text_model_output(:obj:`TFBaseModelOutputWithPooling`):
            The output of the :class:`~transformers.TFCLIPTextModel`.
        vision_model_output(:obj:`TFBaseModelOutputWithPooling`):
            The output of the :class:`~transformers.TFCLIPVisionModel`.
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

    def build(self, input_shape: tf.TensorShape):

        # Q: should we use normal initializer?
        # initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        initializer = get_initializer(self.config.initializer_range)

        self.class_embedding = self.add_weight(
            shape=(self.embed_dim,), initializer=initializer, trainable=True, name="class_embedding"
        )

        with tf.name_scope("patch_embedding"):

            # shape = (kernel_size[1], kernel_size[0], in_channels, out_channels)
            # This is in the inverse order of `torch.nn.Conv2d.weight` format, which is necessary to make weight loading (PT->TF) work.
            # See `transformers.modeling_tf_pytorch_utils.load_pytorch_weights_in_tf2_model` for more details (places involving `transpose`).
            # In TensorFlow, `Conv2D.kernel` expects (kernel_size[0], kernel_size[1], in_channels, out_channels),
            # so we need to perform a transpose (1st & 2nd axes) before using `tf.nn.conv2d` in `self.call`.
            self.conv_kernel = self.add_weight(
                shape=(self.patch_size, self.patch_size, 3, self.embed_dim),
                initializer=get_initializer(self.config.initializer_range),
                name="kernel",
            )

        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.num_positions, self.embed_dim),
                initializer=get_initializer(self.initializer_range),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:

        batch_size, num_channels, height, width = shape_list(pixel_values)

        # When running on CPU, `tf.nn.conv2d` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # shape = (kernel_size[0], kernel_size[1], in_channels=num_channels, out_channels=embed_dim)
        # This is the format `tf.nn.conv2d` expects.
        filters = tf.transpose(self.conv_kernel, perm=(1, 0, 2, 3))

        # Conv2D
        # shape = (batch_size, out_height, out_width, out_channels=embed_dim)
        patch_embeds = (
            tf.nn.conv2d(
                input=pixel_values,
                filters=filters,
                strides=self.patch_size,
                padding="VALID",
                data_format="NHWC",
            )
        )

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        patch_embeds = tf.reshape(tensor=patch_embeds, shape=(batch_size, self.num_patches, -1))

        # add the [CLS] token to the embedded patch tokens
        class_embeds = tf.broadcast_to(self.class_embedding, shape=(batch_size, 1, -1))
        embeddings = tf.concat((class_embeds, patch_embeds), axis=1)

        embeddings = embeddings + self.position_embedding

        return embeddings


class TFCLIPTextEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: CLIPTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.vocab_size = config.vocab_size

    def build(self, input_shape: tf.TensorShape):

        with tf.name_scope("token_embedding"):
            self.token_embedding = self.add_weight(
                shape=(self.vocab_size, self.embed_dim),
                initializer=get_initializer(self.initializer_range),
                trainable=True,
                name="embeddings",
            )

        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.num_positions, self.embed_dim),
                initializer=get_initializer(self.initializer_range),
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
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(params=self.token_embedding, indices=input_ids)

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
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).")

        self.sqrt_att_head_size = math.sqrt(self.head_dim)

        self.q_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="q_proj"
        )
        self.k_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="k_proj"
        )
        self.v_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="v_proj"
        )

        self.dropout = tf.keras.layers.Dropout(rate=config.attention_dropout)

        self.out_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
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
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, embed_dim)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))

        attention_output = self.out_proj(attention_output, training=training)
        # In TFBert, attention weights are returned after dropout.
        # However, in CLIP, they are returned before dropout.
        # Q: What to do?
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs


class TFCLIPMLP(tf.keras.layers.Layer):
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        self.activation_fn = get_tf_activation(config.hidden_act)
        self.fc1 = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="fc1"
        )
        self.fc2 = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="fc2"
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
            hidden_states (:obj:`tf.Tensor`): input to the layer of shape :obj:`(batch, seq_len, embed_dim)`
            attention_mask (:obj:`tf.Tensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            causal_attention_mask (:obj:`tf.Tensor`): causal attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`):
                Whether or not to return the attentions tensors of all attention layers. See ``outputs`` under
                returned tensors for more detail.
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
    Transformer encoder consisting of :obj:`config.num_hidden_layers` self attention layers. Each layer is a
    :class:`~transformers.TFCLIPEncoderLayer`.

    Args:
        config: CLIPConfig
    """
    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        self.layers = [TFCLIPEncoderLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

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
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="final_layer_norm")

    def call(
        self,
        input_ids: TFModelInputType,
        attention_mask: tf.Tensor,
        position_ids: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        input_shape = shape_list(input_ids)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        batch_size, seq_length = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)

        # # expand attention_mask
        # if attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
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
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):

        # set an additive 2D attention mask with all places being masked
        to_mask = tf.constant(-10000.0, shape=(seq_length, seq_length), dtype=dtype)

        # set diagonal & lower triangular parts to 0 (i.e. the places not to be masked)
        # TIP: think the 2D matrix as the space of (query_seq, key_seq)
        to_mask = tf.linalg.band_part(to_mask, -1, 0)

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
        self.text_model.embeddings.token_embedding = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = shape_list(inputs["input_ids"])

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(dims=input_shape, value=1)

        text_model_outputs = self.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        return text_model_outputs


class TFCLIPVisionTransformer(tf.keras.layers.Layer):
    def __init__(self, config: CLIPVisionConfig, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFCLIPVisionEmbeddings(config, name="embeddings")
        self.pre_layrnorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="pre_layrnorm")
        self.encoder = TFCLIPEncoder(config, name="encoder")
        self.post_layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="post_layernorm")

    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:

        embedding_output = self.embeddings(pixel_values=pixel_values)
        embedding_output = self.pre_layrnorm(inputs=embedding_output)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.post_layernorm(inputs=pooled_output)

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

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

    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["pixel_values"] is None:
            raise ValueError("You have to specify pixel_values")

        vision_model_outputs = self.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        return vision_model_outputs


@keras_serializable
class TFCLIPMainLayer(tf.keras.layers.Layer):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                f"config.text_config is expected to be of type CLIPTextConfig but is of type {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                f"config.vision_config is expected to be of type CLIPVisionConfig but is of type {type(config.vision_config)}."
            )

        self.config = config

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim

        self.text_model = TFCLIPTextTransformer(text_config, name="text_model")
        self.vision_model = TFCLIPVisionTransformer(vision_config, name="vision_model")

        self.visual_projection = tf.keras.layers.Dense(
            units=self.projection_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=False,
            name="visual_projection",
        )

        self.text_projection = tf.keras.layers.Dense(
            units=self.projection_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=False,
            name="text_projection",
        )

    def build(self, input_shape: tf.TensorShape):

        self.logit_scale = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
            name="logit_scale",
        )

        super().build(input_shape)

    def get_text_features(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        inputs = input_processing(
            func=self.get_text_features,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = shape_list(inputs["input_ids"])

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(dims=input_shape, value=1)

        text_outputs = self.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(inputs=pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        inputs = input_processing(
            func=self.get_image_features,
            config=self.config,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["pixel_values"] is None:
            raise ValueError("You have to specify pixel_values")

        vision_outputs = self.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(inputs=pooled_output)

        return image_features

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
        **kwargs,
    ) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is None:
            raise ValueError("You have to specify either input_ids")
        if inputs["pixel_values"] is None:
            raise ValueError("You have to specify pixel_values")

        input_shape = shape_list(inputs["input_ids"])

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(dims=input_shape, value=1)

        vision_outputs = self.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        text_outputs = self.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(inputs=image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(inputs=text_embeds)

        # normalized features
        image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord='fro', axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord='fro', axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)

        loss = None
        if inputs["return_loss"]:
            loss = clip_loss(logits_per_text)

        if not inputs["return_dict"]:
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


CLIP_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(input_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.CLIPConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.TFPreTrainedModel.from_pretrained` method to load the
            model weights.
"""

CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`np.ndarray`, :obj:`tf.Tensor`, :obj:`List[tf.Tensor]` :obj:`Dict[str, tf.Tensor]` or :obj:`Dict[str, np.ndarray]` and each example must have the shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        position_ids (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (:obj:`np.ndarray`, :obj:`tf.Tensor`, :obj:`List[tf.Tensor]` :obj:`Dict[str, tf.Tensor]` or :obj:`Dict[str, np.ndarray]` and each example must have the shape :obj:`(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using :class:`~transformers.CLIPFeatureExtractor`. See
            :meth:`transformers.CLIPFeatureExtractor.__call__` for details.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

CLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`np.ndarray`, :obj:`tf.Tensor`, :obj:`List[tf.Tensor]` :obj:`Dict[str, tf.Tensor]` or :obj:`Dict[str, np.ndarray]` and each example must have the shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        pixel_values (:obj:`np.ndarray`, :obj:`tf.Tensor`, :obj:`List[tf.Tensor]` :obj:`Dict[str, tf.Tensor]` or :obj:`Dict[str, np.ndarray]` and each example must have the shape :obj:`(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using :class:`~transformers.CLIPFeatureExtractor`. See
            :meth:`transformers.CLIPFeatureExtractor.__call__` for details.
        attention_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        position_ids (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        return_loss (:obj:`bool`, `optional`):
            Whether or not to return the contrastive loss.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


class TFCLIPTextModel(TFCLIPPreTrainedModel):
    config_class = CLIPTextConfig
    base_model_prefix = "clip_text"

    def __init__(self, config: CLIPTextConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.clip_text = TFCLIPTextMainLayer(config, name="clip_text")

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
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples::

            >>> from transformers import CLIPTokenizer, TFCLIPTextModel

            >>> model = TFCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"],  padding=True, return_tensors="tf")

            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooled_output # pooled (EOS token) states
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.clip_text(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
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


class TFCLIPVisionModel(TFCLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    base_model_prefix = "clip_vision"

    def __init__(self, config: CLIPVisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.clip_vision = TFCLIPVisionMainLayer(config, name="clip_vision")

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples::

            >>> from PIL import Image
            >>> import requests
            >>> from transformers import CLIPProcessor, TFCLIPVisionModel

            >>> model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(images=image, return_tensors="tf")

            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooled_output # pooled CLS states
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.clip_vision(
            pixel_values=inputs["pixel_values"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
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
    base_model_prefix = "clip"

    def __init__(self, config: CLIPConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.clip = TFCLIPMainLayer(config, name="clip")

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
        **kwargs,
    ) -> tf.Tensor:
        r"""
        Returns:
            text_features (:obj:`tf.Tensor` of shape :obj:`(batch_size, output_dim`): The text embeddings
            obtained by applying the projection layer to the pooled output of :class:`~transformers.TFCLIPTextModel`.

        Examples::

            >>> from transformers import CLIPTokenizer, TFCLIPModel

            >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"],  padding=True, return_tensors="tf")
            >>> text_features = model.get_text_features(**inputs)
        """
        inputs = input_processing(
            func=self.get_text_features,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        text_features = self.clip.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
        )

        return text_features

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> tf.Tensor:
        r"""
        Returns:
            image_features (:obj:`tf.Tensor` of shape :obj:`(batch_size, output_dim`): The image embeddings
            obtained by applying the projection layer to the pooled output of :class:`~transformers.TFCLIPVisionModel`.

        Examples::

            >>> from PIL import Image
            >>> import requests
            >>> from transformers import CLIPProcessor, TFCLIPModel

            >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(images=image, return_tensors="tf")

            >>> image_features = model.get_image_features(**inputs)
        """
        inputs = input_processing(
            func=self.get_image_features,
            config=self.config,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        image_features = self.clip.get_image_features(
            pixel_values=inputs["pixel_values"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
        )

        return image_features

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
        **kwargs,
    ) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples::

            >>> from PIL import Image
            >>> import requests
            >>> from transformers import CLIPProcessor, TFCLIPModel

            >>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True)

            >>> outputs = model(**inputs)
            >>> logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            >>> probs = tf.nn.softmax(logits_per_image, axis=1) # we can take the softmax to get the label probabilities

        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        outputs = self.clip(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            return_loss=inputs["return_loss"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
        )

        return outputs

    def serving_output(self, output: TFCLIPOutput) -> TFCLIPOutput:
        return output
