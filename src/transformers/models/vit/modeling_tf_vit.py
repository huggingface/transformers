# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 ViT model. """


import collections.abc
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_vit import ViTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ViTConfig"
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224"


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py


class TFViTEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.patch_embeddings = TFPatchEmbeddings(config, name="patch_embeddings")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def build(self, input_shape: tf.TensorShape):

        num_patches = self.patch_embeddings.num_patches
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size), initializer="zeros", trainable=True, name="cls_token"
        )
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.config.hidden_size),
            initializer="zeros",
            trainable=True,
            name="position_embeddings",
        )

        super().build(input_shape)

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        batch_size, seq_len, dim = shape_list(embeddings)
        npatch = seq_len - 1

        _, N, _ = shape_list(self.position_embeddings)
        N -= 1

        if npatch == N and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(patch_pos_embed, shape=(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)),
            size=(h0, w0),
            method="bicubic",
        )

        shape = shape_list(patch_pos_embed)
        assert h0 == shape[-3] and w0 == shape[-2]
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, training=training
        )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings, training=training)

        return embeddings


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class TFPatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        image_size = to_2tuple(config.image_size)
        patch_size = to_2tuple(config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size
        self.config = config

        self.projection = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer=get_initializer(self.config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if not interpolate_pos_encoding:
            if getattr(height, "numpy", None) and getattr(width, "numpy", None):
                if height != self.image_size[0] or width != self.image_size[1]:
                    raise ValueError(
                        f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
                    )

        # When running on CPU, `tf.keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        projection = self.projection(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        x = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

        return x


class TFViTSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs


class TFViTSelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states


class TFViTAttention(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFViTSelfAttention(config, name="attention")
        self.dense_output = TFViTSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


class TFViTIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TFViTOutput(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = hidden_states + input_tensor

        return hidden_states


class TFViTLayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFViTAttention(config, name="attention")
        self.intermediate = TFViTIntermediate(config, name="intermediate")
        self.vit_output = TFViTOutput(config, name="output")

        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(
            # in ViT, layernorm is applied before self-attention
            input_tensor=self.layernorm_before(inputs=hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(inputs=hidden_states)

        intermediate_output = self.intermediate(hidden_states=layer_output)

        # second residual connection is done here
        layer_output = self.vit_output(
            hidden_states=intermediate_output, input_tensor=hidden_states, training=training
        )
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs


class TFViTEncoder(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.layer = [TFViTLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
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


@keras_serializable
class TFViTMainLayer(tf.keras.layers.Layer):
    config_class = ViTConfig

    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.embeddings = TFViTEmbeddings(config, name="embeddings")
        self.encoder = TFViTEncoder(config, name="encoder")
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        self.pooler = TFViTPooler(config, name="pooler") if add_pooling_layer else None

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if "input_ids" in inputs:
            inputs["pixel_values"] = inputs.pop("input_ids")

        if inputs["pixel_values"] is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(
            pixel_values=inputs["pixel_values"],
            interpolate_pos_encoding=inputs["interpolate_pos_encoding"],
            training=inputs["training"],
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            head_mask=inputs["head_mask"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(inputs=sequence_output)
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None

        if not inputs["return_dict"]:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TFViTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(3, self.config.num_channels, self.config.image_size, self.config.image_size), dtype=tf.float32
        )
        return {"pixel_values": tf.constant(VISION_DUMMY_INPUTS)}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
            }
        ]
    )
    def serving(self, inputs):
        """
        Method used for serving the model.

        Args:
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)


VIT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all
    the tensors in the first argument of the model call function: `model(inputs)`.

    </Tip>

    Args:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the
            model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`): Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See [`ViTFeatureExtractor.__call__`] for details.

        head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class TFViTModel(TFViTPreTrainedModel):
    def __init__(self, config: ViTConfig, *inputs, add_pooling_layer=True, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.vit = TFViTMainLayer(config, add_pooling_layer=add_pooling_layer, name="vit")

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import ViTFeatureExtractor, TFViTModel
        >>> from PIL import Image
        >>> import requests

        >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        >>> model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        >>> inputs = feature_extractor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if "input_ids" in inputs:
            inputs["pixel_values"] = inputs.pop("input_ids")

        outputs = self.vit(
            pixel_values=inputs["pixel_values"],
            head_mask=inputs["head_mask"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            interpolate_pos_encoding=inputs["interpolate_pos_encoding"],
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


class TFViTPooler(tf.keras.layers.Layer):
    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output


@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    VIT_START_DOCSTRING,
)
class TFViTForImageClassification(TFViTPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: ViTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.vit = TFViTMainLayer(config, add_pooling_layer=False, name="vit")

        # Classifier head
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
        **kwargs,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import ViTFeatureExtractor, TFViTForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        >>> model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        >>> inputs = feature_extractor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        ```"""
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        if "input_ids" in inputs:
            inputs["pixel_values"] = inputs.pop("input_ids")

        outputs = self.vit(
            pixel_values=inputs["pixel_values"],
            head_mask=inputs["head_mask"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            interpolate_pos_encoding=inputs["interpolate_pos_encoding"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]
        logits = self.classifier(inputs=sequence_output[:, 0, :])
        loss = None if inputs["labels"] is None else self.compute_loss(labels=inputs["labels"], logits=logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)
