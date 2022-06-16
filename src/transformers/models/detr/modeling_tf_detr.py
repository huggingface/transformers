# coding=utf-8
# Copyright 2021 Facebook AI Research The HuggingFace Inc. team. All rights reserved.
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
""" Tensorflow DETR model."""


import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithCrossAttentions, TFSeq2SeqModelOutput
from ...modeling_tf_utils import TFPreTrainedModel, unpack_inputs
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_detr import DetrConfig


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

if is_vision_available():
    from .feature_extraction_detr import center_to_corners_format


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DetrConfig"
_CHECKPOINT_FOR_DOC = "facebook/detr-resnet-50"

TF_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/detr-resnet-50",
    # See all DETR models at https://huggingface.co/models?filter=detr
]


@dataclass
class TFDetrDecoderOutput(TFBaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`tf.Tensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    intermediate_hidden_states: Optional[tf.Tensor] = None


@dataclass
class TFDetrModelOutput(TFSeq2SeqModelOutput):
    """
    Base class for outputs of the DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each layer plus
            the initial embedding outputs.
        decoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each layer plus
            the initial embedding outputs.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        intermediate_hidden_states (`tf.Tensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    intermediate_hidden_states: Optional[tf.Tensor] = None


@dataclass
class TFDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`TFDetrForObjectDetection`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`tf.Tensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`tf.Tensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~DetrFeatureExtractor.post_process`] to retrieve the unnormalized bounding
            boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each layer plus
            the initial embedding outputs.
        decoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each layer plus
            the initial embedding outputs.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[tf.Tensor] = None
    loss_dict: Optional[Dict] = None
    logits: tf.Tensor = None
    pred_boxes: tf.Tensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[tf.Tensor] = None
    decoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[tf.Tensor]] = None
    cross_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_last_hidden_state: Optional[tf.Tensor] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None


@dataclass
class TFDetrSegmentationOutput(ModelOutput):
    """
    Output type of [`DetrForSegmentation`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`tf.Tensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`tf.Tensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~DetrFeatureExtractor.post_process`] to retrieve the unnormalized bounding
            boxes.
        pred_masks (`tf.Tensor` of shape `(batch_size, num_queries, height/4, width/4)`):
            Segmentation masks logits for all queries. See also [`~DetrFeatureExtractor.post_process_segmentation`] or
            [`~DetrFeatureExtractor.post_process_panoptic`] to evaluate instance and panoptic segmentation masks
            respectively.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each layer plus
            the initial embedding outputs.
        decoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each layer plus
            the initial embedding outputs.
        encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[tf.Tensor] = None
    loss_dict: Optional[Dict] = None
    logits: tf.Tensor = None
    pred_boxes: tf.Tensor = None
    pred_masks: tf.Tensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[tf.Tensor] = None
    decoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    decoder_attentions: Optional[Tuple[tf.Tensor]] = None
    cross_attentions: Optional[Tuple[tf.Tensor]] = None
    encoder_last_hidden_state: Optional[tf.Tensor] = None
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = None
    encoder_attentions: Optional[Tuple[tf.Tensor]] = None


# BELOW: utilities copied from
# https://github.com/facebookresearch/detr/blob/master/backbone.py
class TFDetrFrozenBatchNorm2d(tf.keras.layers.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n: int, **kwargs) -> None:
        super(TFDetrFrozenBatchNorm2d, self).__init__(**kwargs)
        # Set as None first?
        self.gamma = tf.ones(n)
        self.beta = tf.zeros(n)
        self.moving_mean = tf.zeros(n)
        self.moving_variance = tf.ones(n)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        moving_variance = self.moving_variance.reshape(1, -1, 1, 1)
        moving_mean = self.moving_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (moving_variance + epsilon).rsqrt()
        bias = bias - moving_mean * scale
        return x * scale + bias


def replace_batch_norm(m, name=""):  # FIXME - make sure batchnorms become frozen
    for i, layer in enumerate(m.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            bn = layer
            frozen = TFDetrFrozenBatchNorm2d(bn.gamma.shape[0], name=f"{layer.name}.{i}")
            # Identity returns a copy of a tensor with the same elements
            frozen.gamma = tf.identity(bn.gamma)
            frozen.beta = tf.identity(bn.beta)
            frozen.moving_mean = tf.identity(bn.moving_mean)
            frozen.moving_variance = tf.identity(bn.moving_variance)
            m.layers[i] = frozen  # FIXME - this reattribution is messy


# Temproary hack until we have TFResNet
# FIXME - make more generic backbone feature extraction
def create_model(backbone_name):
    if backbone_name != "resnet50":
        raise Exception

    model = tf.keras.applications.resnet50.ResNet50(include_top=False)
    output_layers = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    backbone = tf.keras.models.Model(
        inputs=model.input, outputs=[model.get_layer(layer).output for layer in output_layers], name="model"
    )
    return backbone


class TFDetrTimmConvEncoder(tf.keras.layers.Layer):
    """
    Convolutional encoder (backbone) from the timm library.

    nn.BatchNorm2d layers are replaced by DetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, backbone_name: str, dilation: bool, **kwargs) -> None:
        super().__init__(**kwargs)

        kwargs = {}
        if dilation:
            kwargs["output_stride"] = 16

        backbone = create_model(backbone_name)  # FIXME
        # backbone = create_model(name, pretrained=True, features_only=True, out_indices=(1, 2, 3, 4), **kwargs)
        # replace batch norm by frozen batch norm
        # with torch.no_grad():  # FIXME
        replace_batch_norm(backbone)
        self.model = backbone
        # self.model.feature_info.channels() #FIXME - don't make hard coded
        self.intermediate_channel_sizes = [256, 512, 1024, 2048]

        # if "resnet" in name:
        #     for name, parameter in self.model.named_parameters():
        #         if "layer2" not in name and "layer3" not in name and "layer4" not in name:
        #             parameter.requires_grad_(False)

    def call(self, pixel_values: tf.Tensor, pixel_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        # send pixel_values through the model to get list of feature maps
        # FIXME - make model take BCHW?
        # FIXME - compare resnet implementations
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))
        features = self.model(pixel_values, training=training)

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = tf.image.resize(pixel_mask[:, :, :, None], size=feature_map.shape[1:3])
            mask = tf.squeeze(mask)
            mask = tf.cast(mask, tf.bool)
            feature_map = tf.transpose(feature_map, (0, 3, 1, 2))
            out.append((feature_map, mask))
        return out


class TFDetrConvModel(tf.keras.layers.Layer):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder: tf.keras.layers.Layer, position_embedding, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def call(
        self, pixel_values: tf.Tensor, pixel_mask: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask, training=training)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos_enc = self.position_embedding(feature_map, mask, training=training)
            pos_enc = tf.cast(pos_enc, feature_map.dtype)
            pos.append(pos_enc)

        return out, pos


def _expand_mask(mask: tf.Tensor, dtype: tf.DType, tgt_len: Optional[int] = None) -> tf.Tensor:
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = tf.repeat(mask[:, None, None, :], tgt_len, axis=2)  # FIXME - double check
    expanded_mask = tf.cast(expanded_mask, dtype)

    inverted_mask = 1.0 - expanded_mask
    inverted_mask = tf.cast(inverted_mask, tf.bool)

    # FIXME - don't use experimental
    # FIXME - make sure mask is being set as expected
    expand_mask = tf.where(inverted_mask, inverted_mask, tf.experimental.numpy.finfo(dtype).min)
    return expand_mask


class TFDetrSinePositionEmbedding(tf.keras.layers.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, embedding_dim: int = 64, temperature: int = 10000, normalize: bool = False, scale=None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, pixel_values: tf.Tensor, pixel_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = tf.math.cumsum(tf.cast(pixel_mask, tf.float32), axis=1)  # FIXME dtype=tf.float32
        x_embed = tf.math.cumsum(tf.cast(pixel_mask, tf.float32), axis=2)  # FIXME dtype=tf.float32
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = tf.range(self.embedding_dim, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.embedding_dim)  # FIXME - check torch_int_div

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = tf.stack((tf.math.sin(pos_x[:, :, :, 0::2]), tf.math.cos(pos_x[:, :, :, 1::2])), axis=4)
        pos_x = tf.reshape(pos_x, (*pos_x.shape[:3], -1))
        pos_y = tf.stack((tf.math.sin(pos_y[:, :, :, 0::2]), tf.math.cos(pos_y[:, :, :, 1::2])), axis=4)
        pos_y = tf.reshape(pos_y, (*pos_y.shape[:3], -1))
        pos = tf.concat((pos_y, pos_x), axis=3)
        pos = tf.transpose(pos, (0, 3, 1, 2))
        return pos


class TFDetrLearnedPositionEmbedding(tf.keras.layers.Layer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        self.row_embeddings = tf.keras.layers.Embedding(50, embedding_dim, name="row_embeddings")
        self.column_embeddings = tf.keras.layers.Embedding(50, embedding_dim, name="column_embeddings")

    def call(
        self, pixel_values: tf.Tensor, pixel_mask: Optional[tf.Tensor] = None, training: bool = False
    ) -> tf.Tensor:
        h, w = pixel_values.shape[-2:]
        i = tf.range(w)
        j = tf.range(h)
        x_emb = self.column_embeddings(i)
        y_emb = self.row_embeddings(j)

        x_emb = tf.expand_dims(x_emb, 0)
        x_emb = tf.repeat(x_emb, (h, 1, 1))
        y_emb = tf.expand_dims(y_emb, 1)
        y_emb = tf.repeat(y_emb, (1, w, 1))

        pos = tf.concat([x_emb, y_emb], axis=-1)
        pos = tf.transpose(pos, (2, 0, 1))
        pos = tf.expand_dims(pos, 0)
        pos = tf.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = TFDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = TFDetrLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


class TFDetrAttention(tf.keras.layers.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int) -> tf.Tensor:
        tensor = tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        return tensor

    def with_pos_embed(self, tensor: tf.Tensor, position_embeddings: Optional[tf.Tensor]) -> None:
        return tensor if position_embeddings is None else tensor + position_embeddings

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_embeddings: Optional[tf.Tensor] = None,
        key_value_states: Optional[tf.Tensor] = None,
        key_value_position_embeddings: Optional[tf.Tensor] = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[Tuple[tf.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.shape

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, bsz)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tf.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        value_states = tf.reshape(value_states, proj_shape)

        src_len = key_states.shape[1]

        # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = tf.matmul(query_states, tf.transpose(key_states, (0, 2, 1)))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = tf.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + tf.cast(
                attention_mask, tf.float32
            )
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

        attn_probs = attn_weights  # FIXME - is this right?
        if training:  # FIXME
            attn_probs = tf.nn.dropout(attn_weights, rate=self.dropout)

        attn_output = tf.matmul(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = tf.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class TFDetrEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: DetrConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFDetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            name="self_attn",
        )
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(name="self_attn_layer_norm")
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(name="final_layer_norm")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        position_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_embeddings (`tf.Tensor`, *optional*): position embeddings, to be added to hidden_states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            training=training,
        )

        if training:  # FIXME
            hidden_states = tf.nn.dropout(hidden_states, rate=self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states, training=training)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        if training:  # FIXME
            hidden_states = tf.nn.dropout(hidden_states, rate=self.activation_dropout)

        hidden_states = self.fc2(hidden_states)
        if training:  # FIXME
            hidden_states = tf.nn.dropout(hidden_states, rate=self.dropout)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states, training=training)

        if training:
            if tf.isinf(hidden_states).any() or tf.isnan(hidden_states).any():
                # FIXME - don't use experimental module
                clamp_value = tf.experimental.numpy.finfo(hidden_states.dtype).max - 1000
                hidden_states = tf.clip_by_value(
                    hidden_states, clip_value_min=-clamp_value, clip_value_max=clamp_value
                )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class TFDetrDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: DetrConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embed_dim = config.d_model

        self.self_attn = TFDetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            name="self_attn",
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(name="self_attn_layer_norm")
        self.encoder_attn = TFDetrAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            name="encoder_attn",
        )
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(name="encoder_attn_layer_norm")
        self.fc1 = tf.keras.layers.Dense(config.decoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(name="final_layer_norm")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_embeddings: Optional[tf.Tensor] = None,
        query_position_embeddings: Optional[tf.Tensor] = None,
        encoder_hidden_states: Optional[tf.Tensor] = None,
        encoder_attention_mask: Optional[tf.Tensor] = None,
        output_attentions: Optional[bool] = False,
        training: bool = False,
    ) -> None:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_embeddings (`tf.Tensor`, *optional*):
                position embeddings that are added to the queries and keys
            in the cross-attention layer.
            query_position_embeddings (`tf.Tensor`, *optional*):
                position embeddings that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (`tf.Tensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`tf.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        if training:  # FIXME
            hidden_states = tf.nn.dropout(hidden_states, rate=self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states, training=training)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                position_embeddings=query_position_embeddings,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                key_value_position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                training=training,
            )
            if training:  # FIXME
                hidden_states = tf.nn.dropout(hidden_states, rate=self.dropout)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states, training=training)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        if training:  # FIXME
            hidden_states = tf.nn.dropout(hidden_states, rate=self.activation_dropout)
        hidden_states = self.fc2(hidden_states, training=training)
        if training:  # FIXME
            hidden_states = tf.nn.dropout(hidden_states, rate=self.dropout)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states, training=training)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class TFDetrClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Linear(input_dim, inner_dim, name="dense")
        self.dropout = tf.keras.layers.Dropout(p=pooler_dropout, name="dropout")
        self.out_proj = tf.keras.layers.Linear(inner_dim, num_classes, name="out_proj")

    def call(self, hidden_states: tf.Tensor, training: bool = False):
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.dense(hidden_states)
        hidden_states = tf.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class TFDetrPreTrainedModel(TFPreTrainedModel):
    config_class = DetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network. Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(3, 3, 800, 800),
            dtype=tf.float32,
        )
        return {"pixel_values": tf.constant(VISION_DUMMY_INPUTS)}


DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`DetrFeatureExtractor`]. See [`DetrFeatureExtractor.__call__`] for
            details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        encoder_outputs (`tuple(tuple(tf.Tensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`tf.Tensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class TFDetrEncoder(TFDetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`DetrEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for DETR:

    - position_embeddings are added to the forward pass.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.encoder_layers = [TFDetrEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]

        # in the original DETR, no layernorm is used at the end of the encoder, as "normalize_before" is set to False by default

    def call(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        training: bool = False,
    ):
        r"""
        Args:
            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.

            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:

                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).

                [What are attention masks?](../glossary#attention-mask)

            position_embeddings (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.

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

        hidden_states = inputs_embeds
        if training:  # FIXME
            hidden_states = tf.nn.dropout(hidden_states, rate=self.dropout)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.encoder_layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                # we add position_embeddings as extra input to the encoder_layer
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
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


class TFDetrDecoder(TFDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - position_embeddings and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.decoder_layers = [TFDetrDecoderLayer(config, name=f"layers.{i}") for i in range(config.decoder_layers)]

        # in DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = tf.keras.layers.LayerNormalization(name="layernorm")

        self.gradient_checkpointing = False

    def call(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training: bool = False,
    ):
        r"""
        Args:
            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`tf.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            position_embeddings (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each cross-attention layer.
            query_position_embeddings (`tf.Tensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the queries and keys in each self-attention layer.
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

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.shape[:-1]

        combined_attention_mask = None

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # optional intermediate hidden states
        intermediate = () if self.config.auxiliary_loss else None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.decoder_layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if training and (dropout_probability < self.layerdrop):
                continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                training=training,
            )

            hidden_states = layer_outputs[0]

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states, training=training)
                intermediate += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states, training=training)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # stack intermediate decoder activations
        if self.config.auxiliary_loss:
            intermediate = tf.stack(intermediate)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
                if v is not None
            )
        return TFDetrDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=intermediate,
        )


@add_start_docstrings(
    """
    The bare DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """,
    DETR_START_DOCSTRING,
)
class TFDetrModel(TFDetrPreTrainedModel):
    def __init__(self, config: DetrConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Create backbone + positional encoding
        backbone = TFDetrTimmConvEncoder(config.backbone, config.dilation, name="conv_encoder")
        position_embeddings = build_position_encoding(config)
        self.backbone = TFDetrConvModel(backbone, position_embeddings, name="backbone")

        # Create projection layer
        self.input_projection = tf.keras.layers.Conv2D(config.d_model, kernel_size=1, name="input_projection")

        self.query_position_embeddings = tf.keras.layers.Embedding(
            config.num_queries, config.d_model, name="query_position_embeddings"
        )
        # FIXME - put in build. Needed to gets weights in forward call
        self.query_position_embeddings(tf.random.uniform((1, int(config.num_queries), int(config.d_model))))

        self.encoder = TFDetrEncoder(config, name="encoder")
        self.decoder = TFDetrDecoder(config, name="decoder")

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        pass
        # for name, param in self.detr.backbone.conv_encoder.model.named_parameters():
        #     param.requires_grad_(False)

    def unfreeze_backbone(self):
        pass
        # for name, param in self.detr.backbone.conv_encoder.model.named_parameters():
        #     param.requires_grad_(True)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import DetrFeatureExtractor, DetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = DetrModel.from_pretrained("facebook/detr-resnet-50")
        >>> inputs = feature_extractor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = tf.ones(((batch_size, height, width)))

        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask, training=training)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_map = tf.transpose(feature_map, (0, 2, 3, 1))
        projected_feature_map = self.input_projection(feature_map, training=training)
        projected_feature_map = tf.transpose(projected_feature_map, (0, 3, 1, 2))

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = tf.reshape(projected_feature_map, (*projected_feature_map.shape[:2], -1))
        flattened_features = tf.transpose(flattened_features, (0, 2, 1))

        position_embeddings = position_embeddings_list[-1]
        position_embeddings = tf.reshape(position_embeddings, (*position_embeddings.shape[:2], -1))
        position_embeddings = tf.transpose(position_embeddings, (0, 2, 1))

        flattened_mask = tf.reshape(mask, (mask.shape[0], -1))

        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, sent query embeddings + position embeddings through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.query_position_embeddings.weights[0]
        query_position_embeddings = tf.expand_dims(query_position_embeddings, 0)
        query_position_embeddings = tf.repeat(query_position_embeddings, (batch_size,), axis=0)
        queries = tf.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return TFDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
        )


@add_start_docstrings(
    """
    DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
    such as COCO detection.
    """,
    DETR_START_DOCSTRING,
)
class TFDetrForObjectDetection(TFDetrPreTrainedModel):
    def __init__(self, config: DetrConfig, **kwargs):
        super().__init__(config, **kwargs)

        # DETR encoder-decoder model
        self.model = TFDetrModel(config, name="model")

        # Object detection heads
        # We add one for the "no object" class
        self.class_labels_classifier = tf.keras.layers.Dense(config.num_labels + 1, name="class_labels_classifier")
        self.bbox_predictor = TFDetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3, name="bbox_predictor"
        )

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    # @torch.jit.unused  # FIXME - equivalent in TF?
    def _set_aux_loss(self, outputs_class, outputs_coord) -> List[Dict]:  # FIXME
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training: bool = False,
    ) -> Union[TFDetrObjectDetectionOutput, Tuple]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `tf.Tensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import DetrFeatureExtractor, TFDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = TFDetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # model predicts bounding boxes and corresponding COCO classes
        >>> logits = outputs.logits
        >>> bboxes = outputs.pred_boxes
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output, training=training)
        pred_boxes = tf.sigmoid(self.bbox_predictor(sequence_output, training=training))

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = TFDetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = TFDetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return TFDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings(
    """
    DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top, for tasks
    such as COCO panoptic.

    """,
    DETR_START_DOCSTRING,
)
class TFDetrForSegmentation(TFDetrPreTrainedModel):
    def __init__(self, config: DetrConfig, **kwargs):
        super().__init__(config, **kwargs)

        # object detection model
        self.detr = TFDetrForObjectDetection(config)

        # segmentation head
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        intermediate_channel_sizes = self.detr.model.backbone.conv_encoder.intermediate_channel_sizes

        self.mask_head = TFDetrMaskHeadSmallConv(
            hidden_size + number_of_heads, intermediate_channel_sizes[::-1][-3:], hidden_size, name="mask_head"
        )

        self.bbox_attention = TFDetrMHAttentionMap(
            hidden_size, hidden_size, number_of_heads, dropout=0.0, std=config.init_xavier_std, name="bbox_attention"
        )

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training: bool = False,
    ):
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss, DICE/F-1 loss and Focal loss. List of dicts, each
            dictionary containing at least the following 3 keys: 'class_labels', 'boxes' and 'masks' (the class labels,
            bounding boxes and segmentation masks of an image in the batch respectively). The class labels themselves
            should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)`, the boxes a `tf.Tensor`
            of shape `(number of bounding boxes in the image, 4)` and the masks a `tf.Tensor` of shape `(number of
            bounding boxes in the image, height, width)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import DetrFeatureExtractor, DetrForSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
        >>> model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # model predicts COCO classes, bounding boxes, and masks
        >>> logits = outputs.logits
        >>> bboxes = outputs.pred_boxes
        >>> masks = outputs.pred_masks
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = tf.ones((batch_size, height, width))

        # First, get list of feature maps and position embeddings
        features, position_embeddings_list = self.detr.model.backbone(
            pixel_values, pixel_mask=pixel_mask, training=training
        )

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_map, mask = features[-1]
        batch_size, num_channels, height, width = feature_map.shape
        projected_feature_map = self.detr.model.input_projection(feature_map, training=training)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = tf.reshape(projected_feature_map, (*projected_feature_map[:2], -1))
        flattened_features = tf.transpose(flattened_features, (0, 2, 1))

        position_embeddings = position_embeddings_list[-1]
        position_embeddings = tf.reshape(position_embeddings, (*position_embeddings[:2], -1))
        position_embeddings = tf.transpose(position_embeddings, (0, 2, 1))

        flattened_mask = tf.reshape(mask, (mask.shape[0], -1))

        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.detr.model.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                training=training,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, TFBaseModelOutput):
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, sent query embeddings + position embeddings through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.detr.model.query_position_embeddings.weight
        query_position_embeddings = tf.expand_dims(query_position_embeddings, 0)
        query_position_embeddings = tf.repeat(query_position_embeddings, (batch_size, 1, 1))
        queries = tf.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.detr.model.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = decoder_outputs[0]

        # Sixth, compute logits, pred_boxes and pred_masks
        logits = self.detr.class_labels_classifier(sequence_output)
        pred_boxes = self.detr.bbox_predictor(sequence_output).sigmoid()

        memory = tf.transpose(encoder_outputs[0], (0, 2, 1, 3))
        memory = tf.reshape(memory, (batch_size, self.config.d_model, height, width))
        mask = tf.reshape(flattened_mask, (batch_size, height, width))

        # FIXME h_boxes takes the last one computed, keep this in mind
        # important: we need to reverse the mask, since in the original implementation the mask works reversed
        # bbox_mask is of shape (batch_size, num_queries, number_of_attention_heads in bbox_attention, height/32, width/32)
        bbox_mask = self.bbox_attention(sequence_output, memory, mask=~mask, training=training)

        seg_masks = self.mask_head(
            projected_feature_map, bbox_mask, [features[2][0], features[1][0], features[0][0]], training=training
        )

        pred_masks = tf.reshape(
            seg_masks, (batch_size, self.detr.config.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        )

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = TFDetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality", "masks"]
            criterion = TFDetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
                training=training,
            )
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            outputs_loss["pred_masks"] = pred_masks
            if self.config.auxiliary_loss:
                intermediate = decoder_outputs.intermediate_hidden_states if return_dict else decoder_outputs[-1]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            loss_dict = criterion(outputs_loss, labels, training=training)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            weight_dict["loss_mask"] = self.config.mask_loss_coefficient
            weight_dict["loss_dice"] = self.config.dice_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes, pred_masks) + auxiliary_outputs + decoder_outputs + encoder_outputs
            else:
                output = (logits, pred_boxes, pred_masks) + decoder_outputs + encoder_outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return TFDetrSegmentationOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            pred_masks=pred_masks,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def _expand(tensor, length: int):
    # FIXME - check dimensions
    tensor = tf.expand_dims(tensor, 1)
    tensor = tf.repeat(tensor, (1, int(length), 1, 1, 1))
    tensor = tf.reshape(tensor, (-1, *tensor.shape[1:]))
    return tensor


# taken from https://github.com/facebookresearch/detr/blob/master/models/segmentation.py
class TFDetrMaskHeadSmallConv(tf.keras.layers.Layer):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, **kwargs):
        super().__init__(**kwargs)

        if dim % 8 != 0:
            raise ValueError(
                "The hidden_size + number of attention heads must be divisible by 8 as the number of groups in"
                " GroupNorm is set to 8"
            )

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        self.lay1 = tf.keras.layers.Conv2D(dim, 3, padding=1, name="lay1")
        self.gn1 = tf.keras.layers.GroupNorm(dim, name="gn1")
        self.lay2 = tf.keras.layers.Conv2D(inter_dims[1], 3, padding=1, name="lay2")
        self.gn2 = tf.keras.layers.GroupNorm(inter_dims[1], name="gn2")
        self.lay3 = tf.keras.layers.Conv2D(inter_dims[2], 3, padding=1, name="lay3")
        self.gn3 = tf.keras.layers.GroupNorm(inter_dims[2], name="gn3")
        self.lay4 = tf.keras.layers.Conv2D(inter_dims[3], 3, padding=1, name="lay4")
        self.gn4 = tf.keras.layers.GroupNorm(inter_dims[3], name="gn4")
        self.lay5 = tf.keras.layers.Conv2D(inter_dims[4], 3, padding=1, name="lay5")
        self.gn5 = tf.keras.layers.GroupNorm(inter_dims[4], name="gn5")
        self.out_lay = tf.keras.layers.Conv2D(1, 3, padding=1, name="out_lay")

        self.dim = dim

        self.adapter1 = tf.keras.layers.Conv2D(inter_dims[1], 1, name="adapter1")
        self.adapter2 = tf.keras.layers.Conv2D(inter_dims[2], 1, name="adapter2")
        self.adapter3 = tf.keras.layers.Conv2D(inter_dims[3], 1, name="adapter3")

        for m in self.modules():
            if isinstance(m, tf.keras.layers.Conv2D):
                # FIXME
                pass
                # nn.init.kaiming_uniform_(m.weight, a=1)
                # nn.init.constant_(m.bias, 0)

    def call(self, x: tf.Tensor, bbox_mask: tf.Tensor, fpns: List[tf.Tensor], training: bool = False):
        # here we concatenate x, the projected feature map, of shape (batch_size, d_model, heigth/32, width/32) with
        # the bbox_mask = the attention maps of shape (batch_size, n_queries, n_heads, height/32, width/32).
        # We expand the projected feature map to match the number of heads.
        x = tf.concat([_expand(x, bbox_mask.shape[1]), tf.reshape(bbox_mask, (bbox_mask.shape[0], -1))], 1)

        x = self.lay1(x, training=training)
        x = self.gn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.lay2(x, training=training)
        x = self.gn2(x, training=training)
        x = tf.nn.relu(x)

        cur_fpn = self.adapter1(fpns[0], training=training)
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))

        # x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = tf.image.resize(
            x, size=cur_fpn.shape[-2:], mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )  # FIXME - check these are equivalent
        x = self.lay3(x, training=training)
        x = self.gn3(x, training=training)
        x = tf.nn.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = tf.image.resize(
            x, size=cur_fpn.shape[-2:], mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )  # FIXME - check these are equivalent
        # x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x, training=training)
        x = self.gn4(x, training=training)
        x = tf.nn.relu(x)

        cur_fpn = self.adapter3(fpns[2], training=training)
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = tf.image.resize(
            x, size=cur_fpn.shape[-2:], mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )  # FIXME - check these are equivalent
        x = self.lay5(x, training=training)
        x = self.gn5(x, training=training)
        x = tf.nn.relu(x)

        x = self.out_lay(x, training=training)
        return x


class TFDetrMHAttentionMap(tf.keras.layers.Layer):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        std: float = None,
        **kwargs
    ) -> None:  # FIXME - types
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")

        self.q_linear = tf.keras.layers.Dense(hidden_dim, use_bias=bias, name="q_linear")
        self.k_linear = tf.keras.layers.Dense(hidden_dim, use_bias=bias, name="k_linear")

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def call(self, q, k, mask: Optional[tf.Tensor] = None, training: bool = False):
        q = self.q_linear(q)
        filters = self.k_linear.weight
        filters = tf.expand_dims(filters, -1)
        filters = tf.expand_dims(filters, -1)
        k = tf.nn.conv2d(k, filters, self.k_linear.bias)

        queries_per_head = tf.reshape(q, (q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads))
        keys_per_head = tf.reshape(
            k, (k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        )
        weights = tf.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        if mask is not None:
            weights = tf.where(mask, weights, float("inf"))  # FIXME

        weights = tf.reshape(weights, (*weights.shape[:2], -1))
        weights = tf.reshape(tf.nn.softmax(weights, axis=-1), (weights.shape))
        weights = self.dropout(weights, training=training)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = tf.nn.sigmoid(inputs)
    inputs = tf.reshape(inputs, (inputs.shape[0], -1))
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    # ce_loss= tf.keras.layers.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")  # FIXME
    # ce_loss= tf.keras.losses.BinaryCrossEntropy(from_logits=True, reduction="none")(inputs, targets)
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(targets, inputs)  # FIXME - reduction?
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class TFDetrLoss(tf.keras.layers.Layer):
    """
    This class computes the losses for TFDetrForObjectDetection/DetrForSegmentation. The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each
    pair of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes: int, eos_coef: float, losses: List[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = tf.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # self.register_buffer("empty_weight", empty_weight)  # FIXME - add in build()

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        src_logits = outputs["logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = tf.concat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = tf.ones(src_logits.shape[:2], dtype=tf.int64) * self.num_classes
        target_classes[idx] = target_classes_o

        # loss_ce= tf.keras.layers.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)  # FIXME
        loss_ce = tf.nn.weighted_cross_entropy_with_logits(
            target_classes, tf.transpose(src_logits, (0, 2, 1, 3)), self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        return losses

    # @torch.no_grad()  # FIXME
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        tgt_lengths = tf.constant([len(v["class_labels"]) for v in targets])
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # card_err = tf.keras.layers.functional.l1_loss(card_pred.float(), tgt_lengths.float())  # FIXME
        card_err = tf.keras.lossess.MeanAbsoluteError()(tgt_lengths.float(), card_pred.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = tf.concat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

        # loss_bbox = tf.keras.layers.functional.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = tf.keras.losses.MeanAbsoluteError(target_boxes, src_boxes, reduction="none")  # FIXME

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - tf.linalg.diag(
            generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = tf.cast(target_masks, src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = tf.image.resize(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            method=tf.image.ResizeMethod.BILINEAR,  # align_corners=False FIXME
        )
        src_masks = tf.reshape(src_masks[:, 0], (src_masks[:, 0].shape[0], -1))  # FIXME - get shape in a better way

        target_masks = tf.reshape(target_masks, (target_masks.shape[0], -1))
        target_masks = tf.reshape(target_masks, (src_masks.shape))
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = tf.concat([tf.ones_like(src) * i for i, (src, _) in enumerate(indices)])
        src_idx = tf.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = tf.concat([tf.ones_like(tgt) * i for i, (_, tgt) in enumerate(indices)])
        tgt_idx = tf.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def call(self, outputs: Optional[Dict], targets: List[Dict], training: bool = False):  # FIXME - types
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that len(targets) == batch_size. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, training=training)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = tf.constant([num_boxes], dtype=tf.float32)  # FIXME - check precision torch.float versus tf.float32
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = tf.clip_by_value(num_boxes, clip_min_value=1)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets, training=training)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class TFDetrMLPPredictionHead(tf.keras.layers.Layer):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.mlp_layers = [tf.keras.layers.Dense(k, name=f"layers.{i}") for i, k in enumerate(h + [output_dim])]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for i, layer in enumerate(self.mlp_layers):
            x = tf.nn.relu(layer(x, training=training)) if i < self.num_layers - 1 else layer(x)
        return x


# taken from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
class TFDetrHungarianMatcher(tf.keras.layers.Layer):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 or bbox_cost == 0 or giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    # @torch.no_grad()  # FIXME
    def call(self, outputs: tf.Tensor, targets: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": tf.Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": tf.Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": tf.Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": tf.Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"]
        out_prob = tf.reshape(out_prob, (out_prob.shape[0], -1))
        out_prob = tf.nn.softmax(out_prob, -1)  # [batch_size * num_queries, num_classes]
        out_bbox = tf.reshape(
            outputs["pred_boxes"], (outputs["pred_boxes"].shape[0], -1)
        )  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = tf.concat([v["class_labels"] for v in targets])
        tgt_bbox = tf.concat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -out_prob[:, tgt_ids]

        #  FIXME - write a proper alternative cdist function without list comp
        def l1_dist(a, b):
            return tf.stack([tf.norm(row - b, ord=1, axis=1) for row in a])

        # Compute the L1 cost between boxes
        bbox_cost = l1_dist(out_bbox, tgt_bbox, p=1)  # Check if tf metrics has an equivalent

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = tf.reshape(cost_matrix, (batch_size, num_queries, -1))

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(tf.constant(i, dtype=tf.int64), tf.constant(j, dtype=tf.int64)) for i, j in indices]


# below: bounding box utilities taken from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py


def _upcast(t: tf.Tensor) -> tf.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (tf.float32, tf.float64) else t.float()
    else:
        return t if t.dtype in (tf.int32, tf.int64) else t.int()


def box_area(boxes: tf.Tensor) -> tf.Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`tf.Tensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `tf.Tensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `tf.Tensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = tf.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = tf.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


# below: taken from https://github.com/facebookresearch/detr/blob/master/util/misc.py#L306


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    def __init__(self, tensors, mask: Optional[tf.Tensor]) -> None:
        self.tensors = tensors
        self.mask = mask

    def decompose(self) -> Tuple[Any, Optional[tf.Tensor]]:  # FIXME - type
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[tf.Tensor]) -> NestedTensor:
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        tensor = tf.zeros(batch_shape, dtype=dtype)
        mask = tf.ones((b, h, w), dtype=tf.bool)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)
