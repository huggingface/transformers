# coding=utf-8
# Copyright 2023 Facebook AI Research The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch RT_DETR model."""

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_timm_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from ..auto import AutoBackbone
from .configuration_rt_detr import RT_DETRConfig


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

if is_timm_available():
    from timm import create_model

if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RT_DETRConfig"
_CHECKPOINT_FOR_DOC = "checkpoing/todo"

RT_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "checkpoing/todo",
    # See all RT_DETR models at https://huggingface.co/models?filter=rt_detr
]



@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrDecoderOutput with DETR->RT_DETR,Detr->RT_DETR
class RT_DETRDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the RT_DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrModelOutput with DETR->RT_DETR,Detr->RT_DETR
class RT_DETRModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the RT_DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrObjectDetectionOutput with Detr->RT_DETR,DetrImageProcessor->DetrImageProcessor
class RT_DETRObjectDetectionOutput(ModelOutput):
    """
    Output type of [`RT_DETRForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~DetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrSegmentationOutput with Detr->RT_DETR,DetrImageProcessor->DetrImageProcessor
class RT_DETRSegmentationOutput(ModelOutput):
    """
    Output type of [`RT_DETRForSegmentation`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~DetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_queries, height/4, width/4)`):
            Segmentation masks logits for all queries. See also
            [`~DetrImageProcessor.post_process_semantic_segmentation`] or
            [`~DetrImageProcessor.post_process_instance_segmentation`]
            [`~DetrImageProcessor.post_process_panoptic_segmentation`] to evaluate semantic, instance and panoptic
            segmentation masks respectively.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# BELOW: utilities copied from
# https://github.com/facebookresearch/rt_detr/blob/master/backbone.py
# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->RT_DETR
class RT_DETRFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->RT_DETR
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `RT_DETRFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = RT_DETRFrozenBatchNorm2d(module.num_features)

            new_module.weight.data.copy_(module.weight)
            new_module.bias.data.copy_(module.bias)
            new_module.running_mean.data.copy_(module.running_mean)
            new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# Copied from transformers.models.detr.modeling_detr.DetrConvEncoder with Detr->RT_DETR
class RT_DETRConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by RT_DETRFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            backbone = AutoBackbone.from_config(config.backbone_config)

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->RT_DETR
class RT_DETRConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_len: Optional[int] = None):
    """
    Expands attention_mask from `[batch_size, seq_len]` to `[batch_size, 1, target_seq_len, source_seq_len]`.
    """
    batch_size, source_len = mask.size()
    target_len = target_len if target_len is not None else source_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_len, source_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


# Copied from transformers.models.detr.modeling_detr.DetrSinePositionEmbedding with Detr->RT_DETR
class RT_DETRSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding with Detr->RT_DETR
class RT_DETRLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


# Copied from transformers.models.detr.modeling_detr.build_position_encoding with Detr->RT_DETR
def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = RT_DETRSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = RT_DETRLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


# Copied from transformers.models.detr.modeling_detr.DetrAttention with DETR->RT_DETR,Detr->RT_DETR
class RT_DETRAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the RT_DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
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

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        return tensor if object_queries is None else tensor + object_queries

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        position_embeddings = kwargs.pop("position_ebmeddings", None)
        key_value_position_embeddings = kwargs.pop("key_value_position_embeddings", None)

        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if key_value_position_embeddings is not None and spatial_position_embeddings is not None:
            raise ValueError(
                "Cannot specify both key_value_position_embeddings and spatial_position_embeddings. Please use just spatial_position_embeddings"
            )

        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        if key_value_position_embeddings is not None:
            logger.warning_once(
                "key_value_position_embeddings has been deprecated and will be removed in v4.34. Please use spatial_position_embeddings instead"
            )
            spatial_position_embeddings = key_value_position_embeddings

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before projecting to queries and keys
        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, object_queries)

        # add key-value position embeddings to the key value states
        if spatial_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.detr.modeling_detr.DetrEncoderLayer with Detr->RT_DETR
class RT_DETREncoderLayer(nn.Module):
    def __init__(self, config: RT_DETRConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = RT_DETRAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.detr.modeling_detr.DetrDecoderLayer with Detr->RT_DETR
class RT_DETRDecoderLayer(nn.Module):
    def __init__(self, config: RT_DETRConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = RT_DETRAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = RT_DETRAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*):
                object_queries that are added to the hidden states
            in the cross-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                position embeddings that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            object_queries=query_position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                object_queries=query_position_embeddings,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                spatial_position_embeddings=object_queries,
                output_attentions=output_attentions,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


# Copied from transformers.models.detr.modeling_detr.DetrClassificationHead with Detr->RT_DETR
class RT_DETRClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states




RT_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RT_DETRConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RT_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DetrImageProcessor.__call__`] for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
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


# # Copied from transformers.models.detr.modeling_detr.DetrEncoder with DETR->RT_DETR,Detr->RT_DETR
# class RT_DETREncoder(RT_DETRPreTrainedModel):
#     """
#     Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
#     [`RT_DETREncoderLayer`].

#     The encoder updates the flattened feature map through multiple self-attention layers.

#     Small tweak for RT_DETR:

#     - object_queries are added to the forward pass.

#     Args:
#         config: RT_DETRConfig
#     """

#     def __init__(self, config: RT_DETRConfig):
#         super().__init__(config)

#         self.dropout = config.dropout
#         self.layerdrop = config.encoder_layerdrop

#         self.layers = nn.ModuleList([RT_DETREncoderLayer(config) for _ in range(config.encoder_layers)])

#         # in the original RT_DETR, no layernorm is used at the end of the encoder, as "normalize_before" is set to False by default

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         inputs_embeds=None,
#         attention_mask=None,
#         object_queries=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         **kwargs,
#     ):
#         r"""
#         Args:
#             inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#                 Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.

#             attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:

#                 - 1 for pixel features that are real (i.e. **not masked**),
#                 - 0 for pixel features that are padding (i.e. **masked**).

#                 [What are attention masks?](../glossary#attention-mask)

#             object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#                 Object queries that are added to the queries in each self-attention layer.

#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             output_hidden_states (`bool`, *optional*):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more detail.
#             return_dict (`bool`, *optional*):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         """
#         position_embeddings = kwargs.pop("position_embeddings", None)

#         if kwargs:
#             raise ValueError(f"Unexpected arguments {kwargs.keys()}")

#         if position_embeddings is not None and object_queries is not None:
#             raise ValueError(
#                 "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
#             )

#         if position_embeddings is not None:
#             logger.warning_once(
#                 "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
#             )
#             object_queries = position_embeddings

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         hidden_states = inputs_embeds
#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

#         # expand attention_mask
#         if attention_mask is not None:
#             # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
#             attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

#         encoder_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#         for i, encoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 encoder_states = encoder_states + (hidden_states,)
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             to_drop = False
#             if self.training:
#                 dropout_probability = torch.rand([])
#                 if dropout_probability < self.layerdrop:  # skip the layer
#                     to_drop = True

#             if to_drop:
#                 layer_outputs = (None, None)
#             else:
#                 # we add object_queries as extra input to the encoder_layer
#                 layer_outputs = encoder_layer(
#                     hidden_states,
#                     attention_mask,
#                     object_queries=object_queries,
#                     output_attentions=output_attentions,
#                 )

#                 hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             encoder_states = encoder_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
#         )


# # Copied from transformers.models.detr.modeling_detr.DetrDecoder with DETR->RT_DETR,Detr->RT_DETR
# class RT_DETRDecoder(RT_DETRPreTrainedModel):
#     """
#     Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`RT_DETRDecoderLayer`].

#     The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

#     Some small tweaks for RT_DETR:

#     - object_queries and query_position_embeddings are added to the forward pass.
#     - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

#     Args:
#         config: RT_DETRConfig
#     """

#     def __init__(self, config: RT_DETRConfig):
#         super().__init__(config)
#         self.dropout = config.dropout
#         self.layerdrop = config.decoder_layerdrop

#         self.layers = nn.ModuleList([RT_DETRDecoderLayer(config) for _ in range(config.decoder_layers)])
#         # in RT_DETR, the decoder uses layernorm after the last decoder layer output
#         self.layernorm = nn.LayerNorm(config.d_model)

#         self.gradient_checkpointing = False
#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         inputs_embeds=None,
#         attention_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         object_queries=None,
#         query_position_embeddings=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         **kwargs,
#     ):
#         r"""
#         Args:
#             inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
#                 The query embeddings that are passed into the decoder.

#             attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

#                 - 1 for queries that are **not masked**,
#                 - 0 for queries that are **masked**.

#                 [What are attention masks?](../glossary#attention-mask)
#             encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
#                 Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
#                 of the decoder.
#             encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
#                 Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
#                 in `[0, 1]`:

#                 - 1 for pixels that are real (i.e. **not masked**),
#                 - 0 for pixels that are padding (i.e. **masked**).

#             object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#                 Object queries that are added to the queries and keys in each cross-attention layer.
#             query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
#                 , *optional*): Position embeddings that are added to the values and keys in each self-attention layer.

#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             output_hidden_states (`bool`, *optional*):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more detail.
#             return_dict (`bool`, *optional*):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         """
#         position_embeddings = kwargs.pop("position_embeddings", None)

#         if kwargs:
#             raise ValueError(f"Unexpected arguments {kwargs.keys()}")

#         if position_embeddings is not None and object_queries is not None:
#             raise ValueError(
#                 "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
#             )

#         if position_embeddings is not None:
#             logger.warning_once(
#                 "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
#             )
#             object_queries = position_embeddings

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if inputs_embeds is not None:
#             hidden_states = inputs_embeds
#             input_shape = inputs_embeds.size()[:-1]

#         combined_attention_mask = None

#         if attention_mask is not None and combined_attention_mask is not None:
#             # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
#             combined_attention_mask = combined_attention_mask + _expand_mask(
#                 attention_mask, inputs_embeds.dtype, target_len=input_shape[-1]
#             )

#         # expand encoder attention mask
#         if encoder_hidden_states is not None and encoder_attention_mask is not None:
#             # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
#             encoder_attention_mask = _expand_mask(
#                 encoder_attention_mask, inputs_embeds.dtype, target_len=input_shape[-1]
#             )

#         # optional intermediate hidden states
#         intermediate = () if self.config.auxiliary_loss else None

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

#         for idx, decoder_layer in enumerate(self.layers):
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)
#             if self.training:
#                 dropout_probability = torch.rand([])
#                 if dropout_probability < self.layerdrop:
#                     continue

#             if self.gradient_checkpointing and self.training:

#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         return module(*inputs, output_attentions)

#                     return custom_forward

#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer),
#                     hidden_states,
#                     combined_attention_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     None,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=combined_attention_mask,
#                     object_queries=object_queries,
#                     query_position_embeddings=query_position_embeddings,
#                     encoder_hidden_states=encoder_hidden_states,
#                     encoder_attention_mask=encoder_attention_mask,
#                     output_attentions=output_attentions,
#                 )

#             hidden_states = layer_outputs[0]

#             if self.config.auxiliary_loss:
#                 hidden_states = self.layernorm(hidden_states)
#                 intermediate += (hidden_states,)

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#                 if encoder_hidden_states is not None:
#                     all_cross_attentions += (layer_outputs[2],)

#         # finally, apply layernorm
#         hidden_states = self.layernorm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         # stack intermediate decoder activations
#         if self.config.auxiliary_loss:
#             intermediate = torch.stack(intermediate)

#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
#                 if v is not None
#             )
#         return RT_DETRDecoderOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#             cross_attentions=all_cross_attentions,
#             intermediate_hidden_states=intermediate,
#         )



# @add_start_docstrings(
#     """
#     RT_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
#     such as COCO detection.
#     """,
#     RT_DETR_START_DOCSTRING,
# )
# # Copied from transformers.models.detr.modeling_detr.DetrForObjectDetection with DETR->RT_DETR,Detr->RT_DETR,detr->rt_detr,facebook/detr-resnet-50->checkpoing/todo
# class RT_DETRForObjectDetection(RT_DETRPreTrainedModel):
#     def __init__(self, config: RT_DETRConfig):
#         super().__init__(config)

#         # RT_DETR encoder-decoder model
#         self.model = RT_DETRModel(config)

#         # Object detection heads
#         self.class_labels_classifier = nn.Linear(
#             config.d_model, config.num_labels + 1
#         )  # We add one for the "no object" class
#         self.bbox_predictor = RT_DETRMLPPredictionHead(
#             input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
#         )

#         # Initialize weights and apply final processing
#         self.post_init()

#     # taken from https://github.com/facebookresearch/rt_detr/blob/master/models/rt_detr.py
#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_class, outputs_coord):
#         # this is a workaround to make torchscript happy, as torchscript
#         # doesn't support dictionary with non-homogeneous values, such
#         # as a dict having both a Tensor and a list.
#         return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

#     @add_start_docstrings_to_model_forward(RT_DETR_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=RT_DETRObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         pixel_values: torch.FloatTensor,
#         pixel_mask: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.FloatTensor] = None,
#         encoder_outputs: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[List[dict]] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.FloatTensor], RT_DETRObjectDetectionOutput]:
#         r"""
#         labels (`List[Dict]` of len `(batch_size,)`, *optional*):
#             Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
#             following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
#             respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
#             in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

#         Returns:

#         Examples:

#         ```python
#         >>> from transformers import AutoImageProcessor, RT_DETRForObjectDetection
#         >>> import torch
#         >>> from PIL import Image
#         >>> import requests

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> image_processor = AutoImageProcessor.from_pretrained("checkpoing/todo")
#         >>> model = RT_DETRForObjectDetection.from_pretrained("checkpoing/todo")

#         >>> inputs = image_processor(images=image, return_tensors="pt")
#         >>> outputs = model(**inputs)

#         >>> # convert outputs (bounding boxes and class logits) to COCO API
#         >>> target_sizes = torch.tensor([image.size[::-1]])
#         >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
#         ...     0
#         ... ]

#         >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         ...     box = [round(i, 2) for i in box.tolist()]
#         ...     print(
#         ...         f"Detected {model.config.id2label[label.item()]} with confidence "
#         ...         f"{round(score.item(), 3)} at location {box}"
#         ...     )
#         Detected remote with confidence 0.998 at location [40.16, 70.81, 175.55, 117.98]
#         Detected remote with confidence 0.996 at location [333.24, 72.55, 368.33, 187.66]
#         Detected couch with confidence 0.995 at location [-0.02, 1.15, 639.73, 473.76]
#         Detected cat with confidence 0.999 at location [13.24, 52.05, 314.02, 470.93]
#         Detected cat with confidence 0.999 at location [345.4, 23.85, 640.37, 368.72]
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # First, sent images through RT_DETR base model to obtain encoder + decoder outputs
#         outputs = self.model(
#             pixel_values,
#             pixel_mask=pixel_mask,
#             decoder_attention_mask=decoder_attention_mask,
#             encoder_outputs=encoder_outputs,
#             inputs_embeds=inputs_embeds,
#             decoder_inputs_embeds=decoder_inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         # class logits + predicted bounding boxes
#         logits = self.class_labels_classifier(sequence_output)
#         pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

#         loss, loss_dict, auxiliary_outputs = None, None, None
#         if labels is not None:
#             # First: create the matcher
#             matcher = RT_DETRHungarianMatcher(
#                 class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
#             )
#             # Second: create the criterion
#             losses = ["labels", "boxes", "cardinality"]
#             criterion = RT_DETRLoss(
#                 matcher=matcher,
#                 num_classes=self.config.num_labels,
#                 eos_coef=self.config.eos_coefficient,
#                 losses=losses,
#             )
#             criterion.to(self.device)
#             # Third: compute the losses, based on outputs and labels
#             outputs_loss = {}
#             outputs_loss["logits"] = logits
#             outputs_loss["pred_boxes"] = pred_boxes
#             if self.config.auxiliary_loss:
#                 intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
#                 outputs_class = self.class_labels_classifier(intermediate)
#                 outputs_coord = self.bbox_predictor(intermediate).sigmoid()
#                 auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
#                 outputs_loss["auxiliary_outputs"] = auxiliary_outputs

#             loss_dict = criterion(outputs_loss, labels)
#             # Fourth: compute total loss, as a weighted sum of the various losses
#             weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
#             weight_dict["loss_giou"] = self.config.giou_loss_coefficient
#             if self.config.auxiliary_loss:
#                 aux_weight_dict = {}
#                 for i in range(self.config.decoder_layers - 1):
#                     aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
#                 weight_dict.update(aux_weight_dict)
#             loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

#         if not return_dict:
#             if auxiliary_outputs is not None:
#                 output = (logits, pred_boxes) + auxiliary_outputs + outputs
#             else:
#                 output = (logits, pred_boxes) + outputs
#             return ((loss, loss_dict) + output) if loss is not None else output

#         return RT_DETRObjectDetectionOutput(
#             loss=loss,
#             loss_dict=loss_dict,
#             logits=logits,
#             pred_boxes=pred_boxes,
#             auxiliary_outputs=auxiliary_outputs,
#             last_hidden_state=outputs.last_hidden_state,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )


#### TODO: Rafael
from collections import OrderedDict
import torch.nn.functional as F 


def get_activation(act: str, inplace: bool=True):
    act = act.lower()
    if act == "silu":
        act_func = nn.SiLU()
    elif act == "relu":
        act_func = nn.ReLU()
    elif act == "leaky_relu":
        act_func = nn.LeakyReLU()
    elif act == "silu":
        act_func = nn.SiLU()
    elif act == "gelu":
        act_func = nn.GELU()
    elif act is None:
        act_func = nn.Identity()
    elif isinstance(act, nn.Module):
        act_func = act
    else:
        raise RuntimeError(f"Not valid activation {act}")  
    if hasattr(act_func, "inplace"):
        act_func.inplace = inplace
    return act_func 


class ConvNormLayer(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            channels_in, 
            channels_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(channels_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, channels_in, channels_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = channels_out 

        self.branch2a = ConvNormLayer(channels_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, channels_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(channels_in, channels_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(channels_in, channels_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, channels_in, channels_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(channels_in, channels_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(channels_in, channels_out, 1, stride)

        self.branch2a = ConvNormLayer(channels_in, channels_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(channels_out, channels_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, block, channels_in, channels_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    channels_in, 
                    channels_out,
                    stride=2 if i == 0 and stage_num != 2 else 1, 
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act)
            )

            if i == 0:
                channels_in = channels_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n 

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )


class PResNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        block_nums = config.block_nums
        variant = config.variant
        act_presnet = config.act_presnet
        depth = config.depth
        num_stages = config.num_stages
        return_idx = config.return_idx
        freeze_at = config.freeze_at
        freeze_norm = config.freeze_norm
        pretrained = config.pretrained
        
        channels_in = 64
        if variant in ["c", "d"]:
            conv_def = [
                [3, channels_in // 2, 3, 2, "conv1_1"],
                [channels_in // 2, channels_in // 2, 3, 1, "conv1_2"],
                [channels_in // 2, channels_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, channels_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act_presnet)) for c_in, c_out, k, s, _name in conv_def
        ]))

        channels_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in channels_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, channels_in, channels_out_list[i], block_nums[i], stage_num, act=act_presnet, variant=variant)
            )
            channels_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        # TODO Rafael: How/Where to load weights? 
        # if pretrained:
        #     state = torch.hub.load_state_dict_from_url(donwload_url[depth])
        #     self.load_state_dict(state)
        #     print(f"Load PResNet{depth} state_dict")
            
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class RepVggBlock(nn.Module):
    def __init__(self, channels_in, channels_out, act='relu'):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv1 = ConvNormLayer(channels_in, channels_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(channels_in, channels_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.channels_in, self.channels_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class HybridEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.in_channels = config.in_channels
        self.feat_strides = config.feat_strides
        self.hidden_dim = config.hidden_dim
        num_head = config.num_head
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        enc_act = config.enc_act
        self.use_encoder_idx = config.use_encoder_idx
        self.num_encoder_layers = config.num_encoder_layers
        self.pe_temperature = config.pe_temperature
        expansion = config.expansion
        depth_mult = config.depth_mult
        act_encoder = config.act_encoder
        self.eval_size = config.eval_size

        self.out_channels = [self.hidden_dim for _ in range(len(self.in_channels))]
        self.out_strides = self.feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, self.hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            self.hidden_dim, 
            num_head=num_head,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), self.num_encoder_layers) for _ in range(len(self.use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(self.hidden_dim, self.hidden_dim, 1, 1, act=act_encoder))
            self.fpn_blocks.append(
                CSPRepLayer(self.hidden_dim * 2, self.hidden_dim, round(3 * depth_mult), act=act_encoder, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(self.hidden_dim, self.hidden_dim, 3, 2, act=act_encoder)
            )
            self.pan_blocks.append(
                CSPRepLayer(self.hidden_dim * 2, self.hidden_dim, round(3 * depth_mult), act=act_encoder, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs


class RT_DETRPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RT_DETRConfig
    base_model_prefix = "rt_detr"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        if isinstance(module, RT_DETRMHAttentionMap):
            nn.init.zeros_(module.k_linear.bias)
            nn.init.zeros_(module.q_linear.bias)
            nn.init.xavier_uniform_(module.k_linear.weight, gain=xavier_std)
            nn.init.xavier_uniform_(module.q_linear.weight, gain=xavier_std)
        elif isinstance(module, RT_DETRLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RT_DETRDecoder):
            module.gradient_checkpointing = value


@add_start_docstrings(
    """
    The bare RT_DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """,
    RT_DETR_START_DOCSTRING,
)
# Copied from transformers.models.detr.modeling_detr.DetrModel with DETR->RT_DETR,Detr->RT_DETR,facebook/detr-resnet-50->checkpoing/todo
class RT_DETRModel(RT_DETRPreTrainedModel):
    def __init__(self, config: RT_DETRConfig):
        super().__init__(config)

        self.backbone = PResNet(config)
        self.encoder = HybridEncoder(config)
        # self.decoder = RTDETRTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    @add_start_docstrings_to_model_forward(RT_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RT_DETRModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], RT_DETRModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RT_DETRModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("checkpoing/todo")
        >>> model = RT_DETRModel.from_pretrained("checkpoing/todo")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the last hidden states are the final query embeddings of the Transformer decoder
        >>> # these are of shape (batch_size, num_queries, hidden_size)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 100, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features, object_queries_list = self.backbone(pixel_values, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        projected_feature_map = self.input_projection(feature_map)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, sent query embeddings + object_queries through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return RT_DETRModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
        )
