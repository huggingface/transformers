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
"""PyTorch DETR model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_accelerate_available,
    is_scipy_available,
    is_timm_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from ...utils.backbone_utils import load_backbone
from .configuration_detr import DetrConfig


if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


if is_timm_available():
    from timm import create_model


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DetrConfig"
_CHECKPOINT_FOR_DOC = "facebook/detr-resnet-50"


@dataclass
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
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
class DetrModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
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
class DetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`DetrForObjectDetection`].

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
class DetrSegmentationOutput(ModelOutput):
    """
    Output type of [`DetrForSegmentation`].

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
# https://github.com/facebookresearch/detr/blob/master/backbone.py
class DetrFrozenBatchNorm2d(nn.Module):
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


def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `DetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = DetrFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DetrConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by DetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # For backwards compatibility we have to use the timm library directly instead of the AutoBackbone API
        if config.use_timm_backbone:
            # We default to values which were previously hard-coded. This enables configurability from the config
            # using backbone arguments, while keeping the default behavior the same.
            requires_backends(self, ["timm"])
            kwargs = getattr(config, "backbone_kwargs", {})
            kwargs = {} if kwargs is None else kwargs.copy()
            out_indices = kwargs.pop("out_indices", (1, 2, 3, 4))
            num_channels = kwargs.pop("in_chans", config.num_channels)
            if config.dilation:
                kwargs["output_stride"] = kwargs.get("output_stride", 16)
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=out_indices,
                in_chans=num_channels,
                **kwargs,
            )
        else:
            backbone = load_backbone(config)

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = None
        if config.backbone is not None:
            backbone_model_type = config.backbone
        elif config.backbone_config is not None:
            backbone_model_type = config.backbone_config.model_type
        else:
            raise ValueError("Either `backbone` or `backbone_config` should be provided in the config")

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


class DetrConvModel(nn.Module):
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


class DetrSinePositionEmbedding(nn.Module):
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

        dim_t = torch.arange(self.embedding_dim, dtype=torch.int64, device=pixel_values.device).float()
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DetrLearnedPositionEmbedding(nn.Module):
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


def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = DetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = DetrLearnedPositionEmbedding(n_steps)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


class DetrAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
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

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor]):
        return tensor if object_queries is None else tensor + object_queries

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
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


class DetrEncoderLayer(nn.Module):
    def __init__(self, config: DetrConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DetrAttention(
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


class DetrDecoderLayer(nn.Module):
    def __init__(self, config: DetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = DetrAttention(
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


class DetrPreTrainedModel(PreTrainedModel):
    config_class = DetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DetrConvEncoder", r"DetrEncoderLayer", r"DetrDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        if isinstance(module, DetrMHAttentionMap):
            nn.init.zeros_(module.k_linear.bias)
            nn.init.zeros_(module.q_linear.bias)
            nn.init.xavier_uniform_(module.k_linear.weight, gain=xavier_std)
            nn.init.xavier_uniform_(module.q_linear.weight, gain=xavier_std)
        elif isinstance(module, DetrLearnedPositionEmbedding):
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


class DetrEncoder(DetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`DetrEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for DETR:

    - object_queries are added to the forward pass.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.layers = nn.ModuleList([DetrEncoderLayer(config) for _ in range(config.encoder_layers)])

        # in the original DETR, no layernorm is used at the end of the encoder, as "normalize_before" is set to False by default

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        object_queries=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:

                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).

                [What are attention masks?](../glossary#attention-mask)

            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Object queries that are added to the queries in each self-attention layer.

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
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                # we add object_queries as extra input to the encoder_layer
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    object_queries=object_queries,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class DetrDecoder(DetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # in DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        object_queries=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The query embeddings that are passed into the decoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on certain queries. Mask values selected in `[0, 1]`:

                - 1 for queries that are **not masked**,
                - 0 for queries that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            object_queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Object queries that are added to the queries and keys in each cross-attention layer.
            query_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                , *optional*): Position embeddings that are added to the values and keys in each self-attention layer.

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
            input_shape = inputs_embeds.size()[:-1]

        combined_attention_mask = None

        if attention_mask is not None and combined_attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            combined_attention_mask = combined_attention_mask + _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # optional intermediate hidden states
        intermediate = () if self.config.auxiliary_loss else None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    object_queries=object_queries,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # stack intermediate decoder activations
        if self.config.auxiliary_loss:
            intermediate = torch.stack(intermediate)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
                if v is not None
            )
        return DetrDecoderOutput(
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
class DetrModel(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        backbone = DetrConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = DetrConvModel(backbone, object_queries)

        # Create projection layer
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = DetrEncoder(config)
        self.decoder = DetrDecoder(config)

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

    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrModelOutput, config_class=_CONFIG_FOR_DOC)
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
    ) -> Union[Tuple[torch.FloatTensor], DetrModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        >>> model = DetrModel.from_pretrained("facebook/detr-resnet-50")

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

        return DetrModelOutput(
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
class DetrForObjectDetection(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig):
        super().__init__(config)

        # DETR encoder-decoder model
        self.model = DetrModel(config)

        # Object detection heads
        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels + 1
        )  # We add one for the "no object" class
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DetrObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DetrForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        >>> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.998 at location [40.16, 70.81, 175.55, 117.98]
        Detected remote with confidence 0.996 at location [333.24, 72.55, 368.33, 187.66]
        Detected couch with confidence 0.995 at location [-0.02, 1.15, 639.73, 473.76]
        Detected cat with confidence 0.999 at location [13.24, 52.05, 314.02, 470.93]
        Detected cat with confidence 0.999 at location [345.4, 23.85, 640.37, 368.72]
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
        )

        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = DetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = DetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            criterion.to(self.device)
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

        return DetrObjectDetectionOutput(
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
class DetrForSegmentation(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig):
        super().__init__(config)

        # object detection model
        self.detr = DetrForObjectDetection(config)

        # segmentation head
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        intermediate_channel_sizes = self.detr.model.backbone.conv_encoder.intermediate_channel_sizes

        self.mask_head = DetrMaskHeadSmallConv(
            hidden_size + number_of_heads, intermediate_channel_sizes[::-1][-3:], hidden_size
        )

        self.bbox_attention = DetrMHAttentionMap(
            hidden_size, hidden_size, number_of_heads, dropout=0.0, std=config.init_xavier_std
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DetrSegmentationOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss, DICE/F-1 loss and Focal loss. List of dicts, each
            dictionary containing at least the following 3 keys: 'class_labels', 'boxes' and 'masks' (the class labels,
            bounding boxes and segmentation masks of an image in the batch respectively). The class labels themselves
            should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)`, the boxes a
            `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)` and the masks a
            `torch.FloatTensor` of shape `(number of bounding boxes in the image, height, width)`.

        Returns:

        Examples:

        ```python
        >>> import io
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> import numpy

        >>> from transformers import AutoImageProcessor, DetrForSegmentation
        >>> from transformers.image_transforms import rgb_to_id

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
        >>> model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # Use the `post_process_panoptic_segmentation` method of the `image_processor` to retrieve post-processed panoptic segmentation maps
        >>> # Segmentation results are returned as a list of dictionaries
        >>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(300, 500)])

        >>> # A tensor of shape (height, width) where each value denotes a segment id, filled with -1 if no segment is found
        >>> panoptic_seg = result[0]["segmentation"]
        >>> # Get prediction score and segment_id to class_id mapping of each segment
        >>> panoptic_segments_info = result[0]["segments_info"]
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        # First, get list of feature maps and position embeddings
        features, object_queries_list = self.detr.model.backbone(pixel_values, pixel_mask=pixel_mask)

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_map, mask = features[-1]
        batch_size, num_channels, height, width = feature_map.shape
        projected_feature_map = self.detr.model.input_projection(feature_map)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.detr.model.encoder(
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

        # Fifth, sent query embeddings + position embeddings through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.detr.model.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        queries = torch.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.detr.model.decoder(
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

        sequence_output = decoder_outputs[0]

        # Sixth, compute logits, pred_boxes and pred_masks
        logits = self.detr.class_labels_classifier(sequence_output)
        pred_boxes = self.detr.bbox_predictor(sequence_output).sigmoid()

        memory = encoder_outputs[0].permute(0, 2, 1).view(batch_size, self.config.d_model, height, width)
        mask = flattened_mask.view(batch_size, height, width)

        # FIXME h_boxes takes the last one computed, keep this in mind
        # important: we need to reverse the mask, since in the original implementation the mask works reversed
        # bbox_mask is of shape (batch_size, num_queries, number_of_attention_heads in bbox_attention, height/32, width/32)
        bbox_mask = self.bbox_attention(sequence_output, memory, mask=~mask)

        seg_masks = self.mask_head(projected_feature_map, bbox_mask, [features[2][0], features[1][0], features[0][0]])

        pred_masks = seg_masks.view(batch_size, self.detr.config.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = DetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality", "masks"]
            criterion = DetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            outputs_loss["pred_masks"] = pred_masks
            if self.config.auxiliary_loss:
                intermediate = decoder_outputs.intermediate_hidden_states if return_dict else decoder_outputs[-1]
                outputs_class = self.detr.class_labels_classifier(intermediate)
                outputs_coord = self.detr.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self.detr._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            loss_dict = criterion(outputs_loss, labels)
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

        return DetrSegmentationOutput(
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
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


# taken from https://github.com/facebookresearch/detr/blob/master/models/segmentation.py
class DetrMaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        if dim % 8 != 0:
            raise ValueError(
                "The hidden_size + number of attention heads must be divisible by 8 as the number of groups in"
                " GroupNorm is set to 8"
            )

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, inter_dims[1]), inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(min(8, inter_dims[2]), inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(min(8, inter_dims[3]), inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        # here we concatenate x, the projected feature map, of shape (batch_size, d_model, heigth/32, width/32) with
        # the bbox_mask = the attention maps of shape (batch_size, n_queries, n_heads, height/32, width/32).
        # We expand the projected feature map to match the number of heads.
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = nn.functional.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = nn.functional.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = nn.functional.relu(x)

        x = self.out_lay(x)
        return x


class DetrMHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min)
        weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
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
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class DetrLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

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

    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
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
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
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

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

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

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class DetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# taken from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
class DetrHungarianMatcher(nn.Module):
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

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# below: bounding box utilities taken from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# below: taken from https://github.com/facebookresearch/detr/blob/master/util/misc.py#L306
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)
