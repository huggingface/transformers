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
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from ... import initialization as init
from ...activations import ACT2FN
from ...backbone_utils import load_backbone
from ...loss.loss_utils import dice_loss, sigmoid_focal_loss
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithCrossAttentions,
    Seq2SeqModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    is_accelerate_available,
    is_scipy_available,
    is_vision_available,
    logging,
    requires_backends,
)
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import generalized_box_iou, nested_tensor_from_tensor_list
from .configuration_detr import DetrConfig


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Base class for outputs of the DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.
    """
)
@dataclass
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    r"""
    intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
        Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
        layernorm.
    """

    intermediate_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for outputs of the DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.
    """
)
@dataclass
class DetrModelOutput(Seq2SeqModelOutput):
    r"""
    intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
        Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
        layernorm.
    """

    intermediate_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Output type of [`DetrForObjectDetection`].
    """
)
@dataclass
class DetrObjectDetectionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
        Total loss as a linear combination of a negative log-likelihood (cross-entropy) for class prediction and a
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
        Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
        and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
        `pred_boxes`) for each decoder layer.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    """

    loss: torch.FloatTensor | None = None
    loss_dict: dict | None = None
    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    auxiliary_outputs: list[dict] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring(
    custom_intro="""
    Output type of [`DetrForSegmentation`].
    """
)
@dataclass
class DetrSegmentationOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
        Total loss as a linear combination of a negative log-likelihood (cross-entropy) for class prediction and a
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
    """

    loss: torch.FloatTensor | None = None
    loss_dict: dict | None = None
    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    pred_masks: torch.FloatTensor | None = None
    auxiliary_outputs: list[dict] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    decoder_hidden_states: tuple[torch.FloatTensor] | None = None
    decoder_attentions: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor] | None = None
    encoder_attentions: tuple[torch.FloatTensor] | None = None


class DetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rsqrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.weight = nn.Buffer(torch.ones(n))
        self.bias = nn.Buffer(torch.zeros(n))
        self.running_mean = nn.Buffer(torch.zeros(n))
        self.running_var = nn.Buffer(torch.ones(n))

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

            if module.weight.device != torch.device("meta"):
                new_module.weight.copy_(module.weight)
                new_module.bias.copy_(module.bias)
                new_module.running_mean.copy_(module.running_mean)
                new_module.running_var.copy_(module.running_var)

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

        backbone = load_backbone(config)
        self.intermediate_channel_sizes = backbone.channels

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)

        # We used to load with timm library directly instead of the AutoBackbone API
        # so we need to unwrap the `backbone._backbone` module to load weights without mismatch
        is_timm_model = False
        if hasattr(backbone, "_backbone"):
            backbone = backbone._backbone
            is_timm_model = True
        self.model = backbone

        backbone_model_type = config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if is_timm_model:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values)
        if isinstance(features, dict):
            features = features.feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


class DetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_position_features: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_position_features = num_position_features
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    @staticmethod
    @compile_compatible_method_lru_cache(maxsize=1)
    def build_sine_position_embedding(
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        num_position_features: int,
        normalize: bool = False,
        scale: float | None = None,
        temperature: int = 10000,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = shape
        if mask is None:
            # Without a mask this is just a cumsum over ones, written out as arange
            # instead: inductor's cumsum(ones) rewrite drops the requested dtype
            # (https://github.com/pytorch/pytorch/issues/189518), which breaks
            # float16/bfloat16 under torch.compile — don't revert to cumsum here
            # until that fix is widely released.
            y_embed = torch.arange(1, height + 1, dtype=dtype, device=device)[None, :, None].expand(
                batch_size, height, width
            )
            x_embed = torch.arange(1, width + 1, dtype=dtype, device=device)[None, None, :].expand(
                batch_size, height, width
            )
        else:
            embed_mask = mask.to(dtype)
            y_embed = embed_mask.cumsum(1)
            x_embed = embed_mask.cumsum(2)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_position_features, dtype=torch.int64, device=device).to(dtype)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_position_features)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.build_sine_position_embedding(
            shape, device, dtype, self.num_position_features, self.normalize, self.scale, self.temperature, mask
        )


class DetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    @compile_compatible_method_lru_cache(maxsize=1)
    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: torch.Tensor | None = None,
    ):
        height, width = shape[-2:]
        width_values = torch.arange(width, device=device)
        height_values = torch.arange(height, device=device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(shape[0], 1, 1, 1)
        return pos


# Copied from transformers.models.bert.modeling_bert.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DetrSelfAttention(nn.Module):
    """
    Multi-headed self-attention from 'Attention Is All You Need' paper.

    In DETR, position embeddings are added to both queries and keys (but not values) in self-attention.
    """

    def __init__(
        self,
        config: DetrConfig,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout
        self.is_causal = False

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Position embeddings are added to both queries and keys (but not values).
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_key_input = hidden_states + position_embeddings if position_embeddings is not None else hidden_states

        query_states = self.q_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DetrCrossAttention(nn.Module):
    """
    Multi-headed cross-attention from 'Attention Is All You Need' paper.

    In DETR, queries get their own position embeddings, while keys get encoder position embeddings.
    Values don't get any position embeddings.
    """

    def __init__(
        self,
        config: DetrConfig,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout
        self.is_causal = False

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        encoder_position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Position embeddings logic:
        - Queries get position_embeddings
        - Keys get encoder_position_embeddings
        - Values don't get any position embeddings
        """
        query_input_shape = hidden_states.shape[:-1]
        query_hidden_shape = (*query_input_shape, -1, self.head_dim)

        kv_input_shape = key_value_states.shape[:-1]
        kv_hidden_shape = (*kv_input_shape, -1, self.head_dim)

        query_input = hidden_states + position_embeddings if position_embeddings is not None else hidden_states
        key_input = (
            key_value_states + encoder_position_embeddings
            if encoder_position_embeddings is not None
            else key_value_states
        )

        query_states = self.q_proj(query_input).view(query_hidden_shape).transpose(1, 2)
        key_states = self.k_proj(key_input).view(kv_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(key_value_states).view(kv_hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*query_input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DetrMLP(nn.Module):
    def __init__(self, config: DetrConfig, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class DetrEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DetrConfig):
        super().__init__()
        self.hidden_size = config.d_model
        self.self_attn = DetrSelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = config.dropout
        self.mlp = DetrMLP(config, self.hidden_size, config.encoder_ffn_dim)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            spatial_position_embeddings (`torch.FloatTensor`, *optional*):
                Spatial position embeddings (2D positional encodings of image locations), to be added to both
                the queries and keys in self-attention (but not to values).
        """
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=spatial_position_embeddings,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if not torch.isfinite(hidden_states).all():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class DetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DetrConfig):
        super().__init__()
        self.hidden_size = config.d_model

        self.self_attn = DetrSelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.encoder_attn = DetrCrossAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.mlp = DetrMLP(config, self.hidden_size, config.decoder_ffn_dim)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        spatial_position_embeddings: torch.Tensor | None = None,
        object_queries_position_embeddings: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            spatial_position_embeddings (`torch.FloatTensor`, *optional*):
                Spatial position embeddings (2D positional encodings from encoder) that are added to the keys only
                in the cross-attention layer (not to values).
            object_queries_position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings for the object query slots. In self-attention, these are added to both queries
                and keys (not values). In cross-attention, these are added to queries only (not to keys or values).
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, hidden_size)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=object_queries_position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, _ = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_embeddings=object_queries_position_embeddings,
                encoder_position_embeddings=spatial_position_embeddings,
                **kwargs,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class DetrConvBlock(nn.Module):
    """Basic conv block: Conv3x3 -> GroupNorm -> Activation."""

    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation = ACT2FN[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class DetrFPNFusionStage(nn.Module):
    """Single FPN fusion stage combining low-resolution features with high-resolution FPN features."""

    def __init__(self, fpn_channels: int, current_channels: int, output_channels: int, activation: str = "relu"):
        super().__init__()
        self.fpn_adapter = nn.Conv2d(fpn_channels, current_channels, kernel_size=1)
        self.refine = DetrConvBlock(current_channels, output_channels, activation)

    def forward(self, features: torch.Tensor, fpn_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Current features to upsample, shape (B*Q, current_channels, H_in, W_in)
            fpn_features: FPN features at target resolution, shape (B*Q, fpn_channels, H_out, W_out)

        Returns:
            Fused and refined features, shape (B*Q, output_channels, H_out, W_out)
        """
        fpn_features = self.fpn_adapter(fpn_features)
        features = nn.functional.interpolate(features, size=fpn_features.shape[-2:], mode="nearest")
        return self.refine(fpn_features + features)


class DetrMaskHeadSmallConv(nn.Module):
    """
    Segmentation mask head that generates per-query masks using FPN-based progressive upsampling.

    Combines attention maps (spatial localization) with encoder features (semantics) and progressively
    upsamples through multiple scales, fusing with FPN features for high-resolution detail.
    """

    def __init__(
        self,
        input_channels: int,
        fpn_channels: list[int],
        hidden_size: int,
        activation_function: str = "relu",
    ):
        super().__init__()
        if input_channels % 8 != 0:
            raise ValueError(f"input_channels must be divisible by 8, got {input_channels}")

        self.conv1 = DetrConvBlock(input_channels, input_channels, activation_function)
        self.conv2 = DetrConvBlock(input_channels, hidden_size // 2, activation_function)

        # Progressive channel reduction: /2 -> /4 -> /8 -> /16
        self.fpn_stages = nn.ModuleList(
            [
                DetrFPNFusionStage(fpn_channels[0], hidden_size // 2, hidden_size // 4, activation_function),
                DetrFPNFusionStage(fpn_channels[1], hidden_size // 4, hidden_size // 8, activation_function),
                DetrFPNFusionStage(fpn_channels[2], hidden_size // 8, hidden_size // 16, activation_function),
            ]
        )

        self.output_conv = nn.Conv2d(hidden_size // 16, 1, kernel_size=3, padding=1)

    def forward(
        self,
        features: torch.Tensor,
        attention_masks: torch.Tensor,
        fpn_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            features: Encoder output features, shape (batch_size, hidden_size, H, W)
            attention_masks: Cross-attention maps from decoder, shape (batch_size, num_queries, num_heads, H, W)
            fpn_features: List of 3 FPN features from low to high resolution, each (batch_size, C, H, W)

        Returns:
            Predicted masks, shape (batch_size * num_queries, 1, output_H, output_W)
        """
        num_queries = attention_masks.shape[1]

        # Expand to (batch_size * num_queries) dimension
        features = features.unsqueeze(1).expand(-1, num_queries, -1, -1, -1).flatten(0, 1)
        attention_masks = attention_masks.flatten(0, 1)
        fpn_features = [
            fpn_feat.unsqueeze(1).expand(-1, num_queries, -1, -1, -1).flatten(0, 1) for fpn_feat in fpn_features
        ]

        hidden_states = torch.cat([features, attention_masks], dim=1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)

        for fpn_stage, fpn_feat in zip(self.fpn_stages, fpn_features):
            hidden_states = fpn_stage(hidden_states, fpn_feat)

        return self.output_conv(hidden_states)


class DetrMHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self, query_states: torch.Tensor, key_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        query_hidden_shape = (*query_states.shape[:-1], -1, self.head_dim)
        key_hidden_shape = (key_states.shape[0], -1, self.head_dim, *key_states.shape[-2:])

        query_states = self.q_proj(query_states).view(query_hidden_shape)
        key_states = nn.functional.conv2d(
            key_states, self.k_proj.weight.unsqueeze(-1).unsqueeze(-1), self.k_proj.bias
        ).view(key_hidden_shape)

        batch_size, num_queries, num_heads, head_dim = query_states.shape
        _, _, _, height, width = key_states.shape
        query_shape = (batch_size * num_heads, num_queries, head_dim)
        key_shape = (batch_size * num_heads, height * width, head_dim)
        attn_weights_shape = (batch_size, num_heads, num_queries, height, width)

        query = query_states.transpose(1, 2).contiguous().view(query_shape)
        key = key_states.permute(0, 1, 3, 4, 2).contiguous().view(key_shape)

        attn_weights = (
            (torch.matmul(query * self.scaling, key.transpose(1, 2))).view(attn_weights_shape).transpose(1, 2)
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights.flatten(2), dim=-1).view(attn_weights.size())
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return attn_weights


@auto_docstring
class DetrPreTrainedModel(PreTrainedModel):
    config: DetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = [r"DetrConvEncoder", r"DetrEncoderLayer", r"DetrDecoderLayer"]
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _supports_flex_attn = True  # Uses create_bidirectional_masks for attention masking
    _keys_to_ignore_on_load_unexpected = [
        r"detr\.model\.backbone\.model\.layer\d+\.0\.downsample\.1\.num_batches_tracked"
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        xavier_std = self.config.init_xavier_std

        if isinstance(module, DetrMaskHeadSmallConv):
            # DetrMaskHeadSmallConv uses kaiming initialization for all its Conv2d layers
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_uniform_(m.weight, a=1)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
        elif isinstance(module, DetrMHAttentionMap):
            init.zeros_(module.k_proj.bias)
            init.zeros_(module.q_proj.bias)
            init.xavier_uniform_(module.k_proj.weight, gain=xavier_std)
            init.xavier_uniform_(module.q_proj.weight, gain=xavier_std)
        elif isinstance(module, DetrLearnedPositionEmbedding):
            init.uniform_(module.row_embeddings.weight)
            init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, DetrLoss):
            empty_weight = torch.ones(module.num_classes + 1)
            empty_weight[-1] = module.eos_coef
            init.copy_(module.empty_weight, empty_weight)


class DetrEncoder(DetrPreTrainedModel):
    """
    Transformer encoder that processes a flattened feature map from a vision backbone, composed of a stack of
    [`DetrEncoderLayer`] modules.

    Args:
        config (`DetrConfig`): Model configuration object.
    """

    _can_record_outputs = {"hidden_states": DetrEncoderLayer, "attentions": DetrSelfAttention}

    def __init__(self, config: DetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([DetrEncoderLayer(config) for _ in range(config.encoder_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        spatial_position_embeddings=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:

                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).

                [What are attention masks?](../glossary#attention-mask)
            spatial_position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Spatial position embeddings (2D positional encodings) that are added to the queries and keys in each self-attention layer.
        """
        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        for encoder_layer in self.layers:
            # we add spatial_position_embeddings as extra input to the encoder_layer
            hidden_states = encoder_layer(
                hidden_states, attention_mask, spatial_position_embeddings=spatial_position_embeddings, **kwargs
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


class DetrDecoder(DetrPreTrainedModel):
    """
    Transformer decoder that refines a set of object queries. It is composed of a stack of [`DetrDecoderLayer`] modules,
    which apply self-attention to the queries and cross-attention to the encoder's outputs.

    Args:
        config (`DetrConfig`): Model configuration object.
    """

    _can_record_outputs = {
        "hidden_states": DetrDecoderLayer,
        "attentions": DetrSelfAttention,
        "cross_attentions": DetrCrossAttention,
    }

    def __init__(self, config: DetrConfig):
        super().__init__(config)
        self.dropout = config.dropout

        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        # in DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = nn.LayerNorm(config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        spatial_position_embeddings=None,
        object_queries_position_embeddings=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DetrDecoderOutput:
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

            spatial_position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Spatial position embeddings (2D positional encodings from encoder) that are added to the keys in each cross-attention layer.
            object_queries_position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings for the object query slots that are added to the queries and keys in each self-attention layer.
        """

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        if attention_mask is not None:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )

        # expand encoder attention mask (for cross-attention on encoder outputs)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        # optional intermediate hidden states
        intermediate = () if self.config.auxiliary_loss else None

        # decoder layers

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                spatial_position_embeddings,
                object_queries_position_embeddings,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )

            if self.config.auxiliary_loss:
                hidden_states = self.layernorm(hidden_states)
                intermediate += (hidden_states,)

        # finally, apply layernorm
        hidden_states = self.layernorm(hidden_states)

        # stack intermediate decoder activations
        if self.config.auxiliary_loss:
            intermediate = torch.stack(intermediate)

        return DetrDecoderOutput(last_hidden_state=hidden_states, intermediate_hidden_states=intermediate)


@auto_docstring(
    custom_intro="""
    The bare DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """
)
class DetrModel(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig):
        super().__init__(config)

        self.backbone = DetrConvEncoder(config)

        if config.position_embedding_type == "sine":
            self.position_embedding = DetrSinePositionEmbedding(config.d_model // 2, normalize=True)
        elif config.position_embedding_type == "learned":
            self.position_embedding = DetrLearnedPositionEmbedding(config.d_model // 2)
        else:
            raise ValueError(f"Not supported {config.position_embedding_type}")
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)
        self.input_projection = nn.Conv2d(self.backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        self.encoder = DetrEncoder(config)
        self.decoder = DetrDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_backbone(self):
        for _, param in self.backbone.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for _, param in self.backbone.model.named_parameters():
            param.requires_grad_(True)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        pixel_mask: torch.LongTensor | None = None,
        decoder_attention_mask: torch.FloatTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | DetrModelOutput:
        r"""
        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Mask to avoid performing attention on certain object queries in the decoder. Mask values selected in `[0, 1]`:

            - 1 for queries that are **not masked**,
            - 0 for queries that are **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image. Useful for bypassing the vision backbone.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation. Useful for tasks that require custom query initialization.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DetrModel
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

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
        if pixel_values is None and inputs_embeds is None:
            raise ValueError("You have to specify either pixel_values or inputs_embeds")

        if inputs_embeds is None:
            batch_size, num_channels, height, width = pixel_values.shape
            device = pixel_values.device

            if pixel_mask is None:
                pixel_mask = torch.ones(((batch_size, height, width)), device=device)
            vision_features = self.backbone(pixel_values, pixel_mask)
            feature_map, mask = vision_features[-1]

            # Apply 1x1 conv to map (batch_size, C, H, W) -> (batch_size, hidden_size, H, W), then flatten to (batch_size, HW, hidden_size)
            projected_feature_map = self.input_projection(feature_map)
            flattened_features = projected_feature_map.flatten(2).transpose(1, 2)
            spatial_position_embeddings = (
                self.position_embedding(shape=feature_map.shape, device=device, dtype=pixel_values.dtype, mask=mask)
                .flatten(2)
                .transpose(1, 2)
            )
            flattened_mask = mask.flatten(1)
        else:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            flattened_features = inputs_embeds
            # When using inputs_embeds, we need to infer spatial dimensions for position embeddings
            # Assume square feature map
            seq_len = inputs_embeds.shape[1]
            feat_dim = int(seq_len**0.5)
            # Create position embeddings for the inferred spatial size
            spatial_position_embeddings = (
                self.position_embedding(
                    shape=torch.Size([batch_size, self.config.d_model, feat_dim, feat_dim]),
                    device=device,
                    dtype=inputs_embeds.dtype,
                )
                .flatten(2)
                .transpose(1, 2)
            )
            # If a pixel_mask is provided with inputs_embeds, interpolate it to feat_dim, then flatten.
            if pixel_mask is not None:
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=(feat_dim, feat_dim)).to(torch.bool)[0]
                flattened_mask = mask.flatten(1)
            else:
                # If no mask provided, assume all positions are valid
                flattened_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                spatial_position_embeddings=spatial_position_embeddings,
                **kwargs,
            )

        object_queries_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        # Use decoder_inputs_embeds as queries if provided, otherwise initialize with zeros
        if decoder_inputs_embeds is not None:
            queries = decoder_inputs_embeds
        else:
            queries = torch.zeros_like(object_queries_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=decoder_attention_mask,
            spatial_position_embeddings=spatial_position_embeddings,
            object_queries_position_embeddings=object_queries_position_embeddings,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=flattened_mask,
            **kwargs,
        )

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


def _set_aux_loss(outputs_class, outputs_coord):
    return [{"logits": logits, "pred_boxes": boxes} for logits, boxes in zip(outputs_class[:-1], outputs_coord[:-1])]


def get_weighted_loss(loss_dict, weight_dict):
    return sum(loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict)


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

    def __init__(self, config: DetrConfig):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = config.class_cost
        self.bbox_cost = config.bbox_cost
        self.giou_cost = config.giou_cost
        if self.class_cost == 0 and self.bbox_cost == 0 and self.giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`list[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `list[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
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
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        max_value = torch.finfo(cost_matrix.dtype).max
        cost_matrix = torch.nan_to_num(cost_matrix, nan=max_value, posinf=max_value, neginf=max_value)
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


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
        losses (`list[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, config, losses, matcher):
        super().__init__()
        requires_backends(self, ["scipy"])
        self.matcher = matcher
        self.eos_coef = config.eos_coefficient
        self.num_classes = config.num_labels
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
             targets (`list[dict]`, *optional*):
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


class DetrForObjectDetectionCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": config.bbox_loss_coefficient,
            "loss_giou": config.giou_loss_coefficient,
        }
        self.auxiliary_loss = config.auxiliary_loss
        if self.auxiliary_loss:
            auxiliary_weights = {}
            for index in range(config.decoder_layers - 1):
                auxiliary_weights.update({f"{key}_{index}": value for key, value in self.weight_dict.items()})
            self.weight_dict.update(auxiliary_weights)
        losses = ["labels", "boxes", "cardinality"]
        matcher = DetrHungarianMatcher(config)
        self.loss_module = DetrLoss(config, losses, matcher)

    def forward(self, logits, labels, pred_boxes, outputs_class=None, outputs_coord=None):
        outputs_loss = {}
        auxiliary_outputs = None
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes
        if self.auxiliary_loss:
            auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs

        loss_dict = self.loss_module(outputs_loss, labels)

        loss = get_weighted_loss(loss_dict, self.weight_dict)
        return loss, loss_dict, auxiliary_outputs


class DetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

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


@auto_docstring(
    custom_intro="""
    DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
    such as COCO detection.
    """
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

        self.criterion = DetrForObjectDetectionCriterion(config)
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        decoder_attention_mask: torch.FloatTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | DetrObjectDetectionOutput:
        r"""
        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Mask to avoid performing attention on certain object queries in the decoder. Mask values selected in `[0, 1]`:

            - 1 for queries that are **not masked**,
            - 0 for queries that are **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image. Useful for bypassing the vision backbone.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation. Useful for tasks that require custom query initialization.
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DetrForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

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

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs,
        )

        sequence_output = outputs[0]

        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            loss, loss_dict, auxiliary_outputs = self.criterion(
                logits, labels, pred_boxes, outputs_class, outputs_coord
            )

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


class DetrForSegmentationCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        requires_backends(self, ["scipy"])

        losses = ["labels", "boxes", "cardinality", "masks"]
        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": config.bbox_loss_coefficient,
            "loss_giou": config.giou_loss_coefficient,
            "loss_mask": config.mask_loss_coefficient,
            "loss_dice": config.dice_loss_coefficient,
        }
        self.auxiliary_loss = config.auxiliary_loss
        if self.auxiliary_loss:
            auxiliary_weights = {}
            for index in range(config.decoder_layers - 1):
                auxiliary_weights.update({f"{key}_{index}": value for key, value in self.weight_dict.items()})
            self.weight_dict.update(auxiliary_weights)
        matcher = DetrHungarianMatcher(config)
        self.loss_module = DetrLoss(config, losses, matcher)

    def forward(self, logits, labels, pred_boxes, pred_masks, outputs_class=None, outputs_coord=None):
        outputs_loss = {}
        auxiliary_outputs = None
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes
        outputs_loss["pred_masks"] = pred_masks
        if self.auxiliary_loss:
            auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
            outputs_loss["auxiliary_outputs"] = auxiliary_outputs

        loss_dict = self.loss_module(outputs_loss, labels)

        loss = get_weighted_loss(loss_dict, self.weight_dict)
        return loss, loss_dict, auxiliary_outputs


@auto_docstring(
    custom_intro="""
    DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top, for tasks
    such as COCO panoptic.
    """
)
class DetrForSegmentation(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig):
        super().__init__(config)

        # object detection model
        self.detr = DetrForObjectDetection(config)

        # segmentation head
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        intermediate_channel_sizes = self.detr.model.backbone.intermediate_channel_sizes

        self.mask_head = DetrMaskHeadSmallConv(
            input_channels=hidden_size + number_of_heads,
            fpn_channels=intermediate_channel_sizes[::-1][-3:],
            hidden_size=hidden_size,
            activation_function=config.activation_function,
        )

        self.bbox_attention = DetrMHAttentionMap(hidden_size, number_of_heads, dropout=0.0)

        self.criterion = DetrForSegmentationCriterion(config)
        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        decoder_attention_mask: torch.FloatTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | DetrSegmentationOutput:
        r"""
        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Mask to avoid performing attention on certain object queries in the decoder. Mask values selected in `[0, 1]`:

            - 1 for queries that are **not masked**,
            - 0 for queries that are **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Kept for backward compatibility, but cannot be used for segmentation, as segmentation requires
            multi-scale features from the backbone that are not available when bypassing it with inputs_embeds.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation. Useful for tasks that require custom query initialization.
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss, DICE/F-1 loss and Focal loss. List of dicts, each
            dictionary containing at least the following 3 keys: 'class_labels', 'boxes' and 'masks' (the class labels,
            bounding boxes and segmentation masks of an image in the batch respectively). The class labels themselves
            should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)`, the boxes a
            `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)` and the masks a
            `torch.FloatTensor` of shape `(number of bounding boxes in the image, height, width)`.

        Examples:

        ```python
        >>> import io
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
        >>> from PIL import Image
        >>> import torch
        >>> import numpy

        >>> from transformers import AutoImageProcessor, DetrForSegmentation
        >>> from transformers.image_transforms import rgb_to_id

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

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
        >>> panoptic_seg.shape
        torch.Size([300, 500])
        >>> # Get prediction score and segment_id to class_id mapping of each segment
        >>> panoptic_segments_info = result[0]["segments_info"]
        >>> len(panoptic_segments_info)
        5
        ```"""

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        vision_features = self.detr.model.backbone(pixel_values, pixel_mask)
        feature_map, mask = vision_features[-1]

        # Apply 1x1 conv to map (batch_size, C, H, W) -> (batch_size, hidden_size, H, W), then flatten to (batch_size, HW, hidden_size)
        projected_feature_map = self.detr.model.input_projection(feature_map)
        flattened_features = projected_feature_map.flatten(2).transpose(1, 2)
        spatial_position_embeddings = (
            self.detr.model.position_embedding(
                shape=feature_map.shape, device=device, dtype=pixel_values.dtype, mask=mask
            )
            .flatten(2)
            .transpose(1, 2)
        )
        flattened_mask = mask.flatten(1)

        if encoder_outputs is None:
            encoder_outputs = self.detr.model.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                spatial_position_embeddings=spatial_position_embeddings,
                **kwargs,
            )

        object_queries_position_embeddings = self.detr.model.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        # Use decoder_inputs_embeds as queries if provided, otherwise initialize with zeros
        if decoder_inputs_embeds is not None:
            queries = decoder_inputs_embeds
        else:
            queries = torch.zeros_like(object_queries_position_embeddings)

        decoder_outputs = self.detr.model.decoder(
            inputs_embeds=queries,
            attention_mask=decoder_attention_mask,
            spatial_position_embeddings=spatial_position_embeddings,
            object_queries_position_embeddings=object_queries_position_embeddings,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=flattened_mask,
            **kwargs,
        )

        sequence_output = decoder_outputs[0]

        logits = self.detr.class_labels_classifier(sequence_output)
        pred_boxes = self.detr.bbox_predictor(sequence_output).sigmoid()

        height, width = feature_map.shape[-2:]
        memory = encoder_outputs.last_hidden_state.transpose(1, 2).view(batch_size, self.config.d_model, height, width)
        attention_mask = flattened_mask.view(batch_size, height, width)

        if attention_mask is not None:
            min_dtype = torch.finfo(memory.dtype).min
            attention_mask = torch.where(
                attention_mask.unsqueeze(1).unsqueeze(1),
                torch.full((), 0.0, device=memory.device, dtype=memory.dtype),
                min_dtype,
            )

        bbox_mask = self.bbox_attention(sequence_output, memory, attention_mask=attention_mask)

        seg_masks = self.mask_head(
            features=projected_feature_map,
            attention_masks=bbox_mask,
            fpn_features=[vision_features[2][0], vision_features[1][0], vision_features[0][0]],
        )

        pred_masks = seg_masks.view(batch_size, self.detr.config.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                intermediate = decoder_outputs.intermediate_hidden_states
                outputs_class = self.detr.class_labels_classifier(intermediate)
                outputs_coord = self.detr.bbox_predictor(intermediate).sigmoid()
            loss, loss_dict, auxiliary_outputs = self.criterion(
                logits, labels, pred_boxes, pred_masks, outputs_class, outputs_coord
            )

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


__all__ = [
    "DetrForObjectDetection",
    "DetrForSegmentation",
    "DetrModel",
    "DetrPreTrainedModel",
]
