# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
""" PyTorch RT-DETR model."""
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers import AutoBackbone

from ...activations import ACT2CLS
from ...image_transforms import center_to_corners_format, corners_to_center_format
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_rt_detr import RTDetrConfig


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RTDetrConfig"
# TODO: Replace all occurrences of the checkpoint with the final one
_CHECKPOINT_FOR_DOC = "rafaelpadilla/porting_rt_detr"

RTDETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "rafaelpadilla/porting_rt_detr",
    # See all RTDETR models at https://huggingface.co/models?filter=rt_detr
]


@dataclass
class RTDetrModelOutput(ModelOutput):
    """
    Output type of [`RTDetrModel`].

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
            possible padding). You can use [`~RTDetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized (absolute) bounding boxes.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor`, being one for the output of the embeddings (logits) + one for the boxes + one
            containing the outputs of each layer.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.detr.modeling_detr.NestedTensor
class NestedTensor(object):
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


# Copied from transformers.models.detr.modeling_detr._max_by_axis
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# Copied from transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
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


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
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


# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
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


# Copied from transformers.models.detr.modeling_detr.box_iou
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


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising_queries=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    """
    Creates a contrastive denoising training group using ground-truth samples. It adds noise to labels and boxes.

    Args:
        targets (`List[dict]`): The target objects, each containing 'labels' and 'boxes' for objects in an image.
        num_classes (`int`): Total number of classes in the dataset.
        num_queries (`int`): Number of query slots in the transformer.
        class_embed (`callable`): A function or a model layer to embed class labels.
        num_denoising_queries (`int`, *optional*, defaults to 100): Number of denoising queries.
        label_noise_ratio (`float`, *optional*, defaults to 0.5): Ratio of noise applied to labels.
        box_noise_scale (`float`, *optional*, defaults to 1.0): Scale of noise applied to bounding boxes.
    Returns:
        A tuple containing: input_query_class (`torch.FloatTensor`): Class queries with applied label noise.
        input_query_bbox
            (`torch.FloatTensor`): Bounding box queries with applied box noise. attn_mask (`torch.FloatTensor`):
            Attention mask for separating denoising and reconstruction queries. dn_meta (`dict`): Metadata including
            denoising positive indices, number of groups, and split sizes.
    """

    if num_denoising_queries <= 0:
        return None, None, None, None

    num_ground_truths = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_ground_truths)
    if max_gt_num == 0:
        return None, None, None, None

    num_groups_denoising_queries = num_denoising_queries // max_gt_num
    num_groups_denoising_queries = 1 if num_groups_denoising_queries == 0 else num_groups_denoising_queries
    # pad gt to max_num of a batch
    batch_size = len(num_ground_truths)

    input_query_class = torch.full([batch_size, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([batch_size, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([batch_size, max_gt_num], dtype=torch.bool, device=device)

    for i in range(batch_size):
        num_gt = num_ground_truths[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_groups_denoising_queries])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_groups_denoising_queries, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_groups_denoising_queries])
    # positive and negative mask
    negative_gt_mask = torch.zeros([batch_size, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_groups_denoising_queries, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    denoise_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    denoise_positive_idx = torch.split(
        denoise_positive_idx, [n * num_groups_denoising_queries for n in num_ground_truths]
    )
    # total denoising queries
    num_denoising_queries = int(max_gt_num * 2 * num_groups_denoising_queries)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = center_to_corners_format(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = corners_to_center_format(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    input_query_class = class_embed(input_query_class)

    target_size = num_denoising_queries + num_queries
    attn_mask = torch.full([target_size, target_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising_queries:, :num_denoising_queries] = True

    # reconstructions cannot see each other
    for i in range(num_groups_denoising_queries):
        idx_block_start = max_gt_num * 2 * i
        idx_block_end = max_gt_num * 2 * (i + 1)
        if i == 0:
            attn_mask[idx_block_start:idx_block_end, idx_block_end:num_denoising_queries] = True
        if i == num_groups_denoising_queries - 1:
            attn_mask[idx_block_start:idx_block_end, :idx_block_start] = True
        else:
            attn_mask[idx_block_start:idx_block_end, idx_block_end:num_denoising_queries] = True
            attn_mask[idx_block_start:idx_block_end, :idx_block_start] = True

    dn_meta = {
        "dn_positive_idx": denoise_positive_idx,
        "dn_num_group": num_groups_denoising_queries,
        "dn_num_split": [num_denoising_queries, num_queries],
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta


class RTDetrConvNormLayer(nn.Module):
    def __init__(self, config, channels_in, channels_out, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels_out, config.batch_norm_eps)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


def bias_init_with_prob(prior_prob=0.01):
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Implements the core functionality of deformable attention mechanism.

    This function applies deformable attention to the provided `value` tensor using the specified `sampling_locations`
    and `attention_weights`. It handles multiple levels of features, each with a different spatial shape, and combines
    these features using the deformable attention mechanism.

    Args:
        value (`torch.FloatTensor`): The value tensor with the features on which attention is to be applied.
        value_spatial_shapes (`List[Tuple[int, int]]`): A list of tuples where each tuple represents the spatial shape
                                               (height, width) of the feature map at each level.
        sampling_locations (`torch.FloatTensor`): The sampling locations for applying attention.
        attention_weights (`torch.FloatTensor`):
            The attention weights with shape (batch_size, len_q, num_head, n_levels, n_points).

    Returns:
        The output tensor after applying deformable attention, with shape (batch_size, len_q, num_head * head_dim).
    """

    batch_size, _, num_head, head_dim = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        new_value_list = (
            value_list[level].flatten(2).permute(0, 2, 1).reshape(batch_size * num_head, head_dim, height, width)
        )
        new_sampling_grid = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        new_sampling_value = F.grid_sample(
            new_value_list, new_sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(new_sampling_value)
    # (batch_size, len_q, num_head, n_levels, n_points) -> (batch_size, num_head, len_q, n_levels, n_points)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4)
    # (batch_size, num_head, len_q, n_levels, n_points) -> (batch_size*num_head, 1, len_q, n_levels*n_points)
    attention_weights = attention_weights.reshape(batch_size * num_head, 1, len_q, n_levels * n_points)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(batch_size, num_head * head_dim, len_q)
    )

    return output.permute(0, 2, 1)


class RTDetrTransformerEncoderLayer(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.normalize_before = config.normalize_before

        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, config.dropout, batch_first=True
        )

        self.linear1 = nn.Linear(config.hidden_dim, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_dim)

        self.norm1 = nn.LayerNorm(config.hidden_dim, config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        query = key = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(query, key, value=src, attn_mask=src_mask)

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


class RTDetrTransformerEncoder(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()

        self.layers = nn.ModuleList([RTDetrTransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)])

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        return output


class RTDetrRepVggBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        in_channels = int(config.hidden_dim)
        out_channels = int(config.hidden_dim)
        activation = config.act_encoder
        self.conv1 = RTDetrConvNormLayer(config, in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = RTDetrConvNormLayer(config, in_channels, out_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


class RTDetrCSPRepLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: RTDetrConfig):
        super().__init__()

        in_channels = config.hidden_dim * 2
        out_channels = config.hidden_dim
        num_blocks = 3
        activation = config.act_encoder

        hidden_channels = int(out_channels)
        self.conv1 = RTDetrConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = RTDetrConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[RTDetrRepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = RTDetrConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class RTDetrMSDeformableAttention(nn.Module):
    def __init__(self, config: RTDetrConfig):
        """
        Multi-Scale Deformable Attention Module
        """
        super().__init__()

        self.embed_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.num_levels = config.num_levels
        self.num_points = config.num_decoder_points
        self.total_points = self.num_heads * self.num_levels * self.num_points
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("Relation self.head_dim * num_heads == self.embed_dim does not apply")
        self.sampling_offsets = nn.Linear(self.embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(self.embed_dim, self.total_points)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        batch_size, query_length = query.shape[:2]
        len_v = value.shape[1]
        value = self.value_proj(value)

        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(batch_size, len_v, self.num_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).reshape(
            batch_size, query_length, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            batch_size, query_length, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            batch_size, query_length, self.num_heads, self.num_levels, self.num_points
        )
        # reference_points is a point type to sample feature
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(batch_size, query_length, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        # reference_points is a box type to sample feature
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but got {} instead.".format(reference_points.shape[-1])
            )

        output = deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


class RTDetrTransformerDecoderLayer(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim, config.layer_norm_eps)
        self.cross_attn = RTDetrMSDeformableAttention(config)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.hidden_dim, config.layer_norm_eps)
        # ffn
        self.linear1 = nn.Linear(config.hidden_dim, config.dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_dim)
        self.dropout4 = nn.Dropout(config.dropout)
        self.norm3 = nn.LayerNorm(config.hidden_dim, config.layer_norm_eps)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target,
        reference_points,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        attn_mask=None,
        memory_mask=None,
        query_pos_embed=None,
    ):
        # self attention
        query = keys = self.with_pos_embed(target, query_pos_embed)

        attention_res, _ = self.self_attn(query, keys, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(attention_res)
        target = self.norm1(target)

        # cross attention
        cross_attention_res = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask
        )
        target = target + self.dropout2(cross_attention_res)
        target = self.norm2(target)

        forward_res = self.linear1(target)
        forward_res = self.activation(forward_res)
        forward_res = self.dropout3(forward_res)
        forward_res = self.linear2(forward_res)
        target = target + self.dropout4(forward_res)
        target = self.norm3(target)

        return target


class RTDetrTransformerDecoder(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        self.layers = nn.ModuleList([RTDetrTransformerDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.num_decoder_layers + config.eval_idx

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
    ):
        """
        Forward pass for the RTDetrTransformerDecoder.

        Args:
            target (`torch.FloatTensor`): the input tensor for the target sequences.
            ref_points_unact (`torch.FloatTensor`): unactivated reference points for positional encoding.
            memory (`torch.FloatTensor`):
                the output of the transformer encoder, representing encoded features from the input.
            memory_spatial_shapes (`List[Tuple[int,int]]`): the spatial shape of each feature level in the memory.
            memory_level_start_index (`List[int]`): the starting index of each level in the flattened memory.
            bbox_head (`List[RTDetrMLP]`): a list of bounding box prediction heads for each decoder layer.
            score_head (`List[nn.Linear]`): a list of scoring heads (for class scores) for each decoder layer.
            query_pos_head (`RTDetrMLP`):
                MLP (RTDetrMLP) to generate query positional embeddings from reference points.
            attn_mask (`torch.FloatTensor` of shape [batch_size*num_heads, sequence length, source sequence legth], *optional*):
                attention mask for the target sequences.
            memory_mask (`torch.FloatTensor`, *optional*): mask for the memory sequences.

        Returns:
            tuple containing bounding boxes and logits representing class scores
        """
        output = target
        ref_points = None
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class RTDetrMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + hidden, hidden + [output_dim]))
        self.act = nn.Identity() if act is None else ACT2CLS[act]()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RTDetrTransformer(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()

        feat_channels = config.feat_channels
        feat_strides = config.feat_strides[:]
        num_levels = config.num_levels

        if len(feat_channels) > num_levels:
            raise ValueError("len(feat_channels) must be less than or equal to num_levels")
        if len(feat_strides) != len(feat_channels):
            raise ValueError("len(feat_strides) must be equal to len(feat_channels)")

        # Extends feat_strides list to match the number of levels (num_levels), ensuring that feat_strides has an entry for each level
        # Each added subsequent stride has twice the size of the last one.
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = config.hidden_dim
        self.num_head = config.num_attention_heads
        self.feat_strides = feat_strides
        self.num_levels = config.num_levels
        self.num_classes = config.num_classes
        self.num_queries = config.num_queries
        self.num_decoder_layers = config.num_decoder_layers
        self.image_size = config.image_size
        self.use_aux_loss = config.use_aux_loss
        self.learnt_init_query = config.learnt_init_query
        self.num_denoising = config.num_denoising
        self.label_noise_ratio = config.label_noise_ratio
        self.box_noise_scale = config.box_noise_scale
        feat_channels = config.feat_channels

        # backbone feature projection
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            conv = nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)
            norm = nn.BatchNorm2d(self.hidden_dim, config.batch_norm_eps)
            layer = [("conv", conv), ("norm", norm)]
            sequential_layer = nn.Sequential(OrderedDict(layer))
            self.input_proj.append(sequential_layer)

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            conv = nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)
            norm = nn.BatchNorm2d(self.hidden_dim, config.batch_norm_eps)
            layer = [("conv", conv), ("norm", norm)]
            self.input_proj.append(nn.Sequential(OrderedDict(layer)))
            in_channels = self.hidden_dim

        # transformer module
        self.decoder = RTDetrTransformerDecoder(config)

        # denoising part
        if self.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                self.num_classes + 1, self.hidden_dim, padding_idx=self.num_classes
            )

        # decoder embedding
        if self.learnt_init_query:
            weight_embedding = torch.empty(1, self.num_queries, self.hidden_dim)
            nn.init.normal_(weight_embedding)
            self.weight_embedding = nn.Parameter(weight_embedding, requires_grad=True)

        self.query_pos_head = RTDetrMLP(4, 2 * self.hidden_dim, self.hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.enc_bbox_head = RTDetrMLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.num_classes) for _ in range(self.num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [RTDetrMLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3) for _ in range(self.num_decoder_layers)]
        )

        # init encoder output anchors and valid_mask
        if self.image_size:
            self.anchors, self.valid_mask = self.generate_anchors()

    def generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.image_size[0] / s), int(self.image_size[1] / s)] for s in self.feat_strides]
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, dtype=dtype), torch.arange(end=width, dtype=dtype), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([width, height]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))

        # define the valid range for anchor coordinates
        eps = 1e-2
        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def forward(self, feats, targets=None):
        # get projection features
        projected_features = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(projected_features):
            len_srcs = len(projected_features)
            projected_features.append(self.input_proj[len_srcs](feats[-1]))
            for i in range(len_srcs + 1, self.num_levels):
                projected_features.append(self.input_proj[i](projected_features[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in projected_features:
            height, width = feat.shape[-2:]
            # [batch, channels, height, width] -> [batch, height*width, channel]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([height, width])
            # [level], start index of each level
            level_start_index.append(height * width + level_start_index[-1])

        # [batch, level, channel]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()

        # prepare denoising training
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets=targets,
                num_classes=self.num_classes,
                num_queries=self.num_queries,
                class_embed=self.denoising_class_embed,
                num_denoising_queries=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        batch_size = len(feat_flatten)
        device = feat_flatten.device

        # prepare input for decoder
        if self.training or self.image_size is None:
            anchors, valid_mask = self.generate_anchors(spatial_shapes, device=device)
        else:
            anchors, valid_mask = self.anchors.to(device), self.valid_mask.to(device)

        # use the valid_mask to selectively retain values in the feature map where the mask is `True`
        memory = valid_mask.to(feat_flatten.dtype) * feat_flatten

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1])
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # extract region features
        if self.learnt_init_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        init_ref_points_unact = reference_points_unact.detach()

        # decoder
        out_bboxes, out_logits = self.decoder(
            target=target,
            ref_points_unact=init_ref_points_unact,
            memory=feat_flatten,
            memory_spatial_shapes=spatial_shapes,
            memory_level_start_index=level_start_index,
            bbox_head=self.dec_bbox_head,
            score_head=self.dec_score_head,
            query_pos_head=self.query_pos_head,
            attn_mask=attn_mask,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

        out = {"logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.use_aux_loss:
            out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out["aux_outputs"].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


# Copied from transformers.models.detr.modeling_detr.dice_loss
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


# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
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


class RTDetrLoss(nn.Module):
    """
    This class computes the losses for RTDetr. The process happens in two steps: 1) we compute hungarian assignment
    between ground truth boxes and the outputs of the model 2) we supervise each pair of matched ground-truth /
    prediction (supervise class and box).

    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        weight_dict (`Dict`):
            Dictionary relating each loss with its weights. These losses are configured in RTDetrConf as
            `weight_loss_vfl`, `weight_loss_bbox`, `weight_loss_giou`
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
        alpha (`float`):
            Parameter alpha used to compute the focal loss.
        gamma (`float`):
            Parameter gamma used to compute the focal loss.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
    """

    def __init__(self, config):
        super().__init__()

        self.matcher = RTDetrHungarianMatcher(config)
        self.num_classes = config.num_classes
        self.weight_dict = {
            "loss_vfl": config.weight_loss_vfl,
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
        }
        self.losses = ["vfl", "boxes"]
        self.eos_coef = config.eos_coefficient
        empty_weight = torch.ones(config.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.alpha = config.focal_loss_alpha
        self.gamma = config.focal_loss_gamma

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        if "logits" not in outputs:
            raise KeyError("No predicted logits found in outputs")
        idx = self._get_source_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([_target["boxes"][i] for _target, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["logits"]
        target_classes_original = torch.cat([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_original
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_original = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_original[idx] = ious.to(target_score_original.dtype)
        target_score = target_score_original.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")

        src_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_original = torch.cat([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_original

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes. This is not
        really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
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
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss. Targets dicts must
        contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
        format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(src_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss. Targets dicts must contain the key
        "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
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

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs["logits"]
        idx = self._get_source_permutation_idx(indices)
        target_classes_original = torch.cat([_target["labels"][i] for _target, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_original

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1.0, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_bce": loss}

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

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        if "logits" not in outputs:
            raise KeyError("No logits found in outputs")

        src_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_original = torch.cat([_target["labels"][i] for _target, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_original

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_focal": loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "bce": self.loss_labels_bce,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
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
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

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

        # In case of cdn auxiliary losses. For rtdetr
        if "dn_aux_outputs" in outputs:
            if "dn_meta" not in outputs:
                raise ValueError(
                    "The output must have the 'dn_meta` key. Please, ensure that 'outputs' includes a 'dn_meta' entry."
                )
            indices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class RTDetrPreTrainedModel(PreTrainedModel):
    config_class = RTDetrConfig
    base_model_prefix = "rt_detr"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initalize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
            if hasattr(module, "bias"):
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.fill_(1.0)


class RTDetrHybridEncoder(RTDetrPreTrainedModel):
    """
    Decoder consists of a projection layer, a set of `RTDetrTransformerEncoder`, a top-down Feature Pyramid Network
    (FPN) and a bottom-up Path Aggregation Network (PAN). More details on the paper: https://arxiv.org/abs/2304.08069

    Args:
        config: RTDetrConfig
    """

    def __init__(self, config: RTDetrConfig, in_channels: List[int]):
        super().__init__(config)
        self.in_channels = in_channels
        self.feat_strides = config.feat_strides
        self.hidden_dim = config.hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.pe_temperature = config.pe_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        act_encoder = config.act_encoder
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, self.hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.hidden_dim, config.batch_norm_eps),
                )
            )

        # encoder transformer
        self.encoder = nn.ModuleList([RTDetrTransformerEncoder(config) for _ in range(len(self.encode_proj_layers))])
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                RTDetrConvNormLayer(config, self.hidden_dim, self.hidden_dim, 1, 1, activation=act_encoder)
            )
            self.fpn_blocks.append(RTDetrCSPRepLayer(config))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                RTDetrConvNormLayer(config, self.hidden_dim, self.hidden_dim, 3, 2, activation=act_encoder)
            )
            self.pan_blocks.append(RTDetrCSPRepLayer(config))

    @staticmethod
    def build_2d_sincos_position_embedding(width, height, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(width), dtype=torch.float32)
        grid_h = torch.arange(int(height), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        if len(feats) != len(self.in_channels):
            raise "Relation len(feats) != len(self.in_channels) must apply."
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        for i, enc_ind in enumerate(self.encode_proj_layers):
            height, width = proj_feats[enc_ind].shape[2:]
            # flatten [batch, channel, height, width] to [batch, height*width, channel]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
            if self.training or self.eval_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    width, height, self.hidden_dim, self.pe_temperature
                ).to(src_flatten.device)
            else:
                pos_embed = None

            memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
            proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, height, width).contiguous()

        # broadcasting and fusion
        fpn_feature_maps = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = fpn_feature_maps[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            fpn_feature_maps[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            fps_map = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            fpn_feature_maps.insert(0, fps_map)

        outs = [fpn_feature_maps[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = fpn_feature_maps[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs


RT_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RTDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RT_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`RTDetrImageProcessor.__call__`] for details.
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class RTDetrHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        config: RTDetrConfig
    """

    def __init__(self, config):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.cost_class = config.matcher_class_cost
        self.cost_bbox = config.matcher_bbox_cost
        self.cost_giou = config.matcher_giou_cost

        self.use_focal_loss = config.use_focal_loss
        self.alpha = config.matcher_alpha
        self.gamma = config.matcher_gamma

        if self.cost_class == 0 and self.cost_bbox == 0 and self.cost_giou == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["class_labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["logits"].flatten(0, 1))
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(tgt_bbox))
        # Compute the final cost matrix
        final_cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        final_cost = final_cost.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(final_cost.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@add_start_docstrings(
    """
    RT-DETR Model (consisting of a backbone and encoder-decoder) outputting bounding boxes and logits to be further
    decoded into scores and classes.
    """,
    RT_DETR_START_DOCSTRING,
)
class RTDetrModel(RTDetrPreTrainedModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)

        self.backbone = AutoBackbone.from_config(config.backbone_config)
        # enconder
        self.encoder = RTDetrHybridEncoder(config, in_channels=self.backbone.channels)
        # decoder
        self.decoder = RTDetrTransformer(config)

        self.criterion = RTDetrLoss(config)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    @add_start_docstrings_to_model_forward(RT_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RTDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[List[dict]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], RTDetrModelOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RTDetrModel
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained(_CHECKPOINT_FOR_DOC)
        >>> model = RTDetrModel.from_pretrained(_CHECKPOINT_FOR_DOC)

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 300, 80]

        >>> boxes = outputs.pred_boxes
        >>> list(boxes.shape)
        [1, 300, 4]

        >>> # convert outputs (bounding boxes and class logits) to COCO API
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
        Detected couch with confidence 0.97 at location [0.14, 0.38, 640.13, 476.21]
        Detected cat with confidence 0.96 at location [343.38, 24.28, 640.14, 371.5]
        Detected cat with confidence 0.958 at location [13.23, 54.18, 318.98, 472.22]
        Detected remote with confidence 0.951 at location [40.11, 73.44, 175.96, 118.48]
        Detected remote with confidence 0.924 at location [333.73, 76.58, 369.97, 186.99]
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        features = self.backbone(pixel_values)
        encoder_outputs = self.encoder(features["feature_maps"])
        outputs = self.decoder(encoder_outputs)

        pred_boxes = outputs["pred_boxes"]
        logits = outputs["logits"]

        loss, loss_dict = None, None
        if labels is not None:
            self.criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            loss_dict = self.criterion(outputs_loss, labels)
            # Compute total loss, as a weighted sum of the various losses
            weight_dict = {
                "loss_vfl": self.config.weight_loss_vfl,
                "loss_bbox": self.config.weight_loss_bbox,
                "loss_giou": self.config.weight_loss_giou,
            }
            weight_loss_scaled = {k: v * loss_dict[k] for k, v in weight_dict.items()}

            loss = sum(weight_loss_scaled.values())
            loss_dict = {
                "loss_dict": loss_dict,
                "weight_loss_scaled": weight_loss_scaled,
            }

        encoder_states = encoder_outputs if output_hidden_states else ()

        if not return_dict:
            output = (logits, pred_boxes, encoder_states)
            return ((loss, loss_dict) + output) if loss is not None else output

        return RTDetrModelOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            encoder_hidden_states=encoder_states,
        )
