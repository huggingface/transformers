# coding=utf-8
# Copyright 2023 Meta Platforms, Inc.,
# MMDetection Contributors. (2018). OpenMMLab Detection Toolbox and Benchmark [Computer software]. https://github.com/open-mmlab/mmdetection
# and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Mask R-CNN model."""

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.modules.utils import _pair

from ... import AutoBackbone

# TODO decide whether to define these utilities
from ...assign_result import AssignResult
from ...loss_utils import CrossEntropyLoss, L1Loss, accuracy
from ...modeling_utils import PreTrainedModel
from ...sampling_result import SamplingResult
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchvision_available,
    logging,
    replace_return_docstrings,
)
from .configuration_maskrcnn import MaskRCNNConfig


if is_torchvision_available():
    import torchvision

    from ...mask_target import mask_target
    from ...nms import batched_nms


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "MaskRCNNConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/convnext-tiny-maskrcnn"


MASK_RCNN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnext-tiny-maskrcnn",
    # See all MaskRCNN models at https://huggingface.co/models?filter=convnext_maskrcnn
]


@dataclass
class MaskRCNNRPNOutput(ModelOutput):
    """
    Region Proposal Network (RPN) outputs.

    Args:
        losses (`torch.FloatTensor`):
            Losses of the RPN head.
        proposal_list (`list[`torch.FloatTensor`]`):
            List of proposals, for each example in the batch. Each proposal is a `torch.FloatTensor` of shape
            (num_proposals, 5). Each proposal is of the format (x1, y1, x2, y2, score).
        outs (`tuple(List(torch.FloatTensor)`)):
            Tuple of lists, the first list containing the class logits and the second list containing the box
            predictions.
    """

    losses: torch.FloatTensor = None
    proposal_list: List[torch.FloatTensor] = None
    outs: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MaskRCNNModelOutput(ModelOutput):
    """
    Base class for models that leverage the Mask R-CNN framework.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a combination of various losses.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(num_proposals_per_image stacked on top of each other, num_labels + 1)`):
            Classification logits (including no-object) for all proposals.
        pred_boxes (`torch.FloatTensor` of shape `(num_proposals_per_image stacked on top of each other, num_labels * 4)`):
            Predicted boxes, for each class and each proposal.
        rois (`torch.FloatTensor` of shape `(num_proposals_per_image stacked on top of each other, 5)`):
            Region of interest proposals. Each contains [batch_index, x1, y1, x2, y2].
        proposals (`List[torch.FloatTensor]` of shape `(num_proposals_per_image, 5)`):
            Proposals as predicted by the RPN. Each contains [x1, y1, x2, y2, score].
        fpn_hidden_states (`tuple(torch.FloatTensor)` with length = number of scale levels):
            Hidden states of the FPN (Feature Pyramid Network).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: torch.FloatTensor = None
    loss_dict: dict = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    rois: torch.FloatTensor = None
    proposals: torch.FloatTensor = None
    fpn_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size count)"""
    new_size = (count,) + data.size()[1:]
    ret = data.new_full(new_size, fill)
    ret[inds.type(torch.bool)] = data
    return ret


def select_single_multilevel(multilevel_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient needs to be detached during the training
    of the two-stage model. E.g Cascade Mask R-CNN.

    Args:
        multilevel_tensors (`List[torch.Tensor]`):
            Batch tensor for all scale levels, each is a 4D-tensor.
        batch_id (`int`):
            Batch index.
        detach (`bool`, *optional*, defaults to `True`):
            Whether to detach the gradient.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    if not isinstance(multilevel_tensors, (list, tuple)):
        raise TypeError(f"multilevel_tensors must be a list or tuple, but got {type(multilevel_tensors)}")
    num_levels = len(multilevel_tensors)

    if detach:
        multilevel_tensor_list = [multilevel_tensors[i][batch_id].detach() for i in range(num_levels)]
    else:
        multilevel_tensor_list = [multilevel_tensors[i][batch_id] for i in range(num_levels)]
    return multilevel_tensor_list


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.
    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for level in num_levels:
        end = start + level
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (`torch.Tensor`):
            Flattened anchors, shape (n, 4).
        valid_flags (`torch.Tensor`):
            An existing valid flags of anchors.
        img_shape (`Tuple[int]`):
            Shape of current image.
        allowed_border (`int`, *optional*, defaults to 0):
            The border to allow the valid anchor. Defaults to 0.

    Returns:
        `torch.Tensor`: Flags indicating whether the anchors are inside a valid range.
    """
    img_height, img_width = img_shape[-2:]
    if allowed_border >= 0:
        inside_flags = (
            valid_flags
            & (flat_anchors[:, 0] >= -allowed_border)
            & (flat_anchors[:, 1] >= -allowed_border)
            & (flat_anchors[:, 2] < img_width + allowed_border)
            & (flat_anchors[:, 3] < img_height + allowed_border)
        )
    else:
        inside_flags = valid_flags
    return inside_flags


def bbox2delta(proposals, ground_truth, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
    """Compute deltas of proposals w.r.t. ground truth.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground truth bboxes to get regression target. This
    is the inverse function of [`delta2bbox`].

    Args:
        proposals (`torch.Tensor`):
            Boxes to be transformed, shape (N, ..., 4)
        ground_truth (`torch.Tensor`):
            Gt bboxes to be used as base, shape (N, ..., 4)
        means (`Sequence[float]`, *optional*, defaults to `(0.0, 0.0, 0.0, 0.0)`):
            Denormalizing means for delta coordinates
        stds (`Sequence[float]`, *optional*, defaults to `(1.0, 1.0, 1.0, 1.0)`):
            Denormalizing standard deviation for delta coordinates

    Returns:
       `torch.Tensor`: deltas with shape (N, 4), where columns represent dx, dy, dw, dh.
    """
    if proposals.size() != ground_truth.size():
        raise ValueError("Should have as many proposals as there are ground truths")

    proposals = proposals.float()
    ground_truth = ground_truth.float()

    # predicted boxes
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    # ground truth boxes
    gx = (ground_truth[..., 0] + ground_truth[..., 2]) * 0.5
    gy = (ground_truth[..., 1] + ground_truth[..., 3]) * 0.5
    gw = ground_truth[..., 2] - ground_truth[..., 0]
    gh = ground_truth[..., 3] - ground_truth[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(
    rois,
    deltas,
    means=(0.0, 0.0, 0.0, 0.0),
    stds=(1.0, 1.0, 1.0, 1.0),
    max_shape=None,
    wh_ratio_clip=16 / 1000,
    clip_border=True,
    add_ctr_clamp=False,
    ctr_clamp=32,
):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are network outputs used to shift/scale
    those boxes. This is the inverse function of [`bbox2delta`].

    Args:
        rois (`torch.Tensor`):
            Boxes to be transformed. Has shape (N, 4) with N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        deltas (`torch.Tensor`):
            Encoded offsets relative to each roi. Has shape (N, num_classes * 4) or (N, 4). Offset encoding follows
            https://arxiv.org/abs/1311.2524.
        means (`Sequence[float]`, *optional*, defaults to `(0., 0., 0., 0.)`):
            Denormalizing means for delta coordinates.
        stds (`Sequence[float]`, *optional*, defaults to `(1., 1., 1., 1.)`):
            Denormalizing standard deviation for delta coordinates.
        max_shape (`Tuple[int, int]`, *optional*):
            Maximum bounds for boxes, specifies (H, W). Default None.
        wh_ratio_clip (`float`, *optional*, defaults to 16 / 1000):
            Maximum aspect ratio for boxes.
        clip_border (`bool`, *optional*, defaults to `True`):
            Whether to clip the objects outside the border of the image.
        add_ctr_clamp (`bool`, *optional*, defaults to `False`):
            Whether to add center clamp. When set to True, the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor. Only used by YOLOF.
        ctr_clamp (`int`, *optional*, defaults to 32):
            The maximum pixel shift to clamp. Only used by YOLOF.

    Returns:
        `torch.Tensor`: Boxes with shape (N, num_classes * 4) or (N, 4), where 4 represent top_left_x, top_left_y,
        bottom_right_x, bottom_right_y.
    """
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = (rois_[:, :2] + rois_[:, 2:]) * 0.5
    pwh = rois_[:, 2:] - rois_[:, :2]

    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    if clip_border and max_shape is not None:
        max_shape = max_shape[-2:]
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (`List[Tensor]`):
            A list of bboxes corresponding to a batch of images.

    Returns:
        `torch.Tensor`: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


class MaskRCNNFPN(nn.Module):
    """
    Feature Pyramid Network (FPN).

    This is an implementation of [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144).
    """

    def __init__(self, config, hidden_sizes):
        super().__init__()

        self.num_outs = config.fpn_num_outputs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for hidden_size in hidden_sizes:
            lateral_conv = nn.Conv2d(hidden_size, config.fpn_out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(config.fpn_out_channels, config.fpn_out_channels, kernel_size=3, padding=1)

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, hidden_states: List[torch.Tensor]):
        # build laterals
        laterals = [lateral_conv(hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(laterals[i], size=prev_shape, mode="nearest")

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(nn.functional.max_pool2d(outs[-1], 1, stride=2))

        return outs


class MaskRCNNAnchorGenerator(nn.Module):
    """
    Standard 2D anchor generator.

    Source: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py.
    """

    def __init__(self, config, scale_major=True, centers=None, center_offset=0.0):
        super().__init__()

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in config.anchor_generator_strides]
        self.base_sizes = [min(stride) for stride in self.strides]
        if len(self.base_sizes) != len(self.strides):
            raise ValueError(
                f"The number of strides should be the same as base sizes, got {self.strides} and {self.base_sizes}"
            )

        # calculate scales of anchors
        self.scales = torch.Tensor(config.anchor_generator_scales)
        # calculate ratios of anchors
        self.ratios = torch.Tensor(config.anchor_generator_ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self):
        """
        Returns;:
            `List[int]`: The number of priors (anchors) at a point on the feature grid
        """
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """
        Returns:
            `int`: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            `List[torch.Tensor]`: Base anchors of a feature grid in multiple feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_size, scales=self.scales, ratios=self.ratios, center=center)
            )
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_size, scales, ratios, center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (`int` or `float`):
                Basic size of an anchor.
            scales (`torch.Tensor`):
                Scales of the anchor.
            ratios (`torch.Tensor`):
                The ratio between between the height and width of anchors in a single level.
            center (`Tuple[float]`, *optional*):
                The center of the base anchor related to a single feature grid.

        Returns:
            `torch.Tensor`: Anchors in a single-level feature maps.
        """
        width = base_size
        height = base_size
        if center is None:
            x_center = self.center_offset * width
            y_center = self.center_offset * height
        else:
            x_center, y_center = center

        height_ratios = torch.sqrt(ratios)
        width_ratios = 1 / height_ratios
        if self.scale_major:
            width_scaled = (width * width_ratios[:, None] * scales[None, :]).view(-1)
            height_scaled = (height * height_ratios[:, None] * scales[None, :]).view(-1)
        else:
            width_scaled = (width * scales[:, None] * width_ratios[None, :]).view(-1)
            height_scaled = (height * scales[:, None] * height_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * width_scaled,
            y_center - 0.5 * height_scaled,
            x_center + 0.5 * width_scaled,
            y_center + 0.5 * height_scaled,
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (`torch.Tensor`):
                Grids of x dimension.
            y (`torch.Tensor`):
                Grids of y dimension.
            row_major (`bool`, *optional*, defaults to `True`):
                Whether to return y grids first.

        Returns:
            `Tuple[torch.Tensor]`: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_priors(self, featmap_sizes, dtype=torch.float32, device="cuda"):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (`List[Tuple]`):
                List of feature map sizes in multiple feature levels.
            dtype (`torch.dtype`):
                Dtype of priors. Default: torch.float32.
            device (`str`, *optional*, defaults to `"cuda"`):
                The device where the anchors will be put on.

        Returns:
            `List[torch.Tensor]`: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \ N = width * height * num_base_anchors, width and
                height \ are the sizes of the corresponding feature level, \ num_base_anchors is the number of anchors
                for that level.
        """
        if self.num_levels != len(featmap_sizes):
            raise ValueError(f"Expected {self.num_levels} feature levels, got {len(featmap_sizes)}")
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self, featmap_size, level_idx, dtype=torch.float32, device="cuda"):
        """Generate grid anchors of a single level.
        Note:
            This function is usually called by method `self.grid_priors`.

        Args:
            featmap_size (`Tuple[int]`):
                Size of the feature maps.
            level_idx (`int`):
                The index of corresponding feature map level.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Date type of points.
            device (`str`, *optional*, defaults to `"cuda"`):
                The device the tensor will be put on.

        Returns:
            `torch.Tensor`: Anchors in the overall feature maps.
        """

        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device="cuda"):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (`List[Tuple]`):
                List of feature map sizes in multiple feature levels.
            pad_shape (`Tuple`):
                The padded shape of the image.
            device (`str`, *optional*, defaults to `"cuda"`):
                Device where the anchors will be put on.

        Returns:
            `List[torch.Tensor]`: Valid flags of anchors in multiple levels.
        """
        if self.num_levels != len(featmap_sizes):
            raise ValueError(f"Expected {self.num_levels} feature levels, got {len(featmap_sizes)}")
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[-2:]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w), self.num_base_anchors[i], device=device
            )
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self, featmap_size, valid_size, num_base_anchors, device="cuda"):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (`Tuple[int]`):
                The size of feature maps, arrange as (height, width).
            valid_size (`Tuple[int]`):
                The valid size of the feature maps.
            num_base_anchors (`int`):
                The number of base anchors.
            device (`str`, *optional*, defaults to `"cuda"`):
                Device where the flags will be put on.

        Returns:
            `torch.Tensor`: The valid flags of each anchor in a single level feature map.
        """
        feat_height, feat_width = featmap_size
        valid_height, valid_width = valid_size
        if not (valid_height <= feat_height and valid_width <= feat_width):
            raise ValueError(
                f"valid_size: {valid_size} should be less than "
                f"featmap_size: {featmap_size} in single_level_valid_flags."
            )
        valid_x = torch.zeros(feat_width, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_height, dtype=torch.bool, device=device)
        valid_x[:valid_width] = 1
        valid_y[:valid_height] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0), num_base_anchors).contiguous().view(-1)
        return valid


class MaskRCNNDeltaXYWHBBoxCoder(nn.Module):
    """Delta XYWH BBox coder.
    Following the practice in [R-CNN](https://arxiv.org/abs/1311.2524), this coder encodes bbox (x1, y1, x2, y2) into
    delta (dx, dy, dw, dh) and decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (`Sequence[float]`):
            Denormalizing means of target for delta coordinates.
        target_stds (`Sequence[float]`):
            Denormalizing standard deviation of target for delta coordinates.
        clip_border (`bool`, *optional*, defaults to `True`):
            Whether clip the objects outside the border of the image.
        add_ctr_clamp (`bool`, *optional*, defaults to `False`):
            Whether to add center clamp, when added, the predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF.
        ctr_clamp (`int`, *optional*, defaults to 32):
            The maximum pixel shift to clamp. Only used by YOLOF.
    """

    def __init__(
        self,
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
        clip_border=True,
        add_ctr_clamp=False,
        ctr_clamp=32,
    ):
        super().__init__()
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to transform the `bboxes` into the `gt_bboxes`.

        Args:
            bboxes (`torch.Tensor`):
                Source boxes, e.g., object proposals.
            gt_bboxes (`torch.Tensor`):
                Target of the transformation, e.g., ground-truth boxes.

        Returns:
            `torch.Tensor`: Box transformation deltas
        """

        if bboxes.size(0) != gt_bboxes.size(0):
            raise ValueError("bboxes and gt_bboxes should have same batch size")
        if not (bboxes.size(-1) == gt_bboxes.size(-1) == 4):
            raise ValueError("bboxes and gt_bboxes should have 4 elements in last dimension")
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None, wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (`torch.Tensor`):
                Basic boxes. Shape (batch_size, N, 4) or (N, 4)
            pred_bboxes (`torch.Tensor`):
                Encoded offsets with respect to each roi. Has shape (batch_size, N, num_classes * 4) or (batch_size, N,
                4) or (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H when rois is a grid of anchors.
                Offset encoding follows [1]_.
            max_shape (`Sequence[int]` or `torch.Tensor` or `Sequence[Sequence[int]]`, *optional*):
                Maximum bounds for boxes, specifies (H, W, C) or (H, W). If `bboxes` shape is (B, N, 4), then the
                `max_shape` should be a Sequence[Sequence[int]] and the length of `max_shape` should also be B.
            wh_ratio_clip (`float`, *optional*, defaults to 16 / 1000):
                The allowed ratio between width and height.

        Returns:
            `torch.Tensor`: Decoded boxes.
        """

        if pred_bboxes.size(0) != bboxes.size(0):
            raise ValueError("pred_bboxes and bboxes should have the same first dimension")
        if pred_bboxes.ndim == 3:
            if pred_bboxes.size(1) != bboxes.size(1):
                raise ValueError("pred_bboxes and bboxes should have the same second dimension")

        if pred_bboxes.ndim == 2:
            # single image decode
            decoded_bboxes = delta2bbox(
                bboxes,
                pred_bboxes,
                self.means,
                self.stds,
                max_shape,
                wh_ratio_clip,
                self.clip_border,
                self.add_ctr_clamp,
                self.ctr_clamp,
            )
        else:
            raise ValueError("Predicted boxes should have 2 dimensions")

        return decoded_bboxes


# Everything related to IoU calculator #


def cast_tensor_type(tensor, scale=1.0, dtype=None):
    if dtype == "fp16":
        # scale is for preventing overflows
        tensor = (tensor / scale).half()
    return tensor


def fp16_clamp(tensor, min=None, max=None):
    if not tensor.is_cuda and tensor.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return tensor.float().clamp(min, max).half()

    return tensor.clamp(min, max)


class MaskRCNNBboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, bboxes1, bboxes2, mode="iou", is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (`torch.Tensor`):
                Bboxes having shape (m, 4) in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2, y2, score>
                format.
            bboxes2 (`torch.Tensor`):
                Boxes having shape (m, 4) in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2, score> format,
                or be empty. If `is_aligned ` is `True`, then m and n must be equal.
            mode (`str`, *optional*, defaults to `"iou"`):
                "iou" (intersection over union), "iof" (intersection over foreground), or "giou" (generalized
                intersection over union).
            is_aligned (`bool`, *optional*, defaults to `False`):
                If True, then m and n must be equal. Default False.

        Returns:
            `torch.Tensor : shape (m, n) if `is_aligned ` is False else shape (m,)
        """
        if bboxes1.size(-1) not in [0, 4, 5]:
            raise ValueError(f"bboxes1 must have shape (m, 4) or (m, 5), but got {bboxes1.size()}")
        if bboxes2.size(-1) not in [0, 4, 5]:
            raise ValueError(f"bboxes2 must have shape (n, 4) or (n, 5), but got {bboxes2.size()}")
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == "fp16":
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f"(scale={self.scale}, dtype={self.dtype})"
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """Calculates the overlap between two set of bboxes.

    If `is_aligned` is `False`, then it calculates the overlaps between each bbox of bboxes1 and bboxes2, otherwise the
    overlaps between each aligned pair of bboxes1 and bboxes2.

    For an extensive explanation, we refer to
    https://github.com/open-mmlab/mmdetection/blob/ecac3a77becc63f23d9f6980b2a36f86acd00a8a/mmdet/structures/bbox/bbox_overlaps.py#L13.

    Args:
        bboxes1 (`torch.Tensor`):
            Shape (B, m, 4) in <x1, y1, x2, y2> format or empty. B indicates the batch dim, in shape (B1, B2, ..., Bn).
        bboxes2 (`torch.Tensor`):
            Shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
        mode (`str`, **optional*, defaults to `"iou"`):
            "iou" (intersection over union), "iof" (intersection over foreground) or "giou" (generalized intersection
            over union).
        is_aligned (`bool`, *optional*, defaults to `False`):
            If `True`, then m and n must be equal.
        eps (`float`, *optional*, defaults to 1e-6):
            A value added to the denominator for numerical stability.

    Returns:
        `torch.Tensor`: shape (m, n) if `is_aligned` is `False` else shape (m,)
    """

    if mode not in ["iou", "iof", "giou"]:
        raise ValueError(f"Unsupported mode {mode}")
    # Either the boxes are empty or the length of boxes' last dimension is 4
    if not (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0):
        raise ValueError(f"bboxes1 shape {bboxes1.shape} should have last dimension 4")
    if not (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0):
        raise ValueError(f"bboxes2 shape {bboxes2.shape} should have last dimension 4")

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    if bboxes1.shape[:-2] != bboxes2.shape[:-2]:
        raise ValueError(
            f"bboxes1 and bboxes2 should have same batch dimensions, got {bboxes1.shape[:-2]} and {bboxes2.shape[:-2]}"
        )
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        if rows != cols:
            raise ValueError(f"bboxes1 and bboxes2 should be of same size in aligned mode, got {rows} and {cols}")

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        top_left = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        bottom_right = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        width_height = fp16_clamp(bottom_right - top_left, min=0)
        overlap = width_height[..., 0] * width_height[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_top_left = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_bottom_right = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        top_left = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        bottom_right = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        width_height = fp16_clamp(bottom_right - top_left, min=0)
        overlap = width_height[..., 0] * width_height[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_top_left = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_bottom_right = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_width_height = fp16_clamp(enclosed_bottom_right - enclosed_top_left, min=0)
    enclose_area = enclose_width_height[..., 0] * enclose_width_height[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class MaskRCNNMaxIoUAssigner:
    """Assign a corresponding ground truth bbox or background to each bbox.
    Each proposal will be assigned with `-1`, or a semi-positive integer indicating the ground truth index.
    - -1: negative sample, no assigned ground truth (gt)
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Source:
    https://github.com/open-mmlab/mmdetection/blob/78e3ec8e6adc63763cab4060009e37a5d63c5c7a/mmdet/core/bbox/assigners/max_iou_assigner.py

    Args:
        pos_iou_thr (`float`):
            IoU threshold for positive bboxes.
        neg_iou_thr (`float` or `Tuple`):
            IoU threshold for negative bboxes.
        min_pos_iou (`float`, *optional*, defaults to 0.0):
            Minimum iou for a bbox to be considered as a positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (`bool`, *optional*, defaults to `False`):
            Whether to assign all bboxes with the same highest overlap with some gt to that gt.
        ignore_iof_thr (`float`, *optional*, defaults to -1):
            IoF threshold for ignoring bboxes (if `gt_bboxes_ignore` is specified). Negative values mean not ignoring
            any bboxes.
        ignore_wrt_candidates (`bool`, *optional*, defaults to `True`):
            Whether to compute the iof between `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (`bool`, *optional*, defaults to `True`):
            Whether to allow low quality matches. This is usually allowed for RPN and single stage detectors, but not
            allowed in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (`int`, *optional*, defaults to -1):
            The upper bound of the number of GT for GPU assign. When the number of gt is above this threshold, will
            assign on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(
        self,
        pos_iou_thr,
        neg_iou_thr,
        min_pos_iou=0.0,
        gt_max_assign_all=True,
        ignore_iof_thr=-1,
        ignore_wrt_candidates=True,
        match_low_quality=True,
        gpu_assign_thr=-1,
    ):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = MaskRCNNBboxOverlaps2D()

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        This method assign a gt bbox to every bbox (proposal/anchor), each bbox will be assigned with -1, or a
        semi-positive number. -1 means negative sample, semi-positive number is the index (0-based) of assigned gt. The
        assignment is done in following steps, the order matters.
        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr, assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than one) to itself

        Args:
            bboxes (`torch.Tensor`):
                Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (`torch.Tensor`):
                Ground truth boxes, shape (k, 4).
            gt_bboxes_ignore (`torch.Tensor`, *optional*):
                Ground truth bboxes that are labelled as *ignored*, e.g., crowd boxes in COCO.
            gt_labels (`torch.Tensor`, *optional*):
                Label of gt_bboxes, shape (k, ).

        Returns:
            `AssignResult`: The assign result.
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and bboxes.numel() > 0
        ):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(bboxes, gt_bboxes_ignore, mode="iof")
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(gt_bboxes_ignore, bboxes, mode="iof")
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with ground truths.

        Args:
            overlaps (`torch.Tensor`):
                Overlaps between k gt_bboxes and n bboxes, shape(k, n).
            gt_labels (`torch.Tensor`, *optional*):
                Labels of k gt_bboxes, shape (k, ).

        Returns:
            `AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            if len(self.neg_iou_thr) != 2:
                raise ValueError("`neg_iou_thr` should be a float or tuple of length 2")
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


class MaskRCNNRandomSampler:
    """Random sampler.

    Source: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py

    Args:
        num (`int`):
            Number of samples.
        pos_fraction (`float`):
            Fraction of positive samples
        neg_pos_up (`int`, *optional*, defaults to -1):
            Upper bound number of negative and positive samples.
        add_gt_as_proposals (`bool`, *optional*, defaults to `True`):
            Whether to add ground truth boxes as proposals.
    """

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.
        If `gallery` is a Tensor, the returned indices will be a Tensor; If `gallery` is a ndarray or list, the
        returned indices will be a ndarray.

        Args:
            gallery (Tensor | ndarray | list):
                Indices pool.
            num (int):
                Expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        if not len(gallery) >= num:
            raise ValueError("sample number exceeds population size")

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = "cpu"
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        # This is a temporary fix. We can revert the following code
        # when PyTorch fixes the abnormal return of torch.randperm.
        # See: https://github.com/open-mmlab/mmdetection/pull/5014
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None, **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates, assigning results and ground truth bboxes.

        Args:
            assign_result (`AssignResult`):
                Bbox assigning results.
            boxes (`torch.Tensor`):
                Boxes to be sampled from.
            gt_bboxes (`torch.Tensor`):
                Ground truth bboxes.
            gt_labels (`torch.Tensor`, *optional*):
                Class labels of ground truth bboxes.

        Returns:
            `SamplingResult`: Sampling result.
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError("gt_labels must be given when add_gt_as_proposals is True")
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result


class MaskRCNNRPN(nn.Module):
    """
    Anchor-based Region Proposal Network (RPN). The RPN learns to convert anchors into region proposals, by 1)
    classifying anchors as either positive/negative/neutral (based on IoU overlap with ground-truth boxes) 2) for the
    anchors classified as positive/negative, regressing the anchor box to the ground-truth box.

    RPN was originally proposed in [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
    Networks](https://arxiv.org/abs/1506.01497).

    Source: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/anchor_head.py
    """

    def __init__(self, config, num_classes=1, reg_decoded_bbox=False):
        super().__init__()

        self.config = config

        # anchor generator
        self.prior_generator = MaskRCNNAnchorGenerator(config)
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.bbox_coder = MaskRCNNDeltaXYWHBBoxCoder(
            target_means=config.rpn_bbox_coder_target_means, target_stds=config.rpn_bbox_coder_target_stds
        )

        # layers
        self.use_sigmoid_cls = config.rpn_loss_cls.get("use_sigmoid", False)
        self.num_classes = num_classes
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f"num_classes={num_classes} is too small")

        self.rpn_conv = nn.Conv2d(config.rpn_in_channels, config.rpn_feat_channels, kernel_size=3, padding=1)
        self.rpn_cls = nn.Conv2d(config.rpn_feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(config.rpn_feat_channels, self.num_base_priors * 4, 1)

        self.train_cfg = config.rpn_train_cfg
        self.test_cfg = config.rpn_test_cfg

        # IoU assigner
        self.assigner = MaskRCNNMaxIoUAssigner(
            pos_iou_thr=config.rpn_assigner_pos_iou_thr,
            neg_iou_thr=config.rpn_assigner_neg_iou_thr,
            min_pos_iou=config.rpn_assigner_min_pos_iou,
            match_low_quality=config.rpn_assigner_match_low_quality,
            ignore_iof_thr=config.rpn_assigner_ignore_iof_thr,
        )
        # Sampler
        self.sampler = MaskRCNNRandomSampler(
            num=config.rpn_sampler_num,
            pos_fraction=config.rpn_sampler_pos_fraction,
            neg_pos_ub=config.rpn_sampler_neg_pos_ub,
            add_gt_as_proposals=config.rpn_sampler_add_gt_as_proposals,
        )
        # TODO support PseudoSampler in the future
        self.sampling = True
        self.reg_decoded_bbox = reg_decoded_bbox

        # losses
        # based on config:
        self.loss_cls = CrossEntropyLoss(
            use_sigmoid=True
        )  # this corresponds to dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        self.loss_bbox = L1Loss()  # this corresponds to dict(type='L1Loss', loss_weight=1.0)

    def forward_single(self, hidden_state):
        """Forward feature map of a single scale level."""
        hidden_state = self.rpn_conv(hidden_state)
        hidden_state = nn.functional.relu(hidden_state, inplace=True)
        rpn_cls_score = self.rpn_cls(hidden_state)
        rpn_bbox_pred = self.rpn_reg(hidden_state)
        return rpn_cls_score, rpn_bbox_pred

    def forward_features(self, hidden_states):
        """Forward features from the upstream network.

        Args:
            hidden_states (tuple[torch.FloatTensor]):
                Features from the upstream network, each being a 4D-tensor.
        Returns:
            tuple: A tuple of classification scores and bbox prediction.
                - cls_scores (list[Tensor]):
                    Classification scores for all scale levels, each is a 4D-tensor, the channels number is
                    num_base_priors * num_classes.
                - bbox_preds (list[Tensor]):
                    Box energies / deltas for all scale levels, each is a 4D-tensor, the channels number is
                    num_base_priors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for hidden_state in hidden_states:
            cls_score, bbox_pred = self.forward_single(hidden_state)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds

    def forward(
        self,
        hidden_states,
        img_metas,
        gt_bboxes=None,
        gt_labels=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outs = self.forward_features(hidden_states)

        losses = None
        if gt_bboxes is not None:
            if gt_labels is None:
                loss_inputs = outs + (gt_bboxes, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)

            losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        proposal_list = self.get_bboxes(*outs, img_metas=img_metas, cfg=proposal_cfg)

        if not return_dict:
            output = (
                proposal_list,
                outs,
            )
            return ((losses,) + output) if losses is not None else output

        return MaskRCNNRPNOutput(losses=losses, proposal_list=proposal_list, outs=outs)

    def get_anchors(self, featmap_sizes, img_metas, device="cuda"):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (`List[Tuple]`):
                Multi-level feature map sizes.
            img_metas (`List[Dict]`):
                Image meta info.
            device (`torch.device` or `"str"`, *optional*, defaults to `"cuda"`):
                Device for returned tensors.

        Returns:
            anchor_list (`List[torch.Tensor]`):
                Anchors of each image.
            valid_flag_list (`List[torch.Tensor]`):
                Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_meta in img_metas:
            multi_level_flags = self.prior_generator.valid_flags(featmap_sizes, img_meta["pad_shape"], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(
        self,
        flat_anchors,
        valid_flags,
        gt_bboxes,
        gt_bboxes_ignore,
        gt_labels,
        img_meta,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Compute regression and classification targets for anchors in a single image.

        Args:
            flat_anchors (`torch.Tensor`):
                Multi-level anchors of the image, which are concatenated into a single tensor of shape (num_anchors,
                4).
            valid_flags (`torch.Tensor`):
                Multi level valid flags of the image, which are concatenated into a single tensor of shape
                (num_anchors,).
            gt_bboxes (`torch.Tensor`):
                Ground truth bboxes of the image, shape (num_gts, 4).
            gt_bboxes_ignore (`torch.Tensor`):
                Ground truth bboxes to be ignored, shape (num_ignored_gts, 4).
            gt_labels (`torch.Tensor`):
                Ground truth labels of each box, shape (num_gts,).
            img_meta (`dict`):
                Meta info of the image.
            label_channels (`int`, *optional*, defaults to 1):
                Number of channels of the ground truth labels.
            unmap_outputs (`bool`, *optional*, defaults to `True`):
                Whether to map outputs back to the original set of anchors.

        Returns:
            `tuple` comprising various elements:
                labels_list (`List[torch.Tensor]`):
                    Labels of each level.
                label_weights_list (`List[torch.Tensor]`):
                    Label weights of each level.
                bbox_targets_list (`List[torch.Tensor]`):
                    Bbox targets of each level.
                bbox_weights_list (`List[torch.Tensor]`):
                    Bbox weights of each level
                num_total_pos (`int`):
                    Number of positive samples in all images.
                num_total_neg (`int`):
                    Number of negative samples in all images.
        """
        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_meta["img_shape"][-2:], self.train_cfg["allowed_border"]
        )
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels
        )
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg["pos_weight"] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg["pos_weight"]
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result)

    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
        return_sampling_results=False,
    ):
        """Compute regression and classification targets for anchors in multiple images.

        Args:
            anchor_list (`List[List[torch.Tensor]]`):
                Multi level anchors of each image. The outer list indicates images, and the inner list corresponds to
                feature levels of the image. Each element of the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (`List[List[torch.Tensor]]`):
                Multi level valid flags of each image. The outer list indicates images, and the inner list corresponds
                to feature levels of the image. Each element of the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (`List[torch.Tensor]`):
                Ground truth bboxes of each image.
            img_metas (`List[dict]`):
                Meta info of each image.
            gt_bboxes_ignore_list (`List[torch.Tensor]`):
                Ground truth bboxes to be ignored.
            gt_labels_list (`List[torch.Tensor]`):
                Ground truth labels of each box.
            label_channels (`int`, *optional*, defaults to 1):
                Number of channels in the ground truth labels.
            unmap_outputs (`bool`, *optional*, defaults to `True`):
                Whether to map outputs back to the original set of anchors.

        Returns:
            `tuple` comprising various elements:
                - labels_list (`List[torch.Tensor]`):
                    Labels of each level.
                - label_weights_list (`List[torch.Tensor]`):
                    Label weights of each level.
                - bbox_targets_list (`List[torch.Tensor]`):
                    Bbox targets of each level.
                - bbox_weights_list (`List[torch.Tensor]`):
                    Bbox weights of each level.
                - num_total_pos (`int`):
                    Number of positive samples in all images.
                - num_total_neg (`int`):
                    Number of negative samples in all images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined to properties at each feature map (i.e.
                having HxW dimension). The results will be concatenated after the end.
        """
        num_imgs = len(img_metas)
        if not (len(anchor_list) == len(valid_flag_list) == num_imgs):
            raise ValueError("Must have as many anchors and flags as images")

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            if len(anchor_list[i]) != len(valid_flag_list[i]):
                raise ValueError("Inconsistent num of anchors and flags")
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        pos_inds_list = []
        neg_inds_list = []
        sampling_results_list = []

        for flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta in zip(
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
        ):
            (
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
                pos_inds,
                neg_inds,
                sampling_result,
            ) = self._get_targets_single(
                flat_anchors,
                valid_flags,
                gt_bboxes,
                gt_bboxes_ignore,
                gt_labels,
                img_meta,
                label_channels=label_channels,
                unmap_outputs=unmap_outputs,
            )
            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_bbox_weights.append(bbox_weights)
            pos_inds_list.append(pos_inds)
            neg_inds_list.append(neg_inds)
            sampling_results_list.append(sampling_result)

        # rest_results = []  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list,)

        # for i, r in enumerate(rest_results):  # user-added return values
        #     rest_results[i] = images_to_levels(r, num_level_anchors)

        return res

    def loss_single_scale_level(
        self, cls_score, bbox_pred, anchors, labels, label_weights, bbox_targets, bbox_weights, num_total_samples
    ):
        """Compute loss of a single scale level.

        Args:
            cls_score (`torch.Tensor`):
                Box scores for each scale level. Has shape (N, num_anchors * num_classes, height, width).
            bbox_pred (`torch.Tensor`):
                Box energies / deltas for each scale. Has shape (N, num_anchors * 4, height, width).
            anchors (`torch.Tensor`):
                Box reference for each scale level with shape (N, num_total_anchors, 4).
            labels (`torch.Tensor`):
                Labels of each anchors with shape (N, num_total_anchors).
            label_weights (`torch.Tensor`):
                Label weights of each anchor with shape (N, num_total_anchors)
            bbox_targets (`torch.Tensor`):
                BBox regression targets of each anchor weight shape (N, num_total_anchors, 4).
            bbox_weights (`torch.Tensor`):
                BBox regression loss weights of each anchor with shape (N, num_total_anchors, 4).
            num_total_samples (`int`):
                If sampling, number of total samples equal to the number of total anchors; Otherwise, it is the number
                of positive anchors.

        Returns:
            `dict[str, torch.Tensor]`: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (`List[torch.Tensor]):
                Box scores for each scale level. Has shape (N, num_anchors * num_classes, height, width).
            bbox_preds (`List[torch.Tensor]`):
                Box energies / deltas for each scale level with shape (N, num_anchors * 4, height, width).
            gt_bboxes (`List[torch.Tensor]`):
                Ground truth bboxes for each image with shape (num_gts, 4) in [top_left_x, top_left_y, bottom_right_x,
                bottom_right_y] format.
            gt_labels (`List[torch.Tensor]`):
                Class indices corresponding to each box.
            img_metas (`List[dict]`):
                Meta information of each image, e.g., image size, scaling factor, etc.
            gt_bboxes_ignore (None | `List[torch.Tensor]`, *optional*):
                Specify which bounding boxes can be ignored when computing the loss.

        Returns:
            `dict[str, torch.Tensor]`: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        if len(featmap_sizes) != self.prior_generator.num_levels:
            raise ValueError(
                f"featmap_sizes should have {self.prior_generator.num_levels} "
                f"elements, but got {len(featmap_sizes)}"
            )

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,  # RPN head sets gt_labels = None
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls = []
        losses_bbox = []

        for cls_score, bbox_pred, anchors, labels, label_weights, bbox_targets, bbox_weights in zip(
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
        ):
            loss_cls, loss_bbox = self.loss_single_scale_level(
                cls_score,
                bbox_pred,
                anchors,
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
                num_total_samples=num_total_samples,
            )
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)

        return {"loss_cls": losses_cls, "loss_bbox": losses_bbox}

    def get_bboxes(
        self,
        cls_scores,
        bbox_preds,
        score_factors=None,
        img_metas=None,
        cfg=None,
        rescale=False,
        with_nms=True,
        **kwargs,
    ):
        """Transform network outputs of a batch into bbox results.
        Note: When score_factors is not None, the cls_scores are usually multiplied by it then obtain the real score
        used in NMS, such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (`List[torch.Tensor]`):
                Classification scores for all scale levels, each is a 4D-tensor, has shape (batch_size, num_priors *
                num_classes, height, width).
            bbox_preds (`List[torch.Tensor]`):
                Box energies / deltas for all scale levels, each is a 4D-tensor, has shape (batch_size, num_priors * 4,
                height, width).
            score_factors (`List[torch.Tensor]`, *optional*):
                Score factor for all scale level, each is a 4D-tensor, has shape (batch_size, num_priors * 1, height,
                width).
            img_metas (`List[dict]`, *optional*):
                Image meta info. Default None.
            cfg (`mmcv.Config`, *optional*):
                Test / postprocessing configuration. If None, `test_cfg` is used.
            rescale (`bool`, *optional*, defaults to `False`):
                If True, return boxes in original image space.
            with_nms (`bool`, *optional*, defaults to `True`):
                If `True`, do NMS (non-maximum suppression) before return boxes.

        Returns:
            `List[List[torch.Tensor, torch.Tensor]]`: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns are bounding box positions (top_left_x,
                top_left_y, bottom_right_x, bottom_right_y) and the 5-th column is a score between 0 and 1. The second
                item is a (n,) tensor where each item is the predicted class label of the corresponding box.
        """
        if len(cls_scores) != len(bbox_preds):
            raise ValueError(
                f"The length of cls_scores and bbox_preds should be equal, but got {len(cls_scores)} and {len(bbox_preds)}"
            )

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        multilevel_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_multilevel(cls_scores, img_id)
            bbox_pred_list = select_single_multilevel(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_multilevel(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                score_factor_list,
                multilevel_priors,
                img_meta,
                cfg,
                rescale,
                with_nms,
                **kwargs,
            )
            result_list.append(results)
        return result_list

    def _get_bboxes_single(
        self,
        cls_score_list,
        bbox_pred_list,
        score_factor_list,
        multilevel_anchors,
        img_meta,
        cfg,
        rescale=False,
        with_nms=True,
        **kwargs,
    ):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (`List[torch.Tensor]`):
                Box scores from all scale levels of a single image, each item has shape (num_anchors * num_classes,
                height, width).
            bbox_pred_list (`List[torch.Tensor]`):
                Box energies / deltas from all scale levels of a single image, each item has shape (num_anchors * 4,
                height, width).
            score_factor_list (`List[torch.Tensor]`):
                Score factor from all scale levels of a single image. RPN head does not need this value.
            multilevel_anchors (`List[torch.Tensor]`):
                Anchors of all scale level each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (`mmcv.Config`, *optional*):
                Test / postprocessing configuration. If None, `test_cfg` is used.
            rescale (bool, *optional*, defaults to `False`):
                If `True`, return boxes in original image space.
            with_nms (`bool`, *optional*, defaults to `True`):
                If True, do NMS (non-maximum suppression) before returning boxes.

        Returns:
            `torch.Tensor`:
                Labeled boxes in shape (n, 5), where the first 4 columns are bounding box positions (top_left_x,
                top_left_y, bottom_right_x, bottom_right_y) and the 5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta["img_shape"]

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        multilevel_scores = []
        multilevel_bbox_preds = []
        multilevel_valid_anchors = []
        nms_pre = cfg.get("nms_pre", -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            if rpn_cls_score.size()[-2:] != rpn_bbox_pred.size()[-2:]:
                raise ValueError(
                    f"Last 2 dimensions of cls ({rpn_cls_score.shape}) and bbox ({rpn_bbox_pred.shape}) "
                    "predictions should match!"
                )
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = multilevel_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            multilevel_scores.append(scores)
            multilevel_bbox_preds.append(rpn_bbox_pred)
            multilevel_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), level_idx, dtype=torch.long))

        return self._bbox_post_process(
            multilevel_scores, multilevel_bbox_preds, multilevel_valid_anchors, level_ids, cfg, img_shape
        )

    def _bbox_post_process(
        self, multilevel_scores, multilevel_bboxes, multilevel_valid_anchors, level_ids, cfg, img_shape, **kwargs
    ):
        """Bounding box post-processing method.


        Do the nms operation for bboxes in same level.

        Args:
            multilevel_scores (`List[torch.Tensor]`):
                Box scores from all scale levels of a single image, each item has shape (num_bboxes, ).
            multilevel_bboxes (`List[torch.Tensor]`):
                Decoded bboxes from all scale levels of a single image, each item has shape (num_bboxes, 4).
            multilevel_valid_anchors (`List[torch.Tensor]`):
                Anchors of all scale level each item has shape (num_bboxes, 4).
            level_ids (`List[torch.Tensor]`):
                Indexes from all scale levels of a single image, each item has shape (num_bboxes, ).
            cfg (`mmcv.Config`):
                Test / postprocessing configuration. If None, `self.test_cfg` is used.
            img_shape (`Tuple[int]`):
                The shape of model's input image.

        Returns:
            Tensor:
                Labeled boxes in shape (n, 5), where the first 4 columns are bounding box positions (top_left_x,
                top_left_y, bottom_right_x, bottom_right_y) and the 5-th column is a score between 0 and 1.
        """
        scores = torch.cat(multilevel_scores)
        anchors = torch.cat(multilevel_valid_anchors)
        rpn_bbox_pred = torch.cat(multilevel_bboxes)
        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg["min_bbox_size"] >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg["min_bbox_size"]) & (h > cfg["min_bbox_size"])
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg["nms"])
        else:
            return proposals.new_zeros(0, 5)

        return dets[: cfg["max_per_img"]]


class RoIAlign(nn.Module):
    """
    Wrapper around torchvision's roi_align op.

    Based on:
    https://github.com/open-mmlab/mmcv/blob/d71d067da19d71d79e7b4d7ae967891c7bb00c05/mmcv/ops/roi_align.py#L136
    """

    def __init__(
        self,
        output_size: tuple,
        spatial_scale: float = 1.0,
        sampling_ratio: int = 0,
        pool_mode: str = "avg",
        aligned: bool = True,
    ):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Images.
            rois (`torch.Tensor` of shape `(num_rois, 5)`):
                Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        return torchvision.ops.roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned
        )

    def __repr__(self):
        s = self.__class__.__name__
        s += f"(output_size={self.output_size}, "
        s += f"spatial_scale={self.spatial_scale}, "
        s += f"sampling_ratio={self.sampling_ratio}, "
        s += f"pool_mode={self.pool_mode}, "
        s += f"aligned={self.aligned}, "
        return s


class MaskRCNNSingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map."""

    def __init__(self, roi_layer, out_channels, featmap_strides, finest_scale=56):
        super().__init__()

        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.
        Args:
            layer_cfg (`dict`):
                Dictionary to construct and config RoI layer operation. Options are modules under `mmcv/ops` such as
                `RoIAlign`.
            featmap_strides (`List[int]`):
                The stride of input feature map w.r.t to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature coordinate system.

        Returns:
            `nn.ModuleList`: The RoI extractor modules for each level feature map.
        """

        cfg = layer_cfg.copy()
        cfg.pop("type")
        # we use the RoIAlign op of torchvision inplace of the one in mmcv: https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/roi_align.py
        layer_cls = RoIAlign
        roi_layers = nn.ModuleList([layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map RoI's to corresponding feature levels by scales.
        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (`torch.Tensor`):
                Input RoIs, shape (k, 5).
            num_levels (`int`):
                Total number of levels.

        Returns:
            `torch.Tensor`: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois, roi_scale_factor=None):
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1)
                # select target level rois and reset the rest rois to zero.
                rois_i = rois.clone().detach()
                rois_i *= mask
                mask_exp = mask.expand(*expand_dims).reshape(roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois_i)
                roi_feats_t *= mask_exp
                roi_feats += roi_feats_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.0 + feats[i].sum() * 0.0

        return roi_feats


class MaskRCNNShared2FCBBoxHead(nn.Module):
    """
    A bounding box head with 2 shared fully-connected (fc) layers.

    This class is a simplified version of
    https://github.com/open-mmlab/mmdetection/blob/ca11860f4f3c3ca2ce8340e2686eeaec05b29111/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L11.

    Args:
        config (`MaskRCNNConfig`):
            Model configuration.
        num_branch_fcs (`int`, *optional*, defaults to `2`):
            Number of fully-connected layers in the branch.
        reg_class_agnostic (`bool`, *optional*, defaults to `False`):
            Whether the regression is class agnostic.
        reg_decoded_bbox (`bool`, *optional*, defaults to `False`):
            Whether to apply the regression loss (e.g. `IouLoss`, `GIouLoss`, `DIouLoss`)directly on the decoded
            bounding boxes.
        custom_activation (`bool`, *optional*, defaults to `False`):
            Whether to use a custom activation function.
    """

    def __init__(
        self,
        config,
        num_branch_fcs=2,
        reg_class_agnostic=False,
        reg_decoded_bbox=False,
        custom_activation=False,
    ):
        super().__init__()

        self.roi_feat_size = _pair(config.bbox_head_roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.fc_out_channels = config.bbox_head_fc_out_channels

        last_layer_dim = config.bbox_head_in_channels
        last_layer_dim *= self.roi_feat_area

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        for i in range(num_branch_fcs):
            fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        self.shared_fcs = branch_fcs

        self.relu = nn.ReLU(inplace=True)

        # add class and regression fully-connected layers
        self.num_classes = config.num_labels
        self.fc_cls = nn.Linear(in_features=self.fc_out_channels, out_features=config.num_labels + 1)
        self.fc_reg = nn.Linear(in_features=self.fc_out_channels, out_features=config.num_labels * 4)

        self.bbox_coder = MaskRCNNDeltaXYWHBBoxCoder(
            target_means=config.bbox_head_bbox_coder_target_means, target_stds=config.bbox_head_bbox_coder_target_stds
        )

        # TODO make these configurable in the future
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.custom_activation = custom_activation

        # losses
        # based on config:
        self.loss_cls = CrossEntropyLoss(
            use_sigmoid=False
        )  # this corresponds to dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.loss_bbox = L1Loss()  # this corresponds to dict(type='L1Loss', loss_weight=1.0)

    def forward(self, hidden_states):
        # shared part
        hidden_states = hidden_states.flatten(1)
        for fc in self.shared_fcs:
            hidden_states = self.relu(fc(hidden_states))

        # separate branches
        cls_score = self.fc_cls(hidden_states)
        bbox_pred = self.fc_reg(hidden_states)

        return cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image according to the sampling results.

        Args:
            pos_bboxes (`torch.Tensor`):
                Contains all the positive boxes, has shape (num_pos, 4), the last dimension 4 represents [top_left_x,
                top_left_y, bottom_right_x, bottom_right_y].
            neg_bboxes (`torch.Tensor`):
                Contains all the negative boxes, has shape (num_neg, 4), the last dimension 4 represents [top_left_x,
                top_left_y, bottom_right_x, bottom_right_y].
            pos_gt_bboxes (`torch.Tensor`):
                Contains ground truth boxes for all positive samples, has shape (num_pos, 4), the last dimension 4
                represents [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
            pos_gt_labels (`torch.Tensor`):
                Contains ground truth labels for all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`):
                Training configuration of R-CNN.

        Returns:
            `Tuple[torch.Tensor]`: Ground truth for proposals in a single image. Containing the following tensors:
                - labels (`torch.Tensor`):
                    Ground truth labels for all proposals, has shape (num_proposals,).
                - label_weights (`torch.Tensor`):
                    Labels_weights for all proposals, has shape (num_proposals,).
                - bbox_targets (`torch.Tensor`):
                    Regression target for all proposals, has shape (num_proposals, 4), the last dimension 4 represents
                    [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
                - bbox_weights (`torch.Tensor`):
                    Regression weights for all proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg["pos_weight"] <= 0 else cfg["pos_weight"]
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, concat=True):
        """Calculate the ground truth for all samples in a batch according to the sampling_results. Almost the same as
        the implementation in bbox_head, we passed additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (`List[SamplingResults]`):
                Assign results of all images in a batch after sampling.
            gt_bboxes (`List[torch.Tensor]`):
                Ground truth boxes of all images in a batch, each tensor has shape (num_ground_truths, 4), the last
                dimension 4 represents [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
            gt_labels (`List[torch.Tensor]`):
                Ground truth labels of all images in a batch, each tensor has shape (num_ground_truths,).
            rcnn_train_cfg (`ConfigDict`):
                Training configuration of R-CNN.
            concat (`bool`, *optional*, defaults to `True`):
                Whether to concatenate the results of all the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image. Containing the following list of Tensors:
                - labels (`List[torch.Tensor]`, `torch.Tensor`):
                    Ground truth for all proposals in a batch, each tensor in list has shape (num_proposals,) when
                    `concat=False`, otherwise just a single tensor with shape (num_all_proposals,).
                - label_weights (`List[torch.Tensor]`):
                    Label weights for all proposals in a batch, each tensor in list has shape (num_proposals,) when
                    `concat=False`, otherwise just a single tensor with shape (num_all_proposals,).
                - bbox_targets (`List[torch.Tensor]` or `torch.Tensor`):
                    Regression targets for all proposals in a batch, each tensor in list has shape (num_proposals, 4)
                    when `concat=False`, otherwise just a single tensor with shape (num_all_proposals, 4), the last
                    dimension 4 represents [top_left_x, top_left_y, bottom_right_x, bottom_right_y].
                - bbox_weights (`List[torch.Tensor]` or `torch.Tensor`):
                    Regression weights for all proposals in a batch, each tensor in list has shape (num_proposals, 4)
                    when `concat=False`, otherwise just a single tensor with shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]

        labels = []
        label_weights = []
        bbox_targets = []
        bbox_weights = []

        for pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels in zip(
            pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list
        ):
            labels_, label_weights_, bbox_targets_, bbox_weights_ = self._get_target_single(
                pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg=rcnn_train_cfg
            )
            labels.append(labels_)
            label_weights.append(label_weights_)
            bbox_targets.append(bbox_targets_)
            bbox_weights.append(bbox_weights_)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def loss(
        self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None
    ):
        losses = {}
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override
                )
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses["loss_cls"] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses["acc"] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                        pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]
                    ]
                losses["loss_bbox"] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override,
                )
            else:
                losses["loss_bbox"] = bbox_pred[pos_inds].sum()
        return losses


class MaskRCNNFCNMaskHead(nn.Module):
    """
    Mask head.
    """

    def __init__(self, config, conv_kernel_size=3, scale_factor=2):
        super().__init__()

        self.num_labels = config.num_labels
        self.class_agnostic = False

        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        conv_out_channels = config.mask_head_conv_out_channels
        for i in range(config.mask_head_num_convs):
            in_channels = config.mask_head_in_channels if i == 0 else conv_out_channels
            padding = (conv_kernel_size - 1) // 2
            self.convs.append(nn.Conv2d(in_channels, conv_out_channels, conv_kernel_size, padding=padding))
            self.activations.append(nn.ReLU(inplace=True))
        self.upsample = nn.ConvTranspose2d(
            in_channels=conv_out_channels,
            out_channels=conv_out_channels,
            kernel_size=scale_factor,
            stride=scale_factor,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_logits = nn.Conv2d(conv_out_channels, self.num_labels, 1)

        self.loss_mask = CrossEntropyLoss(use_mask=True, loss_weight=1.0)

    def forward(self, hidden_state):
        for conv, activation in zip(self.convs, self.activations):
            hidden_state = conv(hidden_state)
            hidden_state = activation(hidden_state)

        hidden_state = self.upsample(hidden_state)
        hidden_state = self.relu(hidden_state)
        mask_pred = self.conv_logits(hidden_state)
        return mask_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, mask_pred, mask_targets, labels):
        loss = {}
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets, torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss["loss_mask"] = loss_mask
        return loss


class MaskRCNNRoIHead(nn.Module):
    """
    Mask R-CNN standard Region of Interest (RoI) head including one bbox head and one mask head.

    This head takes the proposals of the RPN + features of the backbone as input and transforms them into a set of
    bounding boxes + classes.
    """

    def __init__(self, config):
        super().__init__()

        self.train_cfg = config.rcnn_train_cfg
        self.test_cfg = config.rcnn_test_cfg
        self.bbox_roi_extractor = MaskRCNNSingleRoIExtractor(
            roi_layer=config.bbox_roi_extractor_roi_layer,
            out_channels=config.bbox_roi_extractor_out_channels,
            featmap_strides=config.bbox_roi_extractor_featmap_strides,
        )
        self.bbox_head = MaskRCNNShared2FCBBoxHead(config)

        self.mask_roi_extractor = MaskRCNNSingleRoIExtractor(
            roi_layer=config.mask_roi_extractor_roi_layer,
            out_channels=config.mask_roi_extractor_out_channels,
            featmap_strides=config.mask_roi_extractor_featmap_strides,
        )
        self.mask_head = MaskRCNNFCNMaskHead(config)

        # assigner
        self.bbox_assigner = MaskRCNNMaxIoUAssigner(
            pos_iou_thr=config.rcnn_assigner_pos_iou_thr,
            neg_iou_thr=config.rcnn_assigner_neg_iou_thr,
            min_pos_iou=config.rcnn_assigner_min_pos_iou,
            match_low_quality=config.rcnn_assigner_match_low_quality,
            ignore_iof_thr=config.rcnn_assigner_ignore_iof_thr,
        )
        # sampler
        self.bbox_sampler = MaskRCNNRandomSampler(
            num=config.rcnn_sampler_num,
            pos_fraction=config.rcnn_sampler_pos_fraction,
            neg_pos_ub=config.rcnn_sampler_neg_pos_ub,
            add_gt_as_proposals=config.rcnn_sampler_add_gt_as_proposals,
        )

        # TODO remove this hardcoded variable
        self.share_roi_extractor = False

    @property
    def with_bbox(self):
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, "mask_head") and self.mask_head is not None

    def _bbox_forward(self, feature_maps, rois):
        """Box head forward function used in both training and testing.

        Args:
            feature_maps (`List[torch.FloatTensor]`):
                Multi-scale feature maps coming from the FPN.
            rois (`torch.FloatTensor`):
                RoIs that are used as input to the box head.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_features = self.bbox_roi_extractor(feature_maps[: self.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_features)

        bbox_results = {"cls_score": cls_score, "bbox_pred": bbox_pred, "bbox_feats": bbox_features}

        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results["cls_score"], bbox_results["bbox_pred"], rois, *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def forward_test_bboxes(self, feature_maps, proposals, rcnn_test_cfg):
        """Test only detected bboxes without augmentation.

        Args:
            feature_maps (`Tuple[torch.Tensor]`):
                Feature maps of all scale levels.
            img_metas (`List[Dict]`):
                Image meta info.
            proposals (`List[torch.Tensor]`):
                Region proposals.
            rcnn_test_cfg (`ConfigDict`):
                Test configuration of R-CNN.

        Returns:
            `Tuple[List[torch.Tensor]`, `List[torch.Tensor]`]: The first list contains the boxes of the corresponding
            image
                in a batch, each tensor has the shape (num_boxes, 5) and last dimension 5 represent (top_left_x,
                top_left_y, bottom_right_x, bottom_right_y, score). Each tensor in the second list contains the labels
                with shape (num_boxes, ). The length of both lists should be equal to batch_size.
        """
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros((0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(feature_maps, rois)
        logits = bbox_results["cls_score"]
        pred_boxes = bbox_results["bbox_pred"]

        return rois, proposals, logits, pred_boxes

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        if not ((rois is not None) ^ (pos_inds is not None and bbox_feats is not None)):
            raise ValueError("Either rois or (pos_inds and bbox_feats) should be specified")
        if rois is not None:
            mask_feats = self.mask_roi_extractor(x[: self.mask_roi_extractor.num_inputs], rois)
            # if self.with_shared_head:
            #     mask_feats = self.shared_head(mask_feats)
        else:
            if bbox_feats is None:
                raise ValueError("bbox_feats must be specified when pos_inds is specified")
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)

        mask_results = {"mask_pred": mask_pred, "mask_feats": mask_feats}
        return mask_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(torch.ones(res.pos_bboxes.shape[0], device=device, dtype=torch.uint8))
                pos_inds.append(torch.zeros(res.neg_bboxes.shape[0], device=device, dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks, self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results["mask_pred"], mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def forward_test_mask(self, hidden_states, scale_factors, det_bboxes, rescale=True):
        """Simple test for mask head without augmentation."""
        # scale_factors = image shapes of images in the batch

        # if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
        #     segm_results = [[[] for _ in range(self.mask_head.num_classes)] for _ in range(num_imgs)]
        # else:

        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        if rescale:
            scale_factors = [scale_factor.to(det_bboxes[0].device) for scale_factor in scale_factors]
        _bboxes = [
            # det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
            det_bboxes[i] * scale_factors[i] if rescale else det_bboxes[i]
            for i in range(len(det_bboxes))
        ]
        mask_rois = bbox2roi(_bboxes)
        mask_results = self._mask_forward(hidden_states, mask_rois)
        mask_pred = mask_results["mask_pred"]

        return mask_pred

    def forward_train(
        self,
        feature_maps,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        **kwargs,
    ):
        """
        Args:
            feature_maps (`List[torch.Tensor]`):
                List of multi-level image features.
            img_metas (`List[Dict]`):
                List of image info dict where each dict. Has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (`List[torch.Tensor]`):
                List of region proposals.
            gt_bboxes (`List[torch.Tensor]`):
                Ground truth bboxes for each image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (`List[Tensor]`):
                Class indices corresponding to each box.
            gt_bboxes_ignore (`List[torch.Tensor]`, *optional*):
                Specify which bounding boxes can be ignored when computing the loss.
            gt_masks (`List[torch.Tensor]`, *optional*) :
                True segmentation masks for each box used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in feature_maps],
                )
                sampling_results.append(sampling_result)

        losses = {}
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(feature_maps, sampling_results, gt_bboxes, gt_labels)

            losses.update(bbox_results["loss_bbox"])

        if self.with_mask:
            mask_results = self._mask_forward_train(
                feature_maps, sampling_results, bbox_results["bbox_feats"], gt_masks, img_metas
            )
            losses.update(mask_results["loss_mask"])

        return losses

    def forward_test(self, hidden_states, proposal_list):
        """Test without augmentation (originally called `simple_test`).

        Source:
        https://github.com/open-mmlab/mmdetection/blob/ca11860f4f3c3ca2ce8340e2686eeaec05b29111/mmdet/models/roi_heads/standard_roi_head.py#L223.

        Args:
            hidden_states (`Tuple[torch.Tensor]`):
                Features from upstream network. Each has shape `(batch_size, num_channels, height, width)`.
            proposal_list (`List[torch.Tensor`]):
                Proposals from RPN head. Each has shape (num_proposals, 5), last dimension 5 represent (top_left_x,
                top_left_y, bottom_right_x, bottom_right_y, score).

        Returns:
            `List[List[np.ndarray]]` or `List[Tuple]`: When no mask branch, it is bbox results of each image and
            classes with type `List[List[np.ndarray]]`. The outer list corresponds to each image. The inner list
            corresponds to each class. When the model has a mask branch, it contains the bbox results and mask results.
            The outer list corresponds to each image, and first element of tuple is bbox results, second element is
            mask results.
        """
        rois, proposals, logits, pred_boxes = self.forward_test_bboxes(hidden_states, proposal_list, self.test_cfg)

        return rois, proposals, logits, pred_boxes


class MaskRCNNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MaskRCNNConfig
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MaskRCNNForObjectDetection):
            module.gradient_checkpointing = value


MASK_RCNN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MaskRCNNConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MASK_RCNN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """
    Mask R-CNN Model (consisting of a backbone, region-proposal network (RPN) and RoI head) for object detection and
    instance segmentation tasks.
    """,
    MASK_RCNN_START_DOCSTRING,
)
class MaskRCNNForObjectDetection(MaskRCNNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.backbone = AutoBackbone.from_config(config.backbone_config)
        self.neck = MaskRCNNFPN(config, self.backbone.channels)
        self.rpn_head = MaskRCNNRPN(config)
        self.roi_head = MaskRCNNRoIHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Source:
        https://github.com/open-mmlab/mmdetection/blob/ff9bc39913cb3ff5dde79d3933add7dc2561bab7/mmdet/models/detectors/base.py#L176

        Args:
            losses (dict):
                Raw output of the network, which usually contain losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \ all the variables to be sent to the
                logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        return loss

    @add_start_docstrings_to_model_forward(MASK_RCNN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskRCNNModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskRCNNModelOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the loss. List of dicts, each dictionary containing at least the following 4 keys:
            'class_labels', 'boxes' (the class labels, bounding boxes and masks of an image in the batch respectively).
            The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes in the
            image,)`, the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)` and the
            `masks` a `torch.FloatTensor` of shape .

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, MaskRCNNForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("nielsr/maskrcnn-convnext-tiny")
        >>> model = MaskRCNNForObjectDetection.from_pretrained("nielsr/maskrcnn-convnext-tiny")

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> # model predicts bounding boxes, binary masks and corresponding class labels
        >>> logits = outputs.logits
        >>> pred_boxes = outputs.pred_boxes
        ```
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # TODO: remove img_metas
        # and figure out where `scale_factor` and `ori_shape` come from (probably test_pipeline)
        if labels is not None:
            img_metas = [
                {"img_shape": (3, *target["size"].tolist()), "pad_shape": (3, *target["size"].tolist())}
                for target in labels
            ]
        else:
            img_metas = [{"img_shape": pixel_values.shape[1:]} for _ in range(pixel_values.shape[0])]

        # send pixel_values through backbone
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # the FPN outputs feature maps at 5 different scales
        feature_maps = outputs.feature_maps if return_dict else outputs[0]
        hidden_states = self.neck(feature_maps)

        # next, RPN computes a tuple of (class, bounding box) features for each of the 5 feature maps
        # rpn_outs[0] are the class features for each of the feature maps
        # rpn_outs[1] are the bounding box features for each of the feature maps

        loss, loss_dict = None, None
        rois, proposals, logits, pred_boxes = None, None, None, None
        if labels is not None:
            loss_dict = {}
            rpn_outputs = self.rpn_head(
                hidden_states,
                img_metas,
                gt_bboxes=[target["boxes"] for target in labels],
                gt_labels=None,  # one explicitly sets them to None in TwoStageDetector
                gt_bboxes_ignore=None,  # TODO remove this
                proposal_cfg=self.config.rpn_proposal,
            )
            loss_dict.update(rpn_outputs.losses)
            # TODO: check for kwargs forwarded here
            roi_losses = self.roi_head.forward_train(
                hidden_states,
                img_metas,
                rpn_outputs.proposal_list,
                gt_bboxes=[target["boxes"] for target in labels],
                gt_labels=[target["class_labels"] for target in labels],
                gt_bboxes_ignore=None,
                gt_masks=[target["masks"] for target in labels],
            )
            loss_dict.update(roi_losses)
            # compute final loss
            loss = self.parse_losses(loss_dict)
        else:
            rpn_outputs = self.rpn_head(hidden_states, img_metas)
            rois, proposals, logits, pred_boxes = self.roi_head.forward_test(hidden_states, rpn_outputs.proposal_list)

        if not return_dict:
            output = (logits, pred_boxes, rois, proposals, hidden_states) + outputs[2:]
            return ((loss, loss_dict) + output) if loss is not None else output

        return MaskRCNNModelOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            rois=rois,
            proposals=proposals,
            fpn_hidden_states=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
