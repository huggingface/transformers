# coding=utf-8
# Copyright 2022 Meta Platforms, Inc.,
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
""" PyTorch ConvNextMaskRCNN model."""


import copy
import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
import torchvision
from torch import nn
from torch.nn.modules.utils import _pair

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_convnext_maskrcnn import ConvNextMaskRCNNConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ConvNextMaskRCNNConfig"
_FEAT_EXTRACTOR_FOR_DOC = "ConvNextFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/convnext-tiny-maskrcnn"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]


CONVNEXTMASKRCNN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnext-tiny-maskrcnn",
    # See all ConvNextMaskRCNN models at https://huggingface.co/models?filter=convnext_maskrcnn
]


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the `func` to multiple inputs and map the multiple outputs of the `func` into different
        list. Each list contains the same type of outputs corresponding to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient needs to be detached during the training
    of the two-stage model. E.g Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id].detach() for i in range(num_levels)]
    else:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list


def bbox2delta(proposals, gt, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
    """Compute deltas of proposals w.r.t. gt.
    Args:
    We usually compute the deltas of x, y, w, h of proposals w.r.t ground truth bboxes to get regression target. This
    is the inverse function of [`delta2bbox`].
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4) gt (Tensor): Gt bboxes to be used as base, shape
        (N, ..., 4) means (Sequence[float]): Denormalizing means for delta coordinates stds (Sequence[float]):
        Denormalizing standard deviation for delta
            coordinates
    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

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
    Args:
    Typically the rois are anchor or proposed bounding boxes and the deltas are network outputs used to shift/scale
    those boxes. This is the inverse function of [`bbox2delta`].
        rois (Tensor): Boxes to be transformed. Has shape (N, 4). deltas (Tensor): Encoded offsets relative to each
        roi.
            Has shape (N, num_classes * 4) or (N, 4). Note N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to avoid being too far away from the center of
            the anchor. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    Returns:
        Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
           represent tl_x, tl_y, br_x, br_y.
    References:
        .. [1] https://arxiv.org/abs/1311.2524
    Example:
        >>> rois = torch.Tensor([[ 0., 0., 1., 1.], >>> [ 0., 0., 1., 1.], >>> [ 0., 0., 1., 1.], >>> [ 5., 5., 5.,
        5.]]) >>> deltas = torch.Tensor([[ 0., 0., 0., 0.], >>> [ 1., 1., 1., 1.], >>> [ 0., 0., 2., -1.], >>> [ 0.7,
        -1.9, -0.5, 0.3]]) >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3)) tensor([[0.0000, 0.0000, 1.0000,
        1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591], [0.0000, 0.3161, 4.1945, 0.6839], [5.0000, 5.0000, 5.0000, 5.0000]])
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
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes


array_like_type = Union[torch.Tensor, np.ndarray]


def nms(
    boxes: array_like_type,
    scores: array_like_type,
    iou_threshold: float,
    offset: int = 0,
    score_threshold: float = 0,
    max_num: int = -1,
) -> Tuple[array_like_type, array_like_type]:
    """Dispatch to either CPU or GPU NMS implementations.
    Arguments:
    The input can be either torch tensor or numpy array. GPU NMS will be used if the input is gpu tensor, otherwise CPU:
    NMS will be used. The returned type will always be the same as inputs.
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4). scores (torch.Tensor or np.ndarray): scores in shape
        (N, ). iou_threshold (float): IoU threshold for NMS. offset (int, 0 or 1): boxes' width or height is (x2 - x1 +
        offset). score_threshold (float): score threshold for NMS. max_num (int): maximum number of boxes after NMS.
    Returns:
        tuple: kept dets (boxes and scores) and indice, which always have the same data type as the input.
    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9], >>> [49.3, 32.9, 51.0, 35.3], >>> [49.2, 31.8, 51.0, 35.4], >>>
        [35.1, 11.5, 39.1, 15.7], >>> [35.6, 11.8, 39.3, 14.2], >>> [35.3, 11.5, 39.9, 14.5], >>> [35.2, 11.7, 39.7,
        15.7]], dtype=np.float32) >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6 >>> dets, inds = nms(boxes, scores, iou_threshold) >>> assert len(inds) == len(dets) ==
        3
    """
    assert isinstance(boxes, (torch.Tensor, np.ndarray))
    assert isinstance(scores, (torch.Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    # TODO use the NMS op of mmcv: https://github.com/open-mmlab/mmcv/blob/d71d067da19d71d79e7b4d7ae967891c7bb00c05/mmcv/ops/nms.py#L28
    # inds = NMSop.apply(boxes, scores, iou_threshold, offset, score_threshold, max_num)
    inds = torchvision.ops.nms(boxes, scores, iou_threshold)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    nms_cfg: Optional[Dict],
    class_agnostic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.
    Modified from [torchvision/ops/boxes.py#L39](https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39). In order to perform NMS independently per
    class, we add an offset to all the boxes. The offset is dependent only on the class idx, and is large enough so
    that boxes from different classes do not overlap. Note:
        In v1.4.1 and later, `batched_nms` supports skipping the NMS and returns sorted raw results when *nms_cfg* is
        None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs, shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when *nms_cfg*
            is None, otherwise it should specify nms type and other parameters like *iou_thr*. Possible keys includes
            the following.
            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the number of boxes is large (e.g., 200k). To
              avoid OOM during training, the users could set *split_thr* to a small value. If the number of boxes is
              greater than the threshold, it will perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes, regardless of the predicted class. Defaults to False.
    Returns:
        tuple: kept dets and indice.
        - boxes (Tensor): Bboxes with score after nms, has shape (num_bboxes, 5). last dimension 5 arrange as (x1, y1,
          x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]], dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop("type", "nms")
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop("split_thr", 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop("max_num", -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, return_inds=False):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.
    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError("[ONNX Error] Can not record NMS as it has not been executed this time")
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.
    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.
    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
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


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class
    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->ConvNextMaskRCNN
class ConvNextMaskRCNNDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->ConvNextMaskRCNN
class ConvNextMaskRCNNLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            x = x.permute(0, 2, 3, 1)
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
        return x


# Copied from transformers.models.convnext.modeling_convnext.ConvNextEmbeddings with ConvNext->ConvNextMaskRCNN
class ConvNextMaskRCNNEmbeddings(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        self.layernorm = ConvNextMaskRCNNLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayer with ConvNext->ConvNextMaskRCNN
class ConvNextMaskRCNNLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextMaskRCNNConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.layernorm = ConvNextMaskRCNNLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = ACT2FN[config.hidden_act]
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_parameter = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )
        self.drop_path = ConvNextMaskRCNNDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# Copied from transformers.models.convnext.modeling_convnext.ConvNextStage with ConvNext->ConvNextMaskRCNN, ConvNeXT->ConvNeXTMaskRCNN
class ConvNextMaskRCNNStage(nn.Module):
    """ConvNeXTMaskRCNN stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextMaskRCNNConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()

        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(
                ConvNextMaskRCNNLayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            )
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(
            *[ConvNextMaskRCNNLayer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states


class ConvNextMaskRCNNEncoder(nn.Module):
    """
    This class isn't copied from modeling_convnext.py as layernorms are added.
    """

    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        drop_path_rates = [
            x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextMaskRCNNStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

        self.layernorms = nn.ModuleList(
            [ConvNextMaskRCNNLayerNorm(i, data_format="channels_first") for i in config.hidden_sizes]
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for i, stage_module in enumerate(self.stages):
            if output_hidden_states:
                if i == 0:
                    # add initial embeddings
                    all_hidden_states = all_hidden_states + (hidden_states,)
                else:
                    all_hidden_states = all_hidden_states + (self.layernorms[i - 1](hidden_states),)

            hidden_states = stage_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (self.layernorms[-1](hidden_states),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class ConvNextMaskRCNNFPN(nn.Module):
    """
    Feature Pyramid Network (FPN).

    This is an implementation of [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144).
    """

    def __init__(self, config):
        super().__init__()

        self.num_outs = config.fpn_num_outputs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for hidden_size in config.hidden_sizes:
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


class ConvNextMaskRCNNAnchorGenerator(nn.Module):
    """
    Standard 2D anchor generator.

    Source: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py.
    """

    def __init__(self, config):
        super().__init__()

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in config.anchor_generator_strides]
        self.base_sizes = [min(stride) for stride in self.strides]
        assert len(self.base_sizes) == len(
            self.strides
        ), f"The number of strides should be the same as base sizes, got {self.strides} and {self.base_sizes}"

        # calculate scales of anchors
        self.scales = torch.Tensor(config.anchor_generator_scales)

        self.ratios = torch.Tensor(config.anchor_generator_ratios)
        # TODO support the following 3 attributes in the config
        self.scale_major = True
        self.centers = None
        self.center_offset = 0.0
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
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
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.
        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws, y_center + 0.5 * hs]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.
        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.
        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
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
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            dtype (`torch.dtype`): Dtype of priors.
                Default: torch.float32.
            device (str): The device where the anchors will be put on.
        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \ N = width * height * num_base_anchors, width and
                height \ are the sizes of the corresponding feature level, \ num_base_anchors is the number of anchors
                for that level.
        """
        assert self.num_levels == len(featmap_sizes)
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
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:*torch.dtype*): Date type of points.Defaults to
                `torch.float32`.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
        Returns:
            torch.Tensor: Anchors in the overall feature maps.
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


class ConvNextMaskRCNNDeltaXYWHBBoxCoder(nn.Module):
    """Delta XYWH BBox coder.
    Following the practice in [R-CNN](https://arxiv.org/abs/1311.2524), this coder encodes bbox (x1, y1, x2, y2) into
    delta (dx, dy, dw, dh) and decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from the original anchor's center. Only used by
            YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
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
        """Get box regression transformation deltas that can be used to
        Args:
        transform the `bboxes` into the `gt_bboxes`.
            bboxes (torch.Tensor): Source boxes, e.g., object proposals. gt_bboxes (torch.Tensor): Target of the
            transformation, e.g.,
                ground-truth boxes.
        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None, wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or (N, num_classes * 4) or (N, 4). Note N = num_anchors *
               W * H when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies (H, W, C) or (H, W). If bboxes shape is
               (B, N, 4), then the max_shape should be a Sequence[Sequence[int]] and the length of max_shape should
               also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.
        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)

        if pred_bboxes.ndim == 2 and not torch.onnx.is_in_onnx_export():
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
            if pred_bboxes.ndim == 3 and not torch.onnx.is_in_onnx_export():
                warnings.warn(
                    "DeprecationWarning: onnx_delta2bbox is deprecated "
                    "in the case of batch decoding and non-ONNX, "
                    "please use “delta2bbox” instead. In order to improve "
                    "the decoding speed, the batch function will no "
                    "longer be supported. "
                )
            raise NotImplementedError("ONNX is not yet supported")
            # decoded_bboxes = onnx_delta2bbox(
            #     bboxes,
            #     pred_bboxes,
            #     self.means,
            #     self.stds,
            #     max_shape,
            #     wh_ratio_clip,
            #     self.clip_border,
            #     self.add_ctr_clamp,
            #     self.ctr_clamp,
            # )

        return decoded_bboxes


class ConvNextMaskRCNNRPN(nn.Module):
    """
    Anchor-based Region Proposal Network (RPN). The RPN learns to convert anchors into region proposals, by

    1) classifying anchors as either positive/negative/neutral (based on IoU overlap with ground-truth boxes) 2) for
    the anchors classified as positive/negative, regressing the anchor box to the ground-truth box.

    RPN was originally proposed in [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
    Networks](https://arxiv.org/abs/1506.01497).
    """

    def __init__(self, config, num_classes=1):
        super().__init__()

        # anchor generator
        self.prior_generator = ConvNextMaskRCNNAnchorGenerator(config)
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.bbox_coder = ConvNextMaskRCNNDeltaXYWHBBoxCoder(
            target_means=config.rpn_bbox_coder_target_means, target_stds=config.rpn_bbox_coder_target_stds
        )

        # layers
        self.use_sigmoid_cls = config.rpn_loss_cls.get("use_sigmoid", False)
        # TODO: fix num_classes
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f"num_classes={num_classes} is too small")

        self.rpn_conv = nn.Conv2d(config.rpn_in_channels, config.rpn_feat_channels, kernel_size=3, padding=1)
        self.rpn_cls = nn.Conv2d(config.rpn_feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(config.rpn_feat_channels, self.num_base_priors * 4, 1)

        self.test_cfg = config.rpn_test_cfg

    def forward_single(self, hidden_state):
        """Forward feature map of a single scale level."""
        hidden_state = self.rpn_conv(hidden_state)
        hidden_state = nn.functional.relu(hidden_state, inplace=True)
        rpn_cls_score = self.rpn_cls(hidden_state)
        rpn_bbox_pred = self.rpn_reg(hidden_state)
        return rpn_cls_score, rpn_bbox_pred

    def forward(self, hidden_states):
        """Forward features from the upstream network.

        Args:
            hidden_states (tuple[torch.Tensor]): Features from the upstream network, each is a 4D-tensor.
        Returns:
            tuple: A tuple of classification scores and bbox prediction.
                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \ is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \ is num_base_priors * 4.
        """
        return multi_apply(self.forward_single, hidden_states)

    def get_bboxes(
        self,
        cls_scores,
        bbox_preds,
        score_factors=None,
        img_metas=None,
        cfg=None,
        rescale=False,
        with_nms=True,
        **kwargs
    ):
        """Transform network outputs of a batch into bbox results.
        Note: When score_factors is not None, the cls_scores are usually multiplied by it then obtain the real score
        used in NMS, such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used. Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns are bounding box positions (tl_x, tl_y,
                br_x, br_y) and the 5-th column is a score between 0 and 1. The second item is a (n,) tensor where each
                item is the predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                score_factor_list,
                mlvl_priors,
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
        mlvl_anchors,
        img_meta,
        cfg,
        rescale=False,
        with_nms=True,
        **kwargs
    ):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the 5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta["img_shape"]

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get("nms_pre", -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
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

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), level_idx, dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds, mlvl_valid_anchors, level_ids, cfg, img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors, level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.
        Args:
        Do the nms operation for bboxes in same level.
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.
        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the 5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
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
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
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


class ConvNextMaskRCNNSingleRoIExtractor(nn.Module):
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
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under `mmcv/ops` such as `RoIAlign`.
            featmap_strides (List[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI coordinate (original image coordinate
                system) to feature coordinate system.
        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy()
        cfg.pop("type")
        # layer_type = cfg.pop("type")
        # TODO: use the RoIAlign op of mmcv: https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/roi_align.py
        # assert hasattr(ops, layer_type)
        # layer_cls = getattr(ops, layer_type)
        layer_cls = RoIAlign
        roi_layers = nn.ModuleList([layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.
        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3
        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.
        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
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
        # TODO: remove this when parrots supports
        if torch.__version__ == "parrots":
            roi_feats.requires_grad = True

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

        print("Roi feats:", roi_feats.shape)
        print(roi_feats[0, 0, :3, :3])

        return roi_feats


class ConvNextMaskRNNShared2FCBBoxHead(nn.Module):
    """
    A bounding box head with 2 shared fully-connected (fc) layers.

    This class is a simplified version of
    https://github.com/open-mmlab/mmdetection/blob/ca11860f4f3c3ca2ce8340e2686eeaec05b29111/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L11.
    """

    def __init__(self, config, roi_feat_size=7, fc_out_channels=1024, num_branch_fcs=2):
        super().__init__()

        # TODO make init attributes configurable
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.fc_out_channels = fc_out_channels

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

        self.bbox_coder = ConvNextMaskRCNNDeltaXYWHBBoxCoder(
            target_means=config.bbox_head_bbox_coder_target_means, target_stds=config.bbox_head_bbox_coder_target_stds
        )

    def forward(self, hidden_states):
        # shared part
        hidden_states = hidden_states.flatten(1)
        for fc in self.shared_fcs:
            hidden_states = self.relu(fc(hidden_states))

        # separate branches
        cls_score = self.fc_cls(hidden_states)
        bbox_pred = self.fc_reg(hidden_states)

        return cls_score, bbox_pred

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # # some loss (Seesaw loss..) may have custom activation
        # if self.custom_cls_channels:
        #     scores = self.loss_cls.get_activation(cls_score)
        # else:
        scores = nn.functional.softmax(cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg["score_thr"], cfg["nms"], cfg["max_per_img"])

            return det_bboxes, det_labels


class ConvNextMaskRCNNRoIHead(nn.Module):
    """
    Mask R-CNN standard Region of Interest (RoI) head including one bbox head and one mask head.

    This head takes the proposals of the RPN + features of the backbone as input and transforms them into a set of
    bounding boxes + classes.
    """

    def __init__(self, config):
        super().__init__()

        self.test_cfg = config.rcnn_test_cfg
        self.bbox_roi_extractor = ConvNextMaskRCNNSingleRoIExtractor(
            roi_layer=config.bbox_roi_extractor_roi_layer,
            out_channels=config.bbox_roi_extractor_out_channels,
            featmap_strides=config.bbox_roi_extractor_featmap_strides,
        )
        self.bbox_head = ConvNextMaskRNNShared2FCBBoxHead(config)

        # TODO: mask roi extractor + mask head
        # self.mask_roi_extractor = ...
        # self.mask_head = ...

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(x[: self.bbox_roi_extractor.num_inputs], rois)
        # if self.with_shared_head:
        #     bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        print("cls_score:", cls_score.shape)
        print("First values of cls_score:", cls_score[:3, :3])
        print("bbox_pred:", bbox_pred.shape)
        print("First values of bbox_pred:", bbox_pred[:3, :3])

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
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

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results["cls_score"]
        bbox_pred = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros((0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg,
                )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def forward_test(self, hidden_states, proposal_list, img_metas, rescale=True):
        """Test without augmentation (originally called `simple_test`).

        Source: https://github.com/open-mmlab/mmdetection/blob/ca11860f4f3c3ca2ce8340e2686eeaec05b29111/mmdet/models/detectors/two_stage.py#L173.

        Args:
            hidden_states (tuple[Tensor]): Features from upstream network. Each has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension 5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.
        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch, it is bbox results of each image and classes
            with type `list[list[np.ndarray]]`. The outer list corresponds to each image. The inner list corresponds to
            each class. When the model has mask branch, it contains bbox results and mask results. The outer list
            corresponds to each image, and first element of tuple is bbox results, second element is mask results.
        """
        det_bboxes, det_labels = self.simple_test_bboxes(
            hidden_states, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )

        print("Det_bboxes:", det_bboxes)
        print("Det_labels:", det_labels)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes) for i in range(len(det_bboxes))
        ]

        # TODO also compute masks

        return bbox_results


# Copied from transformers.models.convnext.modeling_convnext.ConvNextPreTrainedModel with ConvNext->ConvNextMaskRCNN,convnext->convnext_maskrcnn
class ConvNextMaskRCNNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvNextMaskRCNNConfig
    base_model_prefix = "convnext_maskrcnn"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ConvNextMaskRCNNModel):
            module.gradient_checkpointing = value


CONVNEXTMASKRCNN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ConvNextMaskRCNNConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CONVNEXTMASKRCNN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ConvNextMaskRCNN model outputting raw features without any specific head on top.",
    CONVNEXTMASKRCNN_START_DOCSTRING,
)
class ConvNextMaskRCNNModel(ConvNextMaskRCNNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvNextMaskRCNNEmbeddings(config)
        self.encoder = ConvNextMaskRCNNEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVNEXTMASKRCNN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return BaseModelOutputWithNoAttention(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )


class ConvNextMaskRCNNForObjectDetection(ConvNextMaskRCNNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.convnext = ConvNextMaskRCNNModel(config)
        self.neck = ConvNextMaskRCNNFPN(config)
        self.rpn_head = ConvNextMaskRCNNRPN(config)
        self.roi_head = ConvNextMaskRCNNRoIHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # we need the intermediate hidden states
        outputs = self.convnext(pixel_values, output_hidden_states=True, return_dict=return_dict)

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # only keep certain features based on config.backbone_out_indices
        # note that the hidden_states also include the initial embeddings
        hidden_states = [
            feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
        ]

        # the FPN outputs feature maps at 5 different scales
        hidden_states = self.neck(hidden_states)

        print("Neck outputs:")
        for idx, out in enumerate(hidden_states):
            print(out.shape)

        # next, RPN computes a tuple of (class, bounding box) features for each of the 5 feature maps
        # rpn_outs[0] are the class features for each of the feature maps
        # rpn_outs[1] are the bounding box features for each of the feature maps
        rpn_outs = self.rpn_head(hidden_states)

        print(rpn_outs[0][0].shape)
        print(rpn_outs[0][0][0, 0, :3, :3])

        # TODO here: compute loss using losses = self.rpn_head.loss(...)

        # TODO: remove img_metas, compute `img_shape`` based on pixel_values
        # and figure out where `scale_factor` comes from (probably test_pipeline)
        img_metas = [dict(img_shape=(800, 1067, 3), scale_factor=[1.6671875, 1.6666666, 1.6671875, 1.6666666])]
        proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas=img_metas)
        print("Proposal list:")
        print(proposal_list[0][:3, :3])

        # TODO: support training of RoI heads
        results = self.roi_head.forward_test(hidden_states, proposal_list, img_metas=img_metas)

        logits, loss = None, None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported.")
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
