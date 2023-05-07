# coding=utf-8
# Copyright 2022 MMDetection Contributors. OpenMMLab Detection Toolbox and Benchmark [Computer software].
# https://github.com/open-mmlab/mmdetection
# and The HuggingFace Inc. team. All rights reserved.
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
"""Non-maximum suppression (NMS) implementation."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchvision


ArrayType = Union[torch.Tensor, np.ndarray]


def nms(
    boxes: ArrayType,
    scores: ArrayType,
    iou_threshold: float,
    offset: int = 0,
) -> Tuple[ArrayType, ArrayType]:
    """Dispatch to either CPU or GPU NMS implementations.

    Source: https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/nms.py. Removed the `score_threshold`and `max_num`
    arguments as those are only supported by MMCV's NMS implementation and we are using torchvision.ops.nms. See also
    https://github.com/open-mmlab/mmcv/blob/d71d067da19d71d79e7b4d7ae967891c7bb00c05/mmcv/ops/nms.py#L28.

    The input can be either torch tensor or numpy array. GPU NMS will be used if the input is GPU tensor, otherwise
    CPU: NMS will be used. The returned type will always be the same as the inputs.

    Args:
        boxes (`torch.Tensor` or `np.ndarray`):
            Bounding boxes of shape (N, 4) with N = number of objects.
        scores (`torch.Tensor` or `np.ndarray`):
            Scores of shape (N, ).
        iou_threshold (`float`):
            IoU threshold for NMS.
        offset (`int`, *optional*, defaults to 0):
            If set, the bounding boxes' width or height is (x2 - x1 + offset). Can be set to 0 or 1.

    Returns:
        `Tuple`: kept detections (boxes and scores) and indices, which always have the same data type as the input.
    """
    if not isinstance(boxes, (torch.Tensor, np.ndarray)):
        raise ValueError(f"Unsupported type {type(boxes)} for boxes.")
    if not isinstance(scores, (torch.Tensor, np.ndarray)):
        raise ValueError(f"Unsupported type {type(scores)} for scores.")
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)

    if boxes.size(1) != 4:
        raise ValueError(f"Bounding boxes should have shape (N, 4), but got {boxes.shape}")
    if boxes.size(0) != scores.size(0):
        raise ValueError(f"The number of boxes ({boxes.size(0)}) and scores ({scores.size(0)}) should be the same.")
    if offset not in (0, 1):
        raise ValueError(f"Offset should be either 0 or 1, but got {offset}.")

    indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    detections = torch.cat((boxes[indices], scores[indices].reshape(-1, 1)), dim=1)
    if is_numpy:
        detections = detections.cpu().numpy()
        indices = indices.cpu().numpy()
    return detections, indices


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    nms_cfg: Optional[Dict],
    class_agnostic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Performs non-maximum suppression (NMS) in a batched fashion.

    Modified from [torchvision/ops/boxes.py#L39](https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39). In order to perform NMS independently per
    class, we add an offset to all the boxes. The offset is dependent only on the class idx, and is large enough so
    that boxes from different classes do not overlap. Note:

    In v1.4.1 and later, `batched_nms` supports skipping the NMS and returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (`torch.Tensor`):
            Bounding boxes in shape (N, 4) or (N, 5) with N = number of objects.
        scores (`torch.Tensor`):
            Scores in shape (N,).
        idxs (`torch.Tensor`):
            Each index value corresponds to a bbox cluster, and NMS will not be applied between elements of different
            idxs, shape (N, ).
        nms_cfg (dict | optional):
            Supports skipping the nms when *nms_cfg* is None, otherwise it should specify nms type and other parameters
            like *iou_thr*. Possible keys includes the following.
            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the number of boxes is large (e.g., 200k). To
              avoid OOM during training, the users could set *split_thr* to a small value. If the number of boxes is
              greater than the threshold, it will perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (`bool`, *optional*, defaults to `False`):
            If `True`, NMS is class agnostic, i.e. IoU thresholding happens over all boxes, regardless of the predicted
            class.

    Returns:
        `Tuple(torch.Tensor)` comprising various elements:
        - **boxes** (`torch.Tensor`):
            Bboxes with scores after NMS, has shape (num_bboxes, 5). Last dimension 5 arranges as (x1, y1, x2, y2,
            score).
        - **keep** (`torch.Tensor`):
            The indices of remaining boxes in the input boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, indices = scores.sort(descending=True)
        boxes = boxes[indices]
        return torch.cat([boxes, scores[:, None]], -1), indices

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
        detections, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `detections` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = detections[:, -1]
    else:
        max_num = nms_cfg_.pop("max_num", -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            detections, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = detections[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, indices = scores_after_nms[keep].sort(descending=True)
        keep = keep[indices]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def multiclass_nms(
    multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, return_indices=False
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (`torch.Tensor`):
            Shape (N, #class*4) or (N, 4) with N = number of objects.
        multi_scores (`torch.Tensor`):
            Shape (N, #class), where the last column contains scores of the background class, but this will be ignored.
        score_thr (`float`):
            Bounding box threshold, boxes with scores lower than it will not be considered.
        nms_thr (`float`):
            NMS IoU threshold.
        max_num (`int`, *optional*, defaults to -1):
            If there are more than `max_num` bounding boxes after NMS, only top `max_num` will be kept.
        score_factors (`torch.Tensor`, *optional*):
            The factors multiplied to scores before applying NMS.
        return_indices (`bool`, *optional*, defaults to `False`):
            Whether to return the indices of kept bounding boxes.

    Returns:
        `Tuple`: (detections, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). detections are boxes with scores. Labels are 0-based.
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
        indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[indices], scores[indices], labels[indices]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError("[ONNX Error] Can not record NMS as it has not been executed this time")
        detections = torch.cat([bboxes, scores[:, None]], -1)
        if return_indices:
            return detections, labels, indices
        else:
            return detections, labels

    detections, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        detections = detections[:max_num]
        keep = keep[:max_num]

    if return_indices:
        return detections, labels[keep], indices[keep]
    else:
        return detections, labels[keep]
