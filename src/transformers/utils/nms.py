# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

"""NMS (non-maximum suppression) utilities."""

from typing import Dict, Optional, Tuple

from . import is_torch_available, is_torchvision_available


if is_torch_available():
    import torch

if is_torchvision_available():
    import torchvision


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    nms_cfg: Optional[Dict] = None,
    class_agnostic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Performs non-maximum suppression (NMS) in a batched fashion.

    Modified from [torchvision/ops/boxes.py#L39](https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39). In order to perform NMS independently per
    class, we add an offset to all the boxes. The offset is dependent only on the class idx, and is large enough so
    that boxes from different classes do not overlap.

    Note: skipping the NMS is also supported and returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (`torch.Tensor`):
            Bounding boxes in shape (N, 4) or (N, 5) with N = number of objects.
        scores (`torch.Tensor`):
            Scores in shape (N,).
        idxs (`torch.Tensor`):
            Each index value corresponds to a bbox cluster, and NMS will not be applied between elements of different
            idxs, shape (N, ).
        nms_cfg (`dict`, *optional*):
            Supports skipping the nms when *nms_cfg* is None, otherwise it should specify parameters like
            *iou_threshold*. Possible keys includes the following:
            - iou_threshold (float): IoU threshold used for NMS.
            - split_threshold (float): threshold number of boxes. In some cases the number of boxes is large (e.g.,
              200k). To avoid OOM during training, the users could set *split_threshold* to a small value. If the
              number of boxes is greater than the threshold, it will perform NMS on each group of boxes separately and
              sequentially. Defaults to 10000.
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

    nms_op = torchvision.ops.nms

    split_threshold = nms_cfg_.pop("split_threshold", 10000)
    if boxes_for_nms.shape[0] < split_threshold:
        keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        detections = torch.cat((boxes[keep], scores[keep].reshape(-1, 1)), dim=1)
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
            keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            detections = torch.cat((boxes[keep], scores[keep].reshape(-1, 1)), dim=1)
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


def multiclass_nms(multi_bboxes, multi_scores, score_threshold, nms_cfg, max_num=-1, score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (`torch.Tensor`):
            Shape (N, #class*4) or (N, 4) with N = number of objects.
        multi_scores (`torch.Tensor`):
            Shape (N, #class), where the last column contains scores of the background class, but this will be ignored.
        score_threshold (`float`):
            Bounding box threshold, boxes with scores lower than it will not be considered.
        nms_cfg (`dict`):
            NMS configuration.
        max_num (`int`, *optional*, defaults to -1):
            If there are more than `max_num` bounding boxes after NMS, only top `max_num` will be kept.
        score_factors (`torch.Tensor`, *optional*):
            The factors multiplied to scores before applying NMS.

    Returns:
        `Tuple`: (detections, labels, indices), tensors of shape (k, 5),
            (k), and (k), and indices of boxes to keep. detections are boxes with scores. Labels are 0-based.
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

    # remove low scoring boxes
    valid_mask = scores > score_threshold
    # multiply score_factor after threshold to preserve more bboxes, improves mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[indices], scores[indices], labels[indices]

    if bboxes.numel() == 0:
        detections = torch.cat([bboxes, scores[:, None]], -1)
        return detections, labels, indices

    detections, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        detections = detections[:max_num]
        keep = keep[:max_num]

    return detections, labels[keep], indices[keep]
