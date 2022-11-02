# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for ConvNext Mask-RCNN."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torchvision
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import ImageFeatureExtractionMixin
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

logger = logging.get_logger(__name__)


ImageInput = Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]]

ArrayType = Union[torch.Tensor, np.ndarray]

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


def nms(
    boxes: ArrayType,
    scores: ArrayType,
    iou_threshold: float,
    offset: int = 0,
    score_threshold: float = 0,
    max_num: int = -1,
) -> Tuple[ArrayType, ArrayType]:
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

    def decode(self, bboxes, pred_bboxes, max_shape=None, wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor):
                Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (Tensor):
                Encoded offsets with respect to each roi. Has shape (B, N, num_classes * 4) or (B, N, 4) or (N,
                num_classes * 4) or (N, 4). Note N = num_anchors * W * H when rois is a grid of anchors.Offset encoding
                follows [1]_.
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
                # TODO: remove this deprecation
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
        max_shape = max_shape[-2:]
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor):
            N, 1, H, W
        boxes (Tensor):
            N, 4
        img_h (int):
            Height of the image to be pasted.
        img_w (int):
            Width of the image to be pasted.
        skip_empty (bool):
            Only paste masks within the region that tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = nn.functional.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


class ConvNextMaskRCNNFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a ConvNeXT Mask R-CNN feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        test_cfg
            ...
        num_classes
            ...
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        test_cfg,
        num_classes,
        bbox_head_bbox_coder_target_means=[0.0, 0.0, 0.0, 0.0],
        bbox_head_bbox_coder_target_stds=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super().__init__(**kwargs)

        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.bbox_coder = ConvNextMaskRCNNDeltaXYWHBBoxCoder(
            target_means=bbox_head_bbox_coder_target_means, target_stds=bbox_head_bbox_coder_target_stds
        )
        # TODO remove this attribute
        self.class_agnostic = False

    def __call__(self, **kwargs) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s) and optional annotations. Images are by default
        padded up to the largest image in a batch, and a pixel mask is created that indicates which pixels are
        real/which are padding.

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            ...

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            ...
        """
        # Input type checking for clearer error

        raise NotImplementedError("To do")

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
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
                img_shape = img_shape[-2:]
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            # it's here that we create a different amount of objects per image in a batch
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg["score_thr"], cfg["nms"], cfg["max_per_img"])

            return det_bboxes, det_labels

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of model, whose type is Tensor, while for
                multi-scale testing, it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(ndarray | Tensor): If `rescale is True`, box
                coordinates are divided by this scale factor to fit `ori_shape`.
            rescale (bool): If True, the resulting masks will be rescaled to
                `ori_shape`.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the i-th item in that inner list is the mask
                for the i-th box with class label c.
        Example:
            >>> import mmcv >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import * # NOQA >>> N = 7 # N =
            number of extracted ROIs >>> C, H, W = 11, 32, 32 >>> # Create example instance of FCN Mask Head. >>> self
            = FCNMaskHead(num_classes=C, num_convs=0) >>> inputs = torch.rand(N, self.in_channels, H, W) >>> mask_pred
            = self.forward(inputs) >>> # Each input is associated with some bounding box >>> det_bboxes =
            torch.Tensor([[1, 1, 42, 42 ]] * N) >>> det_labels = torch.randint(0, C, size=(N,)) >>> rcnn_test_cfg =
            mmcv.Config({'mask_thr_binary': 0, }) >>> ori_shape = (H * 4, W * 4) >>> scale_factor =
            torch.FloatTensor((1, 1)) >>> rescale = False >>> # Encoded masks are a list for each category. >>>
            encoded_masks = self.get_seg_masks( >>> mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, >>>
            scale_factor, rescale >>> ) >>> assert len(encoded_masks) == C >>> assert sum(list(map(len,
            encoded_masks))) == N
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        # In most cases, scale_factor should have been
        # converted to Tensor when rescale the bbox
        if not isinstance(scale_factor, torch.Tensor):
            if isinstance(scale_factor, float):
                scale_factor = np.array([scale_factor] * 4)
                # TODO: remove this deprecation
                warnings.warn(
                    "Scale_factor should be a Tensor or ndarray with shape (4,), float would be deprecated. "
                )
            assert isinstance(scale_factor, np.ndarray)
            scale_factor = torch.Tensor(scale_factor)

        if rescale:
            img_h, img_w = ori_shape[-2:]
            bboxes = bboxes / scale_factor.to(bboxes)
        else:
            ori_shape = ori_shape[-2:]
            w_scale, h_scale = scale_factor[0], scale_factor[1]
            img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
            img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == "cpu":
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert num_chunks <= N, "Default GPU_MEM_LIMIT is too small; try increasing it"
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg["mask_thr_binary"]
        im_mask = torch.zeros(N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds], bboxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
            )

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes=None,
        scale_factors=None,
        rescale=True,
    ):
        """
        Converts the output of [`ConvNextForObjectDetection`] into the format expected by the COCO api. Only supports
        PyTorch.

        Args:
            outputs ([`ConvNextForObjectDetection`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*, defaults to `None`):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        rois, proposals, cls_score, bbox_pred = outputs.rois, outputs.proposals, outputs.logits, outputs.pred_boxes

        # reshape back to (batch_size*num_proposals_per_image, ...)
        cls_score = cls_score.reshape(-1, cls_score.shape[-1])
        bbox_pred = bbox_pred.reshape(-1, bbox_pred.shape[-1])

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
                if self.test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros((0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    target_sizes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=self.test_cfg,
                )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        # turn into COCO API
        results = []
        for example_idx in range(len(det_bboxes)):
            scores = det_bboxes[example_idx][:, -1]
            boxes = det_bboxes[example_idx][:, :-1]
            labels = det_labels[example_idx]
            results.append(
                {
                    "scores": scores[scores > threshold],
                    "labels": labels[scores > threshold],
                    "boxes": boxes[scores > threshold],
                }
            )

        return results

    def post_process_instance_segmentation(
        self, object_detection_results, mask_pred, target_sizes=None, scale_factors=None, rescale=True
    ):
        det_bboxes = [result["boxes"] for result in object_detection_results]
        det_labels = [result["labels"] for result in object_detection_results]

        # split batch mask prediction back to each image
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

        num_imgs = len(object_detection_results)

        if rescale:
            scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device) for scale_factor in scale_factors]
            _bboxes = [
                # det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
                det_bboxes[i] * scale_factors[i] if rescale else det_bboxes[i]
                for i in range(num_imgs)
            ]

        # apply mask post-processing to each image individually
        segm_results = []
        for i in range(num_imgs):
            if det_bboxes[i].shape[0] == 0:
                segm_results.append([[] for _ in range(self.num_classes)])
            else:
                segm_result = self.get_seg_masks(
                    mask_preds[i],
                    _bboxes[i],
                    det_labels[i],
                    self.test_cfg,
                    target_sizes[i],
                    scale_factors[i],
                    rescale,
                )
                segm_results.append(segm_result)

        return segm_results
