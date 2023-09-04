# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Mask-RCNN."""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import PaddingMode, pad, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_coco_detection_annotations,
    valid_images,
)
from ...utils import ExplicitEnum, TensorType, is_torch_available, is_torchvision_available, logging
from .bbox_coder import MaskRCNNDeltaXYWHBBoxCoder


if is_torch_available():
    import torch
    from torch import nn

if is_torchvision_available():
    import torchvision


logger = logging.get_logger(__name__)


# TODO move this to general utils?
AnnotationType = Dict[str, Union[int, str, List[Dict]]]
ArrayType = Union[torch.Tensor, np.ndarray]


class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"
    COCO_PANOPTIC = "coco_panoptic"


SUPPORTED_ANNOTATION_FORMATS = (AnnotionFormat.COCO_DETECTION,)


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


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

    nms_type = nms_cfg_.pop("type", "nms")
    nms_op = eval(nms_type)

    split_threshold = nms_cfg_.pop("split_threshold", 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_threshold or torch.onnx.is_in_onnx_export():
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


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None):
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

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported in TensorRT
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
        return detections, labels, indices

    detections, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        detections = detections[:max_num]
        keep = keep[:max_num]

    return detections, labels[keep], indices[keep]


# Copied from transformers.models.vilt.image_processing_vilt.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# Copied from transformers.models.vilt.image_processing_vilt.get_max_height_width
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# Copied from transformers.models.detr.image_processing_detr.convert_coco_poly_to_mask
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    Convert a COCO polygon annotation to a mask.

    Args:
        segmentations (`List[List[float]]`):
            List of polygons, each polygon represented by a list of x-y coordinates.
        height (`int`):
            Height of the mask.
        width (`int`):
            Width of the mask.
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("Pycocotools is not installed in your environment.")

    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.any(mask, axis=2)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks


# Copied from transformers.models.detr.image_processing_detr.prepare_coco_detection_annotation
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    Convert the target in COCO format into the format expected by DETR.
    """
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # Get all COCO annotations for the given image.
    annotations = target["annotations"]
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # for conversion to coco api
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    boxes = [obj["bbox"] for obj in annotations]
    # guard against no boxes via resizing
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    if annotations and "keypoints" in annotations[0]:
        keypoints = [obj["keypoints"] for obj in annotations]
        keypoints = np.asarray(keypoints, dtype=np.float32)
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints[keep]

    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    return new_target


# Copied from transformers.models.detr.image_processing_detr.resize_annotation
def resize_annotation(
    annotation: Dict[str, Any],
    orig_size: Tuple[int, int],
    target_size: Tuple[int, int],
    threshold: float = 0.5,
    resample: PILImageResampling = PILImageResampling.NEAREST,
):
    """
    Resizes an annotation to a target size.

    Args:
        annotation (`Dict[str, Any]`):
            The annotation dictionary.
        orig_size (`Tuple[int, int]`):
            The original size of the input image.
        target_size (`Tuple[int, int]`):
            The target size of the image, as returned by the preprocessing `resize` step.
        threshold (`float`, *optional*, defaults to 0.5):
            The threshold used to binarize the segmentation masks.
        resample (`PILImageResampling`, defaults to `PILImageResampling.NEAREST`):
            The resampling filter to use when resizing the masks.
    """
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    new_annotation = {}
    new_annotation["size"] = target_size

    for key, value in annotation.items():
        if key == "boxes":
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            new_annotation["boxes"] = scaled_boxes
        elif key == "area":
            area = value
            scaled_area = area * (ratio_width * ratio_height)
            new_annotation["area"] = scaled_area
        elif key == "masks":
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold
            new_annotation["masks"] = masks
        elif key == "size":
            new_annotation["size"] = target_size
        else:
            new_annotation[key] = value

    return new_annotation


# Copied from transformers.models.detr.image_processing_detr.get_size_with_aspect_ratio
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (height <= width and height == size) or (width <= height and width == size):
        return height, width

    if width < height:
        ow = size
        oh = int(size * height / width)
    else:
        oh = size
        ow = int(size * width / height)
    return (oh, ow)


# Copied from transformers.models.detr.image_processing_detr.get_resize_output_image_size
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int]],
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size. If the desired output size
    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output
    image size is computed by keeping the aspect ratio of the input image size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.
    """
    image_size = get_image_size(input_image, input_data_format)
    if isinstance(size, (list, tuple)):
        return size

    return get_size_with_aspect_ratio(image_size, size, max_size)


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from https://github.com/facebookresearch/detectron2/

    Args:
        masks (`torch.Tensor` of shape `(batch_size, 1, height, width)`:
            Predicted masks.
        boxes (`torch.Tensor` of shape `(batch_size, 4)`:
            Predicted boxes.
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


class MaskRCNNImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Mask R-CNN image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's `(height, width)` dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
            in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image to the largest image in a batch and create a pixel mask. Can be
            overridden by the `do_pad` parameter in the `preprocess` method.
        test_cfg (`Dict`, *optional*, defaults to `{"score_thr": 0.05, "nms": {"type": "nms", "iou_threshold": 0.5},
            "max_per_img": 100, "mask_thr_binary": 0.5}`): Test configuration.
        num_classes (`int`, *optional*, defaults to 80):
            Number of classes in the dataset.
        bbox_head_bbox_coder_target_means (`List[float]`, *optional*, defaults to `[0.0, 0.0, 0.0, 0.0]`):
            Means of the target for bbox head.
        bbox_head_bbox_coder_target_stds (`List[float]`, *optional*, defaults to `[0.1, 0.1, 0.2, 0.2]`):
            Standard deviation of the target for bbox head.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        format: Union[str, AnnotionFormat] = AnnotionFormat.COCO_DETECTION,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        do_pad: bool = True,
        test_cfg: Dict = None,
        num_classes=80,
        bbox_head_bbox_coder_target_means: List[float] = None,
        bbox_head_bbox_coder_target_stds: List[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.format = format
        self.do_resize = do_resize
        self.size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.test_cfg = (
            test_cfg
            if test_cfg is not None
            else {
                "score_thr": 0.05,
                "nms": {"type": "nms", "iou_threshold": 0.5},
                "max_per_img": 100,
                "mask_thr_binary": 0.5,
            }
        )
        self.num_classes = num_classes
        bbox_head_bbox_coder_target_means = (
            bbox_head_bbox_coder_target_means
            if bbox_head_bbox_coder_target_means is not None
            else [0.0, 0.0, 0.0, 0.0]
        )
        bbox_head_bbox_coder_target_stds = (
            bbox_head_bbox_coder_target_stds if bbox_head_bbox_coder_target_stds is not None else [0.1, 0.1, 0.2, 0.2]
        )
        self.bbox_coder = MaskRCNNDeltaXYWHBBoxCoder(
            target_means=bbox_head_bbox_coder_target_means, target_stds=bbox_head_bbox_coder_target_stds
        )
        # TODO remove this attribute
        self.class_agnostic = False

    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        format: Optional[AnnotionFormat] = None,
        return_segmentation_masks: bool = None,
    ) -> Dict:
        """
        Prepare an annotation for feeding into DETR model.
        """
        format = format if format is not None else self.format

        if format == AnnotionFormat.COCO_DETECTION:
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_detection_annotation(image, target, return_segmentation_masks)
        else:
            raise ValueError(f"Format {format} is not supported.")
        return target

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.resize_annotation
    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PILImageResampling = PILImageResampling.NEAREST,
    ) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.
        """
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(image, size["shortest_edge"], size["longest_edge"])
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        image = resize(
            image,
            size=size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor._pad_image
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch.

        Args:
            image (`np.ndarray`):
                Image to pad.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            input_channel_dimension (`ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be inferred from the input image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        pad_size = get_max_height_width(images)

        padded_images = [
            self._pad_image(image, pad_size, constant_values=constant_values, data_format=data_format)
            for image in images
        ]
        data = {"pixel_values": padded_images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        return_segmentation_masks: bool = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,  # PILImageResampling
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        format: Optional[Union[str, AnnotionFormat]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
                List of annotations associated with the image or batch of images. If annotation is for object
                detection, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a
                  dictionary. An image can have no annotations, in which case the list should be empty.
                If annotation is for segmentation, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.
                  An image can have no segments, in which case the list should be empty.
                - "file_name" (`str`): The file name of the image.
            return_segmentation_masks (`bool`, *optional*, defaults to self.return_segmentation_masks):
                Whether to return segmentation masks.
            masks_path (`str` or `pathlib.Path`, *optional*):
                Path to the directory containing the segmentation masks.
            do_resize (`bool`, *optional*, defaults to self.do_resize):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to self.size):
                Size of the image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to self.resample):
                Resampling filter to use when resizing the image.
            do_rescale (`bool`, *optional*, defaults to self.do_rescale):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*, defaults to self.do_normalize):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to self.image_mean):
                Mean to use when normalizing the image.
            image_std (`float` or `List[float]`, *optional*, defaults to self.image_std):
                Standard deviation to use when normalizing the image.
            do_pad (`bool`, *optional*, defaults to self.do_pad):
                Whether to pad the image.
            format (`str` or `AnnotionFormat`, *optional*, defaults to self.format):
                Format of the annotations.
            return_tensors (`str` or `TensorType`, *optional*, defaults to self.return_tensors):
                Type of tensors to return. If `None`, will return the list of images.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """

        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_pad = self.do_pad if do_pad is None else do_pad
        format = self.format if format is None else format

        if do_resize is not None and size is None:
            raise ValueError("Size and max_size must be specified if do_resize is True.")

        if do_rescale is not None and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize is not None and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        images = make_list_of_images(images)
        if annotations is not None and isinstance(annotations, dict):
            annotations = [annotations]

        if annotations is not None and len(images) != len(annotations):
            raise ValueError(
                f"The number of images ({len(images)}) and annotations ({len(annotations)}) do not match."
            )

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        format = AnnotionFormat(format)
        if annotations is not None:
            if format == AnnotionFormat.COCO_DETECTION and not valid_coco_detection_annotations(annotations):
                raise ValueError(
                    "Invalid COCO detection annotations. Annotations must a dict (single image) of list of dicts"
                    "(batch of images) with the following keys: `image_id` and `annotations`, with the latter "
                    "being a list of annotations in the COCO format."
                )
            elif format not in SUPPORTED_ANNOTATION_FORMATS:
                raise ValueError(
                    f"Unsupported annotation format: {format} must be one of {SUPPORTED_ANNOTATION_FORMATS}"
                )

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        # prepare (COCO annotations as a list of Dict -> target as a single Dict per image)
        if annotations is not None:
            prepared_annotations = []
            for image, target in zip(images, annotations):
                target = self.prepare_annotation(
                    image,
                    target,
                    format,
                    return_segmentation_masks=return_segmentation_masks,
                )
                prepared_annotations.append(target)
            annotations = prepared_annotations
            del prepared_annotations

        # transformations
        if do_resize:
            if annotations is not None:
                resized_images, resized_annotations = [], []
                for image, target in zip(images, annotations):
                    orig_size = get_image_size(image)
                    resized_image = self.resize(
                        image, size=size, resample=resample, input_data_format=input_data_format
                    )
                    resized_annotation = self.resize_annotation(target, orig_size, get_image_size(resized_image))
                    resized_images.append(resized_image)
                    resized_annotations.append(resized_annotation)
                images = resized_images
                annotations = resized_annotations
                del resized_images, resized_annotations
            else:
                images = [
                    self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
                    for image in images
                ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        if do_pad:
            # Pads images up to the largest image in the batch
            data = self.pad(images, data_format=data_format)
        else:
            images = [to_channel_dimension_format(image, data_format) for image in images]
            data = {"pixel_values": images}

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        if annotations is not None:
            encoded_inputs["labels"] = [
                BatchFeature(annotation, tensor_type=return_tensors) for annotation in annotations
            ]

        return encoded_inputs

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        # some loss (Seesaw loss..) may have custom activation
        # removed self.custom_cls_channels from original implementation
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
            scale_factor = scale_factor.clone().detach()
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            # it's here that we create a different amount of objects per image in a batch
            detected_bboxes, detected_labels, _ = multiclass_nms(
                bboxes, scores, cfg["score_thr"], cfg["nms"], cfg["max_per_img"]
            )

            return detected_bboxes, detected_labels

    def get_segmentation_masks(
        self, mask_pred, detected_bboxes, detected_labels, rcnn_test_cfg, original_shape, scale_factor, rescale
    ):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (`torch.Tensor or ndarray` of shape `(n, num_classes, height, width)`).
                For single-scale testing, mask_pred is the direct output of model, whose type is `torch.Tensor`, while
                for multi-scale testing, it will be converted to a NumPy array outside of this method.
            detected_bboxes (`torch.Tensor` of shape `(n, 4/5)`):
                Tensor containing detected bounding boxes.
            detected_labels (`torch.Tensor` of shape `(n,)`):
                Tensor containing corresponding labels.
            rcnn_test_cfg (dict):
                R-CNN testing config.
            original_shape (Tuple):
                Original image height and width, shape (2,)
            scale_factor(ndarray | Tensor):
                If `rescale is True`, box coordinates are divided by this scale factor to fit `original_shape`.
            rescale (bool):
                If True, the resulting masks will be rescaled to `original_shape`.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the i-th item in that inner list is the mask
                for the i-th box with class label c.
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = detected_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)]  # background is not included in num_classes
        bboxes = detected_bboxes[:, :4]
        labels = detected_labels

        if rescale:
            img_h, img_w = original_shape[-2:]
            bboxes = bboxes / scale_factor.to(bboxes)
        else:
            original_shape = original_shape[-2:]
            w_scale, h_scale = scale_factor[0], scale_factor[1]
            img_h = np.round(original_shape[0] * h_scale.item()).astype(np.int32)
            img_w = np.round(original_shape[1] * w_scale.item()).astype(np.int32)

        num_masks = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == "cpu":
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = num_masks
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(np.ceil(num_masks * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            if num_chunks > num_masks:
                raise ValueError("Default GPU_MEM_LIMIT is too small; try increasing it")
        chunks = torch.chunk(torch.arange(num_masks, device=device), num_chunks)

        threshold = rcnn_test_cfg["mask_thr_binary"]
        im_mask = torch.zeros(
            num_masks, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
        )

        if not self.class_agnostic:
            mask_pred = mask_pred[range(num_masks), labels][:, None]

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

        for i in range(num_masks):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes=None,
        scale_factors=None,
    ):
        """
        Converts the output of [`MaskRCNNForObjectDetection`] into the format expected by the COCO api. Only supports
        PyTorch.

        Args:
            outputs ([`MaskRCNNForObjectDetection`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.
            scale_factors (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the scale factor
                (height, width) of each image in the batch. If left to None, predictions will not be rescaled.

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

        # calculate img shapes based on target sizes (ori shapes) + scale factors
        img_shapes = []
        for target_size, scale_factor in zip(target_sizes, scale_factors):
            height, width = target_size[-2:]
            img_shape = (3, height * scale_factor[1], width * scale_factor[0])
            img_shapes.append(img_shape)

        # apply bbox post-processing to each image individually
        detected_bboxes = []
        detected_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                detected_bbox = rois[i].new_zeros(0, 5)
                detected_label = rois[i].new_zeros((0,), dtype=torch.long)
                if self.test_cfg is None:
                    detected_bbox = detected_bbox[:, :4]
                    detected_label = rois[i].new_zeros((0, self.bbox_head.fc_cls.out_features))

            else:
                detected_bbox, detected_label = self.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=True,  # we rescale by default
                    cfg=self.test_cfg,
                )
            detected_bboxes.append(detected_bbox)
            detected_labels.append(detected_label)

        # turn into COCO API
        results = []
        for example_idx in range(len(detected_bboxes)):
            scores = detected_bboxes[example_idx][:, -1]
            boxes = detected_bboxes[example_idx][:, :-1]
            labels = detected_labels[example_idx]
            results.append(
                {
                    "scores": scores[scores > threshold],
                    "labels": labels[scores > threshold],
                    "boxes": boxes[scores > threshold],
                }
            )

        return results

    def post_process_instance_segmentation(
        self,
        object_detection_results,
        predicted_masks,
        target_sizes=None,
        scale_factors=None,
    ):
        detected_bboxes = [result["boxes"] for result in object_detection_results]
        detected_labels = [result["labels"] for result in object_detection_results]

        # split batch mask prediction back to each image
        num_mask_roi_per_img = [len(detected_bbox) for detected_bbox in detected_bboxes]
        predicted_masks = predicted_masks.split(num_mask_roi_per_img, 0)

        num_imgs = len(object_detection_results)

        rescale = True  # we rescale by default
        scale_factors = [scale_factor.to(detected_bboxes[0].device) for scale_factor in scale_factors]
        _bboxes = [
            # detected_bboxes[i][:, :4] * scale_factors[i] if rescale else detected_bboxes[i][:, :4]
            detected_bboxes[i] * scale_factors[i] if rescale else detected_bboxes[i]
            for i in range(num_imgs)
        ]

        # apply mask post-processing to each image individually
        segm_results = []
        for i in range(num_imgs):
            if detected_bboxes[i].shape[0] == 0:
                segm_results.append([[] for _ in range(self.num_classes)])
            else:
                segm_result = self.get_segmentation_masks(
                    predicted_masks[i],
                    _bboxes[i],
                    detected_labels[i],
                    self.test_cfg,
                    target_sizes[i],
                    scale_factors[i],
                    rescale=rescale,
                )
                segm_results.append(segm_result)

        return segm_results
