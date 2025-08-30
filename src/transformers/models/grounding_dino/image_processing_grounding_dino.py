# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Deformable DETR."""

import io
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
    center_to_corners_format,
    corners_to_center_format,
    id_to_rgb,
    resize,
    rgb_to_id,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
)
from ...utils import (
    ExplicitEnum,
    is_flax_available,
    is_jax_tensor,
    is_scipy_available,
    is_tf_available,
    is_tf_tensor,
    is_torch_available,
    is_torch_tensor,
    is_vision_available,
    logging,
)


if is_torch_available():
    import torch
    from torch import nn


if is_vision_available():
    import PIL

if is_scipy_available():
    import scipy.special
    import scipy.stats

if TYPE_CHECKING:
    pass


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

AnnotationType = dict[str, Union[int, str, list[dict]]]


class AnnotationFormat(ExplicitEnum):
    COCO_DETECTION = "coco_detection"
    COCO_PANOPTIC = "coco_panoptic"


SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


# Copied from transformers.models.detr.image_processing_detr.get_size_with_aspect_ratio
def get_size_with_aspect_ratio(image_size, size, max_size=None) -> tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    raw_size = None
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            raw_size = max_size * min_original_size / max_original_size
            size = int(round(raw_size))

    if (height <= width and height == size) or (width <= height and width == size):
        oh, ow = height, width
    elif width < height:
        ow = size
        if max_size is not None and raw_size is not None:
            oh = int(raw_size * height / width)
        else:
            oh = int(size * height / width)
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = int(raw_size * width / height)
        else:
            ow = int(size * width / height)

    return (oh, ow)


# Copied from transformers.models.detr.image_processing_detr.get_resize_output_image_size
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, tuple[int, int], list[int]],
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size. If the desired output size
    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output
    image size is computed by keeping the aspect ratio of the input image size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `tuple[int, int]` or `list[int]`):
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


# Copied from transformers.models.detr.image_processing_detr.get_image_size_for_max_height_width
def get_image_size_for_max_height_width(
    input_image: np.ndarray,
    max_height: int,
    max_width: int,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> tuple[int, int]:
    """
    Computes the output image size given the input image and the maximum allowed height and width. Keep aspect ratio.
    Important, even if image_height < max_height and image_width < max_width, the image will be resized
    to at least one of the edges be equal to max_height or max_width.

    For example:
        - input_size: (100, 200), max_height: 50, max_width: 50 -> output_size: (25, 50)
        - input_size: (100, 200), max_height: 200, max_width: 500 -> output_size: (200, 400)

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        max_height (`int`):
            The maximum allowed height.
        max_width (`int`):
            The maximum allowed width.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.
    """
    image_size = get_image_size(input_image, input_data_format)
    height, width = image_size
    height_scale = max_height / height
    width_scale = max_width / width
    min_scale = min(height_scale, width_scale)
    new_height = int(height * min_scale)
    new_width = int(width * min_scale)
    return new_height, new_width


# Copied from transformers.models.detr.image_processing_detr.get_numpy_to_framework_fn
def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    if isinstance(arr, np.ndarray):
        return np.array
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


# Copied from transformers.models.detr.image_processing_detr.safe_squeeze
def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    if axis is None:
        return arr.squeeze()

    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


# Copied from transformers.models.detr.image_processing_detr.normalize_annotation
def normalize_annotation(annotation: dict, image_size: tuple[int, int]) -> dict:
    image_height, image_width = image_size
    norm_annotation = {}
    for key, value in annotation.items():
        if key == "boxes":
            boxes = value
            boxes = corners_to_center_format(boxes)
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            norm_annotation[key] = boxes
        else:
            norm_annotation[key] = value
    return norm_annotation


# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> list[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
def get_max_height_width(
    images: list[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> list[int]:
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


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
def make_pixel_mask(
    image: np.ndarray, output_size: tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


# Copied from transformers.models.detr.image_processing_detr.convert_coco_poly_to_mask
def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    Convert a COCO polygon annotation to a mask.

    Args:
        segmentations (`list[list[float]]`):
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


# Copied from transformers.models.detr.image_processing_detr.prepare_coco_detection_annotation with DETR->GroundingDino
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    Convert the target in COCO format into the format expected by GroundingDino.
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
    iscrowd = np.asarray([obj.get("iscrowd", 0) for obj in annotations], dtype=np.int64)

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
        # Converting the filtered keypoints list to a numpy array
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # Apply the keep mask here to filter the relevant annotations
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    return new_target


# Copied from transformers.models.detr.image_processing_detr.masks_to_boxes
def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    Args:
        masks: masks in format `[number_masks, height, width]` where N is the number of masks

    Returns:
        boxes: bounding boxes in format `[number_masks, 4]` in xyxy format
    """
    if masks.size == 0:
        return np.zeros((0, 4))

    h, w = masks.shape[-2:]
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    # see https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    return np.stack([x_min, y_min, x_max, y_max], 1)


# Copied from transformers.models.detr.image_processing_detr.prepare_coco_panoptic_annotation with DETR->GroundingDino
def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> dict:
    """
    Prepare a coco panoptic annotation for GroundingDino.
    """
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    if "segments_info" in target:
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        masks = rgb_to_id(masks)

        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        if return_masks:
            new_target["masks"] = masks
        new_target["boxes"] = masks_to_boxes(masks)
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


# Copied from transformers.models.detr.image_processing_detr.get_segmentation_image
def get_segmentation_image(
    masks: np.ndarray, input_size: tuple, target_size: tuple, stuff_equiv_classes, deduplicate=False
):
    h, w = input_size
    final_h, final_w = target_size

    m_id = scipy.special.softmax(masks.transpose(0, 1), -1)

    if m_id.shape[-1] == 0:
        # We didn't detect any mask :(
        m_id = np.zeros((h, w), dtype=np.int64)
    else:
        m_id = m_id.argmax(-1).reshape(h, w)

    if deduplicate:
        # Merge the masks corresponding to the same stuff class
        for equiv in stuff_equiv_classes.values():
            for eq_id in equiv:
                m_id[m_id == eq_id] = equiv[0]

    seg_img = id_to_rgb(m_id)
    seg_img = resize(seg_img, (final_w, final_h), resample=PILImageResampling.NEAREST)
    return seg_img


# Copied from transformers.models.detr.image_processing_detr.get_mask_area
def get_mask_area(seg_img: np.ndarray, target_size: tuple[int, int], n_classes: int) -> np.ndarray:
    final_h, final_w = target_size
    np_seg_img = seg_img.astype(np.uint8)
    np_seg_img = np_seg_img.reshape(final_h, final_w, 3)
    m_id = rgb_to_id(np_seg_img)
    area = [(m_id == i).sum() for i in range(n_classes)]
    return area


# Copied from transformers.models.detr.image_processing_detr.score_labels_from_class_probabilities
def score_labels_from_class_probabilities(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = scipy.special.softmax(logits, axis=-1)
    labels = probs.argmax(-1, keepdims=True)
    scores = np.take_along_axis(probs, labels, axis=-1)
    scores, labels = scores.squeeze(-1), labels.squeeze(-1)
    return scores, labels


# Copied from transformers.models.detr.image_processing_detr.post_process_panoptic_sample
def post_process_panoptic_sample(
    out_logits: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    processed_size: tuple[int, int],
    target_size: tuple[int, int],
    is_thing_map: dict,
    threshold=0.85,
) -> dict:
    """
    Converts the output of [`DetrForSegmentation`] into panoptic segmentation predictions for a single sample.

    Args:
        out_logits (`torch.Tensor`):
            The logits for this sample.
        masks (`torch.Tensor`):
            The predicted segmentation masks for this sample.
        boxes (`torch.Tensor`):
            The predicted bounding boxes for this sample. The boxes are in the normalized format `(center_x, center_y,
            width, height)` and values between `[0, 1]`, relative to the size the image (disregarding padding).
        processed_size (`tuple[int, int]`):
            The processed size of the image `(height, width)`, as returned by the preprocessing step i.e. the size
            after data augmentation but before batching.
        target_size (`tuple[int, int]`):
            The target size of the image, `(height, width)` corresponding to the requested final size of the
            prediction.
        is_thing_map (`Dict`):
            A dictionary mapping class indices to a boolean value indicating whether the class is a thing or not.
        threshold (`float`, *optional*, defaults to 0.85):
            The threshold used to binarize the segmentation masks.
    """
    # we filter empty queries and detection below threshold
    scores, labels = score_labels_from_class_probabilities(out_logits)
    keep = (labels != out_logits.shape[-1] - 1) & (scores > threshold)

    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_boxes = center_to_corners_format(boxes[keep])

    if len(cur_boxes) != len(cur_classes):
        raise ValueError("Not as many boxes as there are classes")

    cur_masks = masks[keep]
    cur_masks = resize(cur_masks[:, None], processed_size, resample=PILImageResampling.BILINEAR)
    cur_masks = safe_squeeze(cur_masks, 1)
    b, h, w = cur_masks.shape

    # It may be that we have several predicted masks for the same stuff class.
    # In the following, we track the list of masks ids for each stuff class (they are merged later on)
    cur_masks = cur_masks.reshape(b, -1)
    stuff_equiv_classes = defaultdict(list)
    for k, label in enumerate(cur_classes):
        if not is_thing_map[label]:
            stuff_equiv_classes[label].append(k)

    seg_img = get_segmentation_image(cur_masks, processed_size, target_size, stuff_equiv_classes, deduplicate=True)
    area = get_mask_area(cur_masks, processed_size, n_classes=len(cur_scores))

    # We filter out any mask that is too small
    if cur_classes.size() > 0:
        # We know filter empty masks as long as we find some
        filtered_small = np.array([a <= 4 for a in area], dtype=bool)
        while filtered_small.any():
            cur_masks = cur_masks[~filtered_small]
            cur_scores = cur_scores[~filtered_small]
            cur_classes = cur_classes[~filtered_small]
            seg_img = get_segmentation_image(cur_masks, (h, w), target_size, stuff_equiv_classes, deduplicate=True)
            area = get_mask_area(seg_img, target_size, n_classes=len(cur_scores))
            filtered_small = np.array([a <= 4 for a in area], dtype=bool)
    else:
        cur_classes = np.ones((1, 1), dtype=np.int64)

    segments_info = [
        {"id": i, "isthing": is_thing_map[cat], "category_id": int(cat), "area": a}
        for i, (cat, a) in enumerate(zip(cur_classes, area))
    ]
    del cur_classes

    with io.BytesIO() as out:
        PIL.Image.fromarray(seg_img).save(out, format="PNG")
        predictions = {"png_string": out.getvalue(), "segments_info": segments_info}

    return predictions


# Copied from transformers.models.detr.image_processing_detr.resize_annotation
def resize_annotation(
    annotation: dict[str, Any],
    orig_size: tuple[int, int],
    target_size: tuple[int, int],
    threshold: float = 0.5,
    resample: PILImageResampling = PILImageResampling.NEAREST,
):
    """
    Resizes an annotation to a target size.

    Args:
        annotation (`dict[str, Any]`):
            The annotation dictionary.
        orig_size (`tuple[int, int]`):
            The original size of the input image.
        target_size (`tuple[int, int]`):
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


# Copied from transformers.models.detr.image_processing_detr.binary_mask_to_rle
def binary_mask_to_rle(mask):
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    if is_torch_tensor(mask):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


# Copied from transformers.models.detr.image_processing_detr.convert_segmentation_to_rle
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `list[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings


# Copied from transformers.models.detr.image_processing_detr.remove_low_and_no_objects
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


# Copied from transformers.models.detr.image_processing_detr.check_segment_validity
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # Get the mask associated with the k class
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # Compute the area of all the stuff in query k
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # Eliminate disconnected tiny segments
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# Copied from transformers.models.detr.image_processing_detr.compute_segments
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[set[int]] = None,
    target_size: Optional[tuple[int, int]] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: list[dict] = []

    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # Weigh each mask by its prediction score
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # Keep track of instances of each class
    stuff_memory_list: dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # Check if mask exists and large enough to be a segment
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # Add current object segment to final segmentation map
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    return segmentation, segments


# Copied from transformers.models.owlvit.image_processing_owlvit._scale_boxes
def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


class GroundingDinoImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Grounding DINO image processor.

    Args:
        format (`str`, *optional*, defaults to `AnnotationFormat.COCO_DETECTION`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
            in the `preprocess` method. Available options are:
                - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                    Do NOT keep the aspect ratio.
                - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                    the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                    less or equal to `longest_edge`.
                - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                    aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                    `max_width`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method. Controls whether to normalize the image. Can be overridden by the `do_normalize`
            parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_annotations (`bool`, *optional*, defaults to `True`):
            Controls whether to convert the annotations to the format expected by the DETR model. Converts the
            bounding boxes to the format `(center_x, center_y, width, height)` and in the range `[0, 1]`.
            Can be overridden by the `do_convert_annotations` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
            method. If `True`, padding will be applied to the bottom and right of the image with zeros.
            If `pad_size` is provided, the image will be padded to the specified dimensions.
            Otherwise, the image will be padded to the maximum height and width of the batch.
        pad_size (`dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
            provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
            height and width in the batch.
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.__init__
    def __init__(
        self,
        format: Union[str, AnnotationFormat] = AnnotationFormat.COCO_DETECTION,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_annotations: Optional[bool] = None,
        do_pad: bool = True,
        pad_size: Optional[dict[str, int]] = None,
        **kwargs,
    ) -> None:
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")


        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        size = get_size_dict(size, default_to_square=False)

        # Backwards compatibility
        if do_convert_annotations is None:
            do_convert_annotations = do_normalize

        super().__init__(**kwargs)
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_annotations = do_convert_annotations
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.pad_size = pad_size
        self._valid_processor_keys = [
            "images",
            "annotations",
            "return_segmentation_masks",
            "masks_path",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "do_convert_annotations",
            "image_mean",
            "image_std",
            "do_pad",
            "pad_size",
            "format",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    @classmethod
    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.from_dict with Detr->GroundingDino
    def from_dict(cls, image_processor_dict: dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `GroundingDinoImageProcessor.from_pretrained(checkpoint, size=600,
        max_size=800)`
        """
        image_processor_dict = image_processor_dict.copy()
