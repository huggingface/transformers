# coding=utf-8
# Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for EoMT."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class EoMTImageProcessorFastKwargs(DefaultFastImageProcessorKwargs):
    do_split_image: bool


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
            oh = round(raw_size * height / width)
        else:
            oh = round(size * height / width)
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = round(raw_size * width / height)
        else:
            ow = round(size * width / height)

    return (oh, ow)


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
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # Get the mask associated with the k class
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # Compute the area of all the stuff in query k
    original_mask = mask_probs[k] >= mask_threshold
    original_area = original_mask.sum()

    final_mask = mask_k & original_mask
    final_mask_area = final_mask.sum()

    mask_exists = mask_k_area > 0 and original_area > 0 and final_mask_area > 0

    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, final_mask


def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    stuff_classes,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    target_size: Optional[Tuple[int, int]] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.long, device=mask_probs.device) - 1
    segments: List[Dict] = []

    # Compute per-pixel assignment based on weighted mask scores
    mask_probs = mask_probs.sigmoid()
    mask_labels = (pred_scores[:, None, None] * mask_probs).argmax(0)

    # Keep track of instances of each class
    current_segment_id = 0
    stuff_memory_list: Dict[str, int] = {}

    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()

        # Check if mask exists and large enough to be a segment
        mask_exists, final_mask = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if not mask_exists:
            continue

        if pred_class in stuff_classes:
            if pred_class in stuff_memory_list:
                segmentation[final_mask] = stuff_memory_list[pred_class]
                continue
            else:
                stuff_memory_list[pred_class] = current_segment_id

        segmentation[final_mask] = current_segment_id
        segment_score = round(pred_scores[k].item(), 6)
        segments.append(
            {
                "id": current_segment_id,
                "label_id": pred_class,
                "score": segment_score,
            }
        )
        current_segment_id += 1
    return segmentation, segments


def get_target_size(size_dict: Dict[str, int]) -> Tuple[int, int]:
    """Returns the height and width from a size dict."""
    target_height = size_dict["shortest_edge"]
    target_width = size_dict.get("longest_edge", None) or target_height

    return target_height, target_width


def reorder_crops_and_offsets(
    crops: List[torch.Tensor], offsets: List[List[int]]
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """Sorts crops and offsets according to the original image index."""

    combined = list(zip(offsets, crops))
    combined.sort(key=lambda x: x[0][0])
    sorted_offsets, sorted_crops = zip(*combined)

    return list(sorted_crops), list(sorted_offsets)


@auto_docstring()
class EoMTImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 640, "longest_edge": 640}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None
    do_split_image = False
    valid_kwargs = EoMTImageProcessorFastKwargs

    def __init__(self, **kwargs: Unpack[EoMTImageProcessorFastKwargs]):
        super().__init__(**kwargs)

    def resize(
        self,
        images: torch.Tensor,
        interpolation: "F.InterpolationMode",
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            interpolation (`F.InterpolationMode`):
                The interpolation method to use for resizing. Defaults to bilinear.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        _, _, height, width = images.shape
        output_size = get_size_with_aspect_ratio(
            (height, width), self.size["shortest_edge"], self.size["longest_edge"]
        )

        images = F.resize(
            images,
            size=output_size,
            interpolation=interpolation,
            **kwargs,
        )

        return images

    def _split_image(self, image: ImageInput, image_index: int) -> Tuple[List, List]:
        """Slices an image into overlapping crops for semantic segmentation."""

        crops, patch_offsets = [], []

        image_size = get_image_size(image)
        crop_size = self.size["shortest_edge"]

        longer_side = max(image_size)
        num_crops = math.ceil(longer_side / crop_size)
        total_overlap = num_crops * crop_size - longer_side
        overlap_per_crop = total_overlap / (num_crops - 1) if num_crops > 1 else 0

        for i in range(num_crops):
            start = int(i * (crop_size - overlap_per_crop))
            end = start + crop_size

            if image_size[0] > image_size[1]:
                crop = image[:, start:end, :]
            else:
                crop = image[:, :, start:end]

            crops.append(crop)
            patch_offsets.append([image_index, start, end])

        return crops, patch_offsets

    def _pad(self, images: torch.Tensor) -> torch.Tensor:
        """Pads the image to the target size using zero padding."""
        _, _, height, width = images.shape

        target_height, target_width = get_target_size(self.size)
        pad_h = max(0, target_height - height)
        pad_w = max(0, target_width - width)
        padding = (0, pad_w, 0, pad_h)

        padded_images = torch.nn.functional.pad(images, padding, mode="constant", value=0.0)
        return padded_images

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_split_image: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        mask_labels: Optional[List["torch.Tensor"]] = None,
        class_labels: Optional[Dict[int, int]] = None,
        **kwargs: Unpack[EoMTImageProcessorFastKwargs],
    ) -> BatchFeature:
        """Preprocesses the input images and masks if provided."""
        processed_images, patch_offsets = [], []

        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(images=stacked_images, interpolation=interpolation)
                resized_images_grouped[shape] = stacked_images
        images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(images)
        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            original_indices = [
                original_idx for original_idx, (img_shape, _) in grouped_images_index.items() if img_shape == shape
            ]

            if do_split_image:
                for idx, img in enumerate(stacked_images):
                    crops, offsets = self._split_image(img, original_indices[idx])
                    processed_images.extend(crops)
                    patch_offsets.extend(offsets)
            else:
                stacked_images = self._pad(stacked_images)
                processed_images_grouped[shape] = stacked_images

        if do_split_image:
            images, patch_offsets = reorder_crops_and_offsets(processed_images, patch_offsets)
        else:
            images = reorder_images(processed_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        images = reorder_images(processed_images_grouped, grouped_images_index)

        processed_images = torch.stack(images, dim=0) if return_tensors else images
        data = {"pixel_values": processed_images}

        if do_split_image and patch_offsets:
            data["patch_offsets"] = patch_offsets

        # Only perform padding on masks if they are provided.
        if mask_labels is not None:
            grouped_masks, grouped_masks_index = group_images_by_shape(mask_labels)
            padded_masks_grouped = {}

            for shape, stacked_masks in grouped_masks.items():
                padded = self._pad(stacked_masks)
                padded_masks_grouped[shape] = padded
            processed_masks = reorder_images(padded_masks_grouped, grouped_masks_index)

            data["mask_labels"] = torch.stack(processed_masks, dim=0) if return_tensors else processed_masks
            data["class_labels"] = class_labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[EoMTImageProcessorFastKwargs]) -> BatchFeature:
        # Pop do_center_crop as it's not required for EoMTImageProcessor.
        _ = kwargs.pop("do_center_crop", None)
        return super().preprocess(images, **kwargs)

    def _reverse_semantic_image_preprocessing(
        self,
        segmentation_logits: torch.Tensor,
        patch_offsets: List[Tuple[int, int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """
        Reconstructs full-size semantic segmentation logits from cropped image predictions.

        Args:
            segmentation_logits (`torch.Tensor`):
                A tensor of shape `(num_crops, num_classes, crop_height, crop_width)` representing predicted logits
                for each image crop.
            patch_offsets (`List[Tuple[int, int, int]]`):
                A list of tuples where each tuple contains:
                - `image_index` (int): Index of the original image this crop belongs to.
                - `start` (int): Start pixel index of the crop along the long dimension (height or width).
                - `end` (int): End pixel index of the crop along the long dimension.
            original_image_sizes (`List[Tuple[int, int]]`):
                List of original (height, width) dimensions for each image before preprocessing.
        """
        num_classes = segmentation_logits.shape[1]
        aggregated_logits = []
        crop_counts = []

        for image_size in original_image_sizes:
            height, width = get_size_with_aspect_ratio(
                image_size, self.size["shortest_edge"], self.size["longest_edge"]
            )
            aggregated_logits.append(torch.zeros((num_classes, height, width), device=segmentation_logits.device))
            crop_counts.append(torch.zeros((num_classes, height, width), device=segmentation_logits.device))

        # Stitch crops back into full-sized logit maps
        for crop_idx, (image_idx, crop_start, crop_end) in enumerate(patch_offsets):
            if original_image_sizes[image_idx][0] > original_image_sizes[image_idx][1]:
                aggregated_logits[image_idx][:, crop_start:crop_end, :] += segmentation_logits[crop_idx]
                crop_counts[image_idx][:, crop_start:crop_end, :] += 1
            else:
                aggregated_logits[image_idx][:, :, crop_start:crop_end] += segmentation_logits[crop_idx]
                crop_counts[image_idx][:, :, crop_start:crop_end] += 1

        # Normalize and resize logits to original image size
        reconstructed_logits = []
        for idx, (logit_sum, count) in enumerate(zip(aggregated_logits, crop_counts)):
            averaged_logits = logit_sum / count.clamp(min=1)
            resized_logits = torch.nn.functional.interpolate(
                averaged_logits[None, ...],
                size=original_image_sizes[idx],
                mode="bilinear",
                align_corners=False,
            )[0]

            reconstructed_logits.append(resized_logits)

        return reconstructed_logits

    def _reverse_panoptic_image_preprocessing(
        self,
        segmentation_logits: torch.Tensor,
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """Restores panoptic segmentation logits to their original image resolutions."""

        resized_logits = []

        for idx, original_size in enumerate(original_image_sizes):
            target_height, target_width = get_size_with_aspect_ratio(
                original_size, self.size["shortest_edge"], self.size["longest_edge"]
            )
            cropped_logits = segmentation_logits[idx][:, :target_height, :target_width]
            upsampled_logits = torch.nn.functional.interpolate(
                cropped_logits[None, ...], size=original_size, mode="bilinear", align_corners=False
            )[0]
            resized_logits.append(upsampled_logits)
        return resized_logits

    def post_process_semantic_segmentation(
        self,
        outputs,
        patch_offsets: List[Tuple[int, int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Post-processes model outputs into final semantic segmentation prediction."""

        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]

        size = get_target_size(self.size)
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=size,
            mode="bilinear",
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        output_logits = self._reverse_semantic_image_preprocessing(
            segmentation_logits, patch_offsets, original_image_sizes
        )

        preds = [logit.detach().cpu().argmax(dim=0).numpy() for logit in output_logits]
        return preds

    def post_process_panoptic_segmentation(
        self,
        outputs,
        original_image_sizes,
        stuff_classes: List[int] = [0],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
    ):
        """Post-processes model outputs into final panoptic segmentation prediction."""

        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        size = get_target_size(self.size)
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=size,
            mode="bilinear",
        )

        mask_probs_batch = self._reverse_panoptic_image_preprocessing(masks_queries_logits, original_image_sizes)
        pred_scores_batch, pred_labels_batch = class_queries_logits.softmax(dim=-1).max(-1)

        results: List = []

        for i in range(batch_size):
            mask_probs, pred_scores, pred_labels = remove_low_and_no_objects(
                mask_probs_batch[i], pred_scores_batch[i], pred_labels_batch[i], threshold, num_labels
            )

            # No mask found
            if mask_probs.shape[0] <= 0:
                height, width = original_image_sizes[i] if original_image_sizes is not None else mask_probs.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            segmentation, segments = compute_segments(
                mask_probs=mask_probs,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                stuff_classes=stuff_classes,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_size=original_image_sizes[i] if original_image_sizes is not None else None,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    def post_process_instance_segmentation(
        self,
        outputs,
        original_image_sizes,
        stuff_classes: List[int] = [0],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
    ):
        """Post-processes model outputs into instance segmentation predictions."""

        panoptic_results = self.post_process_panoptic_segmentation(
            outputs=outputs,
            original_image_sizes=original_image_sizes,
            stuff_classes=stuff_classes,
            threshold=threshold,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
        )

        instance_results = []

        for result in panoptic_results:
            segmentation_map = result["segmentation"]
            segments_info = result["segments_info"]

            instance_segmentation = torch.zeros_like(segmentation_map, dtype=torch.int32) - 1
            instance_segments = []

            instance_id = 0
            for segment in segments_info:
                label_id = segment["label_id"]

                if label_id in stuff_classes:
                    continue

                mask = segmentation_map == segment["id"]
                if mask.sum() == 0:
                    continue

                instance_segmentation[mask] = instance_id
                instance_segments.append(
                    {
                        "id": instance_id,
                        "label_id": label_id,
                        "score": segment["score"],
                    }
                )

                instance_id += 1

            instance_results.append({"segmentation": instance_segmentation, "segments_info": instance_segments})
        return instance_results


__all__ = ["EoMTImageProcessorFast"]
