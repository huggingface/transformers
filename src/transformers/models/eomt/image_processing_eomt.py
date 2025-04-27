# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file ehidden_statescept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ehidden_statespress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for EoMT."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    PaddingMode,
    pad,
    resize,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    filter_out_non_signature_kwargs,
    is_torch_available,
    logging,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    import torch.nn.functional as F


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

    # Eliminate disconnected tiny segments
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    print(
        f"Mask sum {mask_k_area}, original area {original_area}, final mask sum {final_mask.sum()}, mask exists {mask_exists}"
    )

    return mask_exists, final_mask


def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    target_size: Tuple[int, int] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.long, device=mask_probs.device)
    segments: List[Dict] = []

    # Compute per-pixel assignment based on weighted mask scores
    mask_probs_item = mask_probs.sigmoid()
    # Weigh each mask by its prediction score
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # Keep track of instances of each class
    current_segment_id = 0
    stuff_memory_list: Dict[str, int] = {}

    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()

        # Check if mask exists and large enough to be a segment
        mask_exists, final_mask = check_segment_validity(
            mask_labels, mask_probs_item, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            segmentation[final_mask] = current_segment_id
            torch.save(final_mask, f"/Users/espm5508/personal/transformers/delete/segments_pr_{k}.pt")
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "score": segment_score,
                }
            )
    return segmentation, segments


class EoMTImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = 640,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.num_labels = num_labels

    def scale_image_size(self, image_size, segmentation_type="semantic"):
        target_h, target_w = self.size["height"], self.size["width"]
        orig_h, orig_w = image_size

        # For semantic segmentation: scale up so that both sides are ≥ target size
        if segmentation_type == "semantic":
            scale_factor = max(target_h / orig_h, target_w / orig_w)
        elif segmentation_type == "instance" or segmentation_type == "panoptic":
            scale_factor = min(target_h / orig_h, target_w / orig_w)
        else:
            raise ValueError(f"Unknown segmentation type: {segmentation_type}")

        output_h = round(orig_h * scale_factor)
        output_w = round(orig_w * scale_factor)

        return (output_h, output_w)

    def resize(
        self,
        image: np.ndarray,
        segmentation_type: str,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> torch.tensor:
        image_size = get_image_size(
            image,
        )

        output_size = self.scale_image_size(image_size, segmentation_type)

        image = resize(
            image=image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            return_numpy=True,
            **kwargs,
        )

        return image

    def _preprocessing_semantic_segmentation(self, image):
        crops, origins = [], []

        image_size = get_image_size(image=image)  # (H, W)
        crop_size = self.size["height"]  # or 'width', both are equal

        long_side = max(image_size)

        num_crops = math.ceil(long_side / crop_size)
        overlap = num_crops * crop_size - long_side
        overlap_per_crop = (overlap / (num_crops - 1)) if num_crops > 1 else 0

        for i in range(num_crops):
            start_idx = int(i * (crop_size - overlap_per_crop))
            end_idx = start_idx + crop_size

            if image_size[0] > image_size[1]:  # taller image
                crop = image[:, start_idx:end_idx, :]
            else:  # wider image
                crop = image[:, :, start_idx:end_idx]

            crops.append(crop)
            origins.append([0, start_idx, end_idx])

        return crops, origins

    def _preprocessing_instance_panoptic_segmentation(self, image):
        height, width = get_image_size(image)
        pad_h = max(0, self.size["height"] - height)
        pad_w = max(0, self.size["width"] - width)

        padding = ((0, pad_h), (0, pad_w))

        # channel axis is last, so no need to override data_format
        padded_image = pad(image=image, padding=padding, mode=PaddingMode.CONSTANT, constant_values=0.0)
        return padded_image

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_type: str,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_resize:
            images = [
                self.resize(
                    image,
                    segmentation_type=segmentation_type,
                    resample=resample,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        transformed_images, origins_list = [], []

        if segmentation_type == "semantic":
            crops_list, origins_list = [], []
            for image in images:
                crops, origins = self._preprocessing_semantic_segmentation(image)
                crops_list.append(crops)
                origins_list.append(origins)

            transformed_images = np.stack(crops_list).squeeze(0)
            origins_list = np.array(origins_list).squeeze(0)
        elif segmentation_type == "instance" or segmentation_type == "panoptic":
            for image in images:
                transformed_image = self._preprocessing_instance_panoptic_segmentation(image)
                transformed_images.append(transformed_image)

        if do_rescale:
            images = [
                self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)
                for image in transformed_images
            ]

        if do_normalize:
            image_mean = np.array(image_mean).reshape(1, -1, 1, 1)
            image_std = np.array(image_std).reshape(1, -1, 1, 1)
            images = (images - image_mean) / image_std

        # # Normalize not working properly fix later
        # if do_normalize:
        #     images = [self.normalize(image, mean=image_mean, std=image_std, input_data_format=ChannelDimension.FIRST) for image in crops_list]

        output = {
            "pixel_values": images,
            "origins": origins_list,
        }
        return BatchFeature(output, tensor_type=return_tensors)

    def _revert_preprocessing_semantic(self, segmentation_logits, origins, original_image_sizes):
        logit_sums, logit_counts = [], []

        for image_size in original_image_sizes:
            height, width = self.scale_image_size(image_size, segmentation_type="semantic")
            logit_sums.append(torch.zeros((segmentation_logits.shape[1], height, width)))
            logit_counts.append(torch.zeros((segmentation_logits.shape[1], height, width)))

        for crop_idx, (image_idx, start, end) in enumerate(origins):
            if original_image_sizes[image_idx][0] > original_image_sizes[image_idx][1]:  # Tall image
                logit_sums[image_idx][:, start:end, :] += segmentation_logits[crop_idx]
                logit_counts[image_idx][:, start:end, :] += 1
            else:  # Wide image
                logit_sums[image_idx][:, :, start:end] += segmentation_logits[crop_idx]
                logit_counts[image_idx][:, :, start:end] += 1

        output_logits = []

        for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts)):
            combined = sums / counts.clamp(min=1)  # avoid division by zero
            combined = F.interpolate(combined[None, ...], size=original_image_sizes[i], mode="bilinear")[0]
            output_logits.append(combined)

        return output_logits

    def _revert_preprocessing_panoptic(self, segmentation_logits, original_image_sizes):
        output_logits = []

        for i, image_size in enumerate(original_image_sizes):
            height, width = self.scale_image_size(image_size, segmentation_type="panoptic")
            image_seg_logits = segmentation_logits[i][:, :height, :width]
            combined = F.interpolate(image_seg_logits[None, ...], size=(height, width), mode="bilinear")[0]
            output_logits.append(combined)

        return output_logits

    def postprocess_semnatic_segmentation(self, outputs, origins, original_image_sizes):
        masks_queries_logits = torch.tensor(outputs[0])  # [batch_size, num_queries, height, width]
        class_queries_logits = torch.tensor(outputs[1])  # [batch_size, num_queries, num_classes+1]

        size = (self.size["height"], self.size["width"])
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=size,
            mode="bilinear",
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        output_logits = self._revert_preprocessing_semantic(segmentation_logits, origins, original_image_sizes)

        preds = output_logits[0].argmax(0).cpu().numpy()
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
        masks_queries_logits = torch.tensor(outputs[0])  # [batch_size, num_queries, height, width]
        class_queries_logits = torch.tensor(outputs[1])  # [batch_size, num_queries, num_classes+1]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        size = (self.size["height"], self.size["width"])
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=size,
            mode="bilinear",
        )

        mask_probs = self._revert_preprocessing_panoptic(masks_queries_logits, original_image_sizes)

        pred_scores, pred_labels = class_queries_logits.softmax(dim=-1).max(-1)

        results = []

        for i in range(batch_size):
            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = (
                    original_image_sizes[i] if original_image_sizes is not None else mask_probs_item.shape[1:]
                )
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            segmentation, segments = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_size=original_image_sizes[i] if original_image_sizes is not None else None,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["EoMTImageProcessor"]
