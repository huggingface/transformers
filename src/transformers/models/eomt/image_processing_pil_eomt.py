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
"""Image processor class for EoMT."""

import math

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode, get_size_with_aspect_ratio
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available
from ...utils.import_utils import requires


if is_torch_available():
    import torch


# Adapted from transformers.models.maskformer.image_processing_maskformer.convert_segmentation_map_to_binary_masks
def convert_segmentation_map_to_binary_masks(
    segmentation_map: np.ndarray,
    instance_id_to_semantic_id: dict[int, int] | None = None,
    ignore_index: int | None = None,
):
    if ignore_index is not None:
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # Get unique ids (class or instance ids based on input)
    all_labels = np.unique(segmentation_map)

    # Drop background label if applicable
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # Generate a binary mask for each object instance
    binary_masks = [(segmentation_map == i) for i in all_labels]

    # Stack the binary masks
    if binary_masks:
        binary_masks = np.stack(binary_masks, axis=0)
    else:
        binary_masks = np.zeros((0, *segmentation_map.shape))

    # Convert instance ids to class ids
    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0])

        for label in all_labels:
            class_id = instance_id_to_semantic_id[label + 1 if ignore_index is not None else label]
            labels[all_labels == label] = class_id - 1 if ignore_index is not None else class_id
    else:
        labels = all_labels

    return binary_masks.astype(np.float32), labels.astype(np.int64)


# Adapted from transformers.models.eomt.image_processing_eomt.check_segment_validity
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


# Adapted from transformers.models.eomt.image_processing_eomt.EomtImageProcessorKwargs
class EomtImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_split_image (`bool`, *optional*, defaults to `self.do_split_image`):
        Whether to split the input images into overlapping patches for semantic segmentation. If set to `True`, the
        input images will be split into patches of size `size["shortest_edge"]` with an overlap between patches.
        Otherwise, the input images will be padded to the target size.
    ignore_index (`int`, *optional*, defaults to `self.ignore_index`):
        Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
        denoted with 0 (background) will be replaced with `ignore_index`.
    """

    do_split_image: bool
    ignore_index: int | None


# Adapted from transformers.models.eomt.image_processing_eomt.compute_segments
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    stuff_classes,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    target_size: tuple[int, int] | None = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.long, device=mask_probs.device) - 1
    segments: list[dict] = []

    # Compute per-pixel assignment based on weighted mask scores
    mask_probs = mask_probs.sigmoid()
    mask_labels = (pred_scores[:, None, None] * mask_probs).argmax(0)

    # Keep track of instances of each class
    current_segment_id = 0
    stuff_memory_list: dict[str, int] = {}

    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()

        # Check if mask exists and large enough to be a segment
        mask_exists, final_mask = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if not mask_exists:
            continue

        if stuff_classes and pred_class in stuff_classes:
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


# Adapted from transformers.models.eomt.image_processing_eomt.get_target_size
def get_target_size(size_dict: dict[str, int]) -> tuple[int, int]:
    """Returns the height and width from a size dict."""
    target_height = size_dict["shortest_edge"]
    target_width = size_dict["longest_edge"] or target_height

    return target_height, target_width


# Adapted from transformers.models.eomt.image_processing_eomt.remove_low_and_no_objects
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


@auto_docstring
@requires(backends=("torch",))
class EomtImageProcessorPil(PilBackend):
    valid_kwargs = EomtImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 640, "longest_edge": 640}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_split_image = False
    do_pad = False
    ignore_index = None

    def __init__(self, **kwargs: Unpack[EomtImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _split_image(self, image: np.ndarray, size: SizeDict, image_index: int) -> tuple[list, list]:
        """Slices an image into overlapping patches for semantic segmentation."""

        patches, patch_offsets = [], []

        height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        patch_size = size.shortest_edge

        longer_side = max(height, width)
        num_patches = math.ceil(longer_side / patch_size)
        total_overlap = num_patches * patch_size - longer_side
        overlap_per_patch = total_overlap / (num_patches - 1) if num_patches > 1 else 0

        for i in range(num_patches):
            start = int(i * (patch_size - overlap_per_patch))
            end = start + patch_size

            if height > width:
                patch = image[:, start:end, :]
            else:
                patch = image[:, :, start:end]

            patches.append(patch)
            patch_offsets.append([image_index, start, end])

        return patches, patch_offsets

    def _pad(self, image: np.ndarray, size: SizeDict) -> np.ndarray:
        """Pads the image to the target size using zero padding."""

        height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        target_height, target_width = get_target_size(
            {"shortest_edge": size.shortest_edge, "longest_edge": size.longest_edge or size.shortest_edge}
        )
        pad_h = max(0, target_height - height)
        pad_w = max(0, target_width - width)

        padding = ((0, pad_h), (0, pad_w))
        padded_image = np_pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=0.0,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )
        return padded_image

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: "list[torch.Tensor] | None" = None,
        instance_id_to_semantic_id: dict[int, int] | None = None,
        **kwargs: Unpack[EomtImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess for corresponding images.
        instance_id_to_semantic_id (`list[dict[int, int]]` or `dict[int, int]`, *optional*):
            A mapping between object instance ids and class ids.
        """
        return super().preprocess(images, segmentation_maps, instance_id_to_semantic_id, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        instance_id_to_semantic_id: dict[int, int] | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        return_tensors: str | TensorType | None,
        **kwargs: Unpack[EomtImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format
        )
        ignore_index = kwargs.pop("ignore_index", None)
        images_kwargs = kwargs.copy()
        processed_images, patch_offsets = self._preprocess(images, **images_kwargs)
        data = {}
        data["pixel_values"] = processed_images
        data["patch_offsets"] = patch_offsets

        if segmentation_maps is not None:
            processed_segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            segmentation_maps_kwargs = kwargs.copy()
            segmentation_maps_kwargs.update(
                {
                    "do_normalize": False,
                    "do_rescale": False,
                    # Nearest interpolation is used for segmentation maps instead of BILINEAR.
                    "resample": PILImageResampling.NEAREST,
                }
            )

            processed_segmentation_maps, _ = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            )
            processed_segmentation_maps = [
                segmentation_map.squeeze(0).astype(np.int64) for segmentation_map in processed_segmentation_maps
            ]

            # Convert to list of binary masks and labels
            mask_labels, class_labels = [], []
            for idx, segmentation_map in enumerate(processed_segmentation_maps):
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id
                # Use instance2class_id mapping per image
                masks, classes = convert_segmentation_map_to_binary_masks(
                    segmentation_map, instance_id, ignore_index=ignore_index
                )

                mask_labels.append(torch.from_numpy(masks))
                class_labels.append(torch.from_numpy(classes))

            # we cannot batch them since they don't share a common class size
            data["mask_labels"] = mask_labels
            data["class_labels"] = class_labels

        if patch_offsets:
            data["patch_offsets"] = [torch.tensor(offsets) for offsets in patch_offsets]

        return BatchFeature(
            data=data,
            tensor_type=return_tensors,
            skip_tensor_conversion=["patch_offsets", "mask_labels", "class_labels"],
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_split_image: bool,
        do_pad: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        **kwargs,
    ):
        """Preprocesses the input images and masks if provided."""
        processed_images, patch_offsets = [], []

        # Resize images
        resized_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample)
            resized_images.append(image)

        # Split images into patches if requested
        if do_split_image:
            for idx, img in enumerate(resized_images):
                patches, offsets = self._split_image(img, size, idx)
                processed_images.extend(patches)
                patch_offsets.extend(offsets)
            images = processed_images
        else:
            images = resized_images

        # Pad images if requested
        if do_pad:
            images = [self._pad(img, size) for img in images]

        # Rescale and normalize
        processed_images = []
        for image in images:
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return processed_images, patch_offsets

    def merge_image_patches(
        self,
        segmentation_logits: "torch.Tensor",
        patch_offsets: list[tuple[int, int, int]],
        target_sizes: list[tuple[int, int]],
        size: dict[str, int],
    ) -> "list[torch.Tensor]":
        """
        Reconstructs full-size semantic segmentation logits from patch predictions.

        Args:
            segmentation_logits (`torch.Tensor`):
                A tensor of shape `(num_patches, num_classes, patch_height, patch_width)` representing predicted logits
                for each image patch.
            patch_offsets (`list[tuple[int, int, int]]`):
                A list of tuples where each tuple contains:
                - `image_index` (int): Index of the original image this patch belongs to.
                - `start` (int): Start pixel index of the patch along the long dimension (height or width).
                - `end` (int): End pixel index of the patch along the long dimension.
            target_sizes (`list[tuple[int, int]]`):
                list of original (height, width) dimensions for each image before preprocessing.
            size (`dict[str, int]`):
                A size dict which was used to resize.
        """
        num_classes = segmentation_logits.shape[1]
        aggregated_logits = []
        patch_counts = []

        for image_size in target_sizes:
            height, width = get_size_with_aspect_ratio(image_size, size["shortest_edge"], size["longest_edge"])
            aggregated_logits.append(torch.zeros((num_classes, height, width), device=segmentation_logits.device))
            patch_counts.append(torch.zeros((num_classes, height, width), device=segmentation_logits.device))

        # Stitch patches back into full-sized logit maps
        for patch_idx, (image_idx, patch_start, patch_end) in enumerate(patch_offsets):
            if target_sizes[image_idx][0] > target_sizes[image_idx][1]:
                aggregated_logits[image_idx][:, patch_start:patch_end, :] += segmentation_logits[patch_idx]
                patch_counts[image_idx][:, patch_start:patch_end, :] += 1
            else:
                aggregated_logits[image_idx][:, :, patch_start:patch_end] += segmentation_logits[patch_idx]
                patch_counts[image_idx][:, :, patch_start:patch_end] += 1

        # Normalize and resize logits to original image size
        reconstructed_logits = []
        for idx, (logit_sum, count) in enumerate(zip(aggregated_logits, patch_counts)):
            averaged_logits = logit_sum / count.clamp(min=1)
            resized_logits = torch.nn.functional.interpolate(
                averaged_logits[None, ...], size=target_sizes[idx], mode="bilinear", align_corners=False
            )[0]

            reconstructed_logits.append(resized_logits)

        return reconstructed_logits

    def unpad_image(
        self, segmentation_logits: "torch.Tensor", target_sizes: list[tuple[int, int]], size: dict[str, int]
    ) -> "list[torch.Tensor]":
        """Restores panoptic segmentation logits to their original image resolutions."""

        resized_logits = []

        for idx, original_size in enumerate(target_sizes):
            target_height, target_width = get_size_with_aspect_ratio(
                original_size, size["shortest_edge"], size["longest_edge"]
            )
            cropped_logits = segmentation_logits[idx][:, :target_height, :target_width]
            upsampled_logits = torch.nn.functional.interpolate(
                cropped_logits[None, ...], size=original_size, mode="bilinear", align_corners=False
            )[0]
            resized_logits.append(upsampled_logits)
        return resized_logits

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple[int, int]], size: dict[str, int] | None = None
    ) -> np.ndarray:
        """Post-processes model outputs into final semantic segmentation prediction."""

        size = size if size is not None else self.size

        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        patch_offsets = outputs.patch_offsets

        output_size = get_target_size(size)
        masks_queries_logits = torch.nn.functional.interpolate(masks_queries_logits, size=output_size, mode="bilinear")

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        if patch_offsets:
            output_logits = self.merge_image_patches(segmentation_logits, patch_offsets, target_sizes, size)
        else:
            output_logits = []

            for idx in range(len(segmentation_logits)):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation_logits[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                output_logits.append(resized_logits[0])

        preds = [logit.argmax(dim=0) for logit in output_logits]
        return preds

    def post_process_panoptic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        stuff_classes: list[int] | None = None,
        size: dict[str, int] | None = None,
    ):
        """Post-processes model outputs into final panoptic segmentation prediction."""

        size = size if size is not None else self.size

        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        output_size = get_target_size(size)
        masks_queries_logits = torch.nn.functional.interpolate(masks_queries_logits, size=output_size, mode="bilinear")

        mask_probs_batch = self.unpad_image(masks_queries_logits, target_sizes, size)
        pred_scores_batch, pred_labels_batch = class_queries_logits.softmax(dim=-1).max(-1)

        results: list = []

        for i in range(batch_size):
            mask_probs, pred_scores, pred_labels = remove_low_and_no_objects(
                mask_probs_batch[i], pred_scores_batch[i], pred_labels_batch[i], threshold, num_labels
            )

            # No mask found
            if mask_probs.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs.shape[1:]
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
                target_size=target_sizes[i] if target_sizes is not None else None,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    def post_process_instance_segmentation(
        self, outputs, target_sizes: list[tuple[int, int]], threshold: float = 0.8, size: dict[str, int] | None = None
    ):
        """Post-processes model outputs into Instance Segmentation Predictions."""

        size = size if size is not None else self.size

        masks_queries_logits = outputs.masks_queries_logits
        class_queries_logits = outputs.class_queries_logits

        output_size = get_target_size(size)
        masks_queries_logits = torch.nn.functional.interpolate(masks_queries_logits, size=output_size, mode="bilinear")

        mask_probs_batch = self.unpad_image(masks_queries_logits, target_sizes, size)

        device = masks_queries_logits.device
        batch_size = class_queries_logits.shape[0]
        num_queries = class_queries_logits.shape[-2]

        results = []

        for i in range(batch_size):
            mask_pred = mask_probs_batch[i]
            mask_class = class_queries_logits[i]

            # Remove the null class `[..., :-1]`
            scores, pred_classes = mask_class.softmax(dim=-1)[..., :-1].max(-1)
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores * mask_scores

            segmentation = torch.zeros(target_sizes[i], device=device) - 1

            instance_maps, segments = [], []
            current_segment_id = 0
            for j in range(num_queries):
                score = pred_scores[j].item()

                if not torch.all(pred_masks[j] == 0) and score >= threshold:
                    segmentation[pred_masks[j] == 1] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[j].item(),
                            "score": round(score, 6),
                        }
                    )
                    current_segment_id += 1
                    instance_maps.append(pred_masks[j])

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["EomtImageProcessorPil"]
