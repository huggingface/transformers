# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for SAM."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    PILImageResampling,
    get_image_size,
)
from ...utils import auto_docstring, is_torch_available


if is_torch_available():
    import torch


class SamFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    SAM-specific kwargs for fast image processor.

    do_pad (`bool`, *optional*, defaults to `self.do_pad`):
        Whether to pad the image to the specified `pad_size`.
    pad_size (`Dict[str, int]`, *optional*, defaults to `self.pad_size`):
        Size of the output image after padding.
    mask_size (`Dict[str, int]`, *optional*, defaults to `self.mask_size`):
        Controls the size of the segmentation map after resize.
    mask_pad_size (`Dict[str, int]`, *optional*, defaults to `self.mask_pad_size`):
        Controls the size of the padding applied to the segmentation map.
    """

    do_pad: Optional[bool]
    pad_size: Optional[Dict[str, int]]
    mask_size: Optional[Dict[str, int]]
    mask_pad_size: Optional[Dict[str, int]]


@auto_docstring()
class SamImageProcessorFast(BaseImageProcessorFast):
    # ✅ ONLY class attributes for defaults (no custom __init__)
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"longest_edge": 1024}
    mask_size = {"longest_edge": 256}
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_pad = True
    pad_size = {"height": 1024, "width": 1024}
    mask_pad_size = {"height": 256, "width": 256}
    do_convert_rgb = True

    # ✅ Custom kwargs ONLY if slow processor needs them
    valid_kwargs = SamFastImageProcessorKwargs

    def _preprocess(self, images, segmentation_maps=None, **kwargs):
        """
        STANDARD FAST PROCESSOR PATTERN - Follow this structure exactly.
        BaseImageProcessorFast handles everything except SAM-specific logic.
        """

        # 🔥 STEP 1: ALWAYS use standard batch processing pattern
        grouped_images, grouped_images_index = group_images_by_shape(images)

        # 🔥 STEP 2: Process each group for maximum efficiency
        processed_groups = {}
        original_sizes = []
        reshaped_sizes = []

        for shape, stacked_images in grouped_images.items():
            # ✅ Images are already stacked by group_images_by_shape
            # stacked_images is already a tensor of shape (batch_size, C, H, W)

            # ✅ Record metadata (required for SAM post-processing)
            batch_size = stacked_images.shape[0]
            batch_original_sizes = [stacked_images.shape[-2:]] * batch_size
            original_sizes.extend(batch_original_sizes)

            # 🎯 STEP 3: Handle ONLY SAM-specific logic (longest_edge resizing)
            if kwargs.get("do_resize", self.do_resize):
                size = kwargs.get("size", self.size)
                # ✅ Handle both dict and SizeDict formats
                longest_edge = None
                if isinstance(size, dict) and "longest_edge" in size:
                    longest_edge = size["longest_edge"]
                elif hasattr(size, "longest_edge") and size.longest_edge is not None:
                    longest_edge = size.longest_edge

                if longest_edge is not None:
                    # ✅ This is the ONLY custom logic needed for SAM
                    current_h, current_w = stacked_images.shape[-2:]
                    target_h, target_w = self._get_preprocess_shape((current_h, current_w), longest_edge)
                    stacked_images = F.interpolate(
                        stacked_images, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )

            # ✅ Record reshaped sizes for post-processing
            batch_reshaped_sizes = [stacked_images.shape[-2:]] * batch_size
            reshaped_sizes.extend(batch_reshaped_sizes)

            # 🔥 STEP 4: LEVERAGE BASE CLASS for standard operations (DON'T REIMPLEMENT)
            stacked_images = self.rescale_and_normalize(  # ✅ Use base class method
                stacked_images,
                do_rescale=kwargs.get("do_rescale", self.do_rescale),
                rescale_factor=kwargs.get("rescale_factor", self.rescale_factor),
                do_normalize=kwargs.get("do_normalize", self.do_normalize),
                image_mean=kwargs.get("image_mean", self.image_mean),
                image_std=kwargs.get("image_std", self.image_std),
            )

            # 🎯 STEP 5: Handle SAM-specific padding (only if needed)
            if kwargs.get("do_pad", self.do_pad):
                pad_size = kwargs.get("pad_size", self.pad_size)
                stacked_images = self._pad_batch(stacked_images, pad_size)  # Minimal helper

            processed_groups[shape] = stacked_images

        # 🔥 STEP 6: ALWAYS reorder back (standard pattern)
        processed_images = reorder_images(processed_groups, grouped_images_index)

        # 🎯 STEP 7: Handle segmentation maps with same batch pattern
        processed_masks = None
        if segmentation_maps is not None:
            processed_masks = self._process_segmentation_maps_batch(segmentation_maps, **kwargs)

        # ✅ STEP 8: Stack processed images into single tensor (standard pattern)
        return_tensors = kwargs.get("return_tensors")
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        # ✅ STEP 9: Return in SAM's expected format
        result = {
            "pixel_values": processed_images,
            "original_sizes": original_sizes,  # ✅ Required for SAM post-processing
            "reshaped_input_sizes": reshaped_sizes,  # ✅ Required for SAM post-processing
        }
        if processed_masks is not None:
            result["labels"] = processed_masks

        return BatchFeature(data=result, tensor_type=return_tensors)

    def _get_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int) -> Tuple[int, int]:
        """
        ✅ MINIMAL HELPER: Calculate target size for SAM's longest edge resizing.
        This is SAM-specific logic that BaseImageProcessorFast doesn't have.
        """
        oldh, oldw = old_shape
        scale = longest_edge * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def _pad_batch(self, batch_images: torch.Tensor, pad_size: Dict[str, int]) -> torch.Tensor:
        """
        ✅ MINIMAL HELPER: Handle SAM's specific padding format.
        BaseImageProcessorFast has pad() but SAM needs this specific format.
        """
        target_h, target_w = pad_size["height"], pad_size["width"]
        current_h, current_w = batch_images.shape[-2:]

        pad_h = max(0, target_h - current_h)
        pad_w = max(0, target_w - current_w)

        if pad_h > 0 or pad_w > 0:
            # torch pad format: (left, right, top, bottom)
            padding = (0, pad_w, 0, pad_h)
            return F.pad(batch_images, padding, mode="constant", value=0)
        return batch_images

    def _process_segmentation_maps_batch(self, segmentation_maps: List, **kwargs) -> List[torch.Tensor]:
        """
        ✅ MINIMAL HELPER: Process segmentation maps with SAM's mask-specific parameters.
        Uses same batch processing pattern but with mask_size and NEAREST interpolation.
        """
        if not segmentation_maps:
            return None

        # Convert to tensors if needed
        if not isinstance(segmentation_maps[0], torch.Tensor):
            segmentation_maps = [torch.from_numpy(np.array(mask)).float() for mask in segmentation_maps]

        # Group by shape for batch processing
        grouped_masks, grouped_masks_index = group_images_by_shape(segmentation_maps)
        processed_groups = {}

        for shape, group_masks in grouped_masks.items():
            stacked_masks = torch.stack(group_masks, dim=0)

            # Handle mask resizing with mask_size
            if kwargs.get("do_resize", self.do_resize):
                mask_size = kwargs.get("mask_size", self.mask_size)
                if "longest_edge" in mask_size:
                    current_h, current_w = stacked_masks.shape[-2:]
                    target_h, target_w = self._get_preprocess_shape((current_h, current_w), mask_size["longest_edge"])
                    stacked_masks = F.interpolate(
                        stacked_masks.unsqueeze(1),  # Add channel dimension
                        size=(target_h, target_w),
                        mode="nearest",
                    ).squeeze(1)  # Remove channel dimension

            # Handle mask padding with mask_pad_size
            if kwargs.get("do_pad", self.do_pad):
                mask_pad_size = kwargs.get("mask_pad_size", self.mask_pad_size)
                stacked_masks = self._pad_batch(stacked_masks.unsqueeze(1), mask_pad_size).squeeze(1)

            processed_groups[shape] = stacked_masks

        # Reorder back to original order
        processed_masks = reorder_images(processed_groups, grouped_masks_index)
        return processed_masks

    def post_process_masks(
        self,
        masks: torch.Tensor,
        original_sizes: List[Tuple[int, int]],
        reshaped_input_sizes: List[Tuple[int, int]],
        mask_threshold: float = 0.0,
        binarize: bool = True,
        pad_size: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Convert model outputs back to original image dimensions.
        Fast version using torch.nn.functional.interpolate instead of PIL.
        """
        pad_size = pad_size or self.pad_size

        # Handle different input shapes
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)  # Add channel dimension

        batch_size = masks.shape[0]
        processed_masks = []

        for i in range(batch_size):
            mask = masks[i]
            original_size = original_sizes[i]
            reshaped_size = reshaped_input_sizes[i]

            # Remove padding first
            mask = mask[..., : reshaped_size[0], : reshaped_size[1]]

            # Resize back to original size
            if mask.shape[-2:] != original_size:
                mask = F.interpolate(
                    mask.unsqueeze(0), size=original_size, mode="bilinear", align_corners=False
                ).squeeze(0)

            # Apply threshold and binarization
            if binarize:
                mask = (mask > mask_threshold).float()

            processed_masks.append(mask)

        return processed_masks

    def generate_crop_boxes(
        self,
        image: torch.Tensor,
        target_size: int,
        crop_n_layers: int = 0,
        overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = None,
        crop_n_points_downscale_factor: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Fast version of crop box generation using pure torch operations.
        """
        # Get image dimensions
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:  # Batch dimension
                orig_h, orig_w = image.shape[-2:]
            else:
                orig_h, orig_w = image.shape[-2:]
        else:
            orig_h, orig_w = get_image_size(image)

        crop_boxes = []
        layer_idxs = []

        # Add full image as first crop
        crop_boxes.append([0, 0, orig_w, orig_h])
        layer_idxs.append(0)

        # Generate crop layers
        for i_layer in range(crop_n_layers):
            n_crops_per_side = 2 ** (i_layer + 1)
            overlap = int(overlap_ratio * target_size)
            crop_width = int(orig_w / n_crops_per_side + overlap)
            crop_height = int(orig_h / n_crops_per_side + overlap)

            crop_width = min(crop_width, orig_w)
            crop_height = min(crop_height, orig_h)

            for i in range(n_crops_per_side):
                for j in range(n_crops_per_side):
                    start_x = int(i * orig_w / n_crops_per_side)
                    start_y = int(j * orig_h / n_crops_per_side)

                    # Adjust to ensure we don't go out of bounds
                    end_x = min(start_x + crop_width, orig_w)
                    end_y = min(start_y + crop_height, orig_h)
                    start_x = max(0, end_x - crop_width)
                    start_y = max(0, end_y - crop_height)

                    crop_boxes.append([start_x, start_y, end_x, end_y])
                    layer_idxs.append(i_layer + 1)

        return crop_boxes, layer_idxs

    def filter_masks(
        self,
        masks: torch.Tensor,
        iou_scores: torch.Tensor,
        original_sizes: List[Tuple[int, int]],
        cropped_boxes: List[List[int]],
        filter_small_disconnected_regions: bool = True,
        use_stability_score: bool = True,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 0.95,
        min_mask_region_area: int = 100,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast version of mask filtering using pure torch operations.
        """
        if masks.numel() == 0:
            return masks, iou_scores

        # Compute stability scores if requested
        if use_stability_score:
            stability_scores = self._compute_stability_score(masks, stability_score_thresh, stability_score_offset)
            scores_to_use = stability_scores
        else:
            scores_to_use = iou_scores

        # Filter by score threshold
        score_mask = scores_to_use > stability_score_thresh
        masks = masks[score_mask]
        iou_scores = iou_scores[score_mask]

        # Filter small disconnected regions if requested
        if filter_small_disconnected_regions and min_mask_region_area > 0:
            # Simple area-based filtering (approximation)
            mask_areas = masks.sum(dim=(-2, -1))
            area_mask = mask_areas > min_mask_region_area
            masks = masks[area_mask]
            iou_scores = iou_scores[area_mask]

        return masks, iou_scores

    def _compute_stability_score(
        self, masks: torch.Tensor, mask_threshold: float, stability_score_offset: float
    ) -> torch.Tensor:
        """
        Compute stability scores for masks using torch operations.
        """
        # Compute areas at different thresholds
        area_thresh = (masks > mask_threshold).sum(dim=(-2, -1)).float()
        area_offset = (masks > (mask_threshold + stability_score_offset)).sum(dim=(-2, -1)).float()

        # Compute stability score
        stability_scores = area_offset / (area_thresh + 1e-8)
        return stability_scores

    def post_process_for_mask_generation(
        self, all_masks, all_scores, all_boxes, crops_nms_thresh, return_tensors="pt"
    ):
        """
        Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

        Args:
            all_masks (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all predicted segmentation masks
            all_scores (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all predicted iou scores
            all_boxes (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all bounding boxes of the predicted masks
            crops_nms_thresh (`float`):
                Threshold for NMS (Non Maximum Suppression) algorithm.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        return self._postprocess_for_mg(all_masks, all_scores, all_boxes, crops_nms_thresh)

    def _postprocess_for_mg(self, rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
        """
        Perform NMS (Non Maximum Suppression) on the outputs.
        Fast version using torch operations.
        """
        from torchvision.ops import batched_nms

        keep_by_nms = batched_nms(
            boxes=mask_boxes.float(),
            scores=iou_scores,
            idxs=torch.zeros(mask_boxes.shape[0]),
            iou_threshold=amg_crops_nms_thresh,
        )

        iou_scores = iou_scores[keep_by_nms]
        rle_masks = [rle_masks[i] for i in keep_by_nms]
        mask_boxes = mask_boxes[keep_by_nms]
        masks = [self._rle_to_mask(rle) for rle in rle_masks]

        return masks, iou_scores, rle_masks, mask_boxes

    def _rle_to_mask(self, rle):
        """Compute a binary mask from an uncompressed RLE."""
        height, width = rle["size"]
        mask = torch.empty(height * width, dtype=torch.bool)
        idx = 0
        parity = False
        for count in rle["counts"]:
            mask[idx : idx + count] = parity
            idx += count
            parity = not parity
        mask = mask.reshape(width, height)
        return mask.transpose(0, 1)  # Reshape to original shape

    def _build_point_grid(self, n_per_side: int) -> torch.Tensor:
        """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
        offset = 1 / (2 * n_per_side)
        points_one_side = torch.linspace(offset, 1 - offset, n_per_side)
        points_x = points_one_side[None, :].repeat(n_per_side, 1)
        points_y = points_one_side[:, None].repeat(1, n_per_side)
        points = torch.stack([points_x, points_y], dim=-1).reshape(-1, 2)
        return points

    def _normalize_coordinates(
        self, target_size: int, coords: torch.Tensor, original_size: Tuple[int, int], is_bounding_box=False
    ) -> torch.Tensor:
        """
        Expects a tensor of length 2 in the final dimension. Requires the original image size in (height, width)
        format.
        """
        old_height, old_width = original_size

        scale = target_size * 1.0 / max(old_height, old_width)
        new_height, new_width = old_height * scale, old_width * scale
        new_width = int(new_width + 0.5)
        new_height = int(new_height + 0.5)

        coords = coords.clone().float()

        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)

        coords[..., 0] = coords[..., 0] * (new_width / old_width)
        coords[..., 1] = coords[..., 1] * (new_height / old_height)

        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        return coords

    def _batched_mask_to_box(self, masks: torch.Tensor):
        """
        Computes the bounding boxes around the given input masks. The bounding boxes are in the XYXY format.
        Return [0,0,0,0] for an empty mask.
        """
        if torch.numel(masks) == 0:
            return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

        # Normalize shape to batch x height x width
        shape = masks.shape
        height, width = shape[-2:]

        # Get top and bottom edges
        in_height, _ = torch.max(masks, dim=-1)
        in_height_coords = in_height * torch.arange(height, device=in_height.device)[None, :]
        bottom_edges, _ = torch.max(in_height_coords, dim=-1)
        in_height_coords = in_height_coords + height * (~in_height)
        top_edges, _ = torch.min(in_height_coords, dim=-1)

        # Get left and right edges
        in_width, _ = torch.max(masks, dim=-2)
        in_width_coords = in_width * torch.arange(width, device=in_width.device)[None, :]
        right_edges, _ = torch.max(in_width_coords, dim=-1)
        in_width_coords = in_width_coords + width * (~in_width)
        left_edges, _ = torch.min(in_width_coords, dim=-1)

        # If the mask is empty the right edge will be to the left of the left edge.
        # Replace these boxes with [0, 0, 0, 0]
        empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
        out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
        out = out * (~empty_filter).unsqueeze(-1)

        # Return to original shape
        out = out.reshape(*shape[:-2], 4)
        return out


__all__ = ["SamImageProcessorFast"]
