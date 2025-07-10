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

import math
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    SizeDict,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    PILImageResampling,
)
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)


# Import torchvision only once with proper availability checks
if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F
    from torchvision.ops.boxes import batched_nms

logger = logging.get_logger(__name__)


class SamFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Additional keyword arguments for SAM image processor.
    """

    mask_size: Optional[Dict[str, int]]
    mask_pad_size: Optional[Dict[str, int]]
    pad_size: Optional[Dict[str, int]]


@add_start_docstrings(
    "Constructs a fast SAM image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain size.
        size (`Dict[str, int]`, *optional*, defaults to `{"longest_edge": 1024}`):
            Size of the output image after resizing. Resizes the longest edge of the image to match
            `size["longest_edge"]` while maintaining the aspect ratio.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input by a certain factor.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            Rescaling factor to apply to the input.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the input.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to normalize the input by.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to normalize the input by.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the input to RGB format.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the input.
        mask_size (`Dict[str, int]`, *optional*, defaults to `{"longest_edge": 256}`):
            Size of the output segmentation map after resizing. Resizes the longest edge of the image to match
            `size["longest_edge"]` while maintaining the aspect ratio.
        mask_pad_size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Size of the output segmentation map after padding.
        pad_size (`Dict[str, int]`, *optional*, defaults to `{"height": 1024, "width": 1024}`):
            Size of the output image after padding.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the parent class.
    """,
)
class SamImageProcessorFast(BaseImageProcessorFast):
    """
    Fast image processor for Segment Anything Model (SAM).
    Uses torch and torchvision for faster image processing operations.
    """

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    do_resize = True
    size = {"longest_edge": 1024}
    mask_size = {"longest_edge": 256}
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_pad = True
    pad_size = {"height": 1024, "width": 1024}
    mask_pad_size = {"height": 256, "width": 256}
    do_convert_rgb = True
    valid_kwargs = SamFastImageProcessorKwargs

    def __init__(self, **kwargs):
        """
        Initialize the SamImageProcessorFast with optional configuration parameters.
        """
        super().__init__(**kwargs)

    def _get_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.

        Args:
            old_shape: Original shape (height, width) of the image
            longest_edge: Target size for the longest edge

        Returns:
            New shape (height, width) maintaining aspect ratio
        """
        oldh, oldw = old_shape
        scale = longest_edge * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def _preprocess(
        self,
        images: List[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool = None,
        crop_size: SizeDict = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        mask_size: Optional[Dict[str, int]] = None,
        mask_pad_size: Optional[Dict[str, int]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        segmentation_maps: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images for the SAM model.
        """
        original_sizes = []
        reshaped_input_sizes = []

        # Store original sizes before any processing
        for image in images:
            original_sizes.append(image.shape[-2:])

        # Group images by shape for batch processing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}

        # Process each group of same-shaped images
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                # Calculate target size for this shape
                target_size = self._get_preprocess_shape(shape[-2:], size["longest_edge"])
                resize_interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
                resized_images = F.resize(stacked_images, target_size, interpolation=resize_interpolation)
            else:
                resized_images = stacked_images

            resized_images_grouped[shape] = resized_images

        # Restore original order
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Store reshaped sizes
        for img in resized_images:
            reshaped_input_sizes.append(img.shape[-2:])

        # Group resized images for further processing
        grouped_resized, grouped_resized_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}

        # Apply rescale, normalize, and padding to each group
        for shape, stacked_images in grouped_resized.items():
            # Apply rescale and normalization
            processed_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            # Apply padding if required
            if do_pad:
                padded_height, padded_width = pad_size["height"], pad_size["width"]
                input_height, input_width = shape[-2:]
                pad_bottom = max(0, padded_height - input_height)
                pad_right = max(0, padded_width - input_width)
                padding = (0, 0, pad_right, pad_bottom)
                processed_images = F.pad(processed_images, padding, fill=0)

            processed_images_grouped[shape] = processed_images

        # Restore original order
        processed_images = reorder_images(processed_images_grouped, grouped_resized_index)

        # Process segmentation maps if provided
        processed_masks = None
        if segmentation_maps is not None:
            if len(segmentation_maps) != len(images):
                raise ValueError(
                    f"Number of segmentation maps ({len(segmentation_maps)}) does not match "
                    f"number of images ({len(images)})"
                )

            # Validate mask sizes
            for i, mask in enumerate(segmentation_maps):
                if mask.dim() == 2:
                    segmentation_maps[i] = mask.unsqueeze(0)

                mask_h, mask_w = segmentation_maps[i].shape[-2:]
                img_h, img_w = original_sizes[i]
                if mask_h != img_h or mask_w != img_w:
                    raise ValueError(
                        f"Segmentation map size ({mask_h}, {mask_w}) does not match image size ({img_h}, {img_w})"
                    )

            # Group masks by shape for batch processing
            grouped_masks, grouped_masks_index = group_images_by_shape(segmentation_maps)
            processed_masks_grouped = {}

            # Process each group of masks
            for shape, stacked_masks in grouped_masks.items():
                if do_resize and mask_size is not None:
                    mask_target_size = self._get_preprocess_shape(shape[-2:], mask_size["longest_edge"])
                    resized_masks = F.resize(
                        stacked_masks.float(), mask_target_size, interpolation=F.InterpolationMode.NEAREST
                    )
                else:
                    resized_masks = stacked_masks

                if do_pad and mask_pad_size is not None:
                    mask_pad_h, mask_pad_w = mask_pad_size["height"], mask_pad_size["width"]
                    mask_h, mask_w = resized_masks.shape[-2:]
                    pad_bottom = max(0, mask_pad_h - mask_h)
                    pad_right = max(0, mask_pad_w - mask_w)
                    padding = (0, 0, pad_right, pad_bottom)
                    resized_masks = F.pad(resized_masks, padding, fill=0)

                processed_masks_grouped[shape] = resized_masks.long()

            # Restore original mask order
            processed_masks = reorder_images(processed_masks_grouped, grouped_masks_index)

        # Stack tensors if required
        if return_tensors:
            processed_images = torch.stack(processed_images, dim=0)
            if processed_masks is not None:
                processed_masks = torch.stack(processed_masks, dim=0)

        return BatchFeature(
            {
                "pixel_values": processed_images,
                "original_sizes": original_sizes,
                "reshaped_input_sizes": reshaped_input_sizes,
                **({"labels": processed_masks} if processed_masks is not None else {}),
            }
        )

    def post_process_masks(
        self,
        masks: torch.Tensor,
        original_sizes: Union[List[Tuple[int, int]], torch.Tensor],
        reshaped_input_sizes: Union[List[Tuple[int, int]], torch.Tensor],
        mask_threshold: float = 0.0,
        binarize: bool = True,
        pad_size: Optional[Dict[str, int]] = None,
        return_tensors: str = "pt",
    ) -> List[torch.Tensor]:
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks: Batch of predicted masks
            original_sizes: Original sizes of each image
            reshaped_input_sizes: Resized input shapes before padding
            mask_threshold: Threshold for binarizing masks
            binarize: Whether to binarize masks
            pad_size: Padding sizes used (defaults to self.pad_size)
            return_tensors: Return format, only "pt" is supported

        Returns:
            List of processed masks matching original image sizes
        """
        if return_tensors != "pt":
            raise ValueError("Only returning PyTorch tensors is currently supported.")

        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size["height"], pad_size["width"])

        if isinstance(original_sizes, torch.Tensor):
            original_sizes = original_sizes.tolist()
        if isinstance(reshaped_input_sizes, torch.Tensor):
            reshaped_input_sizes = reshaped_input_sizes.tolist()

        output_masks = []
        for i, original_size in enumerate(original_sizes):
            if masks[i].dim() == 4:
                mask_batch = masks[i]
            elif masks[i].dim() == 3:
                mask_batch = masks[i].unsqueeze(0)
            else:
                mask_batch = masks[i].unsqueeze(0).unsqueeze(0)

            interpolated_mask = F.resize(
                mask_batch.float(), target_image_size, interpolation=F.InterpolationMode.BILINEAR, antialias=False
            )

            interpolated_mask = interpolated_mask[..., : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1]]

            interpolated_mask = F.resize(
                interpolated_mask, original_size, interpolation=F.InterpolationMode.BILINEAR, antialias=False
            )

            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold

            output_masks.append(interpolated_mask)

        return output_masks

    def generate_crop_boxes(
        self,
        image: torch.Tensor,
        target_size: int,
        crop_n_layers: int = 0,
        overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        crop_n_points_downscale_factor: Optional[int] = 1,
        device: Optional[torch.device] = None,
        return_tensors: str = "pt",
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

        Args:
            image (`torch.Tensor`):
                Input original image tensor
            target_size (`int`):
                Target size of the resized image
            crop_n_layers (`int`, *optional*, defaults to 0):
                Number of crop layers to generate
            overlap_ratio (`float`, *optional*, defaults to 512/1500):
                Degree of overlap between crops
            points_per_crop (`int`, *optional*, defaults to 32):
                Number of points to sample per crop
            crop_n_points_downscale_factor (`int`, *optional*, defaults to 1):
                Factor to reduce points in deeper layers
            device (`torch.device`, *optional*):
                Device to use for computation
            return_tensors (`str`, *optional*, defaults to "pt"):
                Output format, only "pt" is supported

        Returns:
            `Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]`:
                Tuple containing crop boxes, points per crop, cropped images, and input labels
        """
        if device is None:
            device = image.device  # Get device from input image

        image_height, image_width = image.shape[-2:]

        crop_boxes = [[0, 0, image_width, image_height]]

        points_grid = []
        for i in range(crop_n_layers + 1):
            n_points = int(points_per_crop / (crop_n_points_downscale_factor**i))
            points_grid.append(self._build_point_grid(n_points))

        layer_idxs = [0]
        short_side = min(image_height, image_width)

        for i_layer in range(crop_n_layers):
            n_crops_per_side = 2 ** (i_layer + 1)
            overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

            crop_width = int(math.ceil((overlap * (n_crops_per_side - 1) + image_width) / n_crops_per_side))
            crop_height = int(math.ceil((overlap * (n_crops_per_side - 1) + image_height) / n_crops_per_side))

            crop_box_x0 = [int((crop_width - overlap) * i) for i in range(n_crops_per_side)]
            crop_box_y0 = [int((crop_height - overlap) * i) for i in range(n_crops_per_side)]

            for left, top in product(crop_box_x0, crop_box_y0):
                box = [left, top, min(left + crop_width, image_width), min(top + crop_height, image_height)]
                crop_boxes.append(box)
                layer_idxs.append(i_layer + 1)

        cropped_images = []
        total_points_per_crop = []

        for i, crop_box in enumerate(crop_boxes):
            left, top, right, bottom = crop_box
            cropped_im = image[:, top:bottom, left:right]
            cropped_images.append(cropped_im)

            cropped_im_size = (bottom - top, right - left)
            points_scale = torch.tensor(cropped_im_size, device=device).flip(0).unsqueeze(0)

            layer_points = points_grid[layer_idxs[i]]
            if not isinstance(layer_points, torch.Tensor):
                layer_points = torch.tensor(layer_points, device=device, dtype=torch.float32)

            points = layer_points * points_scale

            scale = target_size * 1.0 / max(image_height, image_width)
            new_height, new_width = image_height * scale, image_width * scale
            new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)

            points_tensor = points.clone()
            points_tensor[..., 0] = points_tensor[..., 0] * (new_width / image_width)
            points_tensor[..., 1] = points_tensor[..., 1] * (new_height / image_height)

            total_points_per_crop.append(points_tensor)

        crop_boxes_tensor = torch.tensor(crop_boxes, device=device)

        points_per_crop = torch.stack(total_points_per_crop, dim=0).unsqueeze(0)

        input_labels = torch.ones(points_per_crop.shape[:-1], dtype=torch.int64, device=device)

        return crop_boxes_tensor, points_per_crop, cropped_images, input_labels

    def _build_point_grid(self, n_per_side: int) -> torch.Tensor:
        """
        Generates a 2D grid of points evenly spaced in [0,1]x[0,1].

        Args:
            n_per_side: Despite the name, this is actually the total number
                       of points desired. We calculate sqrt(n_per_side) to get
                       the actual number of points per side.

        Returns:
            Grid of points as a tensor
        """

        actual_points_per_side = int(math.ceil(math.sqrt(n_per_side)))

        offset = 1 / (2 * actual_points_per_side)
        points_one_side = torch.linspace(offset, 1 - offset, actual_points_per_side)
        points_x = points_one_side.unsqueeze(0).repeat(actual_points_per_side, 1)
        points_y = points_one_side.unsqueeze(1).repeat(1, actual_points_per_side)
        points = torch.stack([points_x, points_y], dim=-1).reshape(-1, 2)
        return points

    def filter_masks(
        self,
        masks: torch.Tensor,
        iou_scores: torch.Tensor,
        original_size: Tuple[int, int],
        cropped_box_image: List[int],
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        mask_threshold: float = 0,
        stability_score_offset: int = 1,
        return_tensors: str = "pt",
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
        """
        Filters predicted masks based on various criteria.

        Args:
            masks: Tensor of predicted masks
            iou_scores: Tensor of IoU scores
            original_size: Original image dimensions (height, width)
            cropped_box_image: Coordinates of the crop box [left, top, right, bottom]
            pred_iou_thresh: Threshold for IoU scores
            stability_score_thresh: Threshold for stability scores
            mask_threshold: Threshold for binarizing masks
            stability_score_offset: Offset for stability score calculation
            return_tensors: Output format, only "pt" is supported

        Returns:
            Tuple of filtered mask encodings, scores, and boxes
        """
        if return_tensors != "pt":
            raise ValueError("Only returning PyTorch tensors is currently supported.")

        original_height, original_width = original_size

        if iou_scores.dim() > 1:
            iou_scores = iou_scores.flatten(0, 1)

        if masks.dim() > 3:
            masks = masks.flatten(0, 1)

        batch_size = masks.shape[0]
        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)

        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)

        if stability_score_thresh > 0.0:
            stability_scores = self._compute_stability_score(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)

        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]

        masks = masks > mask_threshold
        converted_boxes = self._batched_mask_to_box(masks)

        keep_mask = ~self._is_box_near_crop_edge(
            converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
        )

        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]

        masks = self._pad_masks(masks, cropped_box_image, original_height, original_width)

        rle_masks = self._mask_to_rle(masks)

        return rle_masks, scores, converted_boxes

    def _compute_stability_score(
        self, masks: torch.Tensor, mask_threshold: float, stability_score_offset: int
    ) -> torch.Tensor:
        """
        Compute the stability score for masks.

        Args:
            masks: Input masks tensor
            mask_threshold: Threshold for mask binarization
            stability_score_offset: Offset for stability computation

        Returns:
            Stability scores tensor
        """
        intersections = (
            (masks > (mask_threshold + stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        )
        unions = (
            (masks > (mask_threshold - stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        )
        stability_scores = intersections / unions
        return stability_scores

    def _batched_mask_to_box(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Convert masks to bounding boxes in XYXY format.

        Args:
            masks: Input binary masks tensor

        Returns:
            Tensor of bounding boxes in XYXY format
        """
        if torch.numel(masks) == 0:
            return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

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

        # Handle empty masks
        empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
        out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
        out = out * (~empty_filter).unsqueeze(-1)

        out = out.reshape(*shape[:-2], 4)
        return out

    def _is_box_near_crop_edge(
        self, boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
    ) -> torch.Tensor:
        """
        Filter masks at the edge of a crop, but not at the edge of the original image.

        Args:
            boxes: Bounding boxes in XYXY format
            crop_box: Crop box coordinates [left, top, right, bottom]
            orig_box: Original image box coordinates [left, top, right, bottom]
            atol: Tolerance for considering boxes near the edge

        Returns:
            Boolean tensor indicating boxes near crop edges
        """
        crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
        orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)

        left, top, _, _ = crop_box
        offset = torch.tensor([[left, top, left, top]], device=boxes.device)
        if len(boxes.shape) == 3:
            offset = offset.unsqueeze(1)
        boxes = (boxes + offset).float()

        near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
        near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
        near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
        return torch.any(near_crop_edge, dim=1)

    def _pad_masks(self, masks: torch.Tensor, crop_box: List[int], orig_height: int, orig_width: int) -> torch.Tensor:
        """
        Pad masks to the original image size.

        Args:
            masks: Masks tensor to pad
            crop_box: Crop box coordinates [left, top, right, bottom]
            orig_height: Original image height
            orig_width: Original image width

        Returns:
            Padded masks tensor
        """
        left, top, right, bottom = crop_box
        if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
            return masks

        pad_x, pad_y = orig_width - (right - left), orig_height - (bottom - top)
        pad = (left, pad_x - left, top, pad_y - top)
        return F.pad(masks, pad, fill=0)

    def _mask_to_rle(self, input_mask: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Convert masks to run-length encoding format.

        Args:
            input_mask: Binary masks tensor

        Returns:
            List of mask encodings in RLE format
        """
        batch_size, height, width = input_mask.shape
        input_mask = input_mask.permute(0, 2, 1).flatten(1)

        diff = input_mask[:, 1:] ^ input_mask[:, :-1]
        change_indices = diff.nonzero()

        out = []
        for i in range(batch_size):
            cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
            if len(cur_idxs) == 0:
                if input_mask[i, 0] == 0:
                    out.append({"size": [height, width], "counts": [height * width]})
                else:
                    out.append({"size": [height, width], "counts": [0, height * width]})
                continue

            btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
            counts = [] if input_mask[i, 0] == 0 else [0]
            counts += [cur_idxs[0].item()] + btw_idxs.tolist() + [height * width - cur_idxs[-1].item()]
            out.append({"size": [height, width], "counts": counts})

        return out

    def _rle_to_mask(self, rle: Dict[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Compute a binary mask from an uncompressed RLE.

        Args:
            rle: RLE encoding of a mask
            device: Optional device for the output mask

        Returns:
            Binary mask tensor
        """
        # Use provided device or default to CPU
        if device is None:
            device = torch.device("cpu")

        height, width = rle["size"]
        mask = torch.zeros(height * width, dtype=torch.bool, device=device)
        idx = 0
        parity = False
        for count in rle["counts"]:
            mask[idx : idx + count] = parity
            idx += count
            parity = not parity
        mask = mask.reshape(width, height)
        return mask.transpose(1, 0)

    def post_process_for_mask_generation(
        self,
        all_masks: List[Dict[str, Any]],
        all_scores: torch.Tensor,
        all_boxes: torch.Tensor,
        crops_nms_thresh: float,
        device: Optional[torch.device] = None,
        return_tensors: str = "pt",
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[Dict[str, Any]], torch.Tensor]:
        """
        Post-processes masks using Non-Maximum Suppression.

        Args:
            all_masks: List of masks in RLE format
            all_scores: IoU scores tensor
            all_boxes: Bounding boxes tensor
            crops_nms_thresh: NMS threshold
            device: Optional device for the output masks
            return_tensors: Output format, only "pt" is supported

        Returns:
            Tuple of filtered masks, scores, RLE encodings, and boxes
        """
        if device is None and len(all_scores) > 0:
            device = all_scores.device

        keep_by_nms = batched_nms(
            boxes=all_boxes.float(),
            scores=all_scores,
            idxs=torch.zeros(all_boxes.shape[0]),
            iou_threshold=crops_nms_thresh,
        )

        iou_scores = all_scores[keep_by_nms]
        rle_masks = [all_masks[i] for i in keep_by_nms]
        mask_boxes = all_boxes[keep_by_nms]

        # Pass device when converting RLE to masks
        masks = [self._rle_to_mask(rle, device=device) for rle in rle_masks]

        return masks, iou_scores, rle_masks, mask_boxes


__all__ = ["SamImageProcessorFast"]
