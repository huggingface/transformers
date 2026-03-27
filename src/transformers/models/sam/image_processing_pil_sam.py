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
"""Image processor class for SAM."""

import math
from collections.abc import Iterable
from copy import deepcopy
from itertools import product
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature, get_size_dict
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
from ...utils import TensorType, auto_docstring, is_torch_available, is_vision_available
from ...utils.import_utils import requires
from .image_processing_sam import SamImageProcessorKwargs


if is_vision_available():
    import PIL


def get_resize_output_image_size(
    input_image: np.ndarray,
    longest_edge: int,
    input_data_format: str | ChannelDimension | None = None,
) -> tuple[int, int]:
    height, width = get_image_size(input_image, input_data_format)
    if height >= width:
        ratio = width * 1.0 / height
        new_height = longest_edge
        new_width = int(new_height * ratio)
    else:
        ratio = height * 1.0 / width
        new_width = longest_edge
        new_height = int(new_width * ratio)
    return (new_height, new_width)


@auto_docstring
@requires(backends=("vision", "torch", "torchvision"))
class SamImageProcessorPil(PilBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"longest_edge": 1024}
    mask_size = {"longest_edge": 256}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    pad_size = {"height": 1024, "width": 1024}
    mask_pad_size = {"height": 256, "width": 256}
    valid_kwargs = SamImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[SamImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        **kwargs: Unpack[SamImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess.
        """
        return super().preprocess(images, segmentation_maps, **kwargs)

    def _standardize_kwargs(
        self,
        mask_size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        mask_pad_size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        if mask_size is not None and not isinstance(mask_size, SizeDict):
            mask_size = SizeDict(**get_size_dict(mask_size, param_name="mask_size"))
        if mask_pad_size is not None and not isinstance(mask_pad_size, SizeDict):
            mask_pad_size = SizeDict(**get_size_dict(mask_pad_size, param_name="mask_pad_size"))

        kwargs["mask_size"] = mask_size
        kwargs["mask_pad_size"] = mask_pad_size

        return kwargs

    def _get_preprocess_shape(self, old_shape: tuple[int, int], longest_edge: int):
        """
        Compute the output size given input size and target long side length.
        """
        oldh, oldw = old_shape
        scale = longest_edge * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest
                edge of the image will be resized to the specified size, while the other edge will be resized to
                maintain the aspect ratio.
            resample (`PILImageResampling | tvF.InterpolationMode | int | None`, *optional*):
                Resampling filter to use when resizing the image.

        Returns:
            `np.ndarray`: The resized image.
        """
        if not size.longest_edge:
            raise ValueError(f"The `size` dictionary must contain the key `longest_edge`. Got {size.keys()}")
        input_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        output_height, output_width = self._get_preprocess_shape(input_size, size.longest_edge)
        return super().resize(
            image, size=SizeDict(height=output_height, width=output_width), resample=resample, **kwargs
        )

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        return_tensors: str | TensorType | None,
        device: Union[str, "torch.device"] | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format
        )
        original_sizes = [image.shape[-2:] for image in images]
        images_kwargs = kwargs.copy()
        pixel_values, reshaped_input_sizes = self._preprocess(images, **images_kwargs)
        data = {
            "pixel_values": pixel_values,
            "original_sizes": original_sizes,
            "reshaped_input_sizes": reshaped_input_sizes,
        }

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
                    "resample": PILImageResampling.NEAREST,
                    "size": segmentation_maps_kwargs.pop("mask_size"),
                    "pad_size": segmentation_maps_kwargs.pop("mask_pad_size"),
                }
            )
            processed_segmentation_maps, _ = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            )
            data["labels"] = [m.squeeze(0).astype(np.int64) for m in processed_segmentation_maps]

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        **kwargs,
    ) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
        """
        Preprocess images using PIL backend.
        """
        processed_images = []
        reshaped_input_sizes = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            reshaped_input_sizes.append(image.shape[-2:])

            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size)

        return processed_images, reshaped_input_sizes

    def generate_crop_boxes(
        self,
        image: "np.ndarray | PIL.Image.Image | torch.Tensor",
        target_size,
        crop_n_layers: int = 0,
        overlap_ratio: float = 512 / 1500,
        points_per_crop: int | None = 32,
        crop_n_points_downscale_factor: list[int] | None = 1,
        device: Optional["torch.device"] = None,
    ):
        """
        Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

        Args:
            image (`np.ndarray | PIL.Image.Image | torch.Tensor`):
                Input original image
            target_size (`int`):
                Target size of the resized image
            crop_n_layers (`int`, *optional*, defaults to 0):
                If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where
                each layer has 2**i_layer number of image crops.
            overlap_ratio (`float`, *optional*, defaults to 512/1500):
                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            points_per_crop (`int`, *optional*, defaults to 32):
                Number of points to sample from each crop.
            crop_n_points_downscale_factor (`list[int]`, *optional*, defaults to 1):
                The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            device (`torch.device`, *optional*, defaults to None):
                Device to use for the computation. If None, cpu will be used.
        """
        image = self.process_image(image)
        crop_boxes, points_per_crop, cropped_images, input_labels = _generate_crop_boxes(
            image,
            target_size,
            crop_n_layers,
            overlap_ratio,
            points_per_crop,
            crop_n_points_downscale_factor,
        )
        if device is None:
            device = torch.device("cpu")
        crop_boxes = torch.tensor(crop_boxes, device=device)
        points_per_crop = torch.tensor(points_per_crop, device=device)
        # cropped_images stays as np.ndarray
        input_labels = torch.tensor(input_labels, device=device)

        return crop_boxes, points_per_crop, cropped_images, input_labels

    def filter_masks(
        self,
        masks,
        iou_scores,
        original_size,
        cropped_box_image,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
    ):
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`torch.Tensor`):
                Input masks.
            iou_scores (`torch.Tensor`):
                List of IoU scores.
            original_size (`tuple[int,int]`):
                Size of the original image.
            cropped_box_image (`np.ndarray`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.
        """
        if not is_torch_available():
            raise ImportError("PyTorch is required for filter_masks")
        original_height, original_width = original_size
        iou_scores = iou_scores.flatten(0, 1)
        masks = masks.flatten(0, 1)

        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError("masks and iou_scores must have the same batch size.")

        if masks.device != iou_scores.device:
            iou_scores = iou_scores.to(masks.device)

        batch_size = masks.shape[0]

        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)

        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)

        # compute stability score
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)

        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]

        # binarize masks
        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box(masks)

        keep_mask = ~_is_box_near_crop_edge(
            converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
        )

        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]

        masks = _pad_masks(masks, cropped_box_image, original_height, original_width)
        # conversion to rle is necessary to run non-maximum suppression
        masks = _mask_to_rle(masks)

        return masks, scores, converted_boxes

    def post_process_masks(
        self,
        masks,
        original_sizes,
        reshaped_input_sizes,
        mask_threshold=0.0,
        binarize=True,
        pad_size=None,
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[torch.Tensor], List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            reshaped_input_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        if not is_torch_available():
            raise ImportError("PyTorch is required for post_process_masks")
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size["height"], pad_size["width"])
        if isinstance(original_sizes, (torch.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()
        if isinstance(reshaped_input_sizes, (torch.Tensor, np.ndarray)):
            reshaped_input_sizes = reshaped_input_sizes.tolist()

        output_masks = []
        for i, original_size in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])
            elif not isinstance(masks[i], torch.Tensor):
                raise TypeError("Input masks should be a list of `torch.tensors` or a list of `np.ndarray`")
            interpolated_mask = F.interpolate(masks[i], target_image_size, mode="bilinear", align_corners=False)
            interpolated_mask = interpolated_mask[..., : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1]]
            interpolated_mask = F.interpolate(interpolated_mask, original_size, mode="bilinear", align_corners=False)
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)

        return output_masks

    def post_process_for_mask_generation(self, all_masks, all_scores, all_boxes, crops_nms_thresh):
        """
        Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

        Args:
            all_masks (`torch.Tensor`):
                List of all predicted segmentation masks
            all_scores (`torch.Tensor`):
                List of all predicted iou scores
            all_boxes (`torch.Tensor`):
                List of all bounding boxes of the predicted masks
            crops_nms_thresh (`float`):
                Threshold for NMS (Non Maximum Suppression) algorithm.
        """
        return _post_process_for_mask_generation(all_masks, all_scores, all_boxes, crops_nms_thresh)


def _compute_stability_score(masks: "torch.Tensor", mask_threshold: float, stability_score_offset: int):
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    )
    unions = (masks > (mask_threshold - stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    stability_scores = intersections / unions
    return stability_scores


def _batched_mask_to_box(masks: "torch.Tensor"):
    """
    Computes the bounding boxes around the given input masks. The bounding boxes are in the XYXY format which
    corresponds the following required indices:
        - LEFT: left hand side of the bounding box
        - TOP: top of the bounding box
        - RIGHT: right of the bounding box
        - BOTTOM: bottom of the bounding box

    Return [0,0,0,0] for an empty mask. For input shape channel_1 x channel_2 x ... x height x width, the output shape
    is channel_1 x channel_2 x ... x 4.

    Args:
        - masks (`torch.Tensor` of shape `(batch, nb_mask, height, width)`)
    """
    # torch.max below raises an error on empty inputs, just skip in this case

    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to Cxheightxwidth
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


def _is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)

    left, top, _, _ = crop_box
    offset = torch.tensor([[left, top, left, top]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    boxes = (boxes + offset).float()

    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def _pad_masks(masks, crop_box: list[int], orig_height: int, orig_width: int):
    left, top, right, bottom = crop_box
    if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_width - (right - left), orig_height - (bottom - top)
    pad = (left, pad_x - left, top, pad_y - top)
    return torch.nn.functional.pad(masks, pad, value=0)


def _generate_crop_boxes(
    image,
    target_size: int,
    crop_n_layers: int = 0,
    overlap_ratio: float = 512 / 1500,
    points_per_crop: int | None = 32,
    crop_n_points_downscale_factor: list[int] | None = 1,
) -> tuple[list[list[int]], list[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

    Args:
        image (`np.ndarray`):
            Image to generate crops for.
        target_size (`int`):
            Size of the smallest crop.
        crop_n_layers (`int`, *optional*):
            If `crops_n_layers>0`, mask prediction will be run again on crops of the image. Sets the number of layers
            to run, where each layer has 2**i_layer number of image crops.
        overlap_ratio (`int`, *optional*):
            Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the
            image length. Later layers with more crops scale down this overlap.
        points_per_crop (`int`, *optional*):
            Number of points to sample per crop.
        crop_n_points_downscale_factor (`int`, *optional*):
            The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
    """

    if isinstance(image, list):
        raise ValueError("Only one image is allowed for crop generation.")
    original_size = image.shape[-2:]

    points_grid = []
    for i in range(crop_n_layers + 1):
        n_points = int(points_per_crop / (crop_n_points_downscale_factor**i))
        points_grid.append(_build_point_grid(n_points))

    crop_boxes, layer_idxs = _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size)

    cropped_images, point_grid_per_crop = _generate_crop_images(
        crop_boxes, image, points_grid, layer_idxs, target_size, original_size
    )
    crop_boxes = np.array(crop_boxes)
    crop_boxes = crop_boxes.astype(np.float32)
    points_per_crop = np.array([point_grid_per_crop])
    points_per_crop = np.transpose(points_per_crop, axes=(0, 2, 1, 3))

    input_labels = np.ones_like(points_per_crop[:, :, :, 0], dtype=np.int64)

    return crop_boxes, points_per_crop, cropped_images, input_labels


def _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size):
    """
    Generates 2 ** (layers idx + 1) crops for each crop_n_layers. Crops are in the XYWH format : The XYWH format
    consists of the following required indices:
        - X: X coordinate of the top left of the bounding box
        - Y: Y coordinate of the top left of the bounding box
        - W: width of the bounding box
        - H: height of the bounding box
    """
    crop_boxes, layer_idxs = [], []
    im_height, im_width = original_size
    short_side = min(im_height, im_width)

    # Original image
    crop_boxes.append([0, 0, im_width, im_height])
    layer_idxs.append(0)
    for i_layer in range(crop_n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_width = int(math.ceil((overlap * (n_crops_per_side - 1) + im_width) / n_crops_per_side))
        crop_height = int(math.ceil((overlap * (n_crops_per_side - 1) + im_height) / n_crops_per_side))

        crop_box_x0 = [int((crop_width - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_height - overlap) * i) for i in range(n_crops_per_side)]

        for left, top in product(crop_box_x0, crop_box_y0):
            box = [left, top, min(left + crop_width, im_width), min(top + crop_height, im_height)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def _build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def _generate_crop_images(
    crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format=None
):
    """
    Takes as an input bounding boxes that are used to crop the image. Based in the crops, the corresponding points are
    also passed.
    """
    cropped_images = []
    total_points_per_crop = []
    for i, crop_box in enumerate(crop_boxes):
        left, top, right, bottom = crop_box
        cropped_im = image[:, top:bottom, left:right]

        cropped_images.append(cropped_im)

        cropped_im_size = cropped_im.shape[-2:]
        points_scale = np.array(cropped_im_size)[None, ::-1]

        points = points_grid[layer_idxs[i]] * points_scale
        normalized_points = _normalize_coordinates(target_size, points, original_size)
        total_points_per_crop.append(normalized_points)

    return cropped_images, total_points_per_crop


def _normalize_coordinates(
    target_size: int, coords: np.ndarray, original_size: tuple[int, int], is_bounding_box=False
) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the original image size in (height, width)
    format.
    """
    old_height, old_width = original_size

    scale = target_size * 1.0 / max(old_height, old_width)
    new_height, new_width = old_height * scale, old_width * scale
    new_width = int(new_width + 0.5)
    new_height = int(new_height + 0.5)

    coords = deepcopy(coords).astype(float)

    if is_bounding_box:
        coords = coords.reshape(-1, 2, 2)

    coords[..., 0] = coords[..., 0] * (new_width / old_width)
    coords[..., 1] = coords[..., 1] * (new_height / old_height)

    if is_bounding_box:
        coords = coords.reshape(-1, 4)

    return coords


def _rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    height, width = rle["size"]
    mask = np.empty(height * width, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity = not parity
    mask = mask.reshape(width, height)
    return mask.transpose()  # Reshape to original shape


def _post_process_for_mask_generation(rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
    """
    Perform NMS (Non Maximum Suppression) on the outputs.

    Args:
            rle_masks (`torch.Tensor`):
                binary masks in the RLE format
            iou_scores (`torch.Tensor` of shape (nb_masks, 1)):
                iou_scores predicted by the model
            mask_boxes (`torch.Tensor`):
                The bounding boxes corresponding to segmentation masks
            amg_crops_nms_thresh (`float`, *optional*, defaults to 0.7):
                NMS threshold.
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required for post_process_for_mask_generation")
    from torchvision.ops.boxes import batched_nms

    keep_by_nms = batched_nms(
        boxes=mask_boxes.float(),
        scores=iou_scores,
        idxs=torch.zeros(mask_boxes.shape[0]),
        iou_threshold=amg_crops_nms_thresh,
    )

    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [_rle_to_mask(rle) for rle in rle_masks]

    return masks, iou_scores, rle_masks, mask_boxes


def _mask_to_rle(input_mask: "torch.Tensor"):
    """
    Encodes masks the run-length encoding (RLE), in the format expected by pycoco tools.
    """
    # Put in fortran order and flatten height and width
    batch_size, height, width = input_mask.shape
    input_mask = input_mask.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(batch_size):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        if len(cur_idxs) == 0:
            # No changes => either all 0 or all 1
            # If the entire mask is 0, RLE is [height*width] or if the entire mask is 1, RLE is [0, height*width].
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


__all__ = ["SamImageProcessorPil"]
