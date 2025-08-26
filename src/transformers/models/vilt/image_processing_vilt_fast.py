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
"""Fast Image processor class for Vilt."""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    get_max_height_width,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling, SizeDict
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

# Set maximum size based on the typical aspect ratio of the COCO dataset
MAX_LONGER_EDGE = 1333
MAX_SHORTER_EDGE = 800


class ViltFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image. If `True`, will pad the images in the batch to the largest height and width
            in the batch. Padding will be applied to the bottom and right with zeros.
        size_divisor (`int`, *optional*, defaults to 32):
            The size to make the height and width divisible by.
        rescale_factor (`float`, *optional*, defaults to 1/255):
            The factor to rescale the image by.
    """

    do_pad: Optional[bool]
    size_divisor: Optional[int]
    rescale_factor: Optional[float]


@auto_docstring
class ViltImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    size_divisor = 32
    do_pad = True
    default_to_square = False
    model_input_names = ["pixel_values", "pixel_mask"]
    valid_kwargs = ViltFastImageProcessorKwargs

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        size_divisor: Optional[int],
        do_pad: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        This method overrides the base class method to include padding and pixel mask generation.
        """
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, interpolation, size_divisor)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Handle padding if required
        data = {}
        if do_pad:
            pixel_values, pixel_mask = self._pad_batch(
                processed_images, return_tensors, disable_grouping=disable_grouping
            )
            data = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        else:
            # If no padding, just return the processed images
            if return_tensors == "pt":
                processed_images = torch.stack(processed_images)
            data = {"pixel_values": processed_images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def resize(
        self,
        images: "torch.Tensor",
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"] = None,
        size_divisor: Optional[int] = None,
    ) -> "torch.Tensor":
        """
        Resize an image or batch of images to specified size.

        Args:
            images (`torch.Tensor`): Image or batch of images to resize.
            size (`dict[str, int]`): Size dictionary with shortest_edge key.
            interpolation (`F.InterpolationMode`, *optional*): Interpolation method to use.
            size_divisor (`int`, *optional*): Value to ensure height/width are divisible by.

        Returns:
            `torch.Tensor`: Resized image or batch of images.
        """
        if interpolation is None:
            interpolation = self.resample

        # Resize with aspect ratio preservation
        shorter = size.shortest_edge
        longer = int(MAX_LONGER_EDGE / MAX_SHORTER_EDGE * shorter)

        heights = images.shape[-2]
        widths = images.shape[-1]

        # Determine the new dimensions
        if heights < widths:
            new_heights = shorter
            new_widths = widths * (shorter / heights)
        else:
            new_heights = heights * (shorter / widths)
            new_widths = shorter

        # Check if the longer side exceeds max size
        if max(new_heights, new_widths) > longer:
            scale = longer / max(new_heights, new_widths)
            new_heights = new_heights * scale
            new_widths = new_widths * scale

        new_heights = int(new_heights + 0.5)
        new_widths = int(new_widths + 0.5)

        # Make dimensions divisible by size_divisor
        if size_divisor is not None:
            new_heights = new_heights // size_divisor * size_divisor
            new_widths = new_widths // size_divisor * size_divisor

        # Resize the image
        return F.resize(images, [new_heights, new_widths], interpolation=interpolation)

    def _pad_batch(
        self,
        images: list["torch.Tensor"],
        return_tensors: Optional[Union[str, TensorType]],
        disable_grouping: Optional[bool],
    ) -> tuple:
        """
        Pad a batch of images to the same size based on the maximum dimensions.

        Args:
            images (`list[torch.Tensor]`): List of images to pad.
            return_tensors (`str` or `TensorType`, *optional*): The type of tensors to return.

        Returns:
            `tuple`: Tuple containing padded images and pixel masks.
        """
        # Calculate global maximum dimensions across all images
        max_size = get_max_height_width(images)

        # Group images by shape before padding
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images = {}
        processed_masks = {}

        for shape, stacked_images in grouped_images.items():
            # Create mask template for efficient masking
            if return_tensors == "pt" and len(stacked_images) > 0:
                device = stacked_images.device
                mask_template = torch.zeros(max_size, dtype=torch.int64, device=device)

            original_size = stacked_images.shape[-2:]
            needs_padding = original_size[0] != max_size[0] or original_size[1] != max_size[1]

            if needs_padding:
                padding_bottom = max_size[0] - original_size[0]
                padding_right = max_size[1] - original_size[1]
                padding = [0, 0, padding_right, padding_bottom]

                padded_images = F.pad(stacked_images, padding, fill=0)
                pixel_mask = mask_template.clone()
                pixel_mask[: original_size[0], : original_size[1]].fill_(1)
                pixel_masks = pixel_mask.unsqueeze(0).repeat(stacked_images.shape[0], 1, 1)
            else:
                padded_images = stacked_images
                pixel_masks = torch.ones(
                    (stacked_images.shape[0], max_size[0], max_size[1]),
                    dtype=torch.int64,
                    device=stacked_images.device,
                )

            # Store processed group
            processed_images[shape] = padded_images
            processed_masks[shape] = pixel_masks

        # Reorder images back to original order
        padded_images = reorder_images(processed_images, grouped_images_index)
        pixel_masks = reorder_images(processed_masks, grouped_images_index)

        # Stack if tensors are requested for final result
        if return_tensors == "pt" and padded_images:
            padded_images = torch.stack(padded_images)
            pixel_masks = torch.stack(pixel_masks)

        return padded_images, pixel_masks


__all__ = ["ViltImageProcessorFast"]
