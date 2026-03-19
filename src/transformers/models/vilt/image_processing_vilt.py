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
"""Image processor class for Vilt."""

import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
    get_max_height_width,
)
from ...processing_utils import ImagesKwargs
from ...utils import (
    TensorType,
    auto_docstring,
    is_torchvision_available,
)


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


# Set maximum size based on the typical aspect ratio of the COCO dataset
MAX_LONGER_EDGE = 1333
MAX_SHORTER_EDGE = 800


class ViltImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
        The size by which to make sure both the height and width can be divided. Only has an effect if `do_resize`
        is set to `True`.
    """

    size_divisor: int


@auto_docstring
class ViltImageProcessor(TorchvisionBackend):
    valid_kwargs = ViltImageProcessorKwargs
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

    def resize(
        self,
        images: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        size_divisor: int | None = None,
    ) -> "torch.Tensor":
        """
        Resize an image or batch of images to specified size.

        Args:
            images (`torch.Tensor`): Image or batch of images to resize.
            size (`SizeDict`): Size dictionary with shortest_edge key.
            resample (`PILImageResampling | tvF.InterpolationMode | int`, *optional*): Interpolation method to use.
            size_divisor (`int`, *optional*): Value to ensure height/width are divisible by.

        Returns:
            `torch.Tensor`: Resized image or batch of images.
        """

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
        return super().resize(images, SizeDict(height=new_heights, width=new_widths), resample=resample)

    def _pad_batch(
        self,
        images: list["torch.Tensor"],
        return_tensors: str | TensorType | None,
        disable_grouping: bool | None,
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

                padded_images = tvF.pad(stacked_images, padding, fill=0)
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

        return padded_images, pixel_masks

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        size_divisor: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, resample, size_divisor)
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


__all__ = ["ViltImageProcessor"]
