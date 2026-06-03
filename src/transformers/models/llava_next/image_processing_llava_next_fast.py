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
"""Fast Image processor class for LLaVa-NeXT."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature, get_patch_output_size, select_best_resolution
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    divide_to_patches,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
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
)


class LlavaNextFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    image_grid_pinpoints (`list[list[int]]`, *optional*):
        A list of possible resolutions to use for processing high resolution images. The best resolution is selected
        based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
        method.
    """

    image_grid_pinpoints: Optional[list[list[int]]]


@auto_docstring
class LlavaNextImageProcessorFast(BaseImageProcessorFast):
    # To be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    valid_kwargs = LlavaNextFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[LlavaNextFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[LlavaNextFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        interpolation: "F.InterpolationMode",
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image ("torch.Tensor"):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            interpolation (`InterpolationMode`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            "torch.Tensor": The resized and padded image.
        """
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        # Resize the image
        resized_image = self.resize(
            image=image,
            size=SizeDict(height=new_height, width=new_width),
            interpolation=interpolation,
        )

        return resized_image

    def _get_padding_size(self, original_resolution: tuple, target_resolution: tuple):
        original_height, original_width = original_resolution
        target_height, target_width = target_resolution
        paste_x, r_x = divmod(target_width - original_width, 2)
        paste_y, r_y = divmod(target_height - original_height, 2)
        return [paste_x, paste_y, paste_x + r_x, paste_y + r_y]

    def _pad_for_patching(
        self, image: "torch.Tensor", target_resolution: tuple, input_data_format: ChannelDimension
    ) -> "torch.Tensor":
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        new_resolution = get_patch_output_size(image, target_resolution, input_data_format)
        padding = self._get_padding_size(new_resolution, target_resolution)

        padded_image = F.pad(image, padding=padding)

        return padded_image

    def _get_image_patches(
        self,
        image: "torch.Tensor",
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        interpolation: "F.InterpolationMode",
    ) -> list["torch.Tensor"]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image ("torch.Tensor"):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            interpolation (`"InterpolationMode"`):
                Resampling filter to use if resizing the image.

        Returns:
            list["torch.Tensor"]: A list of NumPy arrays containing the processed image patches.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, interpolation=interpolation, input_data_format=ChannelDimension.FIRST
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=ChannelDimension.FIRST)
        patches = divide_to_patches(padded_image, patch_size=patch_size)
        resized_original_image = F.resize(image, size=size, interpolation=interpolation)

        image_patches = [resized_original_image] + patches

        return image_patches

    def _pad_for_batching(
        self,
        pixel_values: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`list[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)

        Returns:
            list[`torch.Tensor`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
            for image in pixel_values
        ]

        return pixel_values

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        image_grid_pinpoints: list[list[int]],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_pad: bool,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        image_sizes = []
        # Determine the size tuple
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)

        # Determine the patch size
        if crop_size and crop_size.height:
            patch_size = crop_size.height
        elif size and size.height:
            patch_size = size.height
        else:
            patch_size = size.shortest_edge

        for image in images:
            image_patches = self._get_image_patches(
                image,
                image_grid_pinpoints,
                size=size_tuple,
                patch_size=patch_size,
                interpolation=interpolation,
            )

            # Group images by size for batched processing
            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(
                image_patches, disable_grouping=disable_grouping
            )
            for shape, stacked_image_patches in grouped_image_patches.items():
                if do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        interpolation=interpolation,
                    )
                if do_center_crop:
                    stacked_image_patches = self.center_crop(stacked_image_patches, crop_size)
                # Fused rescale and normalize
                stacked_image_patches = self.rescale_and_normalize(
                    stacked_image_patches, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                processed_image_patches_grouped[shape] = stacked_image_patches
            processed_image_patches = reorder_images(processed_image_patches_grouped, grouped_image_patches_index)
            processed_image_patches = (
                torch.stack(processed_image_patches, dim=0) if return_tensors else processed_image_patches
            )
            processed_images.append(processed_image_patches)
            image_sizes.append(get_image_size(image, ChannelDimension.FIRST))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


__all__ = ["LlavaNextImageProcessorFast"]
