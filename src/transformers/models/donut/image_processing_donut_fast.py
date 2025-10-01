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
"""Fast Image processor class for Donut."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import BaseImageProcessorFast, BatchFeature, DefaultFastImageProcessorKwargs
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ImageInput, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
)


logger = logging.get_logger(__name__)


class DonutFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
        do_thumbnail (`bool`, *optional*, defaults to `self.do_thumbnail`):
            Whether to resize the image using thumbnail method.
        do_align_long_axis (`bool`, *optional*, defaults to `self.do_align_long_axis`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
    """

    do_thumbnail: Optional[bool]
    do_align_long_axis: Optional[bool]


@auto_docstring
class DonutImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 2560, "width": 1920}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_thumbnail = True
    do_align_long_axis = False
    do_pad = True
    valid_kwargs = DonutFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[DonutFastImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        if isinstance(size, (tuple, list)):
            size = size[::-1]
        kwargs["size"] = size
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[DonutFastImageProcessorKwargs]) -> BatchFeature:
        if "size" in kwargs:
            size = kwargs.pop("size")
            if isinstance(size, (tuple, list)):
                size = size[::-1]
            kwargs["size"] = size
        return super().preprocess(images, **kwargs)

    def align_long_axis(
        self,
        image: "torch.Tensor",
        size: SizeDict,
    ) -> "torch.Tensor":
        """
        Align the long axis of the image to the longest axis of the specified size.

        Args:
            image (`torch.Tensor`):
                The image to be aligned.
            size (`dict[str, int]`):
                The size `{"height": h, "width": w}` to align the long axis to.

        Returns:
            `torch.Tensor`: The aligned image.
        """
        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            height_dim, width_dim = image.dim() - 2, image.dim() - 1
            image = torch.rot90(image, 3, dims=[height_dim, width_dim])

        return image

    def pad_image(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        random_padding: bool = False,
    ) -> "torch.Tensor":
        """
        Pad the image to the specified size.

        Args:
            image (`torch.Tensor`):
                The image to be padded.
            size (`dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
            random_padding (`bool`, *optional*, defaults to `False`):
                Whether to use random padding or not.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        output_height, output_width = size.height, size.width
        input_height, input_width = image.shape[-2:]

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        if random_padding:
            pad_top = torch.random.randint(low=0, high=delta_height + 1)
            pad_left = torch.random.randint(low=0, high=delta_width + 1)
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return F.pad(image, padding)

    def pad(self, *args, **kwargs):
        logger.info("pad is deprecated and will be removed in version 4.27. Please use pad_image instead.")
        return self.pad_image(*args, **kwargs)

    def thumbnail(
        self,
        image: "torch.Tensor",
        size: SizeDict,
    ) -> "torch.Tensor":
        """
        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
        corresponding dimension of the specified size.

        Args:
            image (`torch.Tensor`):
                The image to be resized.
            size (`dict[str, int]`):
                The size `{"height": h, "width": w}` to resize the image to.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                The resampling filter to use.
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        # We always resize to the smallest of either the input or output size.
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return image

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        return self.resize(
            image,
            size=SizeDict(width=width, height=height),
            interpolation=F.InterpolationMode.BICUBIC,
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        do_thumbnail: bool,
        do_align_long_axis: bool,
        do_pad: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_align_long_axis:
                stacked_images = self.align_long_axis(image=stacked_images, size=size)
            if do_resize:
                shortest_edge = min(size.height, size.width)
                stacked_images = self.resize(
                    image=stacked_images, size=SizeDict(shortest_edge=shortest_edge), interpolation=interpolation
                )
            if do_thumbnail:
                stacked_images = self.thumbnail(image=stacked_images, size=size)
            if do_pad:
                stacked_images = self.pad_image(image=stacked_images, size=size, random_padding=False)

            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["DonutImageProcessorFast"]
