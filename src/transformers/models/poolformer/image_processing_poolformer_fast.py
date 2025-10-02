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
"""Fast Image processor class for PoolFormer."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import BaseImageProcessorFast, BatchFeature, DefaultFastImageProcessorKwargs
from ...image_transforms import (
    ChannelDimension,
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size_for_max_height_width,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)


class PoolFormerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
        crop_pct (`float`, *optional*, defaults to `self.crop_pct`):
            Percentage of the image to crop. Only has an effect if `do_resize` is set to `True`.
    """

    crop_pct: Optional[float]


@auto_docstring
class PoolFormerImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    crop_pct = 0.9
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    valid_kwargs = PoolFormerFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[PoolFormerFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[PoolFormerFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        crop_pct: Optional[float] = None,
        interpolation: Optional["F.InterpolationMode"] = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image.

        If crop_pct is unset:
            - size is `{"height": h, "width": w}`: the image is resized to `(h, w)`.
            - size is `{"shortest_edge": s}`: the shortest edge of the image is resized to s whilst maintaining the
              aspect ratio.

        if crop_pct is set:
            - size is `{"height": h, "width": w}`: the image is resized to `(int(floor(h/crop_pct)),
              int(floor(w/crop_pct)))`
            - size is `{"height": c, "width": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)`
              whilst maintaining the aspect ratio.
            - size is `{"shortest_edge": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)`
              whilst maintaining the aspect ratio.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            crop_pct (`float`, *optional*):
                Percentage of the image that will be cropped from the center. If set, the image is resized
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if crop_pct is not None:
            if size.shortest_edge:
                scale_size = int(size.shortest_edge / crop_pct)
            elif size.height and size.width:
                if size.height == size.width:
                    scale_size = int(size.height / crop_pct)
                else:
                    scale_size = (int(size.height / crop_pct), int(size.width / crop_pct))
            else:
                raise ValueError(f"Invalid size for resize: {size}")

            new_size = get_resize_output_image_size(
                image,
                size=scale_size,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        else:
            if size.shortest_edge and size.longest_edge:
                # Resize the image so that the shortest edge or the longest edge is of the given size
                # while maintaining the aspect ratio of the original image.
                new_size = get_size_with_aspect_ratio(
                    image.size()[-2:],
                    size.shortest_edge,
                    size.longest_edge,
                )
            elif size.shortest_edge:
                new_size = get_resize_output_image_size(
                    image,
                    size=size.shortest_edge,
                    default_to_square=False,
                    input_data_format=ChannelDimension.FIRST,
                )
            elif size.max_height and size.max_width:
                new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
            elif size.height and size.width:
                new_size = (size.height, size.width)
            else:
                raise ValueError(
                    "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                    f" {size}."
                )
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)

    def center_crop(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`"torch.Tensor"`):
                Image to center crop.
            size (`dict[str, int]`):
                Size of the output image.

        Returns:
            `torch.Tensor`: The center cropped image.
        """
        if size.height is None or size.width is None:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        image_height, image_width = image.shape[-2:]
        crop_height, crop_width = size.height, size.width

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            image = F.pad(image, padding_ltrb, fill=0)  # PIL uses fill value 0
            image_height, image_width = image.shape[-2:]
            if crop_width == image_width and crop_height == image_height:
                return image

        crop_top = int((image_height - crop_height) / 2.0)
        crop_left = int((image_width - crop_width) / 2.0)
        return F.crop(image, crop_top, crop_left, crop_height, crop_width)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        crop_pct: float,
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
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images, size=size, crop_pct=crop_pct, interpolation=interpolation
                )
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


__all__ = ["PoolFormerImageProcessorFast"]
