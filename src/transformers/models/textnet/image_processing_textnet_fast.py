# coding=utf-8
# Copyright 2025 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for TextNet."""

import enum
from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, DefaultFastImageProcessorKwargs
from ...image_transforms import get_resize_output_image_size, group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import auto_docstring, is_torch_available, is_torchvision_available, is_torchvision_v2_available, TensorType

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class TextNetFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    size_divisor (`int`, *optional*, defaults to 32):
        Ensures height and width are rounded to a multiple of this value after resizing.
    """

    size_divisor: Optional[int]


@auto_docstring
class TextNetImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 640}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    size_divisor = 32
    valid_kwargs = TextNetFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[TextNetFastImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)
    
    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[TextNetFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)
    
    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        size_divisor: int = 32,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.
            size_divisor (`int`, *optional*, defaults to 32):
                Ensures height and width are rounded to a multiple of this value after resizing.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
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
        
        # ensure height and width are divisible by size_divisor
        height, width = new_size
        if height % size_divisor != 0:
            height += size_divisor - (height % size_divisor)
        if width % size_divisor != 0:
            width += size_divisor - (width % size_divisor)
        new_size = (height, width)

        # This is a workaround to avoid a bug in torch.compile when dealing with uint8 on AMD MI3XX GPUs
        # Tracked in PyTorch issue: https://github.com/pytorch/pytorch/issues/155209
        # TODO: remove this once the bug is fixed (detected with torch==2.7.0+git1fee196, torchvision==0.22.0+9eb57cd)
        if torch.compiler.is_compiling() and is_rocm_platform():
            return self.compile_friendly_resize(image, new_size, interpolation, antialias)
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        size_divisor: int,
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
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation, size_divisor=size_divisor)
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

    


__all__ = ["TextNetImageProcessorFast"]
