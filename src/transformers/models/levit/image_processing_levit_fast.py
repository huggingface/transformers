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
"""Fast Image processor class for LeViT."""

from typing import Optional

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import BaseImageProcessorFast, SizeDict
from ...image_transforms import (
    ChannelDimension,
    get_resize_output_image_size,
)
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...utils import auto_docstring


@auto_docstring
class LevitImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize an image.

        If size is a dict with keys "width" and "height", the image will be resized to `(size["height"],
        size["width"])`.

        If size is a dict with key "shortest_edge", the shortest edge value `c` is rescaled to `int(c * (256/224))`.
        The smaller edge of the image will be matched to this value i.e, if height > width, then image will be rescaled
        to `(size["shortest_edge"] * height / width, size["shortest_edge"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Size of the output image after resizing. If size is a dict with keys "width" and "height", the image
                will be resized to (height, width). If size is a dict with key "shortest_edge", the shortest edge value
                `c` is rescaled to int(`c` * (256/224)). The smaller edge of the image will be matched to this value
                i.e, if height > width, then image will be rescaled to (size * height / width, size).
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BICUBIC`):
                Resampling filter to use when resiizing the image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BICUBIC
        if size.shortest_edge:
            shortest_edge = int((256 / 224) * size["shortest_edge"])
            new_size = get_resize_output_image_size(
                image, size=shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                f"Size dict must have keys 'height' and 'width' or 'shortest_edge'. Got {size.keys()} {size.keys()}."
            )
        return F.resize(
            image,
            size=new_size,
            interpolation=interpolation,
            **kwargs,
        )


__all__ = ["LevitImageProcessorFast"]
