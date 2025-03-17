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
"""Fast Image processor class for DeepseekVL."""

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_transforms import get_size_with_aspect_ratio
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling, SizeDict, get_image_size
from ...utils import (
	add_start_docstrings,
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


@add_start_docstrings(
    "Constructs a fast DeepseekVL image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class DeepseekVLImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = 1024
    do_resize = True
    do_rescale = True
    do_normalize = False
    background_color = [122, 116, 104]

    def pad_to_square(
        self,
        image: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Pads an image to a square based on the longest edge.

        Args:
            image (`torch.Tensor`):
                Image to pad.
        Returns:
            `torch.Tensor`: The padded image.
        """
        height, width = image.shape[-2:]
        max_dim = max(height, width)
        paste_x_left = (max_dim - width) // 2
        paste_y_left = (max_dim - height) // 2
        paste_x_right = max_dim - width - paste_x_left
        paste_y_right = max_dim - height - paste_y_left
        return F.pad(
            image, padding=[paste_x_left, paste_y_left, paste_x_right, paste_y_right], fill=self.background_color
        )

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        size = get_size_with_aspect_ratio(image.shape[-2:], size["height"], size["height"])

        image =  F.resize(image, size, interpolation=interpolation, antialias=antialias)
        image = self.pad_to_square(image)

        return image


__all__ = ["DeepseekVLImageProcessorFast"]
