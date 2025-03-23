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
from typing import Tuple, Union

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling, SizeDict
from ...utils import (
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
)


if is_vision_available():
    from ...image_utils import PILImageResampling

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
    r"""
    min_size (`int`, *optional*, defaults to 14):
        The minimum allowed size for the resized image. Ensures that neither the height nor width
        falls below this value after resizing.
    """

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = False
    min_size = 14

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.background_color = tuple([int(x * 255) for x in self.image_mean])

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize and pad an image to a square based on the longest edge in `size`.

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

        height, width = image.size()[-2:]
        max_size = max(height, width)

        if size["height"] != size["width"]:
            raise ValueError(
                f"Output height and width must be the same. Got height={size['height']} and width={size['width']}"
            )
        size = size["height"]

        delta = size / max_size
        # Largest side becomes `size` and the other side is scaled according to the aspect ratio.
        output_size_nonpadded = [
            max(int(height * delta), self.min_size),
            max(int(width * delta), self.min_size),
        ]

        image = F.resize(image, output_size_nonpadded, interpolation=interpolation, antialias=antialias)
        image = self.pad_to_square(image, background_color=self.background_color)
        return image

    def pad_to_square(
        self,
        image: "torch.Tensor",
        background_color: Union[int, Tuple[int, int, int]] = 0,
    ) -> "torch.Tensor":
        """
        Pads an image to a square based on the longest edge.

        Args:
            image (`np.ndarray`):
                The image to pad.
            background_color (`int` or `Tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding. Can be an integer for single channel or a
                tuple of integers representing for multi-channel images. If passed as integer
                in mutli-channel mode, it will default to `0` in subsequent channels.

        Returns:
            `torch.Tensor`: The padded image.
        """
        batch_size, num_channels, height, width = image.size()

        if height == width:
            return image

        max_dim = max(height, width)

        # Ensure background_color is the correct shape
        if isinstance(background_color, int):
            background_color = [background_color] * num_channels
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        result = torch.zeros((batch_size, num_channels, max_dim, max_dim), dtype=image.dtype)
        for i, color in enumerate(background_color):
            result[:, i, :, :] = color
        if width > height:
            start = (max_dim - height) // 2
            result[:, :, start : start + height, :] = image
        else:
            start = (max_dim - width) // 2
            result[:, :, :, start : start + width] = image

        return result


__all__ = ["DeepseekVLImageProcessorFast"]
