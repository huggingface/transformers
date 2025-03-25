# coding=utf-8
# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = False

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
        output_size_nonpadded = [round(height * delta), round(width * delta)]

        image = F.resize(image, output_size_nonpadded, interpolation=interpolation, antialias=antialias)
        image = self.pad_to_square(image)
        return image

    def pad_to_square(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Pads an image to a square based on the longest edge.

        Args:
            image (`np.ndarray`):
                The image to pad.

        Returns:
            `torch.Tensor`: The padded image.
        """
        batch_size, num_channels, height, width = image.size()

        if height == width:
            return image

        max_dim = max(height, width)
        result = torch.zeros((batch_size, num_channels, max_dim, max_dim), dtype=image.dtype)
        for i, color in enumerate(self.image_mean):
            result[:, i, :, :] = int(color * 255)
        if width > height:
            start = (max_dim - height) // 2
            result[:, :, start : start + height, :] = image
        else:
            start = (max_dim - width) // 2
            result[:, :, :, start : start + width] = image

        return result


__all__ = ["DeepseekVLImageProcessorFast"]
