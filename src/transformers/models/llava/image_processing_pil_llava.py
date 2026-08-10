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
"""Image processor class for LLaVa."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


@auto_docstring
class LlavaImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_pad = False
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: int | tuple[int, int, int] = 0,
    ) -> np.ndarray:
        """
        Pads an image to a square based on the longest edge.

        Args:
            image (`np.ndarray`):
                The image to pad. Shape: (num_channels, height, width) - always channels_first in backend.
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding.

        Returns:
            `np.ndarray`: The padded image.
        """
        # Backend always uses channels_first format: (num_channels, height, width)
        num_channels, height, width = image.shape

        if height == width:
            return image

        max_dim = max(height, width)

        # Ensure background_color is the correct shape
        if isinstance(background_color, int):
            background_color = [background_color]
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        result = np.zeros((num_channels, max_dim, max_dim), dtype=image.dtype)
        for i, color in enumerate(background_color):
            result[i, :, :] = color
        if width > height:
            start = (max_dim - height) // 2
            result[:, start : start + height, :] = image
        else:
            start = (max_dim - width) // 2
            result[:, :, start : start + width] = image

        return result

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image in images:
            # Apply pad_to_square first if needed (before resize)
            if do_pad:
                background_color = tuple(int(x * 255) for x in image_mean) if image_mean else 0
                image = self.pad_to_square(image, background_color=background_color)

            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)

            if do_center_crop:
                image = self.center_crop(image, crop_size)

            if do_rescale:
                image = self.rescale(image, rescale_factor)

            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["LlavaImageProcessorPil"]
