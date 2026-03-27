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

from collections.abc import Iterable
from typing import Any

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode, pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
    get_max_height_width,
)
from ...utils import (
    TensorType,
    auto_docstring,
)
from ...utils.import_utils import requires
from .image_processing_vilt import ViltImageProcessorKwargs


# Set maximum size based on the typical aspect ratio of the COCO dataset
MAX_LONGER_EDGE = 1333
MAX_SHORTER_EDGE = 800


def max_across_indices(values: Iterable[Any]) -> list[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def make_pixel_mask(
    image: np.ndarray, output_size: tuple[int, int], input_data_format: str | ChannelDimension | None = None
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


def get_resize_output_image_size(
    input_image: np.ndarray,
    shorter: int = 800,
    longer: int = 1333,
    size_divisor: int = 32,
    input_data_format: str | ChannelDimension | None = None,
) -> tuple[int, int]:
    input_height, input_width = get_image_size(input_image, input_data_format)
    min_size, max_size = shorter, longer

    scale = min_size / min(input_height, input_width)

    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size

    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width

    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor

    return new_height, new_width


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class ViltImageProcessorPil(PilBackend):
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
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | int | None" = None,
        size_divisor: int | None = None,
    ) -> np.ndarray:
        """
        Resize an image to specified size.

        Args:
            image (`np.ndarray`): Image to resize.
            size (`SizeDict`): Size dictionary with shortest_edge key.
            resample (`PILImageResampling | int`, *optional*): Interpolation method to use.
            size_divisor (`int`, *optional*): Value to ensure height/width are divisible by.

        Returns:
            `np.ndarray`: Resized image.
        """
        if not hasattr(size, "shortest_edge") or size.shortest_edge is None:
            raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size}")
        shorter = size.shortest_edge
        longer = int(MAX_LONGER_EDGE / MAX_SHORTER_EDGE * shorter)
        output_size = get_resize_output_image_size(
            image,
            shorter=shorter,
            longer=longer,
            size_divisor=size_divisor or self.size_divisor,
            input_data_format=ChannelDimension.FIRST,
        )

        return super().resize(image, SizeDict(height=output_size[0], width=output_size[1]), resample=resample)

    def _pad_batch(
        self,
        images: list[np.ndarray],
        return_tensors: str | TensorType | None,
    ) -> tuple:
        """
        Pad a batch of images to the same size based on the maximum dimensions.

        Args:
            images (`list[np.ndarray]`): List of images to pad.
            return_tensors (`str` or `TensorType`, *optional*): The type of tensors to return.

        Returns:
            `tuple`: Tuple containing padded images and pixel masks.
        """
        # Calculate global maximum dimensions across all images
        max_size = get_max_height_width(images, input_data_format=ChannelDimension.FIRST)

        padded_images = []
        pixel_masks = []

        for image in images:
            input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
            needs_padding = input_height != max_size[0] or input_width != max_size[1]

            if needs_padding:
                pad_bottom = max_size[0] - input_height
                pad_right = max_size[1] - input_width
                padding = ((0, pad_bottom), (0, pad_right))

                padded_image = pad(
                    image,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=0,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                pixel_mask = make_pixel_mask(image, max_size, input_data_format=ChannelDimension.FIRST)
            else:
                padded_image = image
                pixel_mask = np.ones(max_size, dtype=np.int64)

            padded_images.append(padded_image)
            pixel_masks.append(pixel_mask)

        return padded_images, pixel_masks

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        return_tensors: str | TensorType | None,
        size_divisor: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample, size_divisor)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        # Handle padding if required
        data = {}
        if do_pad:
            pixel_values, pixel_mask = self._pad_batch(processed_images, return_tensors)
            data = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        else:
            data = {"pixel_values": processed_images}

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["ViltImageProcessorPil"]
