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

from ...image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    PythonBackend,
    TorchVisionBackend,
)
from ...image_transforms import (
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PILImageResampling,
    SizeDict,
)
from ...utils import TensorType, auto_docstring, is_torchvision_available, logging


if is_torchvision_available():
    import torch
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


class LlavaTorchVisionBackend(TorchVisionBackend):
    def pad_to_square(
        self,
        images: "torch.Tensor",
        background_color: int | tuple[int, int, int] = 0,
    ) -> "torch.Tensor":
        """
        Pads an image to a square based on the longest edge.

        Args:
            images (`torch.Tensor`):
                The images to pad. Shape: (batch_size, num_channels, height, width) or (num_channels, height, width).
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding. Can be an integer for single channel or a
                tuple of integers representing for multi-channel images. If passed as integer
                in multi-channel mode, it will default to `0` in subsequent channels.
        Returns:
            `torch.Tensor`: The padded images.
        """
        height, width = images.shape[-2:]

        if height == width:
            return images

        num_channels = images.shape[1]
        if isinstance(background_color, int):
            background_color = [background_color] + [0] * (num_channels - 1)
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        max_dim = max(height, width)
        paste_x_left = (max_dim - width) // 2
        paste_y_left = (max_dim - height) // 2
        paste_x_right = max_dim - width - paste_x_left
        paste_y_right = max_dim - height - paste_y_left
        padded_images = tvF.pad(
            images, padding=[paste_x_left, paste_y_left, paste_x_right, paste_y_right], fill=background_color
        )

        return padded_images

    def preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_pad:
                stacked_images = self.pad_to_square(
                    images=stacked_images, background_color=tuple(int(x * 255) for x in self.image_mean)
                )
            resized_images_grouped[shape] = stacked_images
        padded_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for batched resizing
        # Needed in case do_pad is False, or padding returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(padded_images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
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

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


class LlavaPythonBackend(PythonBackend):
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

    def preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling" | "tvF.InterpolationMode" | int | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
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


@auto_docstring(custom_intro="Constructs a LLaVa image processor.")
class LlavaImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    _backend_classes = {
        "torchvision": LlavaTorchVisionBackend,
        "python": LlavaPythonBackend,
    }

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


__all__ = ["LlavaImageProcessor"]
