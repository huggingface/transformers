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

from typing import Union

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
    from torchvision.transforms.v2 import functional as F

logger = logging.get_logger(__name__)


class LlavaTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for LLaVA with custom pad_to_square and preprocessing order."""

    def pad_to_square(
        self,
        images: "torch.Tensor",
        background_color: Union[int, tuple[int, int, int]] = 0,
    ) -> "torch.Tensor":
        """
        Pads images to a square based on the longest edge.

        Args:
            images (`torch.Tensor`):
                The images to pad. Shape: (batch_size, num_channels, height, width) or (num_channels, height, width).
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding.

        Returns:
            `torch.Tensor`: The padded images.
        """
        # Handle both batched and single image cases
        if images.ndim == 3:
            images = images.unsqueeze(0)
            was_single = True
        else:
            was_single = False

        height, width = images.shape[-2:]
        num_channels = images.shape[1]

        # Convert to Python ints if they're tensor scalars
        height = int(height)
        width = int(width)

        if height == width:
            return images.squeeze(0) if was_single else images

        # Ensure background_color is the correct shape
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

        padded_images = F.pad(
            images, padding=[paste_x_left, paste_y_left, paste_x_right, paste_y_right], fill=background_color
        )

        return padded_images.squeeze(0) if was_single else padded_images

    def preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: Union["PILImageResampling", "F.InterpolationMode", int, None],
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
        """
        Custom preprocessing for LLaVA: pad_to_square -> resize -> center_crop -> rescale_and_normalize.
        """
        # Apply pad_to_square first if needed (before resize)
        if do_pad:
            # Get background color from image_mean (converted to 0-255 range)
            background_color = tuple(int(x * 255) for x in image_mean) if image_mean else 0
            padded_images = []
            for image in images:
                padded_image = self.pad_to_square(image, background_color=background_color)
                padded_images.append(padded_image)
            images = padded_images

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            # Convert lists to tuples for lru_cache compatibility
            image_mean_tuple = tuple(image_mean) if isinstance(image_mean, list) else image_mean
            image_std_tuple = tuple(image_std) if isinstance(image_std, list) else image_std
            stacked_images = self._rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean_tuple, image_std_tuple
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


class LlavaPythonBackend(PythonBackend):
    """Python backend for LLaVA with custom pad_to_square and preprocessing order."""

    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: Union[int, tuple[int, int, int]] = 0,
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
        resample: Union["PILImageResampling", "F.InterpolationMode", int, None],
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
        """
        Custom preprocessing for LLaVA: pad_to_square -> resize -> center_crop -> rescale -> normalize.
        """
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
