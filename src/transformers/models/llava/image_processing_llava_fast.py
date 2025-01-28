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
"""Fast Image processor class for LLaVa."""

from typing import Dict, List, Optional, Tuple, Union

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_transforms import (
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...utils import (
    TensorType,
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
    "Constructs a fast Llava image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        do_pad (`bool`, *optional*, defaults to `self.do_pad`):
            Whether to pad the image to a square based on the longest edge. Can be overridden by the `do_pad` parameter
    """,
)
class LlavaImageProcessorFast(BaseImageProcessorFast):
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
    valid_extra_kwargs = ["do_pad"]

    def __init__(
        self,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        default_to_square: bool = None,
        resample: Union[PILImageResampling, "F.InterpolationMode"] = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = None,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        do_convert_rgb: bool = None,
        do_pad: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(
            do_resize=do_resize,
            size=size,
            default_to_square=default_to_square,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            do_pad=do_pad,
            **kwargs,
        )

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image to a square based on the longest edge. Can be overridden by the `do_pad` parameter
        """,
    )
    def preprocess(self, *args, **kwargs) -> BatchFeature:
        return super().preprocess(*args, **kwargs)

    def pad_to_square(
        self,
        images: "torch.Tensor",
        background_color: Union[int, Tuple[int, int, int]] = 0,
    ) -> "torch.Tensor":
        """
        Pads an image to a square based on the longest edge.

        Args:
            images (`np.ndarray`):
                The images to pad.
            background_color (`int` or `Tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding. Can be an integer for single channel or a
                tuple of integers representing for multi-channel images. If passed as integer
                in mutli-channel mode, it will default to `0` in subsequent channels.
        Returns:
            `torch.Tensor`: The padded images.
        """
        height, width = get_image_size(images, ChannelDimension.FIRST)

        if height == width:
            return images

        num_channels = images.shape[1] if len(images.shape) == 4 else images.shape[0]
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

        return padded_images

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_pad: bool,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
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
        grouped_images, grouped_images_index = group_images_by_shape(padded_images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
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


__all__ = ["LlavaImageProcessorFast"]
