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
"""Fast Image processor class for OWLv2."""

import warnings
from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
)
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)
from ..owlvit.image_processing_owlvit_fast import OwlViTImageProcessorFast


class Owlv2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs): ...


@auto_docstring
class Owlv2ImageProcessorFast(OwlViTImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 960, "width": 960}
    rescale_factor = 1 / 255
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    valid_kwargs = Owlv2FastImageProcessorKwargs
    crop_size = None
    do_center_crop = None

    def __init__(self, **kwargs: Unpack[Owlv2FastImageProcessorKwargs]):
        BaseImageProcessorFast.__init__(self, **kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Owlv2FastImageProcessorKwargs]):
        return BaseImageProcessorFast.preprocess(self, images, **kwargs)

    def _pad_images(self, images: "torch.Tensor", constant_value: float = 0.5) -> "torch.Tensor":
        """
        Pad an image with zeros to the given size.
        """
        height, width = images.shape[-2:]
        size = max(height, width)
        pad_bottom = size - height
        pad_right = size - width

        padding = (0, 0, pad_right, pad_bottom)
        padded_image = F.pad(images, padding, fill=constant_value)
        return padded_image

    def pad(
        self,
        images: list["torch.Tensor"],
        disable_grouping: Optional[bool],
        constant_value: float = 0.5,
        **kwargs,
    ) -> list["torch.Tensor"]:
        """
        Unlike the Base class `self.pad` where all images are padded to the maximum image size,
        Owlv2 pads an image to square.
        """
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self._pad_images(
                stacked_images,
                constant_value=constant_value,
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        return processed_images

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        anti_aliasing: bool = True,
        anti_aliasing_sigma=None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image as per the original implementation.

        Args:
            image (`Tensor`):
                Image to resize.
            size (`dict[str, int]`):
                Dictionary containing the height and width to resize the image to.
            anti_aliasing (`bool`, *optional*, defaults to `True`):
                Whether to apply anti-aliasing when downsampling the image.
            anti_aliasing_sigma (`float`, *optional*, defaults to `None`):
                Standard deviation for Gaussian kernel when downsampling the image. If `None`, it will be calculated
                automatically.
        """
        output_shape = (size.height, size.width)

        input_shape = image.shape

        # select height and width from input tensor
        factors = torch.tensor(input_shape[2:]).to(image.device) / torch.tensor(output_shape).to(image.device)

        if anti_aliasing:
            if anti_aliasing_sigma is None:
                anti_aliasing_sigma = ((factors - 1) / 2).clamp(min=0)
            else:
                anti_aliasing_sigma = torch.atleast_1d(anti_aliasing_sigma) * torch.ones_like(factors)
                if torch.any(anti_aliasing_sigma < 0):
                    raise ValueError("Anti-aliasing standard deviation must be greater than or equal to zero")
                elif torch.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                    warnings.warn(
                        "Anti-aliasing standard deviation greater than zero but not down-sampling along all axes"
                    )
            if torch.any(anti_aliasing_sigma == 0):
                filtered = image
            else:
                kernel_sizes = 2 * torch.ceil(3 * anti_aliasing_sigma).int() + 1

                filtered = F.gaussian_blur(
                    image, (kernel_sizes[0], kernel_sizes[1]), sigma=anti_aliasing_sigma.tolist()
                )

        else:
            filtered = image

        out = F.resize(filtered, size=(size.height, size.width), antialias=False)

        return out

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_pad: bool,
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
        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            # Rescale images before other operations as done in original implementation
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, False, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        if do_pad:
            processed_images = self.pad(processed_images, constant_value=0.5, disable_grouping=disable_grouping)

        grouped_images, grouped_images_index = group_images_by_shape(
            processed_images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                resized_stack = self.resize(
                    image=stacked_images,
                    size=size,
                    interpolation=interpolation,
                    input_data_format=ChannelDimension.FIRST,
                )
                resized_images_grouped[shape] = resized_stack
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, False, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["Owlv2ImageProcessorFast"]
