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
"""Fast Image processor class for Swin2SR."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature, ChannelDimension, get_image_size
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
)
from ...utils.deprecation import deprecate_kwarg


logger = logging.get_logger(__name__)


class Swin2SRFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    size_divisor (`int`, *optional*, defaults to `8`):
        The size of the sliding window for the local attention. It will be used to pad the image
        to the size divisible by `size_divisor`
    """

    size_divisor: Optional[int]


@auto_docstring
class Swin2SRImageProcessorFast(BaseImageProcessorFast):
    do_rescale = True
    rescale_factor = 1 / 255
    do_pad = True
    size_divisor = 8
    valid_kwargs = Swin2SRFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Swin2SRFastImageProcessorKwargs]):
        pad_size = kwargs.pop("pad_size", None)
        kwargs.setdefault("size_divisor", pad_size)
        super().__init__(**kwargs)

    @property
    def pad_size(self):
        logger.warning(
            "`self.pad_size` attribute is deprecated and will be removed in v5. Use `self.size_divisor` instead",
        )
        return self.size_divisor

    @pad_size.setter
    def pad_size(self, value):
        logger.warning(
            "`self.pad_size` attribute is deprecated and will be removed in v5. Use `self.size_divisor` instead",
        )
        self.size_divisor = value

    def preprocess(self, images: ImageInput, **kwargs: Unpack[Swin2SRFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    @deprecate_kwarg("size", version="v5", new_name="size_divisor")
    def pad(self, images: "torch.Tensor", size_divisor: int) -> "torch.Tensor":
        """
        Pad an image to make the height and width divisible by `size_divisor`.

        Args:
            images (`torch.Tensor`):
                Images to pad.
            size_divisor (`int`):
                The size to make the height and width divisible by.

        Returns:
            `torch.Tensor`: The padded images.
        """
        height, width = get_image_size(images, ChannelDimension.FIRST)
        pad_height = (height // size_divisor + 1) * size_divisor - height
        pad_width = (width // size_divisor + 1) * size_divisor - width

        return F.pad(
            images,
            (0, 0, pad_width, pad_height),
            padding_mode="symmetric",
        )

    @deprecate_kwarg("pad_size", version="v5", new_name="size_divisor")
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_rescale: bool,
        rescale_factor: float,
        do_pad: bool,
        size_divisor: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_image_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_rescale:
                stacked_images = self.rescale(stacked_images, scale=rescale_factor)
            if do_pad:
                stacked_images = self.pad(stacked_images, size_divisor=size_divisor)
            processed_image_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_image_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["Swin2SRImageProcessorFast"]
