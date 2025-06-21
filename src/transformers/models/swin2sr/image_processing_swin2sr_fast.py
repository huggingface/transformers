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


class Swin2SRFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    do_pad (`bool`, *optional*, defaults to `True`):
        Whether to pad the image to make the height and width divisible by `window_size`.
    pad_size (`int`, *optional*, defaults to `8`):
        The size of the sliding window for the local attention.
    """

    do_pad: Optional[bool]
    pad_size: Optional[int]


@auto_docstring
class Swin2SRImageProcessorFast(BaseImageProcessorFast):
    do_rescale = True
    rescale_factor = 1 / 255
    do_pad = True
    pad_size = 8
    valid_kwargs = Swin2SRFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Swin2SRFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def preprocess(self, images: ImageInput, **kwargs: Unpack[Swin2SRFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def pad(self, images: "torch.Tensor", size: int) -> "torch.Tensor":
        """
        Pad an image to make the height and width divisible by `size`.

        Args:
            images (`torch.Tensor`):
                Images to pad.
            size (`int`):
                The size to make the height and width divisible by.

        Returns:
            `torch.Tensor`: The padded images.
        """
        height, width = get_image_size(images, ChannelDimension.FIRST)
        pad_height = (height // size + 1) * size - height
        pad_width = (width // size + 1) * size - width

        return F.pad(
            images,
            (0, 0, pad_width, pad_height),
            padding_mode="symmetric",
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_rescale: bool,
        rescale_factor: float,
        do_pad: bool,
        pad_size: int,
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
                stacked_images = self.pad(stacked_images, size=pad_size)
            processed_image_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_image_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["Swin2SRImageProcessorFast"]
