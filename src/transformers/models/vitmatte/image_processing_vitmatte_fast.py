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
"""Fast Image processor class for ViTMatte."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    get_image_size,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    filter_out_non_signature_kwargs,
    logging,
)


logger = logging.get_logger(__name__)


class VitMatteFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    size_divisor (`int`, *optional*, defaults to 32):
        The width and height of the image will be padded to be divisible by this number.
    """

    size_divisor: Optional[int]


@auto_docstring
class VitMatteImageProcessorFast(BaseImageProcessorFast):
    do_rescale: bool = True
    rescale_factor: Union[int, float] = 1 / 255
    do_normalize: bool = True
    image_mean: Optional[Union[float, list[float]]] = IMAGENET_STANDARD_MEAN
    image_std: Optional[Union[float, list[float]]] = IMAGENET_STANDARD_STD
    do_pad: bool = True
    size_divisor: int = 32
    valid_kwargs = VitMatteFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[VitMatteFastImageProcessorKwargs]) -> None:
        size_divisibility = kwargs.pop("size_divisibility", None)
        kwargs.setdefault("size_divisor", size_divisibility)
        super().__init__(**kwargs)

    @property
    def size_divisibility(self):
        logger.warning(
            "`self.size_divisibility` attribute is deprecated and will be removed in v5. Use `self.size_divisor` instead"
        )
        return self.size_divisor

    @size_divisibility.setter
    def size_divisibility(self, value):
        logger.warning(
            "`self.size_divisibility` attribute is deprecated and will be removed in v5. Use `self.size_divisor` instead"
        )
        self.size_divisor = value

    def _pad_image(
        self,
        images: torch.Tensor,
        size_divisibility: int = 32,
    ) -> torch.Tensor:
        """
        Pads an image or batched images constantly so that width and height are divisible by size_divisibility

        Args:
            image (`torch.Tensor`):
                Image to pad.
            size_divisibility (`int`, *optional*, defaults to 32):
                The width and height of the image will be padded to be divisible by this number.
        """
        height, width = get_image_size(images, channel_dim=ChannelDimension.FIRST)

        pad_height = 0 if height % size_divisibility == 0 else size_divisibility - height % size_divisibility
        pad_width = 0 if width % size_divisibility == 0 else size_divisibility - width % size_divisibility

        if pad_width + pad_height > 0:
            padding = (0, 0, pad_width, pad_height)
            images = F.pad(images, padding)

        return images

    @auto_docstring
    def preprocess(
        self,
        images: list["torch.Tensor"],
        trimaps: list["torch.Tensor"],
        **kwargs: Unpack[VitMatteFastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        trimaps (`list[torch.Tensor]`):
            The trimaps to preprocess.
        """
        return super().preprocess(images, trimaps, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        trimaps: ImageInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[VitMatteFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        trimaps = self._prepare_image_like_inputs(images=trimaps, expected_ndims=2, device=device)

        return self._preprocess(images, trimaps, **kwargs)

    @filter_out_non_signature_kwargs()
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        trimaps: list["torch.Tensor"],
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_pad: Optional[bool] = None,
        size_divisor: Optional[int] = None,
        disable_grouping: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        grouped_trimaps, grouped_trimaps_index = group_images_by_shape(trimaps, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape in grouped_images:
            stacked_images = grouped_images[shape]
            stacked_trimaps = grouped_trimaps[shape]
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            stacked_trimaps = self.rescale_and_normalize(
                stacked_trimaps, do_rescale, rescale_factor, False, image_mean, image_std
            )
            stacked_images = torch.cat([stacked_images, stacked_trimaps], dim=1)
            if do_pad:
                stacked_images = self._pad_image(stacked_images, size_divisor)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["VitMatteImageProcessorFast"]
