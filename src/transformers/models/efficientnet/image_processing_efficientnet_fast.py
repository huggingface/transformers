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
"""Fast Image processor class for EfficientNet."""

from functools import lru_cache
from typing import Optional, Union

from ...image_processing_utils_fast import BaseImageProcessorFast, BatchFeature, DefaultFastImageProcessorKwargs
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ImageInput, PILImageResampling, SizeDict
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


class EfficientNetFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
        rescale_offset (`bool`, *optional*, defaults to `self.rescale_offset`):
            Whether to rescale the image between [-max_range/2, scale_range/2] instead of [0, scale_range].
        include_top (`bool`, *optional*, defaults to `self.include_top`):
            Normalize the image again with the standard deviation only for image classification if set to True.
    """

    rescale_offset: bool
    include_top: bool


@auto_docstring
class EfficientNetImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.NEAREST
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 346, "width": 346}
    crop_size = {"height": 289, "width": 289}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    rescale_factor = 1 / 255
    rescale_offset = False
    do_normalize = True
    include_top = True
    valid_kwargs = EfficientNetFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[EfficientNetFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def rescale(
        self,
        image: "torch.Tensor",
        scale: float,
        offset: Optional[bool] = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Rescale an image by a scale factor.

        If `offset` is `True`, the image has its values rescaled by `scale` and then offset by 1. If `scale` is
        1/127.5, the image is rescaled between [-1, 1].
            image = image * scale - 1

        If `offset` is `False`, and `scale` is 1/255, the image is rescaled between [0, 1].
            image = image * scale

        Args:
            image (`torch.Tensor`):
                Image to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            offset (`bool`, *optional*):
                Whether to scale the image in both negative and positive directions.

        Returns:
            `torch.Tensor`: The rescaled image.
        """

        rescaled_image = image * scale

        if offset:
            rescaled_image -= 1

        return rescaled_image

    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        device: Optional["torch.device"] = None,
        rescale_offset: Optional[bool] = False,
    ) -> tuple:
        if do_rescale and do_normalize and not rescale_offset:
            # Fused rescale and normalize
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale

    def rescale_and_normalize(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
        rescale_offset: bool = False,
    ) -> "torch.Tensor":
        """
        Rescale and normalize images.
        """
        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device,
            rescale_offset=rescale_offset,
        )
        # if/elif as we use fused rescale and normalize if both are set to True
        if do_rescale:
            images = self.rescale(images, rescale_factor, rescale_offset)
        if do_normalize:
            images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)

        return images

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        rescale_offset: bool,
        do_normalize: bool,
        include_top: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
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
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std, rescale_offset
            )
            if include_top:
                stacked_images = self.normalize(stacked_images, 0, image_std)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[EfficientNetFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)


__all__ = ["EfficientNetImageProcessorFast"]
