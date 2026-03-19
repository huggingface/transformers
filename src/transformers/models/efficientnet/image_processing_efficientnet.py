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
"""Image processor class for EfficientNet."""

from functools import lru_cache
from typing import Optional

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available


if is_torchvision_available():
    import torch
    from torchvision.transforms.v2 import functional as tvF


class EfficientNetImageProcessorKwargs(ImagesKwargs, total=False):
    """
    rescale_offset (`bool`, *optional*, defaults to `self.rescale_offset`):
        Whether to rescale the image between [-max_range/2, scale_range/2] instead of [0, scale_range].
    include_top (`bool`, *optional*, defaults to `self.include_top`):
        Normalize the image again with the standard deviation only for image classification if set to True.
    """

    rescale_offset: bool
    include_top: bool


@auto_docstring
class EfficientNetImageProcessor(TorchvisionBackend):
    """Torchvision backend for EfficientNet with rescale offset and include_top."""

    valid_kwargs = EfficientNetImageProcessorKwargs

    resample = PILImageResampling.BICUBIC
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

    def __init__(self, **kwargs: Unpack[EfficientNetImageProcessorKwargs]):
        super().__init__(**kwargs)

    def rescale(
        self,
        image: "torch.Tensor",
        scale: float,
        offset: bool = False,
        **kwargs,
    ) -> "torch.Tensor":
        """Rescale by scale; if offset=True then image = image * scale - 1."""
        rescaled = image * scale
        if offset:
            rescaled -= 1
        return rescaled

    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(
        self,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        device: Optional["torch.device"] = None,
        rescale_offset: bool | None = False,
    ) -> tuple:
        if do_rescale and do_normalize and not rescale_offset:
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale

    def rescale_and_normalize_efficientnet(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
        rescale_offset: bool = False,
    ) -> "torch.Tensor":
        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device,
            rescale_offset=rescale_offset,
        )
        # We can't fuse rescale and normalize when we need to apply the offset (rescale_offset=True)
        if do_rescale:
            images = self.rescale(images, rescale_factor, offset=rescale_offset)
        if do_normalize:
            images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)
        return images

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
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
        rescale_offset: bool = False,
        include_top: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for EfficientNet."""
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            stacked_images = self.rescale_and_normalize_efficientnet(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std, rescale_offset
            )
            if include_top:
                stacked_images = self.normalize(stacked_images, 0, image_std)
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["EfficientNetImageProcessor"]
