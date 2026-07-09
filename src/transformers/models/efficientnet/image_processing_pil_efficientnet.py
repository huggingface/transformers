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

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


# Adapted from transformers.models.efficientnet.image_processing_efficientnet.EfficientNetImageProcessorKwargs
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
class EfficientNetImageProcessorPil(PilBackend):
    """PIL backend for EfficientNet with rescale offset and include_top."""

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
        image: np.ndarray,
        scale: float,
        offset: bool = False,
    ) -> np.ndarray:
        """Rescale by scale; if offset=True then image = image * scale - 1."""
        rescaled = super().rescale(image, scale=scale)
        if offset:
            rescaled -= 1
        return rescaled

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        return_tensors: str | TensorType | None,
        rescale_offset: bool = False,
        include_top: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for EfficientNet."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor, offset=rescale_offset)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            if include_top:
                image = self.normalize(image, 0, image_std)
            processed_images.append(image)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["EfficientNetImageProcessorPil"]
