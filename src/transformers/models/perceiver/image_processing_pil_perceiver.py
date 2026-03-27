# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Perceiver."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


@auto_docstring
class PerceiverImageProcessorPil(PilBackend):
    """PIL backend for Perceiver with custom center crop."""

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 224, "width": 224}
    crop_size = {"height": 256, "width": 256}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

    def center_crop(
        self,
        image: np.ndarray,
        size: SizeDict,
        crop_size: SizeDict,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to ((size.height / crop_size.height) * min_dim, (size.width / crop_size.width) * min_dim),
        where min_dim is the minimum of the image height and width.
        If the requested crop size exceeds the image dimensions along any edge, the image is padded with zeros before
        center cropping.
        """
        if size.height is None or size.width is None:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        if crop_size.height is None or crop_size.width is None:
            raise ValueError(f"The crop_size dictionary must have keys 'height' and 'width'. Got {crop_size.keys()}")
        height, width = image.shape[-2:]
        min_dim = min(height, width)
        cropped_height = int((size.height / crop_size.height) * min_dim)
        cropped_width = int((size.width / crop_size.width) * min_dim)
        return super().center_crop(
            image,
            SizeDict(height=cropped_height, width=cropped_width),
            **kwargs,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
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
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for Perceiver: center_crop -> resize -> rescale and normalize."""
        processed_images = []
        for image in images:
            if do_center_crop:
                image = self.center_crop(image, size=size, crop_size=crop_size)
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["PerceiverImageProcessorPil"]
