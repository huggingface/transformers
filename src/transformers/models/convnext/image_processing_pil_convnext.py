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
"""Image processor class for ConvNeXT."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import get_resize_output_image_size
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


# Adapted from transformers.models.convnext.image_processing_convnext.ConvNextImageProcessorKwargs
class ConvNextImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    crop_pct (`float`, *optional*, defaults to `self.crop_pct`):
        Percentage of the image to crop. Only has an effect if size < 384.
    """

    crop_pct: float


@auto_docstring
class ConvNextImageProcessorPil(PilBackend):
    """PIL backend for ConvNeXT with custom resize."""

    valid_kwargs = ConvNextImageProcessorKwargs

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 384}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    crop_pct = 224 / 256

    def __init__(self, **kwargs: Unpack[ConvNextImageProcessorKwargs]):
        super().__init__(**kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | None",
        crop_pct: float = 224 / 256,
        **kwargs,
    ) -> np.ndarray:
        """Resize with crop_pct support."""
        if not size.shortest_edge:
            raise ValueError(f"Size dictionary must contain 'shortest_edge' key. Got {size.keys()}")
        shortest_edge = size.shortest_edge

        if shortest_edge < 384:
            # maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
            resize_shortest_edge = int(shortest_edge / crop_pct)
            resize_size = get_resize_output_image_size(
                image, size=resize_shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            image = super().resize(
                image,
                size=SizeDict(height=resize_size[0], width=resize_size[1]),
                resample=resample,
                **kwargs,
            )
            # then crop to (shortest_edge, shortest_edge)
            return super().center_crop(
                image,
                size=SizeDict(height=shortest_edge, width=shortest_edge),
                **kwargs,
            )
        else:
            # warping (no cropping) when evaluated at 384 or larger
            return super().resize(
                image,
                size=SizeDict(height=shortest_edge, width=shortest_edge),
                resample=resample,
                **kwargs,
            )

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
        crop_pct: float = 224 / 256,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for ConvNeXT."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample, crop_pct)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["ConvNextImageProcessorPil"]
