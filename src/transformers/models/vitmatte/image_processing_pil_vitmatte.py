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
"""Image processor class for ViTMatte."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available


if is_torch_available():
    pass


# Adapted from transformers.models.vitmatte.image_processing_vitmatte.VitMatteImageProcessorKwargs
class VitMatteImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
        The width and height of the image will be padded to be divisible by this number.
    """

    size_divisor: int


@auto_docstring
class VitMatteImageProcessorPil(PilBackend):
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_pad = True
    size_divisor = 32
    valid_kwargs = VitMatteImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[VitMatteImageProcessorKwargs]) -> None:
        size_divisibility = kwargs.pop("size_divisibility", None)
        if size_divisibility is not None:
            kwargs.setdefault("size_divisor", size_divisibility)
        super().__init__(**kwargs)

    def pad_image(
        self,
        image: np.ndarray,
        size_divisor: int = 32,
    ) -> np.ndarray:
        """
        Args:
            image (`np.ndarray`):
                Image to pad.
            size_divisor (`int`, *optional*, defaults to 32):
                The width and height of the image will be padded to be divisible by this number.
        """
        height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        pad_height = 0 if height % size_divisor == 0 else size_divisor - height % size_divisor
        pad_width = 0 if width % size_divisor == 0 else size_divisor - width % size_divisor
        if pad_width + pad_height > 0:
            padding = ((0, pad_height), (0, pad_width))
            image = np_pad(
                image,
                padding=padding,
                mode=PaddingMode.CONSTANT,
                constant_values=0,
                data_format=ChannelDimension.FIRST,
                input_data_format=ChannelDimension.FIRST,
            )

        return image

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        trimaps: ImageInput,
        **kwargs: Unpack[VitMatteImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        trimaps (`ImageInput`):
            The trimaps to preprocess.
        """
        return super().preprocess(images, trimaps, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        trimaps: ImageInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: str | None = None,
        **kwargs: Unpack[VitMatteImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        trimaps = self._prepare_image_like_inputs(images=trimaps, expected_ndims=2, device=device)

        return self._preprocess(images, trimaps, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        trimaps: list[np.ndarray],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        size_divisor: int | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image, trimap in zip(images, trimaps):
            if do_rescale:
                image = self.rescale(image, rescale_factor)
                trimap = self.rescale(trimap, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            # Concatenate images and trimaps along channel dimension
            # trimap is already (1, H, W) from _prepare_image_like_inputs with expected_ndims=2
            if trimap.ndim == 3 and trimap.shape[0] == 1:
                image = np.concatenate([image, trimap], axis=0)
            else:
                image = np.concatenate([image, np.expand_dims(trimap, axis=0)], axis=0)
            if do_pad:
                image = self.pad_image(image, size_divisor)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["VitMatteImageProcessorPil"]
