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
from typing import Optional, Union, Unpack

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast, \
    DefaultFastImageProcessorKwargs, BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling, SizeDict, \
    ChannelDimension, ImageInput
from ...utils import add_start_docstrings, TensorType, is_torch_available
from ...image_processing_utils import (
    BatchFeature,
)
from ...processing_utils import Unpack

if is_torch_available():
    import torch

class EfficientNetFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    rescale_offset: bool
    include_top: bool

@add_start_docstrings(
    "Constructs a fast EfficientNet image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
    rescale_offset (`bool`, defaults to `False`):
        Whether to rescale the image between [-max_range/2, scale_range/2] instead of [0, scale_range].
    include_top (`bool`, defaults to `True`):
        Normalize the image again with the standard deviation only for image classification if set to True.
    """
)
class EfficientNetImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.Image.NEAREST
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 346, "width": 346}
    crop_size = {"height": 289, "width": 289}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    rescale_factor = 1 / 255
    rescale_offset: bool = False
    include_top: bool = True
    valid_kwargs = EfficientNetFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[EfficientNetFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
        rescale_offset (`bool`, *optional*, defaults to `self.rescale_offset`):
            Whether to rescale the image between [-max_range/2, scale_range/2] instead of [0, scale_range].
        include_top (`bool`, *optional*, defaults to `self.include_top`):
            Normalize the image again with the standard deviation only for image classification if set to True.
        """,
    )
    def preprocess(self, images: ImageInput, **kwargs: Unpack[EfficientNetFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def offset(
        self,
        image: "torch.Tensor",
        rescale_factor: float,
    ) -> "torch.Tensor":

        if rescale_factor not in (1 / 127, 1 / 255):
            raise ValueError(f"Rescale offset is only supported for scale 1/127 or 1/255, got {rescale_factor}")

        offset = 1 if rescale_factor == 1 / 127 else 0.5
        rescaled = image - offset

        return rescaled

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
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        rescale_offset: bool = False,
        include_top: bool = True,
    ) -> BatchFeature:

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if rescale_offset:
                stacked_images = self.offset(stacked_images, rescale_factor)

            if include_top:
                stacked_images = self.normalize(stacked_images, 0, image_std)

            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)




__all__ = ["EfficientNetImageProcessorFast"]
