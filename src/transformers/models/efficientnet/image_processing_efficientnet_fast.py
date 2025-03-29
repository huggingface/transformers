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

from typing import Optional, Union

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
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
    """,
)
class EfficientNetImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.NEAREST
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

    def rescale(
        self,
        image: "torch.Tensor",
        scale: float,
        offset: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> "torch.Tensor":
        """Rescale an image by a scale factor.

        If `offset` is `True`, the image has its values rescaled by `scale` and then offset by 1. If `scale` is
        1/127.5, the image is rescaled between [-1, 1].
        image = image * scale - 1

        If `offset` is `False`, and `scale` is 1/255, the image is rescaled between [0, 1].
        image = image * scale

        Args:
        image (`np.ndarray`):
            Image to rescale.
        scale (`int` or `float`):
            Scale to apply to the image.
        offset (`bool`, default to True):
            Whether to scale the image in both negative and positive directions.
        dtype (`torch.dtype`, default to `torch.float32`):
            Data type of the rescaled image.
        """
        rescaled = image.to(dtype=torch.float64) * scale

        if offset:
            rescaled = rescaled - 1

        return rescaled.to(dtype=dtype)

    def rescale_and_normalize(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
        rescale_offset: bool = True,
    ) -> "torch.Tensor":
        """
        Rescale and normalize images.
        """
        if do_rescale:
            images = self.rescale(image=images, scale=rescale_factor, offset=rescale_offset)

        if do_normalize:
            images = self.normalize(image=images, mean=image_mean, std=image_std)

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
                images=stacked_images,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                rescale_offset=rescale_offset,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
            )

            if include_top:
                stacked_images = self.normalize(stacked_images, mean=0, std=image_std)

            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["EfficientNetImageProcessorFast"]
