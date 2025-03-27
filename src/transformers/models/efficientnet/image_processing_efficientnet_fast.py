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

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast, BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling, SizeDict
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


@add_start_docstrings(
    "Constructs a fast EfficientNet image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
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
    rescale_factor = 1 / 255
    rescale_offset = False
    do_normalize = True
    include_top = True


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
    return_tensors: Optional[Union[str, TensorType]],
    **kwargs,
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
        if include_top:
            images = self.normalize(images.to(dtype=torch.float32), 0, image_std)
        processed_images_grouped[shape] = stacked_images

    processed_images = reorder_images(processed_images_grouped, grouped_images_index)
    processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

    return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def rescale(
        self,
        image: "torch.Tensor",
        scale: Union[int, float],
        offset: bool = True,
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
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            offset (`bool`, *optional*):
                Whether to scale the image in both negative and positive directions.

        Returns:
            `torch.Tensor`: The rescaled image.
        """

        rescaled_image = image * scale

        if offset:
            rescaled_image = rescaled_image - 1

        return rescaled_image


__all__ = ["EfficientNetImageProcessorFast"]
