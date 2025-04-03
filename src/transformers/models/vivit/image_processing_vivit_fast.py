# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Fast Image processor class for Vivit."""

from typing import Optional, Union
from transformers.image_transforms import group_images_by_shape, reorder_images
from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast, DefaultFastImageProcessorKwargs
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ImageInput, PILImageResampling, SizeDict
from ...utils import TensorType, add_start_docstrings, is_torch_available
from ...processing_utils import Unpack
from ...image_processing_utils import (
    BatchFeature,
)

if is_torch_available():
    import torch

@add_start_docstrings(
    "Constructs a fast Vivit image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class VivitImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 256}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None

    def __init__(self, **kwargs: Unpack[DefaultFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def preprocess(self, images: ImageInput, **kwargs: Unpack[DefaultFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs, num_frames=10)
    
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
        num_frames: int,
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
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = [processed_images[i : i + num_frames] for i in range(0, len(processed_images), num_frames)]
        processed_images = [torch.stack(batch) for batch in processed_images]
        processed_images = torch.stack(processed_images)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)
    
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


__all__ = ["VivitImageProcessorFast"]
