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
"""Fast Image processor class for MobileViT."""

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import PILImageResampling
from ...utils import add_start_docstrings
from ...image_processing_utils import BatchFeature
import torch


@add_start_docstrings(
    "Constructs a fast MobileViT image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class MobileViTImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BILINEAR
    image_mean = None
    image_std = None
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 256, "width": 256}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = None
    do_convert_rgb = None
    do_flip_channel_order = True

    def flip_channel_order(self, image):
        # Check if we have 3 or more channels
        if image.shape[0] >= 3:
            # Flip only the first 3 channels (RGB → BGR)
            flipped = image.clone()
            flipped[0:3] = image[[2, 1, 0], ...]
            return flipped
        # For grayscale or other formats, return as is
        return image

    def _preprocess(
        self,
        images,
        do_resize=True,
        size=None,
        interpolation=None,
        do_rescale=True,
        rescale_factor=None,
        do_center_crop=True,
        crop_size=None,
        do_flip_channel_order=True,
        input_data_format=None,
        do_convert_rgb=False,
        return_tensors=None,
        do_normalize=None,
        image_mean=None,
        image_std=None
    ):
        processed_images = []
        
        # 기본값 설정
        if do_normalize is None:
            do_normalize = self.do_normalize
        if image_mean is None and hasattr(self, "image_mean"):
            image_mean = self.image_mean
        if image_std is None and hasattr(self, "image_std"):
            image_std = self.image_std
        
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, interpolation=interpolation)
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor)
            if do_center_crop:
                image = self.center_crop(image=image, size=crop_size)
            if do_flip_channel_order:
                image = self.flip_channel_order(image=image)
            if do_normalize and image_mean is not None and image_std is not None:
                image = self.normalize(image=image, mean=image_mean, std=image_std)
            if do_convert_rgb:
                image = self.convert_to_rgb(image)
            processed_images.append(image)
        
        return BatchFeature(data={"pixel_values": torch.stack(processed_images)}, tensor_type=return_tensors)

__all__ = ["MobileViTImageProcessorFast"]
