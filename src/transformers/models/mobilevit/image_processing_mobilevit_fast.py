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

import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import PILImageResampling
from ...utils import auto_docstring


@auto_docstring
class MobileViTImageProcessorFast(BaseImageProcessorFast):
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
        image_std=None,
    ):
        processed_images = []

        if do_normalize is None:
            do_normalize = self.do_normalize
        if image_mean is None and hasattr(self, "image_mean"):
            image_mean = self.image_mean
        if image_std is None and hasattr(self, "image_std"):
            image_std = self.image_std

        # Group images by shape for more efficient batch processing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}

        # Process each group of images with the same shape
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images

        # Reorder images to original sequence
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group again after resizing (in case resize produced different sizes)
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(image=stacked_images, size=crop_size)
            if do_rescale:
                stacked_images = self.rescale(image=stacked_images, scale=rescale_factor)
            if do_flip_channel_order:
                # For batched images, we need to handle them all at once
                if stacked_images.ndim > 3 and stacked_images.shape[1] >= 3:
                    # Flip RGB → BGR for batched images
                    flipped = stacked_images.clone()
                    flipped[:, 0:3] = stacked_images[:, [2, 1, 0], ...]
                    stacked_images = flipped
            if do_normalize and image_mean is not None and image_std is not None:
                stacked_images = self.normalize(image=stacked_images, mean=image_mean, std=image_std)
            if do_convert_rgb:
                stacked_images = self.convert_to_rgb(stacked_images)

            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Stack all processed images if return_tensors is specified
        if return_tensors is not None:
            processed_images = torch.stack(processed_images, dim=0)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_semantic_segmentation(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for MobileViTImageProcessorFast.")


__all__ = ["MobileViTImageProcessorFast"]
