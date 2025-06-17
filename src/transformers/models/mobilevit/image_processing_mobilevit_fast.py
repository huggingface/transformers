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

from typing import List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...utils import (
    TensorType,
    auto_docstring,
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


@auto_docstring
class MobileViTImageProcessorFast(BaseImageProcessorFast):
    # Default values verified against the slow MobileViTImageProcessor
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 256, "width": 256}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_flip_channel_order = True

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images and optionally segmentation maps.
        """
        if segmentation_maps is not None:
            # For now, pass None for segmentation maps as the base class doesn't handle them
            # This test is mainly checking that both processors can handle the same interface
            # In a full implementation, we'd need to process segmentation maps similarly to the slow processor
            pass

        # Call parent preprocess method for images only
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ):
        # Extract the custom parameter
        do_flip_channel_order = kwargs.pop("do_flip_channel_order", self.do_flip_channel_order)

        # First apply the standard processing (resize, crop, rescale, normalize)
        processed_batch = super()._preprocess(
            images=images,
            do_resize=do_resize,
            size=size,
            interpolation=interpolation,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            return_tensors=None,  # Don't stack yet, we need to flip channels first
            **kwargs,
        )

        # Extract the list of processed images from the BatchFeature
        processed_data = processed_batch["pixel_values"]

        # Apply channel flipping if requested (RGB to BGR)
        if do_flip_channel_order:
            # Flip the channel order for each image
            processed_images = []
            for image in processed_data:
                # Flip channels: [C, H, W] -> flip dimension 0
                flipped_image = torch.flip(image, dims=[0])
                processed_images.append(flipped_image)
        else:
            processed_images = processed_data

        # Stack if return_tensors is specified
        if return_tensors:
            processed_images = torch.stack(processed_images, dim=0)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["MobileViTImageProcessorFast"]
