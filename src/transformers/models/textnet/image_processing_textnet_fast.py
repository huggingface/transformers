# coding=utf-8
# Copyright 2025 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for TextNet."""

import enum
from typing import Optional

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, DefaultFastImageProcessorKwargs
from ...image_transforms import get_resize_output_image_size, group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import auto_docstring, is_torch_available, is_torchvision_available, is_torchvision_v2_available


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class TextNetFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    size_divisor (`int`, *optional*, defaults to 32):
        Ensures height and width are rounded to a multiple of this value after resizing.
    """

    size_divisor: Optional[int]


@auto_docstring
class TextNetImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 640}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    size_divisor = 32
    valid_kwargs = TextNetFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[TextNetFastImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        size_divisor: int,
        disable_grouping: Optional[bool],
        **kwargs,
    ) -> BatchFeature:
        if do_resize:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
            resized_images_grouped = {}

            for shape, stacked_images in grouped_images.items():
                if size.shortest_edge:
                    new_size = get_resize_output_image_size(
                        stacked_images[0],
                        size=size.shortest_edge,
                        default_to_square=False,
                        input_data_format=ChannelDimension.FIRST,
                    )
                else:
                    raise ValueError(f"Size must contain 'shortest_edge' key. Got {size}.")
                # ensure height and width are divisible by size_divisor
                height, width = new_size
                if height % size_divisor != 0:
                    height += size_divisor - (height % size_divisor)
                if width % size_divisor != 0:
                    width += size_divisor - (width % size_divisor)

                new_size_dict = SizeDict(height=height, width=width)

                stacked_images = self.resize(image=stacked_images, size=new_size_dict, interpolation=interpolation)
                resized_images_grouped[shape] = stacked_images

            images = reorder_images(resized_images_grouped, grouped_images_index)

        # set do_resize to False since we have already resized the images
        return super()._preprocess(
            images=images,
            do_resize=False,
            size=size,
            interpolation=interpolation,
            disable_grouping=disable_grouping,
            **kwargs,
        )

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[TextNetFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def to_dict(self) -> dict:
        """
        Return a dict that will yield the same config as the slow processor.

        This ensures serialization compatibility between slow and fast versions by:
        1. Using the slow processor name in image_processor_type
        2. Converting resample from enum to int value
        """
        config = super().to_dict()

        # Use slow processor name for compatibility
        config["image_processor_type"] = "TextNetImageProcessor"

        # Convert enum to int for resample
        if isinstance(config.get("resample"), enum.Enum):
            config["resample"] = config["resample"].value

        return config


__all__ = ["TextNetImageProcessorFast"]
