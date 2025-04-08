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
"""Fast Image processor class for DPT."""

import math
from typing import Optional, Union
from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ChannelDimension, PILImageResampling, SizeDict, get_image_size, infer_channel_dimension_format
from ...utils import add_start_docstrings, is_torchvision_available, is_torchvision_v2_available, is_torch_available

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

@add_start_docstrings(
    "Constructs a fast DPT image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class DPTImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True

    # SPECIFIED IN SLOW IMAGE PROCESSOR FOR DPT, NOT INCLUDED IN GENERATED:
    # rescaled factiore is specified in BaseImageProcessorFast, do_pad is not given default value there
    do_pad = False
    rescale_factor = 1 / 255
    # In BaseImageProcessorFast this is called size_divisibility (I think)
    size_divisor = None

    # NOT SPECIFIED IN SLOW IMAGE PROCESSOR FOR DPT:
    # default_to_square = None (<-- this broke one of the tests as it override this value to None)
    # crop_size = None
    # do_center_crop = None
    # do_convert_rgb = None

    def pad_image(
        self,
        image: "torch.Tensor",
        size_divisor: int,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> "torch.Tensor":
        """
        Center pad an image to be a multiple of `size_divisor`.

        Args:
            image (`torch.Tensor`):
                Image to pad.
            size_divisor (`int`):
                The width and height of the image will be padded to a multiple of this number.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """

        def _get_pad(size, size_divisor):
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right
        
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # TODO reject if channels_last (torchvision only support channels_first )

        height, width = get_image_size(image, input_data_format)

        pad_top, pad_bottom = _get_pad(height, size_divisor)
        pad_left, pad_right = _get_pad(width, size_divisor)

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return F.pad(image, padding)

__all__ = ["DPTImageProcessorFast"]
