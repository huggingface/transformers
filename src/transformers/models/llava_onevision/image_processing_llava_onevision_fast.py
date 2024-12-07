# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for LLaVa-Onevision."""

from ...image_processing_utils_fast import BaseImageProcessorFast, LlavaPatchingMixin
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling


class LlavaOnevisionImageProcessorFast(LlavaPatchingMixin, BaseImageProcessorFast):
    r"""
    Constructs a fast LLaVa-Onevisino-Video video processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        image_grid_pinpoints (`List` *optional*, defaults to `[[672, 336], [336, 672], [672, 672], [336, 1008], [1008, 336]]`):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processinf videos.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    # To be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    image_grid_pinpoints = [
        [384, 384],
        [384, 768],
        [384, 1152],
        [384, 1536],
        [384, 1920],
        [384, 2304],
        [768, 384],
        [768, 768],
        [768, 1152],
        [768, 1536],
        [768, 1920],
        [768, 2304],
        [1152, 384],
        [1152, 768],
        [1152, 1152],
        [1152, 1536],
        [1152, 1920],
        [1152, 2304],
        [1536, 384],
        [1536, 768],
        [1536, 1152],
        [1536, 1536],
        [1536, 1920],
        [1536, 2304],
        [1920, 384],
        [1920, 768],
        [1920, 1152],
        [1920, 1536],
        [1920, 1920],
        [1920, 2304],
        [2304, 384],
        [2304, 768],
        [2304, 1152],
        [2304, 1536],
        [2304, 1920],
        [2304, 2304],
    ]


__all__ = ["LlavaOnevisionImageProcessorFast"]
