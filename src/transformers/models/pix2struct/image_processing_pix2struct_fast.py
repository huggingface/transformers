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
"""Fast Image processor class for Pix2Struct."""

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS
    )
from ...utils import add_start_docstrings

import torch

from typing import Any, Optional, TypedDict, Union

from ...image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)

from ...image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
    group_images_by_shape,
    reorder_images,
)

from ...image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    SizeDict,
    get_image_size,
    get_image_size_for_max_height_width,
    get_image_type,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    validate_kwargs,
    validate_preprocess_arguments,
)

from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Any, Optional, TypedDict, Union

import numpy as np

from ...image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)

class Pix2StructFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    patch_size: Optional[dict[str, int]]
    max_patches: Optional[int]
    is_vqa: Optional[bool]

@add_start_docstrings(
    "Constructs a fast Pix2Struct image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class Pix2StructImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed

    do_normalize = True
    do_convert_rgb = True
    patch_size = {"height": 16, "width": 16}
    max_patches = 2048
    is_vqa = False

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
                The patch size to use for the image. According to Pix2Struct paper and code, the patch size is 16x16.
            max_patches (`int`, *optional*, defaults to 2048):
                The maximum number of patches to extract from the image as per the [Pix2Struct
                paper](https://arxiv.org/pdf/2210.03347.pdf).
            is_vqa (`bool`, *optional*, defaults to `False`):
                Whether or not the image processor is for the VQA task. If `True` and `header_text` is passed in, text is
                rendered onto the input images.
        """,
    )
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Pix2StructFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

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
        **kwargs,
    ) -> BatchFeature:

        # make image list

        # convert rgb if needed
        # 
        # convert numpy array
        # 
        # render header if is_vqa, render-text first, resize original image accordingly and paste with the header part 
        # header-image可以批量生产，size需要由image与header-image一起决定，resize只有在image shape相同，resize shape也相同的情况下才能使用v2的api
        # 
        # normalize if needed
        # 
        # extract flattened patches 

        # images 为经过rgb转化，input-format调整，并且放置到device上的list tensor


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
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

__all__ = ["Pix2StructImageProcessorFast"]
