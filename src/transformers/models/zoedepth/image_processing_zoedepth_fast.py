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
"""Fast Image processor class for ZoeDepth."""

import math
from typing import Optional, Union
import numpy as np

from transformers.image_transforms import get_size_with_aspect_ratio, group_images_by_shape, reorder_images
from transformers.processing_utils import Unpack
from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast, BatchFeature, DefaultFastImageProcessorKwargs
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ChannelDimension, PILImageResampling, SizeDict, get_image_size, get_image_size_for_max_height_width, infer_channel_dimension_format
from ...utils import add_start_docstrings, is_torchvision_available, is_torchvision_v2_available, is_torch_available, TensorType

if is_torch_available():
    import torch

if is_torchvision_available():
    from ...image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

def constrain_to_multiple_of(val, multiple, min_val=0):
    x = (np.round(val / multiple) * multiple).astype(int)

    if x < min_val:
        x = math.ceil(val / multiple) * multiple

    return x

# Logic adapted from torchvision resizing logic: https://github.com/pytorch/vision/blob/511924c1ced4ce0461197e5caa64ce5b9e558aab/torchvision/transforms/functional.py#L366
def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, tuple[int, int], list[int], tuple[int]],
    default_to_square: bool = True,
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    multiple: int = None,
    keep_aspect_ratio: bool = False,
) -> tuple:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]` or List[int] or `Tuple[int]`):
            The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be matched to
            this.

            If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
            `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to this
            number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
        default_to_square (`bool`, *optional*, defaults to `True`):
            How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a square
            (`size`,`size`). If set to `False`, will replicate
            [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
            with support for resizing only the smallest edge and providing an optional `max_size`.
        max_size (`int`, *optional*):
            The maximum allowed for the longer edge of the resized image: if the longer edge of the image is greater
            than `max_size` after being resized according to `size`, then the image is resized again so that the longer
            edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller edge may be shorter
            than `size`. Only used if `default_to_square` is `False`.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `tuple`: The target (height, width) dimension of the output image after resizing.
    """
    if input_data_format is None:
        # We assume that all images have the same channel dimension format.
        input_data_format = infer_channel_dimension_format(input_image[0])
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            output_height, output_width = size
        
            # determine new height and width
            # QUESTION: Is it ok to default this to ChannelDimension.FIRST?
            input_height, input_width = get_image_size(input_image, input_data_format)
            scale_height = output_height / input_height
            scale_width = output_width / input_width

            if keep_aspect_ratio:
                # scale as little as possible
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            new_height = constrain_to_multiple_of(scale_height * input_height, multiple=multiple)
            new_width = constrain_to_multiple_of(scale_width * input_width, multiple=multiple)
            return (new_height, new_width)
        elif len(size) == 1:
            # Perform same logic as if size was an int
            size = size[0]
        else:
            raise ValueError("size must have 1 or 2 elements if it is a list or tuple")

    if default_to_square:
        return (size, size)
    height, width = get_image_size(input_image, input_data_format)
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size

    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    if max_size is not None:
        if max_size <= requested_new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the requested "
                f"size for the smaller edge size = {size}"
            )
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size

    new_long = constrain_to_multiple_of(new_long, multiple=multiple)
    new_short = constrain_to_multiple_of(new_short, multiple=multiple)

    return (new_long, new_short) if width <= height else (new_short, new_long)

class ZoeDepthFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    ensure_multiple_of: Optional[int] = None
    keep_aspect_ratio: bool = False

@add_start_docstrings(
    "Constructs a fast ZoeDepth image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class ZoeDepthImageProcessorFast(BaseImageProcessorFast):
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
    size = {"height": 384, "width": 512}
    do_resize = True
    do_rescale = True
    do_normalize = True
    valid_kwargs = ZoeDepthFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[ZoeDepthFastImageProcessorKwargs]):
        print(f"input_data_format: {kwargs.get('input_data_format')}")
        super().__init__(**kwargs)
    
    # def preprocess(self, images: ImageInput, **kwargs: Unpack[ZoeDepthFastImageProcessorKwargs]) -> BatchFeature:
    #         return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        ensure_multiple_of: int = 1,
        keep_aspect_ratio: bool = False,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                image,
                size=size.shortest_edge,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        elif size.max_height and size.max_width:
            # // TODO left off here
            new_size = get_image_size_for_max_height_width(image.size()[-2:], output_size=size, multiple=ensure_multiple_of)
        elif size.height and size.width:
            new_size = get_resize_output_image_size(image, size=(size.height, size.width), multiple=ensure_multiple_of, keep_aspect_ratio=keep_aspect_ratio)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)
    
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
        ensure_multiple_of: int,
        keep_aspect_ratio,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation, ensure_multiple_of=ensure_multiple_of, keep_aspect_ratio=keep_aspect_ratio)
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

__all__ = ["ZoeDepthImageProcessorFast"]
