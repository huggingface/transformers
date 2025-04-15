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
from collections.abc import Iterable
from typing import Optional, Tuple, Union

import numpy as np

from transformers.image_processing_base import BatchFeature
from transformers.image_transforms import (
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
    group_images_by_shape,
    reorder_images,
)

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    get_image_size_for_max_height_width,
    infer_channel_dimension_format,
    is_pil_image,
)
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


def get_output_image_size_for_ensure_multiple(
    input_image: "torch.Tensor",
    output_size: Union[int, Iterable[int]],
    keep_aspect_ratio: bool,
    multiple: int,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    def constrain_to_multiple_of(val, multiple, min_val=0, max_val=None):
        x = round(val / multiple) * multiple

        if max_val is not None and x > max_val:
            x = math.floor(val / multiple) * multiple

        if x < min_val:
            x = math.ceil(val / multiple) * multiple

        return x

    output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    input_height, input_width = get_image_size(input_image, input_data_format)
    output_height, output_width = output_size

    # determine new height and width
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


class DPTFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    size_divisor: Optional[int]
    do_pad: Optional[bool]
    ensure_multiple_of: Optional[int]
    keep_aspect_ratio: Optional[bool]
    segmentation_maps: Optional[ImageInput] = (None,)


DPT_IMAGE_PROCESSOR_FAST_KWARGS_DOCSTRING = """
 Args:
    ensure_multiple_of (`int`, *optional*, defaults to 1):
    If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overidden
    by `ensure_multiple_of` in `preprocess`.
"""


@add_start_docstrings(
    "Constructs a fast DPT image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    DPT_IMAGE_PROCESSOR_FAST_KWARGS_DOCSTRING,
)
class DPTImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = False
    rescale_factor = 1 / 255
    ensure_multiple_of = 1
    keep_aspect_ratio = False

    valid_kwargs = DPTFastImageProcessorKwargs

    # Overrides BaseImageProcessor `__call__` so that segmentation maps can be passed
    # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.__call__
    def __call__(self, images, segmentation_maps=None, **kwargs):
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

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
        size_divisor: Optional[int],
        do_pad: bool,
        ensure_multiple_of: Optional[int],
        keep_aspect_ratio: bool = False,
        segmentation_maps: Optional[ImageInput] = None,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        processed_images = self._preprocess_images(
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
            return_tensors=return_tensors,
            size_divisor=size_divisor,
            do_pad=do_pad,
            ensure_multiple_of=ensure_multiple_of,
            keep_aspect_ratio=keep_aspect_ratio,
            **kwargs,
        )

        data = {"pixel_values": processed_images}

        if segmentation_maps is not None:
            isList = False
            added_dimension = False
            if isinstance(segmentation_maps, list):
                isList = True
                # Batched input as a list of PIL images, no added dimension is needed.
                if not is_pil_image(segmentation_maps[0]) and segmentation_maps[0].ndim == 2:
                    segmentation_maps = [map.unsqueeze(0) for map in segmentation_maps]
                    added_dimension = True
                elif is_pil_image(segmentation_maps[0]):
                    added_dimension = True
            elif is_pil_image(segmentation_maps):
                added_dimension = True
            elif not is_pil_image(segmentation_maps) and segmentation_maps.ndim == 2:
                segmentation_maps = segmentation_maps.unsqueeze(0)
                added_dimension = True

            segmentation_maps = self._prepare_input_images(segmentation_maps)
            processed_maps = self._preprocess_images(
                images=segmentation_maps,
                do_resize=do_resize,
                size=size,
                interpolation=interpolation,
                do_center_crop=do_center_crop,
                crop_size=crop_size,
                do_rescale=False,
                rescale_factor=rescale_factor,
                do_normalize=False,
                image_mean=image_mean,
                image_std=image_std,
                return_tensors=return_tensors,
                size_divisor=size_divisor,
                do_pad=do_pad,
                ensure_multiple_of=ensure_multiple_of,
                keep_aspect_ratio=keep_aspect_ratio,
            )
            if added_dimension and isList:
                processed_maps = processed_maps.squeeze(1).long()
            elif added_dimension:
                processed_maps = processed_maps.squeeze(0).long()
            data["labels"] = processed_maps

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _preprocess_images(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        do_center_crop: bool,
        do_pad: bool,
        return_tensors: bool,
        do_rescale: bool,
        rescale_factor: float,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        ensure_multiple_of: Optional[int],
        keep_aspect_ratio: bool,
        crop_size: SizeDict,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        size_divisor: Optional[int],
        **kwargs,
    ) -> "torch.Tensor":
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images,
                    size=size,
                    interpolation=interpolation,
                    ensure_multiple_of=ensure_multiple_of,
                    keep_aspect_ratio=keep_aspect_ratio,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # TODO: use batched tensor method, eg pad_images, or a version of pad_image that can detect if images are batched
            if do_pad:
                stacked_images = torch.stack([self.pad_image(image, size_divisor) for image in stacked_images])
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return processed_images

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

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        ensure_multiple_of: Optional[int] = None,
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
            antialias (`bool`, *optional*, defaults to `True`):
                Whether to use antialiasing when resizing the image
            ensure_multiple_of (`int`, *optional*):
                If `do_resize` is `True`, the image is resized to a size that is a multiple of this value

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
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
        elif ensure_multiple_of > 1:
            new_size = get_output_image_size_for_ensure_multiple(
                image, (size.height, size.width), keep_aspect_ratio=keep_aspect_ratio, multiple=ensure_multiple_of
            )
        elif size.height and size.width:
            new_size = (size.height, size.width)

        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)


__all__ = ["DPTImageProcessorFast"]
