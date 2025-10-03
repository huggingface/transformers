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
"""Fast Image processor class for Nougat."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_transforms import (
    get_resize_output_image_size,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)


class NougatFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
    do_crop_margin (`bool`, *optional*, defaults to `True`):
            Whether to crop the image margins.
    do_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the image using thumbnail method.
    do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
    """

    do_crop_margin: Optional[bool]
    do_thumbnail: Optional[bool]
    do_align_long_axis: Optional[bool]


@auto_docstring
class NougatImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 896, "width": 672}
    do_resize: bool = (True,)
    do_normalize: bool = True
    do_thumbnail: bool = True
    do_align_long_axis: bool = False
    do_pad: bool = True
    do_rescale = True
    do_crop_margin: bool = True
    valid_kwargs = NougatFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[NougatFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[NougatFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def python_find_non_zero(
        self,
        image: "torch.Tensor",
    ):
        """This is a reimplementation of a findNonZero function equivalent to cv2."""

        non_zero_indices = torch.nonzero(image, as_tuple=False)
        idxvec = non_zero_indices[:, [2, 1]]
        idxvec = idxvec.reshape(-1, 1, 2)
        return idxvec

    def python_bounding_rect(self, coordinates):
        """This is a reimplementation of a BoundingRect function equivalent to cv2."""

        min_values = torch.amin(coordinates, axis=(0, 1)).to(torch.int)
        max_values = torch.amax(coordinates, axis=(0, 1)).to(torch.int)

        x_min, y_min = min_values[0], min_values[1]
        width = max_values[0] - x_min + 1
        height = max_values[1] - y_min + 1
        return x_min, y_min, width, height

    def crop_margin(
        self,
        image: "torch.Tensor",
        gray_threshold: int = 200,
    ) -> "torch.Tensor":
        """
        Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the
        threshold).

        Args:
            image (`torch.Tensor`):
                The image to be cropped.
            gray_threshold (`int`, *optional*, defaults to `200`)
                Value below which pixels are considered to be gray.
        """
        data = F.rgb_to_grayscale(image, num_output_channels=1)

        max_val = torch.max(data)
        min_val = torch.min(data)

        if max_val == min_val:
            return image
        data = (data - min_val) / (max_val - min_val) * 255
        gray = data < gray_threshold
        coords = self.python_find_non_zero(gray)
        x_min, y_min, width, height = self.python_bounding_rect(coords)
        image = image[:, y_min : y_min + height, x_min : x_min + width]

        return image

    def align_long_axis(
        self,
        image: "torch.Tensor",
        size: SizeDict,
    ) -> "torch.Tensor":
        """
        Align the long axis of the image to the longest axis of the specified size.

        Args:
            image (`torch.Tensor`):
                The image to be aligned.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to align the long axis to.
        Returns:
            `torch.Tensor`: The aligned image.
        """
        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = torch.rot90(image, 3, dims=[1, 2])

        return image

    def thumbnail(
        self,
        image: "torch.Tensor",
        size: SizeDict,
    ) -> "torch.Tensor":
        """
        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
        corresponding dimension of the specified size.

        Args:
            image (`torch.tensor`):
                The image to be resized.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to resize the image to.
        """

        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        # We always resize to the smallest of either the input or output size.
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return image

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        new_size = (height, width)

        return F.resize(image, new_size, interpolation=F.InterpolationMode.BICUBIC)

    def pad_images(
        self,
        image: "torch.Tensor",
        size: SizeDict,
    ) -> "torch.Tensor":
        """
        Pads a batch of images to the specified size at the top, bottom, left and right.

        Args:
            image (`torch.tensor`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
        """
        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        pad_top = delta_height // 2
        pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return F.pad(image, padding)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"] = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BICUBIC`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BICUBIC

        shortest_edge = min(size["height"], size["width"])

        new_size = get_resize_output_image_size(
            image, size=shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
        )
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        do_align_long_axis: bool,
        do_thumbnail: bool,
        do_pad: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_crop_margin: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: bool,
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Crop images
        images = [self.crop_margin(image) for image in images]

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_align_long_axis:
                stacked_images = self.align_long_axis(image=stacked_images, size=size)
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size)
            if do_thumbnail:
                stacked_images = self.thumbnail(image=stacked_images, size=size)
            if do_pad:
                stacked_images = self.pad_images(image=stacked_images, size=size)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["NougatImageProcessorFast"]
