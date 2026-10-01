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
"""Image processor class for Donut."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


# Adapted from transformers.models.donut.image_processing_donut.DonutImageProcessorKwargs
class DonutImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_thumbnail (`bool`, *optional*, defaults to `self.do_thumbnail`):
        Whether to resize the image using thumbnail method.
    do_align_long_axis (`bool`, *optional*, defaults to `self.do_align_long_axis`):
        Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
    """

    do_thumbnail: bool
    do_align_long_axis: bool


@auto_docstring
class DonutImageProcessorPil(PilBackend):
    """PIL backend for Donut with align_long_axis, thumbnail, and pad_image."""

    valid_kwargs = DonutImageProcessorKwargs

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 2560, "width": 1920}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_thumbnail = True
    do_align_long_axis = False
    do_pad = True

    def __init__(self, **kwargs: Unpack[DonutImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        if isinstance(size, (tuple, list)):
            size = size[::-1]
        if size is not None:
            kwargs["size"] = size
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[DonutImageProcessorKwargs],
    ) -> BatchFeature:
        kwargs = dict(kwargs)
        if "size" in kwargs:
            size = kwargs["size"]
            if isinstance(size, (tuple, list)):
                kwargs["size"] = size[::-1]
        return super().preprocess(images, **kwargs)

    def align_long_axis(
        self,
        image: np.ndarray,
        size: SizeDict,
    ) -> np.ndarray:
        """Align the long axis of the image to the longest axis of the specified size."""
        from ...image_utils import ChannelDimension

        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        output_height, output_width = size.height, size.width

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3, axes=(1, 2))

        return image

    def pad_image(
        self,
        image: np.ndarray,
        size: SizeDict,
        random_padding: bool = False,
    ) -> np.ndarray:
        """Pad the image to the specified size."""
        from ...image_transforms import PaddingMode
        from ...image_transforms import pad as np_pad
        from ...image_utils import ChannelDimension

        output_height, output_width = size.height, size.width
        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        if random_padding:
            pad_top = int(np.random.randint(low=0, high=delta_height + 1))
            pad_left = int(np.random.randint(low=0, high=delta_width + 1))
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        # pad() expects (height_pad, width_pad) and adds channel dimension
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return np_pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=0,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def thumbnail(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | None" = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize the image to make a thumbnail."""
        from ...image_utils import ChannelDimension

        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        output_height, output_width = size.height, size.width

        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return image

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        return self.resize(
            image,
            size=SizeDict(width=width, height=height),
            resample=resample or PILImageResampling.BICUBIC,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        return_tensors: str | TensorType | None,
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for Donut."""
        processed_images = []
        for image in images:
            if do_align_long_axis:
                image = self.align_long_axis(image, size)
            if do_resize:
                shortest_edge = min(size.height, size.width)
                image = self.resize(image, SizeDict(shortest_edge=shortest_edge), resample)
            if do_thumbnail:
                image = self.thumbnail(image, size, resample)
            if do_pad:
                image = self.pad_image(image, size, random_padding=False)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["DonutImageProcessorPil"]
