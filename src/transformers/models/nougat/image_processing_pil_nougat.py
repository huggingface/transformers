# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Nougat."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import (
    get_resize_output_image_size,
    pad,
    to_channel_dimension_format,
    to_pil_image,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)
from ...utils.import_utils import requires
from .image_processing_nougat import NougatImageProcessorKwargs


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class NougatImageProcessorPil(PilBackend):
    valid_kwargs = NougatImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 896, "width": 672}
    do_resize = True
    do_normalize = True
    do_thumbnail = True
    do_align_long_axis = False
    do_pad = True
    do_rescale = True
    do_crop_margin = True

    def __init__(self, **kwargs: Unpack[NougatImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[NougatImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def python_find_non_zero(self, image: np.ndarray):
        """This is a reimplementation of a findNonZero function equivalent to cv2."""
        non_zero_indices = np.column_stack(np.nonzero(image))
        idxvec = non_zero_indices[:, [1, 0]]
        idxvec = idxvec.reshape(-1, 1, 2)
        return idxvec

    def python_bounding_rect(self, coordinates):
        """This is a reimplementation of a BoundingRect function equivalent to cv2."""
        min_values = np.min(coordinates, axis=(0, 1)).astype(int)
        max_values = np.max(coordinates, axis=(0, 1)).astype(int)
        x_min, y_min = min_values[0], min_values[1]
        width = max_values[0] - x_min + 1
        height = max_values[1] - y_min + 1
        return x_min, y_min, width, height

    def crop_margin(
        self,
        image: np.ndarray,
        gray_threshold: int = 200,
    ) -> np.ndarray:
        """
        Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the
        threshold).

        Args:
            image (`np.ndarray`):
                The image to be cropped.
            gray_threshold (`int`, *optional*, defaults to `200`)
                Value below which pixels are considered to be gray.
        """
        image_pil = to_pil_image(image, input_data_format=ChannelDimension.FIRST)
        data = np.array(image_pil.convert("L")).astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return image
        data = (data - min_val) / (max_val - min_val) * 255
        gray = data < gray_threshold
        coords = self.python_find_non_zero(gray)
        x_min, y_min, width, height = self.python_bounding_rect(coords)
        image_pil = image_pil.crop((x_min, y_min, x_min + width, y_min + height))
        image = np.array(image_pil).astype(np.uint8)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=ChannelDimension.LAST)
        return image

    def align_long_axis(
        self,
        image: np.ndarray,
        size: SizeDict,
    ) -> np.ndarray:
        """
        Align the long axis of the image to the longest axis of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be aligned.
            size (`SizeDict`):
                The size to align the long axis to.
        Returns:
            `np.ndarray`: The aligned image.
        """
        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        output_height, output_width = size.height, size.width

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3, axes=(1, 2))

        return image

    def thumbnail(
        self,
        image: np.ndarray,
        size: SizeDict,
    ) -> np.ndarray:
        """
        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
        corresponding dimension of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be resized.
            size (`SizeDict`):
                The size to resize the image to.
        """

        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
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

        # Use np_resize for exact dimensions; self.resize uses shortest-edge logic and would produce
        # different output due to rounding in get_resize_output_image_size.
        return super().resize(
            image, SizeDict(height=height, width=width), resample=PILImageResampling.BICUBIC, reducing_gap=2.0
        )

    def pad_images(
        self,
        image: np.ndarray,
        size: SizeDict,
    ) -> np.ndarray:
        """
        Pads a batch of images to the specified size at the top, bottom, left and right.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`SizeDict`):
                The size to pad the image to.
        """

        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        output_height, output_width = size.height, size.width

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        pad_top = delta_height // 2
        pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return pad(image, padding, input_data_format=ChannelDimension.FIRST, data_format=ChannelDimension.FIRST)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | int | None" = None,
        reducing_gap: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size.height, size.width)`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Size of the output image.
            resample (`PILImageResampling | int`, *optional*):
                Resampling filter to use when resizing the image.
        Returns:
            `np.ndarray`: The resized image.
        """

        shortest_edge = min(size.height, size.width)

        new_size = get_resize_output_image_size(
            image, size=shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
        )
        return super().resize(
            image,
            SizeDict(height=new_size[0], width=new_size[1]),
            resample=resample,
            reducing_gap=reducing_gap,
            **kwargs,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        do_align_long_axis: bool = False,
        do_thumbnail: bool = True,
        do_crop_margin: bool = True,
        do_pad: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image in images:
            if do_crop_margin:
                image = self.crop_margin(image)
            if do_align_long_axis:
                image = self.align_long_axis(image, size)
            if do_resize:
                image = self.resize(image, size, resample)
            if do_thumbnail:
                image = self.thumbnail(image, size)
            if do_pad:
                image = self.pad_images(image, size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["NougatImageProcessorPil"]
