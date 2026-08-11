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
"""Image processor class for Pixtral."""

import math

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import PaddingMode, pad
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


# Adapted from transformers.models.pixtral.image_processing_pixtral.PixtralImageProcessorKwargs
class PixtralImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`Union[dict[str, int], int]` *optional*, defaults to `{"height": 16, "width": 16}`):
        Size of the patches in the model, used to calculate the output image size.
    """

    patch_size: dict[str, int] | int


# Adapted from transformers.models.pixtral.image_processing_pixtral._num_image_tokens
def _num_image_tokens(image_size: tuple[int, int], patch_size: tuple[int, int]) -> int:
    """
    Calculate the number of image tokens given the image size and patch size.

    Args:
        image_size (`tuple[int, int]`):
            The size of the image as `(height, width)`.
        patch_size (`tuple[int, int]`):
            The patch size as `(height, width)`.

    Returns:
        `int`: The number of image tokens.
    """
    height, width = image_size
    patch_height, patch_width = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
    num_width_tokens = (width - 1) // patch_width + 1
    num_height_tokens = (height - 1) // patch_height + 1
    return num_height_tokens, num_width_tokens


# Adapted from transformers.models.pixtral.image_processing_pixtral.get_resize_output_image_size
def get_resize_output_image_size(
    input_image: ImageInput,
    size: int | tuple[int, int] | list[int] | tuple[int],
    patch_size: int | tuple[int, int] | list[int] | tuple[int],
    input_data_format: str | ChannelDimension | None = None,
) -> tuple:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.

    Args:
        input_image (`ImageInput`):
            The image to resize.
        size (`int` or `tuple[int, int]`):
            Max image size an input image can be. Must be a dictionary with the key "longest_edge".
        patch_size (`int` or `tuple[int, int]`):
            The patch_size as `(height, width)` to use for resizing the image. If patch_size is an integer, `(patch_size, patch_size)`
            will be used
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `tuple`: The target (height, width) dimension of the output image after resizing.
    """
    max_height, max_width = size if isinstance(size, (tuple, list)) else (size, size)
    patch_height, patch_width = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
    height, width = get_image_size(input_image, input_data_format)

    ratio = max(height / max_height, width / max_width)

    if ratio > 1:
        # Original implementation uses `round` which utilises bankers rounding, which can lead to surprising results
        # Here we use floor to ensure the image is always smaller than the given "longest_edge"
        height = int(math.floor(height / ratio))
        width = int(math.floor(width / ratio))

    num_height_tokens, num_width_tokens = _num_image_tokens((height, width), (patch_height, patch_width))
    return num_height_tokens * patch_height, num_width_tokens * patch_width


@auto_docstring
class PixtralImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    patch_size = {"height": 16, "width": 16}
    size = {"longest_edge": 1024}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = PixtralImageProcessorKwargs

    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(self, **kwargs: Unpack[PixtralImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[PixtralImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        patch_size: SizeDict,
        resample: "PILImageResampling | None" = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The longest edge is resized to size["longest_edge"], with aspect ratio preserved.
        Output dimensions are aligned to patch_size.
        """
        if size.longest_edge:
            size_tuple = (size.longest_edge, size.longest_edge)
        elif size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            raise ValueError("size must contain either 'longest_edge' or 'height' and 'width'.")

        if patch_size.height and patch_size.width:
            patch_size_tuple = (patch_size.height, patch_size.width)
        else:
            raise ValueError("patch_size must contain 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image, size=size_tuple, patch_size=patch_size_tuple, input_data_format=ChannelDimension.FIRST
        )
        return super().resize(
            image, size=SizeDict(height=output_size[0], width=output_size[1]), resample=resample, **kwargs
        )

    def _pad_for_batching(
        self,
        pixel_values: list[np.ndarray],
        image_sizes: list[tuple[int, int]],
    ) -> np.ndarray:
        """Pad images to form a batch of same shape."""
        max_shape = (max(s[0] for s in image_sizes), max(s[1] for s in image_sizes))
        padded = []
        for img, size in zip(pixel_values, image_sizes):
            pad_h = max_shape[0] - size[0]
            pad_w = max_shape[1] - size[1]
            padded_img = pad(
                img,
                padding=((0, pad_h), (0, pad_w)),
                mode=PaddingMode.CONSTANT,
                constant_values=0,
                input_data_format=ChannelDimension.FIRST,
            )
            padded.append(padded_img)
        return np.stack(padded)

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
        return_tensors: str | TensorType | None,
        patch_size: dict[str, int] | SizeDict | None = None,
        **kwargs,
    ) -> BatchFeature:
        patch_size = get_size_dict(patch_size or self.patch_size, default_to_square=True)
        patch_size_sd = SizeDict(**patch_size)

        processed_images = []
        batch_image_sizes = []

        for image in images:
            if do_resize:
                image = self.resize(image, size=size, patch_size=patch_size_sd, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            processed_images.append(image)
            batch_image_sizes.append(get_image_size(image, channel_dim=ChannelDimension.FIRST))

        padded_images = self._pad_for_batching(
            pixel_values=processed_images,
            image_sizes=batch_image_sizes,
        )

        return BatchFeature(
            data={"pixel_values": padded_images, "image_sizes": batch_image_sizes}, tensor_type=return_tensors
        )


__all__ = ["PixtralImageProcessorPil"]
