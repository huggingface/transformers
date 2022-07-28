# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import PIL

from transformers.utils.import_utils import is_flax_available, is_tf_available, is_torch_available

from .image_utils import (
    ChannelDimension,
    get_image_size,
    infer_channel_dimension_format,
    is_jax_tensor,
    is_tf_tensor,
    is_torch_tensor,
)


if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf
    if is_flax_available():
        import jax.numpy as jnp


def to_channel_dimension_format(image: np.ndarray, channel_dim: Union[ChannelDimension, str]) -> np.ndarray:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`numpy.ndarray`):
            The image to convert to the PIL Image format.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.

    Returns:
        image: A converted np.ndarray.
    """
    current_channel_dim = infer_channel_dimension_format(image)
    target_channel_dim = ChannelDimension(channel_dim)
    if current_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        return image.transpose((2, 0, 1))

    if target_channel_dim == ChannelDimension.LAST:
        return image.transpose((1, 2, 0))

    raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))


def to_pil_image(
    image: Union[np.ndarray, PIL.Image.Image, "torch.Tensor", "tf.Tensor"], rescale=None
) -> PIL.Image.Image:
    """
    Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
    needed.

    Args:
        image (`PIL.Image.Image`, `numpy.ndarray`, `torch.Tensor`, `tf.Tensor`):
            The image to convert to the PIL Image format.
        rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default
            to `True` if the image type is a floating type, `False` otherwise.
    """
    if isinstance(image, PIL.Image.Image):
        return image

    if is_torch_tensor(image) or is_tf_tensor(image):
        image = image.numpy()
    elif is_jax_tensor(image):
        image = np.array(image)

    if not isinstance(image, np.ndarray):
        raise ValueError("Input image type not supported: {}".format(type(image)))

    # If the channel as been moved to first dim, we put it back at the end.
    image = to_channel_dimension_format(image, ChannelDimension.LAST)

    # PIL.Image can only store uint8 values, so we rescale the image to be between 0 and 255 if needed.
    rescale = isinstance(image.flat[0], float) if rescale is None else rescale
    if rescale:
        rescale = image * 255
    image = image.astype(np.uint8)
    return PIL.Image.fromarray(image)


def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    default_to_square: bool = True,
    max_size: int = None,
) -> np.ndarray:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]` or List[int] or Tuple[int]):
            The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be matched to
            this.

            If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
            `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to this
            number. i.e, if height > width, then image will be rescaled to (size * height / width, size).
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            The filter to user for resampling.
        default_to_square (`bool`, *optional*, defaults to `True`):
            How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a square
            (`size`,`size`). If set to `False`, will replicate
            [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
            with support for resizing only the smallest edge and providing an optional `max_size`.
        max_size (`int`, *optional*, defaults to `None`):
            The maximum allowed for the longer edge of the resized image: if the longer edge of the image is greater
            than `max_size` after being resized according to `size`, then the image is resized again so that the longer
            edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller edge may be shorter
            than `size`. Only used if `default_to_square` is `False`.
    """
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            return size
        elif len(size) == 1:
            # Perform same logic as if size was an int
            size = size[0]
        else:
            raise ValueError("size must have 1 or 2 elements if it is a list or tuple")

    if default_to_square:
        return (size, size)

    height, width = get_image_size(input_image)
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size

    if short == requested_new_short:
        return (height, width)

    new_short, new_long = requested_new_short, int(requested_new_short * long / short)

    if max_size is not None:
        if max_size <= requested_new_short:
            raise ValueError(
                f"max_size = {max_size} must be strictly greater than the requested "
                f"size for the smaller edge size = {size}"
            )
        if new_long > max_size:
            new_short, new_long = int(max_size * new_short / new_long), max_size

    return (new_short, new_long) if width <= height else (new_long, new_short)


def resize(
    image,
    size: Tuple[int, int],
    resample=PIL.Image.Resampling.BILINEAR,
    data_format: Optional[ChannelDimension] = None,
) -> np.np.ndarray:
    """
    Resizes `image` to (h, w) specified by `size` using the PIL library.

    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to resize.
        size (`Tuple[int, int]`):
            The size to use for resizing the image.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            The filter to user for resampling.
        data_format (`ChannelDimension`, *optional*, defaults to `None`):
            The channel dimension format of the output image. If `None`, will use the inferred format from the input.

    Returns:
        image: A resized np.ndarray.
    """
    if not len(size) == 2:
        raise ValueError("size must have 2 elements")

    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    data_format = infer_channel_dimension_format(image) if data_format is None else data_format

    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    if not isinstance(image, PIL.Image.Image):
        image = to_pil_image(image)
    # PIL images are in the format (width, height)
    h, w = size
    resized_image = image.resize((w, h), resample=resample)
    resized_image = np.array(resized_image)
    resized_image = to_channel_dimension_format(resized_image, data_format)
    return resized_image
