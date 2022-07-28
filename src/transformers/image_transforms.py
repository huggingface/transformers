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
    infer_channel_dimension,
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


def to_pil_image(
    image: Union[np.ndarray, PIL.Image.Image, "torch.Tensor", "tf.Tensor", "jnp.ndarray"],
    channel_dim: Optional[ChannelDimension] = None,
    rescale=None,
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
    channel_dim = infer_channel_dimension(image) if channel_dim is None else channel_dim
    if channel_dim == ChannelDimension.CHANNEL_FIRST:
        image = image.transpose((1, 2, 0))

    # PIL.Image can only store uint8 values, so we rescale the image to be between 0 and 255 if needed.
    rescale = isinstance(image.flat[0], float) if rescale is None else rescale
    if rescale:
        rescale = image * 255
    image = image.astype(np.uint8)
    return PIL.Image.fromarray(image)


def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int]],
    default_to_square: bool = True,
    max_size: int = None,
) -> np.ndarray:
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


def resize(image, size: Tuple[int, int], resample=PIL.Image.BILINEAR):
    """
    Resizes `image`. Enforces conversion of input to PIL.Image.

    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to resize.
        size (`Tuple[int, int]`):
            The size to use for resizing the image.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            The filter to user for resampling.

    Returns:
        image: A resized np.ndarray.
    """
    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    if not isinstance(image, PIL.Image.Image):
        image = to_pil_image(image)
    resized_image = image.resize(size, resample=resample)
    return resized_image.numpy()
