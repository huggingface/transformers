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

import warnings
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

import numpy as np

from transformers.image_utils import PILImageResampling
from transformers.utils.import_utils import is_flax_available, is_tf_available, is_torch_available, is_vision_available


if is_vision_available():
    import PIL

    from .image_utils import (
        ChannelDimension,
        get_channel_dimension_axis,
        get_image_size,
        infer_channel_dimension_format,
        is_jax_tensor,
        is_tf_tensor,
        is_torch_tensor,
        to_numpy_array,
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
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.

    Returns:
        `np.ndarray`: The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    current_channel_dim = infer_channel_dimension_format(image)
    target_channel_dim = ChannelDimension(channel_dim)
    if current_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image


def rescale(
    image: np.ndarray, scale: float, data_format: Optional[ChannelDimension] = None, dtype=np.float32
) -> np.ndarray:
    """
    Rescales `image` by `scale`.

    Args:
        image (`np.ndarray`):
            The image to rescale.
        scale (`float`):
            The scale to use for rescaling the image.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the image. If not provided, it will be the same as the input image.
        dtype (`np.dtype`, *optional*, defaults to `np.float32`):
            The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature
            extractors.

    Returns:
        `np.ndarray`: The rescaled image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    rescaled_image = image * scale
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format)
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def to_pil_image(
    image: Union[np.ndarray, PIL.Image.Image, "torch.Tensor", "tf.Tensor", "jnp.Tensor"],
    do_rescale: Optional[bool] = None,
) -> PIL.Image.Image:
    """
    Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
    needed.

    Args:
        image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor` or `tf.Tensor`):
            The image to convert to the `PIL.Image` format.
        do_rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default
            to `True` if the image type is a floating type, `False` otherwise.

    Returns:
        `PIL.Image.Image`: The converted image.
    """
    if isinstance(image, PIL.Image.Image):
        return image

    # Convert all tensors to numpy arrays before converting to PIL image
    if is_torch_tensor(image) or is_tf_tensor(image):
        image = image.numpy()
    elif is_jax_tensor(image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError("Input image type not supported: {}".format(type(image)))

    # If the channel as been moved to first dim, we put it back at the end.
    image = to_channel_dimension_format(image, ChannelDimension.LAST)

    # PIL.Image can only store uint8 values, so we rescale the image to be between 0 and 255 if needed.
    do_rescale = isinstance(image.flat[0], float) if do_rescale is None else do_rescale
    if do_rescale:
        image = rescale(image, 255)
    image = image.astype(np.uint8)
    return PIL.Image.fromarray(image)


def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    default_to_square: bool = True,
    max_size: Optional[int] = None,
) -> tuple:
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

    Returns:
        `tuple`: The target (height, width) dimension of the output image after resizing.
    """
    if isinstance(size, (tuple, list)):
        if len(size) == 2:
            return tuple(size)
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

    return (new_long, new_short) if width <= height else (new_short, new_long)


def resize(
    image,
    size: Tuple[int, int],
    resample=PILImageResampling.BILINEAR,
    data_format: Optional[ChannelDimension] = None,
    return_numpy: bool = True,
) -> np.ndarray:
    """
    Resizes `image` to (h, w) specified by `size` using the PIL library.

    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to resize.
        size (`Tuple[int, int]`):
            The size to use for resizing the image.
        resample (`int`, *optional*, defaults to `PIL.Image.Resampling.BILINEAR`):
            The filter to user for resampling.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If `None`, will use the inferred format from the input.
        return_numpy (`bool`, *optional*, defaults to `True`):
            Whether or not to return the resized image as a numpy array. If False a `PIL.Image.Image` object is
            returned.

    Returns:
        `np.ndarray`: The resized image.
    """
    if not len(size) == 2:
        raise ValueError("size must have 2 elements")

    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    data_format = infer_channel_dimension_format(image) if data_format is None else data_format

    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    if not isinstance(image, PIL.Image.Image):
        # PIL expects image to have channels last
        image = to_channel_dimension_format(image, ChannelDimension.LAST)
        image = to_pil_image(image)
    height, width = size
    # PIL images are in the format (width, height)
    resized_image = image.resize((width, height), resample=resample)

    if return_numpy:
        resized_image = np.array(resized_image)
        resized_image = to_channel_dimension_format(resized_image, data_format)
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[ChannelDimension] = None,
) -> np.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`np.ndarray`):
            The image to normalize.
        mean (`float` or `Iterable[float]`):
            The mean to use for normalization.
        std (`float` or `Iterable[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If `None`, will use the inferred format from the input.
    """
    if isinstance(image, PIL.Image.Image):
        warnings.warn(
            "PIL.Image.Image inputs are deprecated and will be removed in v4.26.0. Please use numpy arrays instead.",
            FutureWarning,
        )
        # Convert PIL image to numpy array with the same logic as in the previous feature extractor normalize -
        # casting to numpy array and dividing by 255.
        image = to_numpy_array(image)
        image = rescale(image, scale=1 / 255)

    input_data_format = infer_channel_dimension_format(image)
    channel_axis = get_channel_dimension_axis(image)
    num_channels = image.shape[channel_axis]

    if isinstance(mean, Iterable):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels

    if isinstance(std, Iterable):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels

    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    image = to_channel_dimension_format(image, data_format) if data_format is not None else image
    return image


def center_crop(
    image: np.ndarray,
    size: Tuple[int, int],
    data_format: Optional[Union[str, ChannelDimension]] = None,
    return_numpy: Optional[bool] = None,
) -> np.ndarray:
    """
    Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to
    the size given, it will be padded (so the returned result will always be of size `size`).

    Args:
        image (`np.ndarray`):
            The image to crop.
        size (`Tuple[int, int]`):
            The target size for the cropped image.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
        return_numpy (`bool`, *optional*):
            Whether or not to return the cropped image as a numpy array. Used for backwards compatibility with the
            previous ImageFeatureExtractionMixin method.
                - Unset: will return the same type as the input image.
                - `True`: will return a numpy array.
                - `False`: will return a `PIL.Image.Image` object.
    Returns:
        `np.ndarray`: The cropped image.
    """
    if isinstance(image, PIL.Image.Image):
        warnings.warn(
            "PIL.Image.Image inputs are deprecated and will be removed in v4.26.0. Please use numpy arrays instead.",
            FutureWarning,
        )
        image = to_numpy_array(image)
        return_numpy = False if return_numpy is None else return_numpy
    else:
        return_numpy = True if return_numpy is None else return_numpy

    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    if not isinstance(size, Iterable) or len(size) != 2:
        raise ValueError("size must have 2 elements representing the height and width of the output image")

    input_data_format = infer_channel_dimension_format(image)
    output_data_format = data_format if data_format is not None else input_data_format

    # We perform the crop in (C, H, W) format and then convert to the output format
    image = to_channel_dimension_format(image, ChannelDimension.FIRST)

    orig_height, orig_width = get_image_size(image)
    crop_height, crop_width = size

    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        image = image[..., top:bottom, left:right]
        image = to_channel_dimension_format(image, output_data_format)
        return image

    # Otherwise, we may need to pad if the image is too small. Oh joy...
    new_height = max(crop_height, orig_height)
    new_width = max(crop_width, orig_width)
    new_shape = image.shape[:-2] + (new_height, new_width)
    new_image = np.zeros_like(image, shape=new_shape)

    # If the image is too small, pad it with zeros
    top_pad = (new_height - orig_height) // 2
    bottom_pad = top_pad + orig_height
    left_pad = (new_width - orig_width) // 2
    right_pad = left_pad + orig_width
    new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

    top += top_pad
    bottom += top_pad
    left += left_pad
    right += left_pad

    new_image = new_image[..., max(0, top) : min(new_height, bottom), max(0, left) : min(new_width, right)]
    new_image = to_channel_dimension_format(new_image, output_data_format)

    if not return_numpy:
        new_image = to_pil_image(new_image)

    return new_image
