# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for GLPN."""

from typing import List, Optional, Union

import numpy as np
import PIL.Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import rescale, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class GLPNImageProcessor(BaseImageProcessor):
    r"""
    Constructs a GLPN image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions, rounding them down to the closest multiple of
            `size_divisor`. Can be overridden by `do_resize` in `preprocess`.
        size_divisor (`int`, *optional*, defaults to 32):
            When `do_resize` is `True`, images are resized so their height and width are rounded down to the closest
            multiple of `size_divisor`. Can be overridden by `size_divisor` in `preprocess`.
        resample (`PIL.Image` resampling filter, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Can be
            overridden by `do_rescale` in `preprocess`.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size_divisor: int = 32,
        resample=PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        **kwargs,
    ) -> None:
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.size_divisor = size_divisor
        self.resample = resample
        super().__init__(**kwargs)

    def resize(
        self, image: np.ndarray, size_divisor: int, resample, data_format: Optional[ChannelDimension] = None, **kwargs
    ) -> np.ndarray:
        """
        Resize the image, rounding the (height, width) dimensions down to the closest multiple of size_divisor.

        If the image is of dimension (3, 260, 170) and size_divisor is 32, the image will be resized to (3, 256, 160).

        Args:
            image (`np.ndarray`):
                The image to resize.
            size_divisor (`int`):
                The image is resized so its height and width are rounded down to the closest multiple of
                `size_divisor`.
            resample:
                `PIL.Image` resampling filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If `None`, the channel dimension format of the input
                image is used. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        height, width = get_image_size(image)
        # Rounds the height and width down to the closest multiple of size_divisor
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        image = resize(image, (new_h, new_w), resample=resample, data_format=data_format, **kwargs)
        return image

    def rescale(
        self, image: np.ndarray, scale: float, data_format: Optional[ChannelDimension] = None, **kwargs
    ) -> np.ndarray:
        """
        Rescale the image by the given scaling factor `scale`.

        Args:
            image (`np.ndarray`):
                The image to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If `None`, the channel dimension format of the input
                image is used. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The rescaled image.
        """
        return rescale(image=image, scale=scale, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: Union["PIL.Image.Image", TensorType, List["PIL.Image.Image"], List[TensorType]],
        do_resize: Optional[bool] = None,
        size_divisor: Optional[int] = None,
        resample=None,
        do_rescale: Optional[bool] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess the given images.

        Args:
            images (`PIL.Image.Image` or `TensorType` or `List[np.ndarray]` or `List[TensorType]`):
                The image or images to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the input such that the (height, width) dimensions are a multiple of `size_divisor`.
            size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
                When `do_resize` is `True`, images are resized so their height and width are rounded down to the
                closest multiple of `size_divisor`.
            resample (`PIL.Image` resampling filter, *optional*, defaults to `self.resample`):
                `PIL.Image` resampling filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - `None`: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample

        if do_resize and size_divisor is None:
            raise ValueError("size_divisor is required for resizing")

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError("Invalid image(s)")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(img) for img in images]

        if do_resize:
            images = [self.resize(image, size_divisor=size_divisor, resample=resample) for image in images]

        if do_rescale:
            images = [self.rescale(image, scale=1 / 255) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
