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
"""Image processor class for Swin2SR."""

from typing import List, Optional, Union

import numpy as np
import PIL.Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ChannelDimension, is_batched, to_numpy_array, valid_images
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class Swin2SRImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Swin2SR image processor.

    Args:
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int` or `Tuple(int)`, *optional*, defaults to 224):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if `do_resize` is
            set to `True`.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels of the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input by subtracting a mean and multiplying by range.
    """

    model_input_names = ["pixel_values"]

    def __init__(self, do_pad=True, size=8, num_channels=3, do_normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.do_pad = do_pad
        self.size = size
        self.num_channels = num_channels
        self.do_normalize = do_normalize
        self.mean = (0.4488, 0.4371, 0.4040) if num_channels == 3 else (0.0, 0.0, 0.0)
        self.range = 1.0 if num_channels == 3 else 255.0

    def pad(image: np.ndarray) -> np.ndarray:
        return -1

    def normalize(self, image: np.ndarray) -> np.ndarray:
        image = (image - self.mean) * self.range

        return image

    def preprocess(
        self,
        images: Union["PIL.Image.Image", TensorType, List["PIL.Image.Image"], List[TensorType]],
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
        """
        if not valid_images(images):
            raise ValueError("Invalid image(s)")

        if not is_batched:
            images = [images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(img) for img in images]

        # transformations (padding + normalization)
        if self.do_pad and self.size is not None:
            images = [self.pad(image=image, size=self.size) for image in images]

        if self.do_normalize:
            images = [self.normalize(image=image) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
