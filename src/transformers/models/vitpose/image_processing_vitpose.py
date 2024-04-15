# coding=utf-8
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
"""Feature extractor class for ViTPose."""

import math
from typing import Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ImageFeatureExtractionMixin,
    ImageInput,
    is_torch_tensor,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """
    Source: https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/post_processing/post_transforms.py

    Calculate the transformation matrix under the constraint of unbiased. Paper ref: Huang et al. The Devil is in the
    Details: Delving into Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * math.cos(theta) + 0.5 * size_input[1] * math.sin(theta) + 0.5 * size_target[0]
    )
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * math.sin(theta) - 0.5 * size_input[1] * math.cos(theta) + 0.5 * size_target[1]
    )
    return matrix


class ViTPoseImageProcessor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a ViTPose image processor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_affine_transform (`bool`, *optional*, defaults to `True`):
            Whether to apply an affine transformation to the input images.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self, do_affine_transform=True, do_rescale=True, do_normalize=True, image_mean=None, image_std=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.do_affine_transform = do_affine_transform
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def affine_transform(self, image):
        raise NotImplementedError("To do")

        # transformation = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)

        # image = image.transform(transformation, Image.AFFINE, resample=Image.BILINEAR)

        # return image

    def __call__(
        self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
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
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (affine transformation + rescaling + normalization)
        if self.do_affine_transform:
            images = [self.affine_transform(image) for image in images]
        if self.do_rescale:
            images = [self.to_numpy_array(image=image) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
