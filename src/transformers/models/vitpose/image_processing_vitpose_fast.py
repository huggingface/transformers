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
"""Fast Image processor class for VitPose."""
from typing import Optional, Union
from ...image_processing_utils_fast import BaseImageProcessorFast, BatchFeature
from ...image_transforms import to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import auto_docstring, TensorType, is_scipy_available, is_torch_available, is_vision_available, logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import math

if is_torch_available():
    import torch

if is_vision_available():
    import PIL

if is_scipy_available():
    from scipy.linalg import inv
    from scipy.ndimage import affine_transform, gaussian_filter

logger = logging.get_logger(__name__)

def box_to_center_and_scale(
    box: Union[Tuple, List, np.ndarray],
    image_width: int,
    image_height: int,
    normalize_factor: float = 200.0,
    padding_factor: float = 1.25,
):
    """
    Encodes a bounding box in COCO format into (center, scale).

    Args:
        box (`Tuple`, `List`, or `np.ndarray`):
            Bounding box in COCO format (top_left_x, top_left_y, width, height).
        image_width (`int`):
            Image width.
        image_height (`int`):
            Image height.
        normalize_factor (`float`):
            Width and height scale factor.
        padding_factor (`float`):
            Bounding box padding factor.

    Returns:
        tuple: A tuple containing center and scale.

        - `np.ndarray` [float32](2,): Center of the bbox (x, y).
        - `np.ndarray` [float32](2,): Scale of the bbox width & height.
    """

    top_left_x, top_left_y, width, height = box[:4]
    aspect_ratio = image_width / image_height
    center = np.array([top_left_x + width * 0.5, top_left_y + height * 0.5], dtype=np.float32)

    if width > aspect_ratio * height:
        height = width * 1.0 / aspect_ratio
    elif width < aspect_ratio * height:
        width = height * aspect_ratio

    scale = np.array([width / normalize_factor, height / normalize_factor], dtype=np.float32)
    scale = scale * padding_factor

    return center, scale

def get_warp_matrix(theta: float, size_input: np.ndarray, size_dst: np.ndarray, size_target: np.ndarray):
    """
    Calculate the transformation matrix under the constraint of unbiased. Paper ref: Huang et al. The Devil is in the
    Details: Delving into Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Source: https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/post_processing/post_transforms.py

    Args:
        theta (`float`):
            Rotation angle in degrees.
        size_input (`np.ndarray`):
            Size of input image [width, height].
        size_dst (`np.ndarray`):
            Size of output image [width, height].
        size_target (`np.ndarray`):
            Size of ROI in input plane [w, h].

    Returns:
        `np.ndarray`: A matrix for transformation.
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


def scipy_warp_affine(src, M, size):
    """
    This function implements cv2.warpAffine function using affine_transform in scipy. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html and https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html for more details.

    Note: the original implementation of cv2.warpAffine uses cv2.INTER_LINEAR.
    """
    channels = [src[..., i] for i in range(src.shape[-1])]

    # Convert to a 3x3 matrix used by SciPy
    M_scipy = np.vstack([M, [0, 0, 1]])
    # If you have a matrix for the ‘push’ transformation, use its inverse (numpy.linalg.inv) in this function.
    M_inv = inv(M_scipy)
    M_inv[0, 0], M_inv[0, 1], M_inv[1, 0], M_inv[1, 1], M_inv[0, 2], M_inv[1, 2] = (
        M_inv[1, 1],
        M_inv[1, 0],
        M_inv[0, 1],
        M_inv[0, 0],
        M_inv[1, 2],
        M_inv[0, 2],
    )

    new_src = [affine_transform(channel, M_inv, output_shape=size, order=1) for channel in channels]
    new_src = np.stack(new_src, axis=-1)
    return new_src

@auto_docstring
class VitPoseImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 256, "width": 192}
    crop_size = None
    do_affine_transform = True
    do_rescale = True
    do_normalize = True


    def affine_transform(
        self,
        image: np.array,
        center: Tuple[float],
        scale: Tuple[float],
        rotation: float,
        size: Dict[str, int],
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.array:
        """
        Apply an affine transformation to an image.

        Args:
            image (`np.array`):
                Image to transform.
            center (`Tuple[float]`):
                Center of the bounding box (x, y).
            scale (`Tuple[float]`):
                Scale of the bounding box with respect to height/width.
            rotation (`float`):
                Rotation angle in degrees.
            size (`Dict[str, int]`):
                Size of the destination image.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format of the output image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image.
        """

        data_format = input_data_format if data_format is None else data_format

        size = (size["width"], size["height"])

        # one uses a pixel standard deviation of 200 pixels
        transformation = get_warp_matrix(rotation, center * 2.0, np.array(size) - 1.0, scale * 200.0)

        # input image requires channels last format
        image = (
            image
            if input_data_format == ChannelDimension.LAST
            else to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)
        )
        image = scipy_warp_affine(src=image, M=transformation, size=(size[1], size[0]))

        image = to_channel_dimension_format(image, data_format, ChannelDimension.LAST)

        return image

    def _preprocess(
        self,
        images: ImageInput,
        boxes: Union[List[List[float]], np.ndarray],
        do_affine_transform: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.

            boxes (`List[List[List[float]]]` or `np.ndarray`):
                List or array of bounding boxes for each image. Each box should be a list of 4 floats representing the bounding
                box coordinates in COCO format (top_left_x, top_left_y, width, height).

            do_affine_transform (`bool`, *optional*, defaults to `self.do_affine_transform`):
                Whether to apply an affine transformation to the input images.
            size (`Dict[str, int]` *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
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
        do_affine_transform = do_affine_transform if do_affine_transform is not None else self.do_affine_transform
        size = size if size is not None else self.size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if isinstance(boxes, list) and len(images) != len(boxes):
            raise ValueError(f"Batch of images and boxes mismatch : {len(images)} != {len(boxes)}")
        elif isinstance(boxes, np.ndarray) and len(images) != boxes.shape[0]:
            raise ValueError(f"Batch of images and boxes mismatch : {len(images)} != {boxes.shape[0]}")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        # transformations (affine transformation + rescaling + normalization)
        if self.do_affine_transform:
            new_images = []
            for image, image_boxes in zip(images, boxes):
                for box in image_boxes:
                    center, scale = box_to_center_and_scale(
                        box,
                        image_width=size["width"],
                        image_height=size["height"],
                        normalize_factor=self.normalize_factor,
                    )
                    transformed_image = self.affine_transform(
                        image, center, scale, rotation=0, size=size, input_data_format=input_data_format
                    )
                    new_images.append(transformed_image)
            images = new_images

        # For batch processing, the number of boxes must be consistent across all images in the batch.
        # When using a list input, the number of boxes can vary dynamically per image.
        # The image processor creates pixel_values of shape (batch_size*num_persons, num_channels, height, width)

        all_images = []
        for image in images:
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            all_images.append(image)
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            for image in all_images
        ]

        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
    

__all__ = ["VitPoseImageProcessorFast"]
