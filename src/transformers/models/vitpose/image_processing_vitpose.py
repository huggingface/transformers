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
"""Image processor class for ViTPose."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import box_to_center_and_scale, to_channel_dimension_format
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
from ...utils import TensorType, is_scipy_available, is_torch_available, is_vision_available, logging


if is_torch_available():
    import torch

if is_vision_available():
    import PIL

if is_scipy_available():
    from scipy.linalg import inv
    from scipy.ndimage import affine_transform, gaussian_filter

logger = logging.get_logger(__name__)


def coco_to_pascal_voc(bboxes: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from the COCO format to the Pascal VOC format.

    In other words, converts from (top_left_x, top_left_y, width, height) format
    to (top_left_x, top_left_y, bottom_right_x, bottom_right_y).

    Args:
        bboxes (`np.ndarray` of shape `(batch_size, 4)):
            Bounding boxes in COCO format.

    Returns:
        `np.ndarray` of shape `(batch_size, 4) in Pascal VOC format.
    """
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0] - 1
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1] - 1

    return bboxes


def get_keypoint_predictions(heatmaps):
    """Get keypoint predictions from score maps.

    Args:
        heatmaps (`np.ndarray` of shape `(batch_size, num_keypoints, height, width)`):
            Model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - coords (np.ndarray[N, K, 2]):
            Predicted keypoint location.
        - scores (np.ndarray[N, K, 1]):
            Scores (confidence) of the keypoints.
    """
    if not isinstance(heatmaps, np.ndarray):
        raise ValueError("Heatmaps should be np.ndarray")
    if heatmaps.ndim != 4:
        raise ValueError("Heatmaps should be 4-dimensional")

    batch_size, num_keypoints, _, width = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_keypoints, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((batch_size, num_keypoints, 1))
    scores = np.amax(heatmaps_reshaped, 2).reshape((batch_size, num_keypoints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = preds[:, :, 1] // width

    preds = np.where(np.tile(scores, (1, 1, 2)) > 0.0, preds, -1)
    return preds, scores


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp.

    Paper references:
    - Huang et al. The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    - Zhang et al. Distribution-Aware Coordinate Representation for Human Pose Estimation (CVPR 2020).

    Args:
        coords (`np.ndarray` of shape `(num_persons, num_keypoints, 2)`):
            Initial coordinates of human pose.
        batch_heatmaps (`np.ndarray` of shape `(batch_size, num_keypoints, height, width)`):
            Batched heatmaps as predicted by the model.
            A batch_size of 1 is used for the bottom up paradigm where all persons share the same heatmap.
            A batch_size of `num_persons` is used for the top down paradigm where each person has its own heatmaps.
        kernel (`int`, *optional*, defaults to 3):
            Gaussian kernel size (K) for modulation.

    Returns:
        `np.ndarray` of shape `(num_persons, num_keypoints, 2)` ):
            Refined coordinates.
    """
    if not isinstance(coords, np.ndarray):
        coords = coords.cpu().numpy()
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    batch_size, num_keypoints, height, width = batch_heatmaps.shape
    num_coords = coords.shape[0]
    if not (batch_size == 1 or batch_size == num_coords):
        raise ValueError("The batch size of heatmaps should be 1 or equal to the batch size of coordinates.")
    radius = int((kernel - 1) // 2)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            gaussian_filter(heatmap, sigma=0.8, output=heatmap, radius=(radius, radius), axes=(0, 1))
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    # calculate indices for coordinates
    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (width + 2)
    index += (width + 2) * (height + 2) * np.arange(0, batch_size * num_keypoints).reshape(-1, num_keypoints)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + width + 2]
    ix1y1 = batch_heatmaps_pad[index + width + 3]
    ix1_y1_ = batch_heatmaps_pad[index - width - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - width]

    # calculate refined coordinates using Newton's method
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(num_coords, num_keypoints, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(num_coords, num_keypoints, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


def transform_preds(coords, center, scale, output_size):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (`np.ndarray[K, ndims]`):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (`np.ndarray[2,]`):
            Center of the bounding box (x, y).
        scale (`np.ndarray[2,]`):
            Scale of the bounding box wrt [width, height].
        output_size (`np.ndarray[2,]):
            Size of the destination heatmaps in (height, width) format.

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    if coords.shape[1] not in (2, 4, 5):
        raise ValueError("Coordinates need to have either 2, 4 or 5 dimensions.")
    if len(center) != 2:
        raise ValueError("Center needs to have 2 elements, one for x and one for y.")
    if len(scale) != 2:
        raise ValueError("Scale needs to consist of a width and height")
    if len(output_size) != 2:
        raise ValueError("Output size needs to consist of a height and width")

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    # We use unbiased data processing
    scale_y = scale[1] / (output_size[0] - 1.0)
    scale_x = scale[0] / (output_size[1] - 1.0)

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


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
    This function implements cv2.warpAffine used in the original implementation using scipy.

    Note: the original implementation uses cv2.INTER_LINEAR.
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


class ViTPoseImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ViTPose image processor.

    Args:
        do_affine_transform (`bool`, *optional*, defaults to `True`):
            Whether to apply an affine transformation to the input images.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 256, "width": 192}`):
            Resolution of the image after `affine_transform` is applied. Only has an effect if `do_affine_transform` is set to `True`. Can
            be overriden by `size` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overriden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.485, 0.456, 0.406]`, *optional*):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.229, 0.224, 0.225]`, *optional*):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_affine_transform: bool = True,
        size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_affine_transform = do_affine_transform
        self.size = size if size is not None else {"height": 256, "width": 192}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

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

        # cv2 requires channels last format
        image = (
            image
            if input_data_format == ChannelDimension.LAST
            else to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)
        )
        image = scipy_warp_affine(src=image, M=transformation, size=(size[1], size[0]))

        image = to_channel_dimension_format(image, data_format, ChannelDimension.LAST)

        return image

    def preprocess(
        self,
        images: ImageInput,
        boxes: Union[List[List[float]], np.ndarray],
        do_affine_transform: bool = None,
        size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
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

            boxes (`List[List[float]]` or `np.ndarray`):
                List or array of bounding boxes for each image. Each box should be a list of 4 floats representing the bounding
                box coordinates in COCO format (x, y, w, h).

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
                        box, image_width=size["width"], image_height=size["height"]
                    )
                    transformed_image = self.affine_transform(
                        image, center, scale, rotation=0, size=size, input_data_format=input_data_format
                    )
                    new_images.append(transformed_image)
            images = new_images

        # since the number of boxes can differ per image, the image processor takes a list
        # rather than a numpy array of boxes
        # it currently creates pixel_values of shape (batch_size*num_persons, num_channels, height, width)

        if self.do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]
        if self.do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def keypoints_from_heatmaps(
        self,
        heatmaps,
        center,
        scale,
        kernel=11,
    ):
        """
        Get final keypoint predictions from heatmaps and transform them back to
        the image.

        Args:
            heatmaps (`np.ndarray` of shape `(batch_size, num_keypoints, height, width])`):
                Model predicted heatmaps.
            center (np.ndarray[N, 2]):
                Center of the bounding box (x, y).
            scale (np.ndarray[N, 2]):
                Scale of the bounding box wrt height/width.
            kernel (int):
                Gaussian kernel size (K) for modulation, which should match the heatmap gaussian sigma when training.
                K=17 for sigma=3 and k=11 for sigma=2.

        Returns:
            tuple: A tuple containing keypoint predictions and scores.

            - preds (np.ndarray[batch_size, num_keypoints, 2]):
                Predicted keypoint location in images.
            - scores (np.ndarray[batch_size, num_keypoints, 1]):
                Scores (confidence) of the keypoints.
        """
        # Avoid mutation
        heatmaps = heatmaps.numpy().copy()

        batch_size, _, height, width = heatmaps.shape

        coords, scores = get_keypoint_predictions(heatmaps)

        preds = post_dark_udp(coords, heatmaps, kernel=kernel)

        # Transform back to the image
        for i in range(batch_size):
            preds[i] = transform_preds(preds[i], center=center[i], scale=scale[i], output_size=[height, width])

        return preds, scores

    def post_process_pose_estimation(self, outputs, boxes, kernel_size=11):
        """
        Transform the heatmaps into keypoint predictions and transform them back to the image.

        Args:
            outputs (torch.Tensor):
                Model outputs.
            boxes (torch.Tensor of shape [batch_size, 4]):
                Bounding boxes.
            kernel_size (`int`, *optional*, defaults to 11):
                Gaussian kernel size (K) for modulation.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the keypoints and boxes for an image
            in the batch as predicted by the model.
        """

        # First compute centers and scales for each bounding box
        batch_size = len(outputs.heatmaps)
        centers = np.zeros((batch_size, 2), dtype=np.float32)
        scales = np.zeros((batch_size, 2), dtype=np.float32)
        for i in range(batch_size):
            width, height = self.size["width"], self.size["height"]
            center, scale = box_to_center_and_scale(boxes[i], image_width=width, image_height=height)
            centers[i, :] = center
            scales[i, :] = scale

        preds, scores = self.keypoints_from_heatmaps(outputs.heatmaps, centers, scales, kernel=kernel_size)

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = scores
        all_boxes[:, 0:2] = centers[:, 0:2]
        all_boxes[:, 2:4] = scales[:, 0:2]
        all_boxes[:, 4] = np.prod(scales * 200.0, axis=1)

        poses = torch.Tensor(all_preds)

        bboxes_xyxy = torch.Tensor(coco_to_pascal_voc(all_boxes))

        pose_results: List[Dict[str, torch.Tensor]] = []
        for pose, bbox_xyxy in zip(poses, bboxes_xyxy):
            pose_result = {}
            pose_result["keypoints"] = pose
            pose_result["bbox"] = bbox_xyxy
            pose_results.append(pose_result)

        return pose_results
