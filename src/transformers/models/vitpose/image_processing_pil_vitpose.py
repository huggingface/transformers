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
"""Image processor class for VitPose."""

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_scipy_available, logging
from ...utils.import_utils import requires


if is_scipy_available():
    from scipy.ndimage import affine_transform, gaussian_filter

inv = np.linalg.inv

if TYPE_CHECKING:
    from .modeling_vitpose import VitPoseEstimatorOutput

logger = logging.get_logger(__name__)


# Adapted from transformers.models.vitpose.image_processing_vitpose.VitPoseImageProcessorKwargs
class VitPoseImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_affine_transform (`bool`, *optional*):
        Whether to apply an affine transformation to the input images based on the bounding boxes.
    normalize_factor (`float`, *optional*, defaults to `200.0`):
        Width and height scale factor used for normalization when computing center and scale from bounding boxes.
    """

    do_affine_transform: bool | None
    normalize_factor: float | None


# Adapted from transformers.models.vitpose.image_processing_vitpose.box_to_center_and_scale
# inspired by https://github.com/ViTAE-Transformer/ViTPose/blob/d5216452796c90c6bc29f5c5ec0bdba94366768a/mmpose/datasets/datasets/base/kpt_2d_sview_rgb_img_top_down_dataset.py#L132
def box_to_center_and_scale(
    box: tuple | list | np.ndarray,
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


# Adapted from transformers.models.vitpose.image_processing_vitpose.coco_to_pascal_voc
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


# Adapted from transformers.models.vitpose.image_processing_vitpose.get_keypoint_predictions
def get_keypoint_predictions(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get keypoint predictions from score maps.

    Args:
        heatmaps (`np.ndarray` of shape `(batch_size, num_keypoints, height, width)`):
            Model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - coords (`np.ndarray` of shape `(batch_size, num_keypoints, 2)`):
            Predicted keypoint location.
        - scores (`np.ndarray` of shape `(batch_size, num_keypoints, 1)`):
            Scores (confidence) of the keypoints.
    """
    if not isinstance(heatmaps, np.ndarray):
        raise TypeError("Heatmaps should be np.ndarray")
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


# Adapted from transformers.models.vitpose.image_processing_vitpose.get_warp_matrix
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


# Adapted from transformers.models.vitpose.image_processing_vitpose.post_dark_unbiased_data_processing
def post_dark_unbiased_data_processing(coords: np.ndarray, batch_heatmaps: np.ndarray, kernel: int = 3) -> np.ndarray:
    """DARK post-pocessing. Implemented by unbiased_data_processing.

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
    batch_size, num_keypoints, height, width = batch_heatmaps.shape
    num_coords = coords.shape[0]
    if not (batch_size == 1 or batch_size == num_coords):
        raise ValueError("The batch size of heatmaps should be 1 or equal to the batch size of coordinates.")
    radius = int((kernel - 1) // 2)
    batch_heatmaps = np.array(
        [
            [gaussian_filter(heatmap, sigma=0.8, radius=(radius, radius), axes=(0, 1)) for heatmap in heatmaps]
            for heatmaps in batch_heatmaps
        ]
    )
    batch_heatmaps = np.clip(batch_heatmaps, 0.001, 50)
    batch_heatmaps = np.log(batch_heatmaps)

    batch_heatmaps_pad = np.pad(batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").flatten()
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


# Adapted from transformers.models.vitpose.image_processing_vitpose.scipy_warp_affine
def scipy_warp_affine(src, M, size):
    """
    This function implements cv2.warpAffine function using affine_transform in scipy. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html and https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html for more details.

    Note: the original implementation of cv2.warpAffine uses cv2.INTER_LINEAR.
    """
    channels = [src[..., i] for i in range(src.shape[-1])]

    # Convert to a 3x3 matrix used by SciPy
    M_scipy = np.vstack([M, [0, 0, 1]])
    # If you have a matrix for the 'push' transformation, use its inverse (numpy.linalg.inv) in this function.
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


# Adapted from transformers.models.vitpose.image_processing_vitpose.transform_preds
def transform_preds(coords: np.ndarray, center: np.ndarray, scale: np.ndarray, output_size: np.ndarray) -> np.ndarray:
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (`np.ndarray` of shape `(num_keypoints, ndims)`):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (`np.ndarray` of shape `(2,)`):
            Center of the bounding box (x, y).
        scale (`np.ndarray` of shape `(2,)`):
            Scale of the bounding box wrt original image of width and height.
        output_size (`np.ndarray` of shape `(2,)`):
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


@auto_docstring
class VitPoseImageProcessorPil(PilBackend):
    """PIL backend for VitPose with affine transform."""

    valid_kwargs = VitPoseImageProcessorKwargs
    model_input_names = ["pixel_values"]

    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 256, "width": 192}
    do_rescale = True
    do_normalize = True
    do_affine_transform = True
    normalize_factor = 200.0

    def __init__(self, **kwargs: Unpack[VitPoseImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        boxes: list[list[list[float]]] | np.ndarray,
        **kwargs: Unpack[VitPoseImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        boxes (`list[list[list[float]]]` or `np.ndarray`):
            List or array of bounding boxes for each image. Each box should be a list of 4 floats representing the
            bounding box coordinates in COCO format (top_left_x, top_left_y, width, height).
        """
        return super().preprocess(images, boxes, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        boxes: list[list[list[float]]] | np.ndarray | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Handle extra inputs beyond images."""
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        # Pass boxes to backend preprocess
        kwargs["boxes"] = boxes
        return self._preprocess(images, **kwargs)

    def affine_transform(
        self, image: np.ndarray, center: tuple[float], scale: tuple[float], rotation: float, size: SizeDict
    ) -> np.ndarray:
        """Apply an affine transformation to an image."""
        size_tuple = (size.width, size.height)
        transformation = get_warp_matrix(rotation, center * 2.0, np.array(size_tuple) - 1.0, scale * 200.0)
        # scipy_warp_affine expects (H, W, C) - channels_last format
        if image.ndim == 3 and image.shape[0] <= 4 and image.shape[0] < image.shape[1]:
            # channels_first (C, H, W) -> channels_last (H, W, C)
            image = image.transpose(1, 2, 0)
        transformed = scipy_warp_affine(src=image, M=transformation, size=(size.height, size.width))
        # Convert back to channels_first (C, H, W) - PilBackend pipeline expects it
        transformed = transformed.transpose(2, 0, 1)
        return transformed

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        return_tensors: str | TensorType | None,
        do_affine_transform: bool = True,
        normalize_factor: float = 200.0,
        boxes: list | np.ndarray | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for VitPose."""
        if boxes is not None and do_affine_transform:
            transformed_images = []
            for image, image_boxes in zip(images, boxes):
                for box in image_boxes:
                    center, scale = box_to_center_and_scale(
                        box, image_width=size.width, image_height=size.height, normalize_factor=normalize_factor
                    )
                    transformed_image = self.affine_transform(image, center, scale, rotation=0, size=size)
                    transformed_images.append(transformed_image)
            images = transformed_images

        processed_images = []
        for image in images:
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def keypoints_from_heatmaps(self, heatmaps: np.ndarray, center: np.ndarray, scale: np.ndarray, kernel: int = 11):
        """Get final keypoint predictions from heatmaps and transform them back to the image."""
        batch_size, _, height, width = heatmaps.shape
        coords, scores = get_keypoint_predictions(heatmaps)
        preds = post_dark_unbiased_data_processing(coords, heatmaps, kernel=kernel)
        for i in range(batch_size):
            preds[i] = transform_preds(preds[i], center=center[i], scale=scale[i], output_size=[height, width])
        return preds, scores

    @requires(backends=("torch",))
    def post_process_pose_estimation(
        self,
        outputs: "VitPoseEstimatorOutput",
        boxes: list[list[list[float]]] | np.ndarray,
        kernel_size: int = 11,
        threshold: float | None = None,
        target_sizes: TensorType | list[tuple] | None = None,
    ):
        """
        Transform the heatmaps into keypoint predictions and transform them back to the image.

        Args:
            outputs (`VitPoseEstimatorOutput`):
                VitPoseForPoseEstimation model outputs.
            boxes (`list[list[list[float]]]` or `np.ndarray`):
                List or array of bounding boxes for each image. Each box should be a list of 4 floats representing the bounding
                box coordinates in COCO format (top_left_x, top_left_y, width, height).
            kernel_size (`int`, *optional*, defaults to 11):
                Gaussian kernel size (K) for modulation.
            threshold (`float`, *optional*, defaults to None):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will be resize with the default value.
        Returns:
            `list[list[Dict]]`: A list of dictionaries, each dictionary containing the keypoints and boxes for an image
            in the batch as predicted by the model.
        """
        import torch

        batch_size, num_keypoints, _, _ = outputs.heatmaps.shape
        if target_sizes is not None and batch_size != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        centers = np.zeros((batch_size, 2), dtype=np.float32)
        scales = np.zeros((batch_size, 2), dtype=np.float32)
        flattened_boxes = list(itertools.chain(*boxes))
        for i in range(batch_size):
            if target_sizes is not None:
                image_width, image_height = target_sizes[i][0], target_sizes[i][1]
                scale_factor = np.array([image_width, image_height, image_width, image_height])
                flattened_boxes[i] = flattened_boxes[i] * scale_factor
            width, height = self.size["width"], self.size["height"]
            center, scale = box_to_center_and_scale(flattened_boxes[i], image_width=width, image_height=height)
            centers[i, :] = center
            scales[i, :] = scale
        preds, scores = self.keypoints_from_heatmaps(
            outputs.heatmaps.cpu().numpy(), centers, scales, kernel=kernel_size
        )
        all_boxes = np.zeros((batch_size, 4), dtype=np.float32)
        all_boxes[:, 0:2] = centers[:, 0:2]
        all_boxes[:, 2:4] = scales[:, 0:2]
        poses = torch.tensor(preds)
        scores = torch.tensor(scores)
        labels = torch.arange(0, num_keypoints)
        bboxes_xyxy = torch.tensor(coco_to_pascal_voc(all_boxes))
        results: list[list[dict[str, torch.Tensor]]] = []
        pose_bbox_pairs = zip(poses, scores, bboxes_xyxy)
        for image_bboxes in boxes:
            image_results: list[dict[str, torch.Tensor]] = []
            for _ in image_bboxes:
                pose, score, bbox_xyxy = next(pose_bbox_pairs)
                score = score.squeeze()
                keypoints_labels = labels
                if threshold is not None:
                    keep = score > threshold
                    pose = pose[keep]
                    score = score[keep]
                    keypoints_labels = keypoints_labels[keep]
                pose_result = {"keypoints": pose, "scores": score, "labels": keypoints_labels, "bbox": bbox_xyxy}
                image_results.append(pose_result)
            results.append(image_results)
        return results


__all__ = ["VitPoseImageProcessorPil"]
