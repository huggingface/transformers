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
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

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
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, is_scipy_available, logging, requires_backends
from ...utils.import_utils import requires
from .image_processing_vitpose import (
    VitPoseImageProcessorKwargs,
    box_to_center_and_scale,
    coco_to_pascal_voc,
    get_keypoint_predictions,
    get_warp_matrix,
    post_dark_unbiased_data_processing,
    scipy_warp_affine,
    transform_preds,
)


if is_scipy_available():
    pass

if TYPE_CHECKING:
    from .modeling_vitpose import VitPoseEstimatorOutput

logger = logging.get_logger(__name__)


@auto_docstring
@requires(backends=("torch", "torchvision"))
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
        device: Union[str, "torch.device"] | None = None,
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
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
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
        requires_backends(self, "torch")
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
