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

import itertools
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
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


if TYPE_CHECKING:
    from .modeling_vitpose import VitPoseEstimatorOutput


@auto_docstring
class VitPoseImageProcessorFast(BaseImageProcessorFast):
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 256, "width": 192}
    do_rescale = True
    do_normalize = True
    do_affine_transform = True
    normalize_factor = 200.0
    valid_kwargs = VitPoseImageProcessorKwargs
    model_input_names = ["pixel_values"]

    def torch_affine_transform(
        self,
        image: torch.Tensor,
        center: tuple[float],
        scale: tuple[float],
        rotation: float,
        size: SizeDict,
    ) -> torch.Tensor:
        """
        Apply an affine transformation to a torch tensor image.

        Args:
            image (`torch.Tensor`):
                Image tensor of shape (C, H, W) to transform.
            center (`tuple[float]`):
                Center of the bounding box (x, y).
            scale (`tuple[float]`):
                Scale of the bounding box with respect to height/width.
            rotation (`float`):
                Rotation angle in degrees.
            size (`SizeDict`):
                Size of the destination image.

        Returns:
            `torch.Tensor`: The transformed image.
        """
        transformation = get_warp_matrix(
            rotation, center * 2.0, np.array((size.width, size.height)) - 1.0, scale * 200.0
        )
        # Convert tensor to numpy (channels last) for scipy_warp_affine
        image_np = image.permute(1, 2, 0).cpu().numpy()
        transformed_np = scipy_warp_affine(src=image_np, M=transformation, size=(size.height, size.width))

        # Convert back to torch tensor (channels first)
        transformed = torch.from_numpy(transformed_np).permute(2, 0, 1).to(image.device)

        return transformed

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        boxes: Union[list[list[float]], np.ndarray],
        **kwargs: Unpack[VitPoseImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        boxes (`list[list[list[float]]]` or `np.ndarray`):
            List or array of bounding boxes for each image. Each box should be a list of 4 floats representing the
            bounding box coordinates in COCO format (top_left_x, top_left_y, width, height).
        """
        return super().preprocess(images, boxes, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        boxes: Union[list, np.ndarray],
        do_affine_transform: bool,
        size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, tuple[float]],
        image_std: Union[float, tuple[float]],
        disable_grouping: bool,
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images with affine transformations based on bounding boxes.
        """
        if len(images) != len(boxes):
            raise ValueError(f"Number of images and boxes must match: {len(images)} != {len(boxes)}")

        # Apply affine transformation for each image and each box
        if do_affine_transform:
            transformed_images = []
            for image, image_boxes in zip(images, boxes):
                for box in image_boxes:
                    center, scale = box_to_center_and_scale(
                        box,
                        image_width=size.width,
                        image_height=size.height,
                        normalize_factor=self.normalize_factor,
                    )
                    transformed_image = self.torch_affine_transform(image, center, scale, rotation=0, size=size)
                    transformed_images.append(transformed_image)
            images = transformed_images

        # Group images by shape for efficient batch processing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Apply rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Stack into batch tensor

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def keypoints_from_heatmaps(
        self,
        heatmaps: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        kernel: int = 11,
    ):
        """
        Get final keypoint predictions from heatmaps and transform them back to
        the image.

        Args:
            heatmaps (`np.ndarray` of shape `(batch_size, num_keypoints, height, width])`):
                Model predicted heatmaps.
            center (`np.ndarray` of shape `(batch_size, 2)`):
                Center of the bounding box (x, y).
            scale (`np.ndarray` of shape `(batch_size, 2)`):
                Scale of the bounding box wrt original images of width and height.
            kernel (int, *optional*, defaults to 11):
                Gaussian kernel size (K) for modulation, which should match the heatmap gaussian sigma when training.
                K=17 for sigma=3 and k=11 for sigma=2.

        Returns:
            tuple: A tuple containing keypoint predictions and scores.

            - preds (`np.ndarray` of shape `(batch_size, num_keypoints, 2)`):
                Predicted keypoint location in images.
            - scores (`np.ndarray` of shape `(batch_size, num_keypoints, 1)`):
                Scores (confidence) of the keypoints.
        """
        batch_size, _, height, width = heatmaps.shape

        coords, scores = get_keypoint_predictions(heatmaps)

        preds = post_dark_unbiased_data_processing(coords, heatmaps, kernel=kernel)

        # Transform back to the image
        for i in range(batch_size):
            preds[i] = transform_preds(preds[i], center=center[i], scale=scale[i], output_size=[height, width])

        return preds, scores

    def post_process_pose_estimation(
        self,
        outputs: "VitPoseEstimatorOutput",
        boxes: Union[list[list[list[float]]], np.ndarray],
        kernel_size: int = 11,
        threshold: Optional[float] = None,
        target_sizes: Optional[Union[TensorType, list[tuple]]] = None,
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

        # First compute centers and scales for each bounding box
        batch_size, num_keypoints, _, _ = outputs.heatmaps.shape

        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

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
                # Unpack the next pose and bbox_xyxy from the iterator
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


__all__ = ["VitPoseImageProcessorFast"]
