# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Image processor class for VitPose."""

import itertools
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F

from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
)
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


class VitPoseImageProcessorFast(BaseImageProcessorFast):
    """
    Fast PyTorch VitPose image processor.

    Args:
        do_affine_transform (bool, optional, defaults to True):
            Whether to apply affine transform on input images.
        size (dict[str, int], optional, defaults to {"height": 20, "width": 20}):
            Resolution of output image after affine transform.
        do_rescale (bool, optional, defaults to True):
            Whether to scale pixel values to [0, 1].
        rescale_factor (float, optional, defaults to 1/255):
            Rescaling factor if do_rescale is True.
        do_normalize (bool, optional, defaults to True):
            Whether to normalize images.
        image_mean (list[float], optional, defaults to ImageNet mean):
            Mean for normalization per channel.
        image_std (list[float], optional, defaults to ImageNet std):
            Std dev for normalization per channel.
        normalize_factor (float, optional, defaults to 200.0):
            Normalization factor for scaling in box_to_center_and_scale and transform_preds.
    """

    model_input_names = ["pixel_values"]

    resample = None  # Not used in fast version, placeholder for interface

    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 20, "width": 20}
    do_affine_transform: bool = True
    do_rescale: bool = True
    rescale_factor: float = 1 / 255
    do_normalize: bool = True
    normalize_factor = 200.0

    def __init__(
        self,
        do_affine_transform: bool = True,
        size: Optional[dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        normalize_factor: float = 200.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_affine_transform = do_affine_transform
        self.size = size if size is not None else self.size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else self.image_mean
        self.image_std = image_std if image_std is not None else self.image_std
        self.normalize_factor = normalize_factor

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        Returns:
            dict: Dictionary of all the attributes that make up this processor instance.
        """
        return {
            "_processor_class": None,
            "image_processor_type": "VitPoseImageProcessor",
            "do_affine_transform": self.do_affine_transform,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "normalize_factor": self.normalize_factor,
        }

    def to_channel_dimension_format_fast(
        self,
        image: "torch.Tensor",
        channel_dim: Union[str, ChannelDimension],
        input_channel_dim: Optional[Union[str, ChannelDimension]] = None,
    ) -> "torch.Tensor":
        if input_channel_dim is None:
            input_channel_dim = self.infer_channel_dimension_format_fast(image)

        if input_channel_dim == channel_dim:
            return image

        if channel_dim == ChannelDimension.FIRST:
            if image.shape[-1] in [1, 3, 4]:  # (H, W, C) -> (C, H, W)
                return image.permute(2, 0, 1)
            elif image.shape[0] in [1, 3, 4]:  # (C, H, W) - already correct
                return image
            else:  # (H, C, W) -> (C, H, W)
                return image.permute(1, 0, 2)
        elif channel_dim == ChannelDimension.LAST:
            if image.shape[0] in [1, 3, 4]:  # (C, H, W) -> (H, W, C)
                return image.permute(1, 2, 0)
            elif image.shape[-1] in [1, 3, 4]:  # (H, W, C) - already correct
                return image
            else:  # (H, C, W) -> (H, W, C)
                return image.permute(0, 2, 1)
        else:
            raise ValueError(f"Unsupported channel dimension: {channel_dim}")

    def box_to_center_and_scale(
        self,
        box: torch.Tensor,
        image_width: int,
        image_height: int,
        normalize_factor: float = 200.0,
        padding_factor: float = 1.25,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if box.shape[-1] != 4:
            raise ValueError(
                f"Box must have 4 elements (top_left_x, top_left_y, width, height), got shape {box.shape}"
            )
        top_left_x = box[0] * image_width
        top_left_y = box[1] * image_height
        width = box[2] * image_width
        height = box[3] * image_height
        aspect_ratio = image_width / image_height
        center = torch.tensor(
            [top_left_x + 0.5 * width, top_left_y + 0.5 * height], dtype=torch.float32, device=box.device
        )

        if width > aspect_ratio * height:
            height = width / aspect_ratio
        elif width < aspect_ratio * height:
            width = height * aspect_ratio

        scale = torch.tensor(
            [width / normalize_factor, height / normalize_factor], dtype=torch.float32, device=box.device
        )
        scale *= padding_factor
        return center, scale

    def get_keypoint_predictions(
        self,
        heatmaps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_keypoints, height, width = heatmaps.shape
        heatmaps_reshaped = heatmaps.view(batch_size, num_keypoints, -1)
        maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
        maxvals = maxvals.unsqueeze(-1)  # (batch_size, num_keypoints, 1)
        idx = idx.unsqueeze(-1).repeat(1, 1, 2)  # repeat for x,y

        coords = idx.clone().float()
        coords[:, :, 0] = (coords[:, :, 0] % width).float()
        coords[:, :, 1] = (coords[:, :, 1] // width).float()

        # Set coordinates to -1 where maxvals <= 0.0 to indicate invalid keypoints
        coords = torch.where(maxvals > 0.0, coords, torch.full_like(coords, -1))
        return coords, maxvals

    def post_dark_unbiased_data_processing(
        self,
        coords: torch.Tensor,
        batch_heatmaps: torch.Tensor,
        kernel: int = 3,
    ) -> torch.Tensor:
        batch_size, num_keypoints, height, width = batch_heatmaps.shape
        num_coords = coords.shape[0]
        expected_coords = batch_size * num_keypoints
        if num_coords != expected_coords:
            raise ValueError(
                f"Number of coordinates ({num_coords}) must equal batch_size * num_keypoints ({expected_coords})"
            )

        radius = (kernel - 1) // 2

        # Create Gaussian kernel
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=batch_heatmaps.device)
        sigma = 0.8
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

        # Pad & smooth heatmaps
        padding = radius
        heatmaps_padded = F.pad(batch_heatmaps, (padding, padding, padding, padding), mode="replicate")
        heatmaps_smoothed = F.conv2d(
            heatmaps_padded.view(-1, 1, height + 2 * padding, width + 2 * padding), kernel_2d, padding=0
        )
        heatmaps_smoothed = heatmaps_smoothed.view(batch_size, num_keypoints, height, width)
        heatmaps_smoothed = torch.clamp(heatmaps_smoothed, min=0.001)
        heatmaps_log = heatmaps_smoothed.log()

        # Pad for indexing
        heatmaps_log_padded = F.pad(heatmaps_log, (1, 1, 1, 1), mode="replicate").view(-1)

        coords_x = coords[..., 0] + 1
        coords_y = coords[..., 1] + 1

        base = (width + 2) * (height + 2)
        batch_kp_idx = (
            torch.arange(batch_size * num_keypoints, device=coords.device)
            .unsqueeze(1)
            .repeat(1, num_coords // batch_size)
            .view(-1)
        )
        indices = coords_x + coords_y * (width + 2) + base * batch_kp_idx
        indices = indices.long()

        i_ = heatmaps_log_padded[indices]
        ix1 = heatmaps_log_padded[indices + 1]
        ix1_ = heatmaps_log_padded[indices - 1]
        iy1 = heatmaps_log_padded[indices + (width + 2)]
        iy1_ = heatmaps_log_padded[indices - (width + 2)]
        ix1y1 = heatmaps_log_padded[indices + (width + 2) + 1]
        ix1_y1_ = heatmaps_log_padded[indices - (width + 2) - 1]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = torch.stack([dx, dy], dim=1).view(num_coords, num_keypoints, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.25 * (ix1y1 - ix1 - iy1 + 2 * i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = torch.stack([dxx, dxy, dxy, dyy], dim=1).view(num_coords, num_keypoints, 2, 2)

        eye_eps = torch.eye(2, device=coords.device).unsqueeze(0).unsqueeze(0) * torch.finfo(torch.float32).eps
        hessian_inv = torch.linalg.inv(hessian + eye_eps)

        delta = torch.matmul(hessian_inv, derivative).squeeze(-1)
        refined_coords = coords - delta
        return refined_coords

    def transform_preds(
        self,
        coords: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if coords.shape[1] not in (2, 3):
            raise ValueError("Coordinates must have 2 (x, y) or 3 (x, y, confidence) dimensions.")
        if len(center) != 2 or len(scale) != 2 or len(output_size) != 2:
            raise ValueError("Center, scale, and output_size must have 2 elements.")

        scale = scale * self.normalize_factor

        scale_y = scale[1] / (output_size[0] - 1.0)
        scale_x = scale[0] / (output_size[1] - 1.0)

        target_coords = torch.ones_like(coords)
        target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - 0.5 * scale[0]
        target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - 0.5 * scale[1]

        return target_coords

    def affine_transform(
        self,
        image: torch.Tensor,
        center: tuple[float, float],
        scale: tuple[float, float],
        rotation: float,
        size: dict[str, int],
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> torch.Tensor:
        """
        Apply affine transform to a torch image tensor.

        Args:
            image (torch.Tensor): Image tensor of shape (C,H,W) or (H,W,C) depending on input_data_format.
            center (tuple): Center coordinates.
            scale (tuple): Scale factors.
            rotation (float): Rotation angle in degrees.
            size (dict): Output size dict with keys "height" and "width".
            data_format (optional): Output channel dimension format.
            input_data_format (optional): Input channel dimension format.

        Returns:
            Transformed image tensor.
        """
        if input_data_format not in [ChannelDimension.FIRST, ChannelDimension.LAST, None]:
            raise ValueError(f"Invalid input_data_format: {input_data_format}")
        if data_format not in [ChannelDimension.FIRST, ChannelDimension.LAST, None]:
            raise ValueError(f"Invalid data_format: {data_format}")
        data_format = input_data_format if data_format is None else data_format
        out_size = (size["width"], size["height"])

        # Adapt image format to (C,H,W) using PyTorch-native method
        if input_data_format != ChannelDimension.FIRST:
            image = self.to_channel_dimension_format_fast(image, ChannelDimension.FIRST, input_data_format)

        num_channels = image.shape[0]  # Preserve input channel count
        theta_rad = math.radians(rotation)
        scale_x = out_size[0] / (scale[0] * self.normalize_factor)
        scale_y = out_size[1] / (scale[1] * self.normalize_factor)

        # Construct affine matrix for grid_sample with shape (1, 2, 3)
        theta = torch.zeros((1, 2, 3), dtype=torch.float32, device=image.device)
        theta[0, 0, 0] = math.cos(theta_rad) * scale_x
        theta[0, 0, 1] = -math.sin(theta_rad) * scale_x
        theta[0, 1, 0] = math.sin(theta_rad) * scale_y
        theta[0, 1, 1] = math.cos(theta_rad) * scale_y
        theta[0, 0, 2] = -center[0] * theta[0, 0, 0] - center[1] * theta[0, 0, 1] + out_size[0] / 2
        theta[0, 1, 2] = -center[0] * theta[0, 1, 0] - center[1] * theta[0, 1, 1] + out_size[1] / 2

        grid = F.affine_grid(theta, size=(1, num_channels, out_size[1], out_size[0]), align_corners=False)
        image = image.unsqueeze(0).float()
        transformed = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
        transformed = transformed.squeeze(0)

        # Convert output format using PyTorch-native method
        if data_format != ChannelDimension.FIRST:
            transformed = self.to_channel_dimension_format_fast(transformed, data_format, ChannelDimension.FIRST)

        return transformed

    def preprocess(
        self,
        images: ImageInput,
        boxes: Union[list[list[float]], torch.Tensor],
        do_affine_transform: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> "BatchFeature":
        do_affine_transform = do_affine_transform if do_affine_transform is not None else self.do_affine_transform
        size = size if size is not None else self.size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = self.make_list_of_images_fast(images)
        if not self.valid_images_fast(images):
            raise ValueError(
                "Invalid image type. Must be a PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if isinstance(boxes, list):
            for image_boxes in boxes:
                if not isinstance(image_boxes, (list, torch.Tensor)):
                    raise ValueError(f"Each element of boxes must be a list or tensor, got {type(image_boxes)}")
            if len(images) != len(boxes):
                raise ValueError(f"Batch of images and boxes mismatch : {len(images)} != {len(boxes)}")
        elif torch.is_tensor(boxes) and len(images) != boxes.shape[0]:
            raise ValueError(f"Batch of images and boxes mismatch : {len(images)} != {boxes.shape[0]}")

        if self.is_scaled_image_fast(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. Set do_rescale=False to avoid double scaling."
            )

        if input_data_format is None:
            input_data_format = self.infer_channel_dimension_format_fast(images[0])

        # Get number of channels from first image
        num_channels = images[0].shape[0] if input_data_format == ChannelDimension.FIRST else images[0].shape[-1]

        # Fix for 4-channel normalization: Ensure image_mean and image_std match the number of channels
        if isinstance(image_mean, (int, float)):
            image_mean = [image_mean] * num_channels
        if isinstance(image_std, (int, float)):
            image_std = [image_std] * num_channels

        # Ensure the lists have the correct length - pad or truncate as needed
        if len(image_mean) < num_channels:
            image_mean = image_mean + [image_mean[-1]] * (num_channels - len(image_mean))
        elif len(image_mean) > num_channels:
            image_mean = image_mean[:num_channels]

        if len(image_std) < num_channels:
            image_std = image_std + [image_std[-1]] * (num_channels - len(image_std))
        elif len(image_std) > num_channels:
            image_std = image_std[:num_channels]

        if do_affine_transform:
            new_images = []
            for image, image_boxes in zip(images, boxes):
                image_tensor = image if torch.is_tensor(image) else torch.tensor(image, dtype=torch.float32)
                if input_data_format == ChannelDimension.FIRST:
                    num_channels, height, width = image_tensor.shape
                else:
                    height, width, num_channels = image_tensor.shape
                for box in image_boxes:
                    box_tensor = torch.tensor(box, dtype=torch.float32) if not torch.is_tensor(box) else box
                    center, scale = self.box_to_center_and_scale(box_tensor, image_width=width, image_height=height)
                    transformed_image = self.affine_transform(
                        image_tensor, center, scale, rotation=0, size=size, input_data_format=input_data_format
                    )
                    new_images.append(transformed_image)
            images = new_images
        else:
            images = [
                image if torch.is_tensor(image) else torch.tensor(image, dtype=torch.float32) for image in images
            ]

        # Apply rescale and normalize after affine transform
        all_images = []
        for image in images:
            # Convert to channels_first for normalization
            current_format = self.infer_channel_dimension_format_fast(image)
            if current_format != ChannelDimension.FIRST:
                image = self.to_channel_dimension_format_fast(image, ChannelDimension.FIRST, current_format)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            if data_format != ChannelDimension.FIRST:
                image = self.to_channel_dimension_format_fast(image, data_format, ChannelDimension.FIRST)
            all_images.append(image)

        # Stack images into a single tensor if return_tensors is specified
        if return_tensors is not None:
            images = torch.stack(all_images)

        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def keypoints_from_heatmaps(
        self,
        heatmaps: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        kernel: int = 11,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = heatmaps.shape

        coords, scores = self.get_keypoint_predictions(heatmaps)

        preds = self.post_dark_unbiased_data_processing(coords, heatmaps, kernel=kernel)

        for i in range(batch_size):
            preds[i] = self.transform_preds(preds[i], center[i], scale[i], (height, width))

        return preds, scores

    def post_process_pose_estimation(
        self,
        outputs,
        boxes: Union[list[list[list[float]]], torch.Tensor],
        kernel_size: int = 11,
        threshold: Optional[float] = None,
        target_sizes: Union[TensorType, list[tuple[int, int]]] = None,
    ):
        if not hasattr(outputs, "heatmaps"):
            raise ValueError("Outputs must have a 'heatmaps' attribute")
        batch_size, num_keypoints, _, _ = outputs.heatmaps.shape

        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError("Number of target sizes must match batch size.")

        centers = torch.zeros((batch_size, 2), dtype=torch.float32)
        scales = torch.zeros((batch_size, 2), dtype=torch.float32)

        if isinstance(boxes, torch.Tensor):
            flattened_boxes = boxes
        else:
            flattened_boxes = list(itertools.chain(*boxes))

        for i in range(batch_size):
            if target_sizes is not None:
                image_width, image_height = target_sizes[i][0], target_sizes[i][1]
                scale_factor = torch.tensor(
                    [image_width, image_height, image_width, image_height], dtype=torch.float32
                )
                flattened_boxes[i] = torch.tensor(flattened_boxes[i], dtype=torch.float32) * scale_factor
            width, height = self.size["width"], self.size["height"]
            center, scale = self.box_to_center_and_scale(flattened_boxes[i], image_width=width, image_height=height)
            centers[i, :] = center
            scales[i, :] = scale

        preds, scores = self.keypoints_from_heatmaps(outputs.heatmaps.cpu(), centers, scales, kernel=kernel_size)

        all_boxes = torch.zeros((batch_size, 4), dtype=torch.float32)
        all_boxes[:, 0:2] = centers
        all_boxes[:, 2:4] = scales

        poses = preds
        scores = scores
        labels = torch.arange(0, num_keypoints).repeat(batch_size, 1)
        bboxes_xyxy = self.coco_to_pascal_voc(all_boxes)

        results = []
        pose_bbox_pairs = zip(poses, scores, bboxes_xyxy)

        for image_bboxes in boxes:
            image_results = []
            for _ in image_bboxes:
                pose, score, bbox_xyxy = next(pose_bbox_pairs)
                score = score.squeeze()
                keypoints_labels = labels[0]  # Use first batch's labels as they are repeated
                if threshold is not None:
                    keep = score > threshold
                    pose = pose[keep]
                    score = score[keep]
                    keypoints_labels = keypoints_labels[keep]
                image_results.append(
                    {"keypoints": pose, "scores": score, "labels": keypoints_labels, "bbox": bbox_xyxy}
                )
            results.append(image_results)

        return results

    @staticmethod
    def coco_to_pascal_voc(bboxes: torch.Tensor) -> torch.Tensor:
        bboxes = bboxes.clone()
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0] - 1
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1] - 1
        return bboxes


__all__ = ["VitPoseImageProcessorFast"]
