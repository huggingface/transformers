# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF

from ...activations import ACT2FN
from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    SizeDict,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_cv2_available,
    logging,
    requires_backends,
)
from ...utils.generic import TensorType
from ..auto import AutoConfig


if is_cv2_available():
    import cv2


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv5_server_det_safetensors")
class PPOCRV5ServerDetConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PPOCRV5ServerDet`]. It is used to instantiate a
    PPOCRV5 Server text detection model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the PPOCRV5 Server Det
    [PaddlePaddle/PP-OCRv5-server-det](https://huggingface.co/PaddlePaddle/PP-OCRv5-server-det) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        interpolate_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation mode used for upsampling or downsampling feature maps in the neck network.
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*):
            The configuration of the backbone model.
        neck_out_channels (`int`, *optional*, defaults to 256):
            The number of output channels from the neck network, responsible for feature fusion and refinement.
        reduce_factor (`int`, *optional*, defaults to 2):
            The channel reduction factor used in the neck blocks to balance performance and complexity.
        intraclass_block_number (`int`, *optional*, defaults to 4):
            The number of Intra-Class Block modules used for enhancing feature representation.
        intraclass_block_config (`dict`, *optional*, defaults to `None`):
            Configuration for the Intra-Class Block modules, if any, used for enhancing feature representation.
        scale_factor (`int`, *optional*, defaults to 2):
            The scaling factor used for spatial resolution adjustments in the feature maps.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function used in the hidden layers. Supported functions include `"relu"`, `"hswish"`, etc.
        kernel_list (`list[int]`, *optional*, defaults to `[3, 2, 2]`):
            The list of kernel sizes for convolutional layers in the head network for multi-scale feature extraction.

    Examples:
    ```python
    >>> from transformers import PPOCRV5ServerDetConfig, PPOCRV5ServerDetForTextDetection
    >>> # Initializing a PPOCRV5 Server Det configuration
    >>> configuration = PPOCRV5ServerDetConfig()
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = PPOCRV5ServerDetForTextDetection(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    sub_configs = {"backbone_config": AutoConfig}
    model_type = "pp_ocrv5_server_det"

    def __init__(
        self,
        interpolate_mode: str = "nearest",
        backbone_config=None,
        neck_out_channels: int = 256,
        reduce_factor: int = 2,
        intraclass_block_number: int = 4,
        intraclass_block_config: dict | None = None,
        scale_factor: int = 2,
        hidden_act: str = "relu",
        kernel_list: list[int] = [3, 2, 2],
        **kwargs,
    ):
        self.interpolate_mode = interpolate_mode

        # ---- backbone ----
        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="hgnet_v2",
            default_config_kwargs={
                "arch": "L",
                "return_idx": [0, 1, 2, 3],
                "freeze_stem_only": True,
                "freeze_at": 0,
                "freeze_norm": True,
                "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )
        self.backbone_config = backbone_config

        # ---- neck ----
        self.neck_out_channels = neck_out_channels
        self.reduce_factor = reduce_factor
        self.intraclass_block_number = intraclass_block_number
        self.intraclass_block_config = intraclass_block_config

        # ---- head ----
        self.scale_factor = scale_factor
        self.hidden_act = hidden_act
        self.kernel_list = kernel_list

        # For object detection pipeline compatibility: single class "text"
        if "id2label" not in kwargs:
            kwargs["id2label"] = {0: "text"}
        if "num_labels" not in kwargs:
            kwargs["num_labels"] = 1

        super().__init__(**kwargs)


class PPOCRV5ServerDetImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    limit_side_len (`int`, *optional*, defaults to `960`):
        Maximum or minimum side length.
    limit_type (`str`, *optional*, defaults to `max`):
        Resizing strategy: "max", "min", or "resize_long".
    max_side_limit (`int`, *optional* defaults to `4000`):
        Maximum allowed side length.
    """

    limit_side_len: int
    limit_type: str
    max_side_limit: int


@auto_docstring(
    custom_intro="""
    """
)
class PPOCRV5ServerDetImageProcessorFast(BaseImageProcessorFast):
    """
    Image processor for PPOCRV5 Server Det model, handling preprocessing (resizing, normalization)
    and post-processing (converting model outputs to text boxes).
    """

    resample = 2
    image_mean = [0.406, 0.456, 0.485]
    image_std = [0.225, 0.224, 0.229]
    size = {"height": 960, "width": 960}
    do_resize = True
    do_rescale = True
    do_normalize = True
    limit_side_len = 960
    limit_type = "max"
    max_side_limit = 4000
    valid_kwargs = PPOCRV5ServerDetImageProcessorKwargs

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        limit_side_len: int,
        limit_type: str,
        max_side_limit: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        target_sizes = []

        # Group images by their original spatial shape to enable batched resizing (optimization for efficiency)
        # [Key Change] Unlike the original implementation, we now track target shapes for each original shape group
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        # Store resized image batches mapped to their original shape keys
        resized_images_grouped = {}
        # [Key Change] Core addition: Mapping from original image shape to target resize shape
        # This dict ensures consistent target shape handling across all subsequent operations (resize/processing)
        target_shape_per_shape = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                resize_size, target_shape = self.get_image_size(
                    stacked_images[0], limit_side_len, limit_type, max_side_limit
                )
                target_shape_per_shape[shape] = target_shape
                stacked_images = self.resize(image=stacked_images, size=resize_size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images

        resized_images = reorder_images(resized_images_grouped, grouped_images_index)
        if do_resize:
            target_sizes = [target_shape_per_shape[grouped_images_index[i][0]] for i in range(len(images))]

        # Group images by size for further processing
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            # BGR to RGB conversion
            stacked_images = stacked_images[:, [2, 1, 0], :, :]
            processed_images_grouped[shape] = stacked_images

        pixel_values = reorder_images(processed_images_grouped, grouped_images_index)

        return BatchFeature(
            data={"pixel_values": pixel_values, "target_sizes": target_sizes},
            tensor_type=return_tensors,
        )

    def _unclip(self, contour_box, unclip_ratio):
        """
        Expands (dilates) a detected text bounding box to recover the full text region.

        Args:
            contour_box (np.ndarray): Input contour of shape (N, 2), where N is the number of points.
            unclip_ratio (float): Expansion ratio, typically greater than 1.0.

        Returns:
            np.ndarray: Expanded contour of shape (M, 2).
        """
        contour_box = np.array(contour_box).reshape(-1, 2)

        contour_area = cv2.contourArea(contour_box)
        contour_perimeter = cv2.arcLength(contour_box, True)

        if contour_perimeter == 0:
            return contour_box

        expansion_distance = contour_area * unclip_ratio / contour_perimeter

        closed_contour = np.concatenate([contour_box, contour_box[0:1]], axis=0)
        expanded_points = []

        for point_idx in range(len(contour_box)):
            current_point = closed_contour[point_idx]
            prev_point = closed_contour[point_idx - 1]
            next_point = closed_contour[point_idx + 1]

            def get_normal_vector(point_a, point_b):
                direction = point_b - point_a
                direction_norm = np.linalg.norm(direction)
                if direction_norm == 0:
                    return np.array([0, 0])
                return np.array([direction[1], -direction[0]]) / direction_norm

            normal_prev_current = get_normal_vector(prev_point, current_point)
            normal_current_next = get_normal_vector(current_point, next_point)

            combined_normal = normal_prev_current + normal_current_next
            cos_angle = np.dot(normal_prev_current, normal_current_next)

            denominator = 1 + cos_angle
            if denominator < 1e-6:
                scale_factor = expansion_distance
            else:
                scale_factor = expansion_distance * np.sqrt(2 / denominator)

            combined_norm = np.linalg.norm(combined_normal) + 1e-6
            new_point = current_point + combined_normal * (scale_factor / combined_norm)
            expanded_points.append(new_point)

        return np.array(expanded_points, dtype=np.float32)

    def _get_mini_boxes(self, contour):
        """
        Computes the minimum-area bounding rectangle for a given contour and returns
        its four corners in a consistent order (top-left, bottom-left, bottom-right, top-right).

        Args:
            contour (np.ndarray): Input contour of shape (N, 1, 2).

        Returns:
            tuple:
                - box (list): List of four corner points in order.
                - short_side_length (float): Length of the shorter side of the bounding rectangle.
        """
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(bounding_box), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def _get_box_score(self, bitmap: np.ndarray, polygon_bounding_box: np.ndarray):
        """
        Computes the mean score of a bounding box region in the prediction map using
        a fast approach with axis-aligned bounding boxes.

        Args:
            bitmap (np.ndarray): Binary or float prediction map of shape (H, W).
            polygon_bounding_box (np.ndarray): Bounding box polygon of shape (N, 2).

        Returns:
            float: Mean score within the bounding box region.
        """
        height, width = bitmap.shape[:2]
        box = polygon_bounding_box.copy()
        xmin = max(0, min(math.floor(box[:, 0].min()), width - 1))
        xmax = max(0, min(math.ceil(box[:, 0].max()), width - 1))
        ymin = max(0, min(math.floor(box[:, 1].min()), height - 1))
        ymax = max(0, min(math.ceil(box[:, 1].max()), height - 1))

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def _boxes_from_bitmap(
        self,
        prediction: np.ndarray,
        bitmap: np.ndarray,
        dest_width: int,
        dest_height: int,
        box_threshold: float,
        unclip_ratio: float,
        min_size: int,
        max_candidates: int,
    ):
        """
        Extracts axis-aligned or rotated bounding boxes from a binary segmentation map.

        Args:
            prediction (np.ndarray): Raw prediction map of shape (H, W).
            bitmap (np.ndarray): Binarized segmentation map of shape (H, W).
            dest_width (int): Original image width for scaling back.
            dest_height (int): Original image height for scaling back.
            box_threshold (float): Score threshold for filtering low-confidence boxes.
            unclip_ratio (float): Expansion ratio for contour unclipping.
            min_size (int): Minimum side length of valid boxes.
            max_candidates (int): Maximum number of contours to process.

        Returns:
            tuple:
                - boxes (np.ndarray): Array of boxes, each of shape (4, 2).
                - scores (list): List of corresponding scores.
        """

        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = outs[1] if len(outs) == 3 else outs[0]

        num_contours = min(len(contours), max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, short_side_length = self._get_mini_boxes(contour)
            if short_side_length < min_size:
                continue
            points = np.array(points)
            score = self._get_box_score(prediction, points.reshape(-1, 2))
            if box_threshold > score:
                continue
            box = self._unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, short_side_length = self._get_mini_boxes(box)
            if short_side_length < min_size + 2:
                continue

            box = np.array(box)
            for i in range(box.shape[0]):
                box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
                box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def get_image_size(
        self,
        image: torch.Tensor,
        limit_side_len: int,
        limit_type: str,
        max_side_limit: int,
    ):
        """
        Computes the target size for resizing an image while preserving aspect ratio.

        Args:
            image (torch.Tensor): Input image.
            limit_side_len (int): Maximum or minimum side length.
            limit_type (str): Resizing strategy: "max", "min", or "resize_long".
            max_side_limit (int): Maximum allowed side length.

        Returns:
            tuple:
                - SizeDict: Target size.
                - torch.Tensor: Original size.
        """
        _, height, width = image.shape
        height, width = int(height), int(width)

        if limit_type == "max":
            if max(height, width) > limit_side_len:
                ratio = float(limit_side_len) / max(height, width)
            else:
                ratio = 1.0
        elif limit_type == "min":
            if min(height, width) < limit_side_len:
                ratio = float(limit_side_len) / min(height, width)
            else:
                ratio = 1.0
        elif limit_type == "resize_long":
            ratio = float(limit_side_len) / max(height, width)
        else:
            raise Exception(f"PPOCRV5ServerDetImageProcessorFast does not support limit type: {limit_type}")

        resize_height = int(height * ratio)
        resize_width = int(width * ratio)

        if max_side_limit is not None and max(resize_height, resize_width) > max_side_limit:
            ratio = float(max_side_limit) / max(resize_height, resize_width)
            resize_height = int(resize_height * ratio)
            resize_width = int(resize_width * ratio)

        resize_height = max(int(round(resize_height / 32) * 32), 32)
        resize_width = max(int(round(resize_width / 32) * 32), 32)

        if resize_height == height and resize_width == width:
            return SizeDict(height=resize_height, width=resize_width), torch.tensor(
                [height, width], dtype=torch.float32, device=image.device
            )

        if resize_width <= 0 or resize_height <= 0:
            return None, (None, None)

        return SizeDict(height=resize_height, width=resize_width), torch.tensor(
            [height, width], dtype=torch.float32, device=image.device
        )

    def post_process_object_detection(
        self,
        predictions,
        threshold: float = 0.3,
        target_sizes: list[tuple[int, int]] | torch.Tensor | None = None,
        box_threshold: float = 0.6,
        max_candidates: int = 1000,
        min_size: int = 3,
        unclip_ratio: float = 1.5,
    ):
        """
        Converts model outputs into detected text boxes in corners format (xmin, ymin, xmax, ymax).

        Args:
            predictions: Model outputs with `logits` attribute (probability maps of shape `(batch_size, 1, H, W)`).
            threshold (float): Binarization threshold.
            target_sizes: Original image sizes (height, width) per image.
            box_threshold (float): Box score threshold.
            max_candidates (int): Maximum number of boxes.
            min_size (int): Minimum box size.
            unclip_ratio (float): Expansion ratio.

        Returns:
            list[dict]: List of detection results per image. Each dict contains:
                - "boxes": `torch.Tensor` of shape `(N, 4)` in corners format (xmin, ymin, xmax, ymax)
                - "scores": `torch.Tensor` of shape `(N,)`
                - "labels": `torch.Tensor` of shape `(N,)` (class id 0 for text)
        """
        if target_sizes is None:
            raise ValueError("target_sizes must be provided for post_process_object_detection")

        requires_backends(self, ["torch", "cv2"])
        device = predictions.logits.device
        results = []
        for prediction, size in zip(predictions.logits, target_sizes):
            prediction = prediction[0, :, :].cpu().detach().numpy()
            size = size.cpu().detach().numpy()
            src_height, src_width = size
            mask = prediction > threshold
            boxes_polygon, scores = self._boxes_from_bitmap(
                prediction, mask, src_width, src_height, box_threshold, unclip_ratio, min_size, max_candidates
            )

            # Convert polygon (N, 4, 2) to axis-aligned corners [xmin, ymin, xmax, ymax]
            if len(boxes_polygon) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                boxes = np.array(
                    [[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in boxes_polygon],
                    dtype=np.float32,
                )

            results.append(
                {
                    "boxes": torch.from_numpy(boxes).to(device),
                    "scores": torch.tensor(scores, dtype=torch.float32, device=device),
                    "labels": torch.zeros(len(scores), dtype=torch.long, device=device),  # Single class: text
                }
            )
        return results


class PPOCRV5ServerDetIntraclassBlock(nn.Module):
    """
    Intra-Class Relationship Block. It uses multi-scale convolution (7x7, 5x5, 3x3)
    and asymmetric kernels (e.g., 7x1, 1x7) to capture long-range spatial dependencies
    within text regions.

    Args:
        intraclass_block_config (`dict`, *optional*):
            Configuration dictionary specifying kernel sizes and paddings for all sub-layers.
        in_channels (`int`, *optional*, defaults to 96):
            Number of channels in the input feature map.
        reduce_factor (`int`, *optional*, defaults to 4):
            The factor used to compress channels for efficiency during relationship modeling.
    """

    def __init__(
        self,
        intraclass_block_config: dict | None = None,
        in_channels: int = 96,
        reduce_factor: int = 4,
    ):
        super().__init__()

        reduced_channels = in_channels // reduce_factor

        self.conv_reduce_channel = nn.Conv2d(in_channels, reduced_channels, *intraclass_block_config["reduce_channel"])

        self.vertical_long_to_small_conv_longratio = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["v_layer_7x1"]
        )
        self.vertical_long_to_small_conv_midratio = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["v_layer_5x1"]
        )
        self.vertical_long_to_small_conv_shortratio = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["v_layer_3x1"]
        )

        self.horizontal_small_to_long_conv_longratior = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["q_layer_1x7"]
        )
        self.horizontal_small_to_long_conv_midratio = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["q_layer_1x5"]
        )
        self.horizontal_small_to_long_conv_shortratio = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["q_layer_1x3"]
        )

        self.symmetric_conv_long_longratior = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["c_layer_7x7"]
        )
        self.symmetric_conv_long_midratio = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["c_layer_5x5"]
        )
        self.symmetric_conv_long_shortratio = nn.Conv2d(
            reduced_channels, reduced_channels, *intraclass_block_config["c_layer_3x3"]
        )

        self.layer = PPOCRV5ServerDetConvBNLayer(
            in_channels=reduced_channels,
            out_channels=in_channels,
            kernel_size=intraclass_block_config["return_channel"][0],
            stride=intraclass_block_config["return_channel"][1],
            padding=intraclass_block_config["return_channel"][2],
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv_reduce_channel(hidden_states)

        hidden_states = (
            self.symmetric_conv_long_longratior(hidden_states)
            + self.vertical_long_to_small_conv_longratio(hidden_states)
            + self.horizontal_small_to_long_conv_longratior(hidden_states)
        )
        hidden_states = (
            self.symmetric_conv_long_midratio(hidden_states)
            + self.vertical_long_to_small_conv_midratio(hidden_states)
            + self.horizontal_small_to_long_conv_midratio(hidden_states)
        )
        hidden_states = (
            self.symmetric_conv_long_shortratio(hidden_states)
            + self.vertical_long_to_small_conv_shortratio(hidden_states)
            + self.horizontal_small_to_long_conv_shortratio(hidden_states)
        )

        hidden_states = self.layer(hidden_states)

        return residual + hidden_states


class PPOCRV5ServerDetNeck(nn.Module):
    """
    Large Kernel Path Aggregation Network (Neck) for PPOCRV5 Server Detection.
    Fuses multi-scale features from backbone stages (stage 2 to stage 5) via top-down and bottom-up paths,
    enhanced with large kernel convolution for better spatial dependency modeling.
    """

    def __init__(self, config):
        super().__init__()
        self.interpolate_mode = config.interpolate_mode

        self.convolution_class = nn.Conv2d

        self.input_channel_adjustment_convolution = nn.ModuleList()
        self.input_feature_projection_convolution = nn.ModuleList()
        self.path_aggregation_head_convolution = nn.ModuleList()
        self.path_aggregation_lateral_convolution = nn.ModuleList()

        backbone_stage_output_channels = config.backbone_config.stage_out_channels

        for backbone_stage_index in range(len(backbone_stage_output_channels)):
            channel_adjustment_convolution = nn.Conv2d(
                in_channels=backbone_stage_output_channels[backbone_stage_index],
                out_channels=config.neck_out_channels,
                kernel_size=1,
                bias=False,
            )
            self.input_channel_adjustment_convolution.append(channel_adjustment_convolution)

            feature_projection_convolution = self.convolution_class(
                in_channels=config.neck_out_channels,
                out_channels=config.neck_out_channels // 4,
                kernel_size=9,
                padding=4,
                bias=False,
            )
            self.input_feature_projection_convolution.append(feature_projection_convolution)

            if backbone_stage_index > 0:
                pan_head_convolution = nn.Conv2d(
                    in_channels=config.neck_out_channels // 4,
                    out_channels=config.neck_out_channels // 4,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                )
                self.path_aggregation_head_convolution.append(pan_head_convolution)

            pan_lateral_convolution = self.convolution_class(
                in_channels=config.neck_out_channels // 4,
                out_channels=config.neck_out_channels // 4,
                kernel_size=9,
                padding=4,
                bias=False,
            )
            self.path_aggregation_lateral_convolution.append(pan_lateral_convolution)

        self.intraclass_blocks = nn.ModuleList()
        for _ in range(config.intraclass_block_number):
            self.intraclass_blocks.append(
                PPOCRV5ServerDetIntraclassBlock(
                    config.intraclass_block_config, config.neck_out_channels // 4, reduce_factor=config.reduce_factor
                )
            )

    def forward(self, backbone_stage_feature_maps: list[torch.Tensor], **kwargs) -> torch.Tensor:
        channel_adjusted = [
            self.input_channel_adjustment_convolution[i](feature)
            for i, feature in enumerate(backbone_stage_feature_maps)
        ]

        top_down = [None] * 4
        top_down[3] = channel_adjusted[3]
        for i in range(2, -1, -1):
            top_down[i] = channel_adjusted[i] + F.interpolate(
                top_down[i + 1], scale_factor=2, mode=self.interpolate_mode
            )

        projected = [
            self.input_feature_projection_convolution[i](top_down[i] if i < 3 else channel_adjusted[3])
            for i in range(4)
        ]

        bottom_up = [None] * 4
        bottom_up[0] = projected[0]
        for i in range(1, 4):
            bottom_up[i] = projected[i] + self.path_aggregation_head_convolution[i - 1](bottom_up[i - 1])

        lateral_refined = [
            self.path_aggregation_lateral_convolution[i](projected[0] if i == 0 else bottom_up[i]) for i in range(4)
        ]

        intraclass_refined = [block(feature) for block, feature in zip(self.intraclass_blocks, lateral_refined)]

        scale_factors = [1, 2, 4, 8]
        upsampled = [
            F.interpolate(feature, scale_factor=scale_factor, mode=self.interpolate_mode)
            if scale_factor > 1
            else feature
            for feature, scale_factor in zip(intraclass_refined, scale_factors)
        ]

        return torch.cat(upsampled[::-1], dim=1)


class PPOCRV5ServerDetConvBNLayer(nn.Module):
    """
    A basic wrapper for Convolution-BatchNorm-Activation, typically used for head components.

    Args:
        in_channels (`int`): Input channel count.
        out_channels (`int`): Output channel count.
        kernel_size (`int`): Size of the kernel.
        stride (`int`): Stride for the convolution.
        padding (`Union[int, str]`): Padding value or strategy.
        groups (`int`, *optional*, defaults to 1): Grouped convolution parameter.
        activation (`str`, *optional*): Type of activation ("relu" or "hardswish").
        bias (`bool`, *optional*, defaults to `True`): Whether to use bias term.
        convolution_transpose (`bool`, *optional*, defaults to `False`): Whether to use a transposed convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 1,
        groups: int = 1,
        activation: str = "relu",
        bias: bool = False,
        convolution_transpose: bool = False,
    ):
        super().__init__()
        if convolution_transpose:
            self.convolution = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
        else:
            self.convolution = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            )

        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.Identity() if activation is None else ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class PPOCRV5ServerDetHead(nn.Module):
    """
    Standard segmentation head for generating probability maps. It uses transposed
    convolution to upsample the feature map back to the original image size.

    Args:
        in_channels (`int`):
            Number of input channels from the neck (e.g., PPOCRV5ServerDetNeck).
        kernel_list (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            List of kernel sizes for the sequence of [Conv2d, ConvTranspose2d, ConvTranspose2d].
    """

    def __init__(
        self,
        in_channels: int,
        kernel_list: list[int] = [3, 2, 2],
    ):
        super().__init__()

        self.layer1 = PPOCRV5ServerDetConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
        )
        self.layer2 = PPOCRV5ServerDetConvBNLayer(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            convolution_transpose=True,
        )

        self.convolution3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.layer1(hidden_states)
        hidden_states = self.layer2(hidden_states)
        feature = hidden_states
        hidden_states = self.convolution3(hidden_states)
        hidden_states = torch.sigmoid(hidden_states)
        return hidden_states, feature


class PPOCRV5ServerDetLocalModule(nn.Module):
    """
    Local Refinement Module that refines the initial probability map by
    concatenating it with higher-resolution features.

    Args:
        in_channels (`int`): Number of channels in the feature map `hidden_states`.
        out_channels (`int`): Hidden channel size for the refinement layers.
        hidden_act (`str`): Activation function name.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_act: str):
        super().__init__()
        self.convolution_backbone = PPOCRV5ServerDetConvBNLayer(
            in_channels=in_channels + 1,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=hidden_act,
        )
        self.convolution_final = nn.Conv2d(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, hidden_states: torch.Tensor, init_map: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.cat([init_map, hidden_states], dim=1)
        # last Conv
        hidden_states = self.convolution_backbone(hidden_states)
        hidden_states = self.convolution_final(hidden_states)
        return hidden_states


class PPOCRV5ServerDetPFHeadLocal(nn.Module):
    """
    PPOCRV5ServerDetPFHeadLocal implements the Progressive Fusion Head with Local refinement,
    the core detection head of PP-OCRv5.
    """

    def __init__(self, config: PPOCRV5ServerDetConfig):
        super().__init__()
        self.binarize_head = PPOCRV5ServerDetHead(in_channels=config.neck_out_channels, kernel_list=config.kernel_list)
        self.upsample_convolution = nn.Upsample(scale_factor=config.scale_factor, mode=config.interpolate_mode)

        self.local_refinement_module = PPOCRV5ServerDetLocalModule(
            config.neck_out_channels // 4, config.neck_out_channels // 4, config.hidden_act
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, feature = self.binarize_head(hidden_states)
        residual = hidden_states
        feature = self.upsample_convolution(feature)
        hidden_states = self.local_refinement_module(feature, hidden_states)
        hidden_states = torch.sigmoid(hidden_states)

        return 0.5 * (residual + hidden_states)


class PPOCRV5ServerDetPreTrainedModel(PreTrainedModel):
    """
    Base class for all PPOCRV5 Server Det pre-trained models. Handles model initialization,
    configuration, and loading of pre-trained weights, following the Transformers library conventions.
    """

    config: PPOCRV5ServerDetConfig
    base_model_prefix = "pp_ocrv5_server_det"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _can_compile_fullgraph = True


@auto_docstring(
    custom_intro="""
    Core PPOCRV5 Server Det model.
    Integration of HGNetV2 (Backbone), PPOCRV5ServerDetNeck (Neck), and PPOCRV5ServerDetPFHeadLocal (Head).
    """
)
class PPOCRV5ServerDetModel(PPOCRV5ServerDetPreTrainedModel):
    def __init__(self, config: PPOCRV5ServerDetConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = PPOCRV5ServerDetNeck(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        backbone_outputs = self.backbone(pixel_values, **kwargs)
        hidden_state = backbone_outputs.feature_maps
        hidden_state = self.neck(hidden_state)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=backbone_outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    Output class for PPOCRV5ServerDetForObjectDetection.
    """
)
@dataclass
class PPOCRV5ServerDetForObjectDetectionOutput(BaseModelOutputWithNoAttention):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, neck_out_channels, height, width)`, *optional*):
        Fused feature maps from the neck (LKPAN) before the detection head.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Tuple of feature maps from backbone stages when `output_hidden_states=True`.
    logits (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`):
        The predicted text mask (binary probability maps). Values in [0, 1] indicate probability of text
        presence at each pixel. Use [`PPOCRV5ServerDetImageProcessorFast.post_process_object_detection`]
        to convert to bounding boxes.
    """

    logits: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    PPOCRV5 Server Det model for object (text) detection tasks. Wraps the core PPOCRV5ServerDetModel
    and returns outputs compatible with the Transformers object detection API.
    """
)
class PPOCRV5ServerDetForObjectDetection(PPOCRV5ServerDetPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPOCRV5ServerDetConfig):
        super().__init__(config)
        self.model = PPOCRV5ServerDetModel(config)
        self.head = PPOCRV5ServerDetPFHeadLocal(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | PPOCRV5ServerDetForObjectDetectionOutput:
        outputs = self.model(pixel_values, **kwargs)
        logits = self.head(outputs.last_hidden_state)

        return PPOCRV5ServerDetForObjectDetectionOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "PPOCRV5ServerDetForObjectDetection",
    "PPOCRV5ServerDetImageProcessorFast",
    "PPOCRV5ServerDetConfig",
    "PPOCRV5ServerDetModel",
    "PPOCRV5ServerDetPreTrainedModel",
]
