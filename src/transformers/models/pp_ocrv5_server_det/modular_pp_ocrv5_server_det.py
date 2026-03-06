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

from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    SizeDict,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    auto_docstring,
    can_return_tuple,
    is_cv2_available,
    logging,
)
from ...utils.generic import TensorType
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig


if is_cv2_available():
    import cv2


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    """
)
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
        intraclblock_config (`dict`, *optional*, defaults to `None`):
            Configuration for the Intra-Class Block modules, if any, used for enhancing feature representation.
        mode (`str`, *optional*, defaults to `"large"`):
            The model scale mode, such as `"large"` or `"small"`, affecting the depth and width of the network.
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
        intraclblock_config: dict | None = None,
        mode: str = "large",
        scale_factor: int = 2,
        hidden_act: str = "relu",
        kernel_list: list[int] = [3, 2, 2],
        **kwargs,
    ) -> None:
        self.mode = mode
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
        self.intraclblock_config = intraclblock_config

        # ---- head ----
        self.scale_factor = scale_factor
        self.hidden_act = hidden_act
        self.kernel_list = kernel_list

        super().__init__(**kwargs)


def polygon_area(box: np.ndarray) -> float:
    x = box[:, 0]
    y = box[:, 1]

    return 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def polygon_arc_length(box: np.ndarray, closed: bool = True) -> float:
    diffs = box[1:] - box[:-1]
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    total_length = np.sum(segment_lengths)

    if closed:
        last_to_first = box[0] - box[-1]
        total_length += np.sqrt(np.sum(last_to_first**2))

    return total_length


def convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.unique(points.astype(np.float64), axis=0)
    if len(points) <= 1:
        return points

    pivot_idx = np.lexsort((points[:, 0], points[:, 1]))[0]
    pivot = points[pivot_idx]

    def polar_angle(p):
        return np.arctan2(p[1] - pivot[1], p[0] - pivot[0])

    sorted_points = sorted(points, key=polar_angle)

    hull = []
    for p in sorted_points:
        while len(hull) >= 2:
            a = hull[-2]
            b = hull[-1]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross <= 1e-8:
                hull.pop()
            else:
                break
        hull.append(p)

    return np.array(hull, dtype=np.float64)


def min_area_rect(contour: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    contour = contour.reshape(-1, 2).astype(np.float64)
    hull = convex_hull(contour)
    n = len(hull)

    if n == 1:
        return (float(hull[0][0]), float(hull[0][1])), (0.0, 0.0), 0.0
    if n == 2:
        dx, dy = hull[1] - hull[0]
        length = np.hypot(dx, dy)
        center = ((hull[0][0] + hull[1][0]) / 2, (hull[0][1] + hull[1][1]) / 2)
        angle = np.arctan2(dy, dx) * 180 / np.pi
        angle = angle - 90 if angle >= 0 else angle
        return (float(center[0]), float(center[1])), (float(length), 0.0), float(angle)

    min_area = float("inf")
    best_rect = None
    j = 1

    for i in range(n):
        p1, p2 = hull[i], hull[(i + 1) % n]
        edge = p2 - p1
        edge_len = np.hypot(edge[0], edge[1])

        if edge_len < 1e-8:
            continue

        edge_normal = np.array([-edge[1], edge[0]]) / edge_len

        while True:
            curr_dot = np.dot(hull[j] - p1, edge_normal)
            next_dot = np.dot(hull[(j + 1) % n] - p1, edge_normal)
            if next_dot > curr_dot + 1e-8:
                j = (j + 1) % n
            else:
                break

        proj = np.dot(hull - p1, edge) / edge_len
        min_proj, max_proj = np.min(proj), np.max(proj)
        width = max_proj - min_proj

        proj_n = np.dot(hull - p1, edge_normal)
        min_proj_n, max_proj_n = np.min(proj_n), np.max(proj_n)
        height = max_proj_n - min_proj_n

        area = width * height
        if area < min_area - 1e-8:
            min_area = area
            center_x = (
                p1[0]
                + edge[0] * (min_proj + max_proj) / (2 * edge_len)
                + edge_normal[0] * (min_proj_n + max_proj_n) / 2
            )
            center_y = (
                p1[1]
                + edge[1] * (min_proj + max_proj) / (2 * edge_len)
                + edge_normal[1] * (min_proj_n + max_proj_n) / 2
            )
            center = (float(center_x), float(center_y))

            angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
            if angle >= 90:
                angle -= 180
            elif angle >= 0:
                angle -= 90
            angle = float(angle)

            if width < height:
                width, height = height, width
                angle -= 90

            best_rect = (center, (float(width), float(height)), angle)

    return best_rect if best_rect else ((0.0, 0.0), (0.0, 0.0), 0.0)


def box_points(rect: tuple) -> np.ndarray:
    center, size, angle = rect
    cx, cy = center
    width, height = size
    angle_rad = angle * np.pi / 180.0

    half_w = width / 2.0
    half_h = height / 2.0

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    pts = [
        (cx - half_w * cos_a - half_h * sin_a, cy - half_w * sin_a + half_h * cos_a),
        (cx + half_w * cos_a - half_h * sin_a, cy + half_w * sin_a + half_h * cos_a),
        (cx + half_w * cos_a + half_h * sin_a, cy + half_w * sin_a - half_h * cos_a),
        (cx - half_w * cos_a + half_h * sin_a, cy - half_w * sin_a - half_h * cos_a),
    ]
    return np.array(pts, dtype=np.float32)


def masked_mean(roi: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(np.uint8)
    roi = roi.astype(np.float64)

    valid_pixels = roi[mask == 1]

    if len(valid_pixels) == 0:
        return 0.0

    return float(np.mean(valid_pixels))


def unclip(box: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """
    Expands (dilates) a detected text bounding box to recover the full text region.
    Args:
        box (np.ndarray): Input contour of shape (N, 2), where N is the number of points.
        unclip_ratio (float): Expansion ratio, typically greater than 1.0.
    Returns:
        np.ndarray: Expanded contour of shape (M, 2).
    """
    box = np.array(box).reshape(-1, 2)

    area = polygon_area(box)
    length = polygon_arc_length(box, True)
    if length == 0:
        return box
    distance = area * unclip_ratio / length

    points = np.concatenate([box, box[0:1]], axis=0)
    new_points = []

    for i in range(len(box)):
        p1 = points[i]
        p0 = points[i - 1]
        p2 = points[i + 1]

        def get_normal(pa, pb):
            direction = pb - pa
            norm = np.linalg.norm(direction)
            if norm == 0:
                return np.array([0, 0])
            return np.array([direction[1], -direction[0]]) / norm

        v1 = get_normal(p0, p1)
        v2 = get_normal(p1, p2)
        combined_v = v1 + v2
        cos_theta = np.dot(v1, v2)

        denom = 1 + cos_theta
        if denom < 1e-6:
            scale = distance
        else:
            scale = distance * np.sqrt(2 / denom)

        new_point = p1 + combined_v * (scale / (np.linalg.norm(combined_v) + 1e-6))
        new_points.append(new_point)

    return np.array(new_points, dtype=np.float32)


def get_mini_boxes(contour: np.ndarray) -> tuple[list[list[float]], float]:
    """
    Computes the minimum-area bounding rectangle for a given contour and returns
    its four corners in a consistent order (top-left, bottom-left, bottom-right, top-right).

    Args:
        contour (np.ndarray): Input contour of shape (N, 1, 2).

    Returns:
        tuple:
            - box (list): List of four corner points in order.
            - sside (float): Length of the shorter side of the bounding rectangle.
    """
    bounding_box = min_area_rect(contour)
    points = sorted(box_points(bounding_box), key=lambda x: x[0])

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


def get_box_score(bitmap: np.ndarray, _box: np.ndarray) -> float:
    """
    Computes the mean score of a bounding box region in the prediction map using
    a fast approach with axis-aligned bounding boxes.

    Args:
        bitmap (np.ndarray): Binary or float prediction map of shape (H, W).
        _box (np.ndarray): Bounding box polygon of shape (N, 2).

    Returns:
        float: Mean score within the bounding box region.
    """
    height, width = bitmap.shape[:2]
    box = _box.copy()
    xmin = max(0, min(math.floor(box[:, 0].min()), width - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), width - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), height - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), height - 1))

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return masked_mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)


def boxes_from_bitmap(
    pred: np.ndarray,
    _bitmap: np.ndarray,
    dest_width: int,
    dest_height: int,
    box_thresh: float,
    unclip_ratio: float,
    min_size: int,
    max_candidates: int,
) -> tuple[np.ndarray, list[float]]:
    """
    Extracts axis-aligned or rotated bounding boxes from a binary segmentation map.

    Args:
        pred (np.ndarray): Raw prediction map of shape (H, W).
        _bitmap (np.ndarray): Binarized segmentation map of shape (H, W).
        dest_width (int): Original image width for scaling back.
        dest_height (int): Original image height for scaling back.
        box_thresh (float): Score threshold for filtering low-confidence boxes.
        unclip_ratio (float): Expansion ratio for contour unclipping.
        min_size (int): Minimum side length of valid boxes.
        max_candidates (int): Maximum number of contours to process.

    Returns:
        tuple:
            - boxes (np.ndarray): Array of boxes, each of shape (4, 2).
            - scores (list): List of corresponding scores.
    """

    bitmap = _bitmap
    height, width = bitmap.shape
    width_scale = dest_width / width
    height_scale = dest_height / height

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 3:
        _, contours, _ = outs[0], outs[1], outs[2]
    elif len(outs) == 2:
        contours, _ = outs[0], outs[1]

    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        score = get_box_score(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        box = unclip(points, unclip_ratio).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        if sside < min_size + 2:
            continue

        box = np.array(box)
        for i in range(box.shape[0]):
            box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
            box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

        boxes.append(box.astype(np.int16))
        scores.append(score)
    return np.array(boxes, dtype=np.int16), scores


def process(
    pred: np.ndarray,
    size: np.ndarray,
    threshold: float,
    box_thresh: float,
    unclip_ratio: float,
    min_size: int,
    max_candidates: int,
) -> tuple[list[np.ndarray] | np.ndarray, list[float]]:
    """
    Main post-processing function to convert model predictions into text boxes.

    Args:
        pred (torch.Tensor): Model output of shape (1, H, W).
        size (torch.Tensor): Original image size (height, width).
        threshold (float): Threshold for binarizing the prediction map.
        box_thresh (float): Score threshold for filtering boxes.
        unclip_ratio (float): Expansion ratio for unclipping.
        min_size (int): Minimum side length of valid boxes.
        max_candidates (int): Maximum number of boxes to extract.

    Returns:
        tuple:
            - boxes (list or np.ndarray): Extracted text boxes.
            - scores (list): Corresponding confidence scores.
    """
    src_h, src_w = size
    mask = pred > threshold
    boxes, scores = boxes_from_bitmap(pred, mask, src_w, src_h, box_thresh, unclip_ratio, min_size, max_candidates)
    return boxes, scores


class PPOCRV5ServerDetImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    limit_side_len (`int`, *optional*, defaults to `960`):
        todo
    limit_type (`str`, *optional*, defaults to `max`):
        todo
    max_side_limit (`int`, *optional* defaults to `4000`):
        todo
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

    def __init__(self, **kwargs: Unpack[PPOCRV5ServerDetImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)

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
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        data = {}
        resize_images, target_sizes = [], []
        if do_resize:
            for image in images:
                size, shape = self.get_image_size(image, self.limit_side_len, self.limit_type, self.max_side_limit)
                image = self.resize(image, size=size, interpolation=interpolation)
                resize_images.append(image)
                target_sizes.append(shape)
            images = resize_images

        processed_images = []
        for image in images:
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            processed_images.append(image)

        images = processed_images
        images = [image[[2, 1, 0], :, :] for image in images]

        data.update({"pixel_values": torch.stack(images, dim=0), "target_sizes": target_sizes})
        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)

        return encoded_inputs

    def post_process_object_detection(
        self,
        preds,
        threshold: float = 0.3,
        target_sizes: list[tuple[int, int]] | torch.Tensor | None = None,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        min_size: int = 3,
        unclip_ratio: float = 1.5,
    ):
        """
        Converts model outputs into detected text boxes.

        Args:
            preds (torch.Tensor): Model outputs.
            threshold (float):Binarization threshold.
            target_sizes (TensorType or list[tuple]): Original image sizes.
            box_thresh (float): Box score threshold.
            max_candidates (int): Maximum number of boxes.
            min_size (int): Minimum box size.
            unclip_ratio (float): Expansion ratio.

        Returns:
            list[dict]: List of detection results.
        """

        results = []
        for pred, size in zip(preds.logits, target_sizes):
            box, score = process(
                pred=pred[0, :, :].cpu().detach().numpy(),
                size=size.cpu().detach().numpy(),
                threshold=threshold,
                box_thresh=box_thresh,
                unclip_ratio=unclip_ratio,
                min_size=min_size,
                max_candidates=max_candidates,
            )

            results.append(
                {
                    "boxes": box,
                    "scores": score,
                }
            )
        return results

    def get_image_size(
        self,
        image: np.ndarray,
        limit_side_len: int,
        limit_type: str,
        max_side_limit: int = 4000,
    ) -> tuple[dict, np.ndarray]:
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
        limit_side_len = limit_side_len or self.limit_side_len
        limit_type = limit_type or self.limit_type
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
            raise Exception(f"not support limit type: {limit_type}")

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
                [height, width], dtype=torch.float32
            )

        if resize_width <= 0 or resize_height <= 0:
            return None, (None, None)

        return SizeDict(height=resize_height, width=resize_width), torch.tensor([height, width], dtype=torch.float32)


class PPOCRV5ServerDetDSConv(nn.Module):
    """
    Depthwise Separable Convolution block with an expanded intermediate state and residual connection.
    This block mimics the inverted residual structure to reduce computation while maintaining capacity.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        kernel_size (`int`):
            Size of the convolving kernel for the depthwise step.
        padding (`Union[int, str]`):
            Padding for the depthwise convolution.
        stride (`int`, *optional*, defaults to 1):
            Stride for the spatial downsampling.
        groups (`int`, *optional*):
            Number of blocked connections. Defaults to `in_channels` for depthwise convolution.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation type, supports `"relu"` or `"hardswish"`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | str,
        stride: int = 1,
        groups: int | None = None,
        hidden_act: str = "relu",
        **kwargs,
    ):
        super().__init__()
        if groups is None:
            groups = in_channels

        if self.hidden_act == "relu":
            self.act = nn.ReLU()
        elif self.hidden_act == "hardswish":
            self.act = nn.Hardswish()
        else:
            print(f"The activation function({self.hidden_act}) is selected incorrectly.")
            exit()

        self.hidden_act = hidden_act
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(num_features=in_channels, momentum=0.9)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(num_features=int(in_channels * 4))

        self.conv3 = nn.Conv2d(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetDSConv.

        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, height, width)`):
                The input feature map.

        Returns:
            `torch.FloatTensor`: Output feature map of shape `(batch_size, out_channels, out_height, out_width)`.
        """
        identity = hidden_state
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.bn1(hidden_state)

        hidden_state = self.conv2(hidden_state)
        hidden_state = self.bn2(hidden_state)
        hidden_state = self.act(hidden_state)

        hidden_state = self.conv3(hidden_state)
        if self._c[0] != self._c[1]:
            hidden_state = hidden_state + self.conv_end(identity)
        return hidden_state


class PPOCRV5ServerDetIntraCLBlock(nn.Module):
    """
    Intra-Class Relationship Block. It uses multi-scale convolutions (7x7, 5x5, 3x3)
    and asymmetric kernels (e.g., 7x1, 1x7) to capture long-range spatial dependencies
    within text regions.

    Args:
        intraclblock_config (`dict`, *optional*):
            Configuration dictionary specifying kernel sizes and paddings for all sub-layers.
        in_channels (`int`, *optional*, defaults to 96):
            Number of channels in the input feature map.
        reduce_factor (`int`, *optional*, defaults to 4):
            The factor used to compress channels for efficiency during relationship modeling.
    """

    def __init__(self, intraclblock_config: dict | None = None, in_channels: int = 96, reduce_factor: int = 4):
        super().__init__()

        reduced_ch = in_channels // reduce_factor

        self.conv1x1_reduce_channel = nn.Conv2d(in_channels, reduced_ch, *intraclblock_config["reduce_channel"])
        self.conv1x1_return_channel = nn.Conv2d(reduced_ch, in_channels, *intraclblock_config["return_channel"])

        self.v_layer_7x1 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["v_layer_7x1"])
        self.v_layer_5x1 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["v_layer_5x1"])
        self.v_layer_3x1 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["v_layer_3x1"])

        self.q_layer_1x7 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["q_layer_1x7"])
        self.q_layer_1x5 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["q_layer_1x5"])
        self.q_layer_1x3 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["q_layer_1x3"])

        self.c_layer_7x7 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["c_layer_7x7"])
        self.c_layer_5x5 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["c_layer_5x5"])
        self.c_layer_3x3 = nn.Conv2d(reduced_ch, reduced_ch, *intraclblock_config["c_layer_3x3"])

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetIntraCLBlock.

        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, height, width)`):
                The input feature map from PPOCRV5ServerDetLKPAN stages.

        Returns:
            `torch.FloatTensor`: Refined feature map with the same shape as input,
                enhanced by spatial relationship modeling.
        """
        identity = hidden_state
        hidden_state = self.conv1x1_reduce_channel(hidden_state)

        hidden_state = self.c_layer_7x7(hidden_state) + self.v_layer_7x1(hidden_state) + self.q_layer_1x7(hidden_state)
        hidden_state = self.c_layer_5x5(hidden_state) + self.v_layer_5x1(hidden_state) + self.q_layer_1x5(hidden_state)
        hidden_state = self.c_layer_3x3(hidden_state) + self.v_layer_3x1(hidden_state) + self.q_layer_1x3(hidden_state)

        hidden_state = self.conv1x1_return_channel(hidden_state)
        hidden_state = self.bn(hidden_state)
        hidden_state = self.relu(hidden_state)

        return identity + hidden_state


class PPOCRV5ServerDetLKPAN(PreTrainedModel):
    """
    Large Kernel Path Aggregation Network (Neck).
    It fuses features from multiple backbone stages (C2-C5) using a combination of
    top-down and bottom-up paths, enhanced by large kernel convolutions.

    Args:
        config (`PPOCRV5ServerDetConfig`):
            Configuration object containing `neck_out_channels`, `mode`, and `interpolate_mode`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.interpolate_mode = config.interpolate_mode

        if config.mode == "lite":
            p_layer = PPOCRV5ServerDetDSConv
        elif config.mode == "large":
            p_layer = nn.Conv2d
        else:
            raise ValueError(f"mode can only be one of ['lite', 'large'], but received {config.mode}")

        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()

        in_channels = config.backbone_config.stage_out_channels
        for i in range(len(in_channels)):
            conv = nn.Conv2d(
                in_channels=in_channels[i], out_channels=config.neck_out_channels, kernel_size=1, bias=False
            )

            self.ins_conv.append(conv)

            inp_conv = p_layer(
                in_channels=config.neck_out_channels,
                out_channels=config.neck_out_channels // 4,
                kernel_size=9,
                padding=4,
                bias=False,
            )

            self.inp_conv.append(inp_conv)

            if i > 0:
                pan_head = nn.Conv2d(
                    in_channels=config.neck_out_channels // 4,
                    out_channels=config.neck_out_channels // 4,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                )
                self.pan_head_conv.append(pan_head)

            pan_lat = p_layer(
                in_channels=config.neck_out_channels // 4,
                out_channels=config.neck_out_channels // 4,
                kernel_size=9,
                padding=4,
                bias=False,
            )
            self.pan_lat_conv.append(pan_lat)

        self.incl1 = PPOCRV5ServerDetIntraCLBlock(
            config.intraclblock_config, config.neck_out_channels // 4, reduce_factor=config.reduce_factor
        )
        self.incl2 = PPOCRV5ServerDetIntraCLBlock(
            config.intraclblock_config, config.neck_out_channels // 4, reduce_factor=config.reduce_factor
        )
        self.incl3 = PPOCRV5ServerDetIntraCLBlock(
            config.intraclblock_config, config.neck_out_channels // 4, reduce_factor=config.reduce_factor
        )
        self.incl4 = PPOCRV5ServerDetIntraCLBlock(
            config.intraclblock_config, config.neck_out_channels // 4, reduce_factor=config.reduce_factor
        )

    @capture_outputs()
    def forward(self, hidden_state: list[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetLKPAN.

        Args:
            hidden_state (`list` of `torch.FloatTensor`):
                Multi-scale features `[c2, c3, c4, c5]` from the backbone.

        Returns:
            `torch.FloatTensor`:
                Fused feature map of shape `(batch_size, neck_out_channels, height/4, width/4)`.
                This tensor is a concatenation of multi-scale refined features, ready for the head.
        """
        c2, c3, c4, c5 = hidden_state

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode=self.interpolate_mode)
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode=self.interpolate_mode)
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode=self.interpolate_mode)

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        p5 = self.incl4(p5)
        p4 = self.incl3(p4)
        p3 = self.incl2(p3)
        p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode=self.interpolate_mode)
        p4 = F.interpolate(p4, scale_factor=4, mode=self.interpolate_mode)
        p3 = F.interpolate(p3, scale_factor=2, mode=self.interpolate_mode)

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return BaseModelOutputWithNoAttention(last_hidden_state=fuse)


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
        if_act (`bool`, *optional*, defaults to `True`): Whether to apply activation.
        hidden_act (`str`, *optional*): Type of activation ("relu" or "hardswish").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | str,
        groups: int = 1,
        if_act: bool = True,
        hidden_act: str | None = None,
    ):
        super().__init__()
        self.if_act = if_act
        self.hidden_act = hidden_act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetHead.

        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, height, width)`):
                Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor of shape `(batch_size, out_channels, out_height, out_width)`.
        """
        hidden_state = self.conv(hidden_state)
        hidden_state = self.bn(hidden_state)
        if self.if_act:
            if self.hidden_act == "relu":
                hidden_state = F.relu(hidden_state)
            elif self.hidden_act == "hardswish":
                hidden_state = F.hardswish(hidden_state)
            else:
                print(f"The activation function({self.hidden_act}) is selected incorrectly.")
                exit()
        return hidden_state


class PPOCRV5ServerDetHead(nn.Module):
    """
    Standard segmentation head for generating probability maps. It uses transposed
    convolutions to upsample the feature map back to the original image size.

    Args:
        in_channels (`int`):
            Number of input channels from the neck (e.g., PPOCRV5ServerDetLKPAN).
        kernel_list (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            List of kernel sizes for the sequence of [Conv2d, ConvTranspose2d, ConvTranspose2d].
    """

    def __init__(
        self,
        in_channels: int,
        kernel_list: list[int] = [3, 2, 2],
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            bias=False,
        )
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4, momentum=0.9)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
        )

        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4, momentum=0.9)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
        )

    def forward(
        self, hidden_state: torch.Tensor, return_feature: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PPOCRV5ServerDetHead.

        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, in_channels, height, width)`):
                Input feature map.
            return_feature (`bool`, *optional*, defaults to `False`):
                Whether to return the intermediate feature map before the final convolution.

        Returns:
            `torch.FloatTensor` or `tuple(torch.FloatTensor, torch.FloatTensor)`:
                - **hidden_state** (`torch.FloatTensor`): Final probability map of shape `(batch_size, 1, H*4, W*4)`.
                - **feature** (`torch.FloatTensor`, *optional*): Intermediate features, returned only if `return_feature` is `True`.
        """
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv_bn1(hidden_state)
        hidden_state = self.relu1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv_bn2(hidden_state)
        hidden_state = self.relu2(hidden_state)
        if return_feature is True:
            feature = hidden_state
        hidden_state = self.conv3(hidden_state)
        hidden_state = torch.sigmoid(hidden_state)
        if return_feature is True:
            return hidden_state, feature
        return hidden_state


class PPOCRV5ServerDetDBHead(nn.Module):
    """
    Differentiable Binarization (DB) Head wrapper.

    Args:
        in_channels (`int`): Input channel depth.
        kernel_list (`List[int]`, *optional*, defaults to `[3, 2, 2]`): Kernel sizes for the internal Head.
    """

    def __init__(self, in_channels: int, kernel_list: list[int] = [3, 2, 2]):
        super().__init__()
        self.binarize = PPOCRV5ServerDetHead(in_channels=in_channels, kernel_list=kernel_list)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate the initial shrink map.

        Args:
            hidden_state (`torch.FloatTensor`): Input feature map.

        Returns:
            `torch.FloatTensor`: Shrink probability map.
        """
        hidden_state = self.binarize(hidden_state)
        return hidden_state


class PPOCRV5ServerDetLocalModule(nn.Module):
    """
    Local Refinement Module that refines the initial probability map by
    concatenating it with higher-resolution features.

    Args:
        in_channels (`int`): Number of channels in the feature map `hidden_state`.
        out_channels (`int`): Hidden channel size for the refinement layers.
        hidden_act (`str`): Activation function name.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_act: str):
        super().__init__()
        self.last_3 = PPOCRV5ServerDetConvBNLayer(in_channels + 1, out_channels, 3, 1, 1, hidden_act=hidden_act)
        self.last_1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, hidden_state: torch.Tensor, init_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state (`torch.FloatTensor`): Upsampled intermediate feature map.
            init_map (`torch.FloatTensor`): Initial probability map (shrink map).

        Returns:
            `torch.FloatTensor`: Refined single-channel logit map.
        """
        hidden_state = torch.cat([init_map, hidden_state], dim=1)
        # last Conv
        hidden_state = self.last_3(hidden_state)
        hidden_state = self.last_1(hidden_state)
        return hidden_state


class PPOCRV5ServerDetPFHeadLocal(PPOCRV5ServerDetDBHead):
    """
    PPOCRV5ServerDetPFHeadLocal implements the Progressive Fusion Head with Local refinement,
    the core detection head of PP-OCRv5.

    Args:
        config (`PPOCRV5ServerDetConfig`):
            Configuration object containing parameters for upsampling, mode selection,
            and refinement hidden channels.
    """

    def __init__(self, config: PPOCRV5ServerDetConfig):
        super().__init__(in_channels=config.neck_out_channels, kernel_list=config.kernel_list)

        self.up_conv = nn.Upsample(scale_factor=config.scale_factor, mode=config.interpolate_mode)
        if config.mode == "large":
            out_channels = config.neck_out_channels // 4
        elif config.mode == "small":
            out_channels = config.neck_out_channels // 8
        else:
            raise ValueError(f"mode must be 'large' or 'small', currently {config.mode}")
        self.cbn_layer = PPOCRV5ServerDetLocalModule(config.neck_out_channels // 4, out_channels, config.hidden_act)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetPFHeadLocal, combining base shrink maps and locally refined maps.

        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, neck_out_channels, H, W)`):
                Fused feature map from the neck.

        Returns:
            `torch.FloatTensor`:
                The final refined text detection probability map, calculated as the
                average of the base map and the refined local map.
        """
        hidden_state, feature = self.binarize(hidden_state, return_feature=True)
        identity = hidden_state
        feature = self.up_conv(feature)
        hidden_state = self.cbn_layer(feature, hidden_state)
        hidden_state = torch.sigmoid(hidden_state)

        return 0.5 * (identity + hidden_state)


@dataclass
class PPOCRV5ServerDetModelOutput(BaseModelOutputWithNoAttention):
    """
    Output class for the PPOCRV5ServerDetModel.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
            Binary segmentation probability maps from the head. Higher values indicate
            higher probability of text presence.
    """

    logits: torch.FloatTensor | None = None


class PPOCRV5ServerDetPreTrainedModel(PreTrainedModel):
    """
    Base class for all PPOCRV5 Server Det pre-trained models. Handles model initialization,
    configuration, and loading of pre-trained weights, following the Transformers library conventions.
    """

    config: PPOCRV5ServerDetConfig
    base_model_prefix = "pp_ocrv5_server_det"
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, PPOCRV5ServerDetConvBNLayer):
            nn.init.kaiming_normal_(module.conv.weight)

        if isinstance(module, PPOCRV5ServerDetHead):
            nn.init.constant_(module.conv_bn1.weight, 1.0)
            nn.init.constant_(module.conv_bn1.bias, 1e-4)
            nn.init.constant_(module.conv_bn2.weight, 1.0)
            nn.init.constant_(module.conv_bn2.bias, 1e-4)
            nn.init.kaiming_uniform_(module.conv2.weight)
            nn.init.kaiming_uniform_(module.conv3.weight)

        if isinstance(module, PPOCRV5ServerDetLKPAN):
            for sub_module in module.modules():
                if isinstance(sub_module, nn.ModuleList):
                    for m in sub_module:
                        nn.init.kaiming_uniform_(m.weight)


@auto_docstring(
    custom_intro="""
    """
)
class PPOCRV5ServerDetModel(PPOCRV5ServerDetPreTrainedModel):
    """
    Core PPOCRV5 Server Det model.
    Integration of HGNetV2 (Backbone), PPOCRV5ServerDetLKPAN (Neck), and PPOCRV5ServerDetPFHeadLocal (Head).
    """

    _can_record_outputs = {"hidden_states": PPOCRV5ServerDetLKPAN}

    def __init__(self, config: PPOCRV5ServerDetConfig):
        """
        Initialize the PPOCRV5ServerDetModel with the specified configuration.

        Args:
            config (PPOCRV5ServerDetConfig): Configuration object containing all model hyperparameters.
        """
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = PPOCRV5ServerDetLKPAN(config)
        self.post_init()

    @capture_outputs
    @can_return_tuple
    def forward(
        self,
        hidden_state: torch.FloatTensor,
        **kwargs,
    ) -> tuple[torch.FloatTensor] | PPOCRV5ServerDetModelOutput:
        """
        Forward pass of the PPOCRV5ServerDetModel.

        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
                Input image pixels.

        """
        hidden_state = self.backbone(hidden_state).feature_maps
        outputs = self.neck(hidden_state)

        return PPOCRV5ServerDetModelOutput(logits=outputs.last_hidden_state)


@auto_docstring(
    custom_intro="""
    """
)
@dataclass
class PPOCRV5ServerDetForObjectDetectionOutput(BaseModelOutputWithNoAttention):
    """
    Output class for PPOCRV5ServerDetForObjectDetection.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
            The predicted text mask.
        last_hidden_state (`torch.FloatTensor`, *optional*):
            Last stage features from the backbone.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate stage features.
    """

    logits: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    """
)
class PPOCRV5ServerDetForObjectDetection(PPOCRV5ServerDetPreTrainedModel):
    """
    PPOCRV5 Server Det model for object (text) detection tasks. Wraps the core PPOCRV5ServerDetModel
    and returns outputs compatible with the Transformers object detection API.
    """

    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPOCRV5ServerDetConfig):
        """
        Initialize the PPOCRV5ServerDetForObjectDetection with the specified configuration.

        Args:
            config (PPOCRV5ServerDetConfig): Configuration object containing all model hyperparameters.
        """
        super().__init__(config)
        self.model = PPOCRV5ServerDetModel(config)
        self.head = PPOCRV5ServerDetPFHeadLocal(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> tuple[torch.FloatTensor] | PPOCRV5ServerDetForObjectDetectionOutput:
        """
        Forward pass of the PPOCRV5 detection model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
                Pixel values of the input images.

        Returns:
            `PPOCRV5ServerDetForObjectDetectionOutput` or `tuple(torch.FloatTensor)`:
                The detection result containing `logits` (segmentation mask).
        """

        outputs = self.model(pixel_values)
        logits = self.head(outputs.logits)

        return PPOCRV5ServerDetForObjectDetectionOutput(
            logits=logits,
        )


__all__ = [
    "PPOCRV5ServerDetForObjectDetection",
    "PPOCRV5ServerDetImageProcessorFast",
    "PPOCRV5ServerDetConfig",
    "PPOCRV5ServerDetModel",
    "PPOCRV5ServerDetPreTrainedModel",
]
