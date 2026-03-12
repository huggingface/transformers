import math
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
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    SizeDict,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_cv2_available,
    logging,
)
from ...utils.generic import TensorType, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig


if is_cv2_available():
    import cv2


logger = logging.get_logger(__name__)


@auto_docstring(
    checkpoint="PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
    custom_args=r"""
    reduction (`int`, *optional*, defaults to 4):
        The reduction factor for feature channel dimensions, used to reduce the number of model parameters and
        computational complexity while maintaining feature representability.
    neck_out_channels (`int`, *optional*, defaults to 96):
        The number of output channels from the neck network, which is responsible for feature fusion and
        refinement before passing features to the head network.
    interpolate_mode (`str`, *optional*, defaults to `"nearest"`):
        The interpolation mode used for upsampling or downsampling feature maps in the neck network. Supported
        modes include `"nearest"` (nearest neighbor interpolation) and `"bilinear"`.
    kernel_list (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
        The list of kernel sizes for convolutional layers in the head network, used for multi-scale feature
        extraction to detect text regions of different sizes.
    layer_list_out_channels (`List[int]`, *optional*, defaults to `[12, 18, 42, 360]`):
        The list of output channels for each backbone stage, used to configure the input channels of the RSE layers
        in the neck network for multi-scale feature fusion.
    """,
)
class PPOCRV5MobileDetConfig(PreTrainedConfig):
    model_type = "pp_ocrv5_mobile_det"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        reduction=4,
        neck_out_channels=96,
        interpolate_mode="nearest",
        kernel_list=[3, 2, 2],
        layer_list_out_channels=[12, 18, 42, 360],
        **kwargs,
    ):
        # ---- Backbone ----
        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="pp_lcnet_v3",
            default_config_kwargs={
                "scale": 0.75,
                "out_features": ["stage2", "stage3", "stage4", "stage5"],
                "out_indices": [2, 3, 4, 5],
                "divisor": 16,
            },
            **kwargs,
        )
        self.backbone_config = backbone_config
        self.reduction = reduction

        # ---- Neck ----
        self.neck_out_channels = neck_out_channels
        self.interpolate_mode = interpolate_mode

        # ---- Head ----
        self.kernel_list = kernel_list
        self.layer_list_out_channels = layer_list_out_channels

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

    pivot_index = np.lexsort((points[:, 0], points[:, 1]))[0]
    pivot_point = points[pivot_index]

    def polar_angle(point):
        return np.arctan2(point[1] - pivot_point[1], point[0] - pivot_point[0])

    sorted_points = sorted(points, key=polar_angle)

    convex_hull_points = []
    for current_point in sorted_points:
        while len(convex_hull_points) >= 2:
            second_last_point = convex_hull_points[-2]
            last_point = convex_hull_points[-1]
            cross_product = (last_point[0] - second_last_point[0]) * (current_point[1] - second_last_point[1]) - (
                last_point[1] - second_last_point[1]
            ) * (current_point[0] - second_last_point[0])
            if cross_product <= 1e-8:
                convex_hull_points.pop()
            else:
                break
        convex_hull_points.append(current_point)

    return np.array(convex_hull_points, dtype=np.float64)


def min_area_rect(contour: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    contour = contour.reshape(-1, 2).astype(np.float64)
    convex_hull_points = convex_hull(contour)
    number_of_hull_points = len(convex_hull_points)

    if number_of_hull_points == 1:
        return (float(convex_hull_points[0][0]), float(convex_hull_points[0][1])), (0.0, 0.0), 0.0
    if number_of_hull_points == 2:
        delta_x, delta_y = convex_hull_points[1] - convex_hull_points[0]
        edge_length = np.hypot(delta_x, delta_y)
        center_point = (
            (convex_hull_points[0][0] + convex_hull_points[1][0]) / 2,
            (convex_hull_points[0][1] + convex_hull_points[1][1]) / 2,
        )
        rotation_angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
        rotation_angle = rotation_angle - 90 if rotation_angle >= 0 else rotation_angle
        return (float(center_point[0]), float(center_point[1])), (float(edge_length), 0.0), float(rotation_angle)

    minimum_area = float("inf")
    best_rectangle = None
    current_j_index = 1

    for current_i_index in range(number_of_hull_points):
        point_one, point_two = (
            convex_hull_points[current_i_index],
            convex_hull_points[(current_i_index + 1) % number_of_hull_points],
        )
        edge_vector = point_two - point_one
        edge_vector_length = np.hypot(edge_vector[0], edge_vector[1])

        if edge_vector_length < 1e-8:
            continue

        edge_normal_vector = np.array([-edge_vector[1], edge_vector[0]]) / edge_vector_length

        while True:
            current_dot_product = np.dot(convex_hull_points[current_j_index] - point_one, edge_normal_vector)
            next_dot_product = np.dot(
                convex_hull_points[(current_j_index + 1) % number_of_hull_points] - point_one, edge_normal_vector
            )
            if next_dot_product > current_dot_product + 1e-8:
                current_j_index = (current_j_index + 1) % number_of_hull_points
            else:
                break

        projection = np.dot(convex_hull_points - point_one, edge_vector) / edge_vector_length
        minimum_projection, maximum_projection = np.min(projection), np.max(projection)
        rectangle_width = maximum_projection - minimum_projection

        normal_projection = np.dot(convex_hull_points - point_one, edge_normal_vector)
        minimum_normal_projection, maximum_normal_projection = np.min(normal_projection), np.max(normal_projection)
        rectangle_height = maximum_normal_projection - minimum_normal_projection

        current_area = rectangle_width * rectangle_height
        if current_area < minimum_area - 1e-8:
            minimum_area = current_area
            center_x_coordinate = (
                point_one[0]
                + edge_vector[0] * (minimum_projection + maximum_projection) / (2 * edge_vector_length)
                + edge_normal_vector[0] * (minimum_normal_projection + maximum_normal_projection) / 2
            )
            center_y_coordinate = (
                point_one[1]
                + edge_vector[1] * (minimum_projection + maximum_projection) / (2 * edge_vector_length)
                + edge_normal_vector[1] * (minimum_normal_projection + maximum_normal_projection) / 2
            )
            center_point = (float(center_x_coordinate), float(center_y_coordinate))

            rotation_angle = np.arctan2(edge_vector[1], edge_vector[0]) * 180 / np.pi
            if rotation_angle >= 90:
                rotation_angle -= 180
            elif rotation_angle >= 0:
                rotation_angle -= 90
            rotation_angle = float(rotation_angle)

            if rectangle_width < rectangle_height:
                rectangle_width, rectangle_height = rectangle_height, rectangle_width
                rotation_angle -= 90

            best_rectangle = (center_point, (float(rectangle_width), float(rectangle_height)), rotation_angle)

    return best_rectangle if best_rectangle else ((0.0, 0.0), (0.0, 0.0), 0.0)


def box_points(rectangle: tuple) -> np.ndarray:
    center_point, rectangle_size, rotation_angle = rectangle
    center_x, center_y = center_point
    rectangle_width, rectangle_height = rectangle_size
    rotation_angle_radians = rotation_angle * np.pi / 180.0

    half_width = rectangle_width / 2.0
    half_height = rectangle_height / 2.0

    cosine_angle = np.cos(rotation_angle_radians)
    sine_angle = np.sin(rotation_angle_radians)

    rectangle_vertices = [
        (
            center_x - half_width * cosine_angle - half_height * sine_angle,
            center_y - half_width * sine_angle + half_height * cosine_angle,
        ),
        (
            center_x + half_width * cosine_angle - half_height * sine_angle,
            center_y + half_width * sine_angle + half_height * cosine_angle,
        ),
        (
            center_x + half_width * cosine_angle + half_height * sine_angle,
            center_y + half_width * sine_angle - half_height * cosine_angle,
        ),
        (
            center_x - half_width * cosine_angle + half_height * sine_angle,
            center_y - half_width * sine_angle - half_height * cosine_angle,
        ),
    ]
    return np.array(rectangle_vertices, dtype=np.float32)


def masked_mean(region_of_interest: np.ndarray, mask_array: np.ndarray) -> float:
    mask_array = mask_array.astype(np.uint8)
    region_of_interest = region_of_interest.astype(np.float64)

    valid_pixel_values = region_of_interest[mask_array == 1]

    if len(valid_pixel_values) == 0:
        return 0.0

    return float(np.mean(valid_pixel_values))


def unclip(bounding_box: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """
    Expands (dilates) a detected text bounding box to recover the full text region.
    Args:
        bounding_box (np.ndarray): Input contour of shape (N, 2), where N is the number of points.
        unclip_ratio (float): Expansion ratio, typically greater than 1.0.
    Returns:
        np.ndarray: Expanded contour of shape (M, 2).
    """
    bounding_box = np.array(bounding_box).reshape(-1, 2)

    polygon_area_value = polygon_area(bounding_box)
    polygon_arc_length_value = polygon_arc_length(bounding_box, True)
    if polygon_arc_length_value == 0:
        return bounding_box
    expansion_distance = polygon_area_value * unclip_ratio / polygon_arc_length_value

    contour_points = np.concatenate([bounding_box, bounding_box[0:1]], axis=0)
    expanded_contour_points = []

    for current_index in range(len(bounding_box)):
        current_point = contour_points[current_index]
        previous_point = contour_points[current_index - 1]
        next_point = contour_points[current_index + 1]

        def get_normal_vector(point_a, point_b):
            direction_vector = point_b - point_a
            vector_norm = np.linalg.norm(direction_vector)
            if vector_norm == 0:
                return np.array([0, 0])
            return np.array([direction_vector[1], -direction_vector[0]]) / vector_norm

        normal_vector_one = get_normal_vector(previous_point, current_point)
        normal_vector_two = get_normal_vector(current_point, next_point)
        combined_normal_vector = normal_vector_one + normal_vector_two
        cosine_theta_value = np.dot(normal_vector_one, normal_vector_two)

        denominator_value = 1 + cosine_theta_value
        if denominator_value < 1e-6:
            scale_factor = expansion_distance
        else:
            scale_factor = expansion_distance * np.sqrt(2 / denominator_value)

        new_contour_point = current_point + combined_normal_vector * (
            scale_factor / (np.linalg.norm(combined_normal_vector) + 1e-6)
        )
        expanded_contour_points.append(new_contour_point)

    return np.array(expanded_contour_points, dtype=np.float32)


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
) -> tuple[list[np.ndarray] | np.ndarray, list[float]]:
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


@auto_docstring(
    custom_intro="""
    """
)
class PPOCRV5MobileDetImageProcessorFast(BaseImageProcessorFast):
    """
    Image processor for PP-OCRv5_mobile_det, handling preprocessing (resizing, normalization)
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

    def __init__(self, **kwargs) -> None:
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
                size, shape = self.get_image_size(
                    image=image,
                    limit_side_len=self.limit_side_len,
                    limit_type=self.limit_type,
                    max_side_limit=self.max_side_limit,
                )
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

    def get_image_size(
        self,
        image: np.ndarray,
        limit_side_len: int,
        limit_type: str,
        max_side_limit: int = 4000,
        **kwargs,
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

    def post_process_object_detection(
        self,
        outputs,
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
        for logit, size in zip(outputs.last_hidden_state, target_sizes):
            logit = logit[0, :, :].cpu().detach().numpy()
            size = size.cpu().detach().numpy()

            src_height, src_width = size
            mask = logit > threshold
            box, score = boxes_from_bitmap(
                logit, mask, src_width, src_height, box_thresh, unclip_ratio, min_size, max_candidates
            )

            results.append({"scores": score, "boxes": box})
        return results


@auto_docstring
class PPOCRV5MobileDetPreTrainedModel(PreTrainedModel):
    config: PPOCRV5MobileDetConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)


class PPOCRV5MobileDetSqueezeExcitationModule(nn.Module):
    """
    Simplified Squeeze-and-Excitation (SE) Module for the neck network.
    Applies channel-wise recalibration with a clamped activation to stabilize training.
    """

    def __init__(self, in_channels, reduction, activation="relu"):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.activation = ACT2FN[activation]

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv2(self.activation(self.conv1(outputs)))
        outputs = torch.clamp(0.2 * outputs + 0.5, min=0.0, max=1.0)
        return inputs * outputs


class PPOCRV5MobileDetResidualSqueezeExcitationLayer(nn.Module):
    """
    Residual Squeeze-and-Excitation (RSE) Layer for the neck network.
    Combines a 1x1/3x3 convolution with an SE Module.
    """

    def __init__(self, in_channels, out_channels, kernel_size, reduction):
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False,
        )
        self.se_block = PPOCRV5MobileDetSqueezeExcitationModule(out_channels, reduction)

    def forward(self, hidden_states):
        hidden_states = self.in_conv(hidden_states)
        hidden_states = hidden_states + self.se_block(hidden_states)

        return hidden_states


class PPOCRV5MobileDetNeck(nn.Module):
    """
    Neck network for PPOCRV5 Mobile Det, responsible for multi-scale feature fusion.
    Uses RSELayers to process backbone features and upsampling to fuse features at the same spatial scale,
    then concatenates the fused features for input to the head network.
    """

    def __init__(self, config: PPOCRV5MobileDetConfig):
        super().__init__()
        self.interpolate_mode = config.interpolate_mode

        self.insert_conv = nn.ModuleList()
        self.input_conv = nn.ModuleList()
        for i in range(len(config.layer_list_out_channels)):
            self.insert_conv.append(
                PPOCRV5MobileDetResidualSqueezeExcitationLayer(
                    config.layer_list_out_channels[i],
                    config.neck_out_channels,
                    1,
                    config.reduction,
                )
            )
            self.input_conv.append(
                PPOCRV5MobileDetResidualSqueezeExcitationLayer(
                    config.neck_out_channels, config.neck_out_channels // 4, 3, config.reduction
                )
            )

    def forward(self, feature_maps):
        fused = [conv(feature) for conv, feature in zip(self.insert_conv, feature_maps)]  # [p2, p3, p4, p5]

        for i in range(2, -1, -1):  # p4 -> p3-> p2
            fused[i] = fused[i] + F.interpolate(fused[i + 1], scale_factor=2, mode=self.interpolate_mode)

        processed = [conv(feat) for conv, feat in zip(self.input_conv, [fused[0], fused[1], fused[2], fused[3]])]
        upsample_scales = [1, 2, 4, 8]  # p2, p3, p4, p5
        processed = [
            F.interpolate(feat, scale_factor=scale, mode=self.interpolate_mode) if scale != 1 else feat
            for feat, scale in zip(processed, upsample_scales)
        ]
        fused_feature_map = torch.cat(processed[::-1], dim=1)  # [p5, p4, p3, p2]
        return fused_feature_map


class PPOCRV5MobileDetHead(nn.Module):
    """
    Head sub-module for PPOCRV5 Mobile Det, responsible for generating text segmentation maps.
    Uses two transposed convolutions for upsampling to recover the original image spatial scale,
    and a sigmoid activation to produce binary segmentation logits.
    """

    def __init__(self, config):
        super().__init__()

        in_channels = config.neck_out_channels
        kernel_list = config.kernel_list
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
        )
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
        )

    def forward(self, hidden_states):
        hidden_states = self.relu1(self.bn1(self.conv1(hidden_states)))
        hidden_states = self.relu2(self.bn2(self.conv2(hidden_states)))
        hidden_states = torch.sigmoid(self.conv3(hidden_states))
        return hidden_states


@auto_docstring(
    custom_intro="""
    Core PP-OCRv5_mobile_det, consisting of Backbone, Neck, and Head networks.
    Generates binary text segmentation maps for text detection tasks.
    """
)
class PPOCRV5MobileDetModel(PPOCRV5MobileDetPreTrainedModel):
    def __init__(self, config: PPOCRV5MobileDetConfig):
        super().__init__(config)

        self.backbone = load_backbone(config)
        out_channels = [self.backbone.num_features[i] for i in self.backbone.out_indices]
        self.layer = nn.ModuleList()
        for idx, out_channel in enumerate(out_channels):
            self.layer.append(nn.Conv2d(out_channel, config.layer_list_out_channels[idx], 1, 1, 0))

        self.neck = PPOCRV5MobileDetNeck(config)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.backbone(hidden_states, **kwargs)
        feature_maps = outputs.feature_maps
        hidden_states = [self.layer[i](feature_maps[i]) for i in range(len(feature_maps))]
        hidden_states = self.neck(hidden_states)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=outputs.hidden_states)


@auto_docstring(
    custom_intro="""
    PP-OCRv5_mobile_det for text detection tasks.
    """
)
class PPOCRV5MobileDetForObjectDetection(PPOCRV5MobileDetPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPOCRV5MobileDetConfig):
        super().__init__(config)
        self.model = PPOCRV5MobileDetModel(config)
        self.head = PPOCRV5MobileDetHead(config)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.model(pixel_values, **kwargs)
        hidden_states = self.head(outputs.last_hidden_state)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=outputs.hidden_states)


__all__ = [
    "PPOCRV5MobileDetForObjectDetection",
    "PPOCRV5MobileDetImageProcessorFast",
    "PPOCRV5MobileDetConfig",
    "PPOCRV5MobileDetModel",
    "PPOCRV5MobileDetPreTrainedModel",
]
