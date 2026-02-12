import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF

from transformers.models.hgnet_v2.modeling_hgnet_v2 import (
    HGNetV2BasicLayer,
    HGNetV2ConvLayer,
    HGNetV2ConvLayerLight,
    HGNetV2Embeddings,
    HGNetV2LearnableAffineBlock,
    HGNetV2Stage,
)

from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import flip_channel_order, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    filter_out_non_signature_kwargs,
    is_cv2_available,
    logging,
)
from ...utils.generic import TensorType

if is_cv2_available():
    import cv2


logger = logging.get_logger(__name__)


@auto_docstring(custom_intro="Configuration for the PPOCRV5 Server Det model.")
class PPOCRV5ServerDetConfig(PreTrainedConfig):
    model_type = "pp_ocrv5_server_det"

    """
    This is the configuration class to store the configuration of a [`PPOCRV5ServerDet`]. It is used to instantiate a
    PPOCRV5 Server text detection model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the PPOCRV5 Server Det
    [PaddlePaddle/PP-OCRv5-server-det](https://huggingface.co/PaddlePaddle/PP-OCRv5-server-det) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        interpolate_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation mode used for upsampling or downsampling feature maps in the neck network.
        stem_channels (`list[int]`, *optional*, defaults to `[3, 32, 48]`):
            The number of output channels for the stem layers in the backbone network.
        stage_in_channels (`list[int]`, *optional*, defaults to `[48, 128, 512, 1024]`):
            Input channel dimensions for each stage in the backbone network.
        stage_mid_channels (`list[int]`, *optional*, defaults to `[48, 96, 192, 384]`):
            Intermediate channel dimensions for each stage in the backbone network, used in bottleneck blocks.
        stage_out_channels (`list[int]`, *optional*, defaults to `[128, 512, 1024, 2048]`):
            Output channel dimensions for each stage in the backbone network.
        stage_num_blocks (`list[int]`, *optional*, defaults to `[1, 1, 3, 1]`):
            Number of blocks in each stage of the backbone network.
        stage_downsample (`list[bool]`, *optional*, defaults to `[False, True, True, True]`):
            Whether to apply downsampling (strided convolution/pooling) at the start of each backbone stage.
        stage_light_block (`list[bool]`, *optional*, defaults to `[False, False, True, True]`):
            Whether to use lightweight blocks (instead of standard blocks) in each backbone stage to reduce computation.
        stage_kernel_size (`list[int]`, *optional*, defaults to `[3, 3, 5, 5]`):
            Kernel size of convolutional layers in each backbone stage for feature extraction.
        stage_num_of_layers (`list[int]`, *optional*, defaults to `[6, 6, 6, 6]`):
            Number of sub-layers within each block of the backbone stages (fixed to 6 for PP-OCRv5).
        use_learnable_affine_block (`bool`, *optional*, defaults to `False`):
            Whether to use Large Adaptive Blocks (LAB) in the backbone architecture.
        use_last_conv (`bool`, *optional*, defaults to `True`):
            Whether to include the last convolutional layer in the backbone feature extractor.
        class_expand (`int`, *optional*, defaults to 2048):
            The expansion factor for the classification layer channels.
        dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for the classification or bottleneck layers to prevent overfitting.
        class_num (`int`, *optional*, defaults to 1000):
            The number of classes for the pre-training task (typically ImageNet-1k).
        out_indices (`list[int]`, *optional*, defaults to `[0, 1, 2, 3]`):
            The indices of the backbone layers from which to extract feature maps for the neck.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input channels (3 for RGB images, 1 for grayscale).
        neck_out_channels (`int`, *optional*, defaults to 256):
            The number of output channels from the neck network, responsible for feature fusion and refinement.
        reduce_factor (`int`, *optional*, defaults to 2):
            The channel reduction factor used in the neck blocks to balance performance and complexity.
        intraclblock_config (`dict`, *optional*, defaults to `None`):
            Configuration for the Intra-Class Block modules, if any, used for enhancing feature representation.
        k (`int`, *optional*, defaults to 50):
            The candidate box number threshold for the head network, controlling the maximum number of text region candidates.
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

    def __init__(
        self,
        interpolate_mode: str = "nearest",
        stem_channels: list[int] = [3, 32, 48],
        stage_in_channels: list[int] = [48, 128, 512, 1024],
        stage_mid_channels: list[int] = [48, 96, 192, 384],
        stage_out_channels: list[int] = [128, 512, 1024, 2048],
        stage_num_blocks: list[int] = [1, 1, 3, 1],
        stage_downsample: list[bool] = [False, True, True, True],
        stage_light_block: list[bool] = [False, False, True, True],
        stage_kernel_size: list[int] = [3, 3, 5, 5],
        stage_numb_of_layers: list[int] = [6, 6, 6, 6],
        use_learnable_affine_block: bool = False,
        use_last_conv: bool = True,
        class_expand: int = 2048,
        dropout_prob: float = 0.0,
        class_num: int = 1000,
        out_indices: list[int] = [0, 1, 2, 3],
        num_channels: int = 3,
        neck_out_channels: int = 256,
        reduce_factor: int = 2,
        intraclblock_config: dict | None = None,
        k: int = 50,
        mode: str = "large",
        scale_factor: int = 2,
        hidden_act: str = "relu",
        kernel_list: list[int] = [3, 2, 2],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.mode = mode
        self.interpolate_mode = interpolate_mode
        # ---- backbone ----
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(stage_in_channels) + 1)]
        self.stem_channels = stem_channels
        self.stage_in_channels = stage_in_channels
        self.stage_mid_channels = stage_mid_channels
        self.stage_out_channels = stage_out_channels
        self.stage_num_blocks = stage_num_blocks
        self.stage_downsample = stage_downsample
        self.stage_light_block = stage_light_block
        self.stage_kernel_size = stage_kernel_size
        self.stage_numb_of_layers = stage_numb_of_layers
        self.use_learnable_affine_block = use_learnable_affine_block
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        self.dropout_prob = dropout_prob
        self.class_num = class_num
        self.out_indices = out_indices
        self.num_channels = num_channels

        # ---- neck ----
        self.neck_out_channels = neck_out_channels
        self.reduce_factor = reduce_factor
        self.intraclblock_config = intraclblock_config

        # ---- head ----
        self.k = k
        self.scale_factor = scale_factor
        self.hidden_act = hidden_act
        self.kernel_list = kernel_list


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
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))

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
) -> tuple[Union[list[np.ndarray], np.ndarray], list[float]]:
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


@auto_docstring(custom_intro="ImageProcessor for the PPOCRV5 Server Det model.")
class PPOCRV5ServerDetImageProcessor(BaseImageProcessor):
    """
    Image Processor for the PPOCRV5 Server Det text detection model.

    This class handles all image preprocessing (resizing, rescaling, normalization, channel flipping)
    and post-processing (converting model logits to detected text boxes) required for the PPOCRV5 Server Det model.
    It ensures input images are formatted correctly for model inference and converts model outputs into human-interpretable
    text bounding boxes.

    Key features:
    - Aspect-ratio preserving image resizing with side length limits.
    - RGB to BGR channel flipping (compatible with PaddlePaddle's original model).
    - Standard image normalization and rescaling.
    - Post-processing to extract quadrilateral or polygonal text boxes from segmentation maps.

    Attributes:
        model_input_names (List[str]): List of input names expected by the model (only "pixel_values" for this processor).
        limit_side_len (int): Maximum/minimum side length for image resizing (depending on `limit_type`).
        limit_type (str): Resizing strategy ("max", "min", or "resize_long").
        max_side_limit (int): Hard maximum limit for the longest image side to prevent excessive memory usage.
        do_resize (bool): Whether to resize input images.
        size (dict[str, int]): Default target size for resizing (height, width).
        resample (PILImageResampling): Resampling mode for image resizing.
        do_rescale (bool): Whether to rescale pixel values from [0, 255] to [0, 1].
        rescale_factor (Union[int, float]): Factor used for pixel value rescaling (1/255 by default).
        do_normalize (bool): Whether to normalize images using mean and standard deviation.
        image_mean (Union[float, List[float]]): Mean values for image normalization (BGR order, compatible with model).
        image_std (Union[float, List[float]]): Standard deviation values for image normalization (BGR order).
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        limit_side_len: int = 960,
        limit_type: str = "max",
        max_side_limit: int = 4000,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 960, "width": 960}

        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.max_side_limit = max_side_limit

        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.resample = resample

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        limit_side_len: int = 960,
        limit_type: str = "max",
        max_side_limit: int = 4000,
        size: Optional[dict[str, int]] = None,
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        size = self.size if size is None else size
        limit_side_len = self.limit_side_len if limit_side_len is None else limit_side_len
        limit_type = self.limit_type if limit_type is None else limit_type
        max_side_limit = max_side_limit if max_side_limit is not None else self.max_side_limit
        do_resize = self.do_resize if do_resize is None else do_resize
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std

        images = make_flat_list_of_images(images)

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            size=size,
            do_resize=do_resize,
            resample=resample,
        )

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        # transformations
        resize_imgs, target_sizes = [], []
        if do_resize:
            for image in images:
                size, shape = self.get_image_size(image, self.limit_side_len, self.limit_type, max_side_limit)
                try:
                    img = resize(
                        image,
                        size=(size["height"], size["width"]),
                        resample=resample,
                        input_data_format=input_data_format,
                    )
                except Exception as e:
                    print(size)
                    raise RuntimeError(f"Failed to resize image: {e}") from e

                resize_imgs.append(img)
                target_sizes.append(shape)
            images = resize_imgs

        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]

        if do_normalize:
            images = [
                self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images
            ]
        # flip color channels from RGB to BGR
        images = [flip_channel_order(image, input_data_format=input_data_format) for image in images]
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        encoded_inputs = BatchFeature(
            data={"pixel_values": images, "target_sizes": target_sizes}, tensor_type=return_tensors
        )

        return encoded_inputs

    def post_process_object_detection(
        self,
        preds,
        threshold: float = 0.3,
        target_sizes: Optional[Union[list[tuple[int, int]], torch.Tensor]] = None,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        min_size: int = 3,
        unclip_ratio: float = 1.5,
    ):
        """
        Converts model outputs into detected text boxes.

        Args:
            preds (torch.Tensor): Model outputs.
            target_sizes (TensorType or list[tuple]): Original image sizes.
            threshold (float): Binarization threshold.
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
            results.append({"scores": score, "boxes": box})
        return results

    def get_image_size(
        self,
        img: np.ndarray,
        limit_side_len: int,
        limit_type: str,
        max_side_limit: int = 4000,
    ) -> tuple[dict, np.ndarray]:
        """
        Computes the target size for resizing an image while preserving aspect ratio.

        Args:
            img (torch.Tensor): Input image.
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
        h, w, c = img.shape

        if limit_type == "max":
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif limit_type == "min":
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        elif limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception("not support limit type, image ")
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        if max(resize_h, resize_w) > max_side_limit:
            ratio = float(max_side_limit) / max(resize_h, resize_w)
            resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        if resize_h == h and resize_w == w:
            return {"height": resize_h, "width": resize_w}, np.array([h, w])

        if int(resize_w) <= 0 or int(resize_h) <= 0:
            return None, (None, None)

        return {"height": resize_h, "width": resize_w}, np.array([h, w])


@auto_docstring(custom_intro="ImageProcessorFast for the PPOCRV5 Server Det model.")
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
        resize_imgs, target_sizes = [], []
        if do_resize:
            for image in images:
                size, shape = self.get_image_size(image, self.limit_side_len, self.limit_type, self.max_side_limit)
                img = self.resize(image, size=size, interpolation=interpolation)
                resize_imgs.append(img)
                target_sizes.append(shape)
            images = resize_imgs

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
        target_sizes: Optional[Union[list[tuple[int, int]], torch.Tensor]] = None,
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
        img: np.ndarray,
        limit_side_len: int,
        limit_type: str,
        max_side_limit: int = 4000,
    ) -> tuple[dict, np.ndarray]:
        """
        Computes the target size for resizing an image while preserving aspect ratio.

        Args:
            img (torch.Tensor): Input image.
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
        _, h, w = img.shape
        h, w = int(h), int(w)

        if limit_type == "max":
            if max(h, w) > limit_side_len:
                ratio = float(limit_side_len) / max(h, w)
            else:
                ratio = 1.0
        elif limit_type == "min":
            if min(h, w) < limit_side_len:
                ratio = float(limit_side_len) / min(h, w)
            else:
                ratio = 1.0
        elif limit_type == "resize_long":
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception(f"not support limit type: {limit_type}")

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        if max_side_limit is not None and max(resize_h, resize_w) > max_side_limit:
            ratio = float(max_side_limit) / max(resize_h, resize_w)
            resize_h = int(resize_h * ratio)
            resize_w = int(resize_w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        if resize_h == h and resize_w == w:
            return SizeDict(height=resize_h, width=resize_w), torch.tensor([h, w], dtype=torch.float32)

        if resize_w <= 0 or resize_h <= 0:
            return None, (None, None)

        return SizeDict(height=resize_h, width=resize_w), torch.tensor([h, w], dtype=torch.float32)


class PPOCRV5ServerDetLearnableAffineBlock(HGNetV2LearnableAffineBlock):
    pass


class PPOCRV5ServerDetConvBNAct(HGNetV2ConvLayer):
    pass


class PPOCRV5ServerDetLightConvBNAct(HGNetV2ConvLayerLight):
    pass


class PPOCRV5ServerDetStemBlock(HGNetV2Embeddings):
    pass


class PPOCRV5ServerDetHGV2_Block(HGNetV2BasicLayer):
    pass


class PPOCRV5ServerDetHGV2_Stage(HGNetV2Stage):
    pass


class PPOCRV5ServerDetHGNetV2(nn.Module):
    """
    PPOCRV5ServerDetHGNetV2 (Paddle High-Performance GPU Network V2) backbone.
    Extracts multi-scale hierarchical features from input images for downstream detection or classification.

    Args:
        config (`PPOCRV5ServerDetConfig`):
            Configuration object containing model hyperparameters:
            - **out_indices**: Indices of stages to return features from.
            - **use_learnable_affine_block**: Global flag for Learnable Affine Block.
            - **use_last_conv**: Whether to apply final global pooling and classification head.
    """

    def __init__(self, config: PPOCRV5ServerDetConfig):
        super().__init__()
        self.use_learnable_affine_block = config.use_learnable_affine_block
        self.use_last_conv = config.use_last_conv
        self.class_expand = config.class_expand
        self.class_num = config.class_num
        self.out_indices = config.out_indices
        self.out_channels = []

        # stem
        self.stem = PPOCRV5ServerDetStemBlock(config)

        # stages
        self.stages = nn.ModuleList([])
        for stage_index in range(len(config.stage_in_channels)):
            resnet_stage = PPOCRV5ServerDetHGV2_Stage(config, stage_index)
            self.stages.append(resnet_stage)
            if stage_index in self.out_indices:
                self.out_channels.append(config.stage_out_channels[stage_index])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if self.use_last_conv:
            self.last_conv = nn.Conv2d(
                in_channels=config.stage_out_channels[-1],
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.act = nn.ReLU()
            if self.use_learnable_affine_block:
                self.lab = PPOCRV5ServerDetLearnableAffineBlock()
            self.dropout = nn.Dropout(p=config.dropout_prob)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(
        self, hidden_state: torch.Tensor, output_hidden_states: bool = False
    ) -> tuple[list[torch.Tensor], torch.Tensor, Optional[tuple[torch.Tensor, ...]]]:
        """
        Forward pass of PPOCRV5ServerDetHGNetV2.

        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
                Input image tensor (pixel values).
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether to return all intermediate stage outputs.

        Returns:
            `tuple(list, torch.FloatTensor, tuple)`:
                - **out** (`list` of `torch.FloatTensor`): Selected multi-scale features for the neck (e.g., c2, c3, c4, c5).
                - **hidden_state** (`torch.FloatTensor`): Final processed feature map from the last stage.
                - **hidden_states** (`tuple` of `torch.FloatTensor`, *optional*): All intermediate states, returned only if `output_hidden_states` is `True`.
        """
        hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        hidden_state = self.stem(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        out = []
        for i, stage in enumerate(self.stages):
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            hidden_state = stage(hidden_state)
            if i in self.out_indices:
                out.append(hidden_state)

        return out, hidden_state, hidden_states


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
        if_act (`bool`, *optional*, defaults to `True`):
            Whether to use an activation function in the bottleneck.
        act (`str`, *optional*, defaults to `"relu"`):
            Activation type, supports `"relu"` or `"hardswish"`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Union[int, str],
        stride: int = 1,
        groups: Optional[int] = None,
        if_act: bool = True,
        act: str = "relu",
        **kwargs,
    ):
        super().__init__()
        if groups is None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetDSConv.

        Args:
            inputs (`torch.FloatTensor` of shape `(batch_size, in_channels, height, width)`):
                The input feature map.

        Returns:
            `torch.FloatTensor`: Output feature map of shape `(batch_size, out_channels, out_height, out_width)`.
        """
        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print(f"The activation function({self.act}) is selected incorrectly.")
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetIntraCLBlock.

        Args:
            x (`torch.FloatTensor` of shape `(batch_size, in_channels, height, width)`):
                The input feature map from PPOCRV5ServerDetLKPAN stages.

        Returns:
            `torch.FloatTensor`: Refined feature map with the same shape as input,
                enhanced by spatial relationship modeling.
        """
        x_new = self.conv1x1_reduce_channel(x)

        x_7 = self.c_layer_7x7(x_new) + self.v_layer_7x1(x_new) + self.q_layer_1x7(x_new)
        x_5 = self.c_layer_5x5(x_7) + self.v_layer_5x1(x_7) + self.q_layer_1x5(x_7)
        x_3 = self.c_layer_3x3(x_5) + self.v_layer_3x1(x_5) + self.q_layer_1x3(x_5)

        x_relation = self.conv1x1_return_channel(x_3)
        x_relation = self.bn(x_relation)
        x_relation = self.relu(x_relation)

        return x + x_relation


class PPOCRV5ServerDetLKPAN(nn.Module):
    """
    Large Kernel Path Aggregation Network (Neck).
    It fuses features from multiple backbone stages (C2-C5) using a combination of
    top-down and bottom-up paths, enhanced by large kernel convolutions.

    Args:
        config (`PPOCRV5ServerDetConfig`):
            Configuration object containing `neck_out_channels`, `mode`, and `interpolate_mode`.
        in_channels (`list` of `int`):
            The channel counts of the input feature maps from the backbone stages.
    """

    def __init__(self, config: Any, in_channels: list[int]):
        super().__init__()
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

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetLKPAN.

        Args:
            x (`list` of `torch.FloatTensor`):
                Multi-scale features `[c2, c3, c4, c5]` from the backbone.

        Returns:
            `torch.FloatTensor`:
                Fused feature map of shape `(batch_size, neck_out_channels, height/4, width/4)`.
                This tensor is a concatenation of multi-scale refined features, ready for the head.
        """
        c2, c3, c4, c5 = x

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
        return fuse


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
        act (`str`, *optional*): Type of activation ("relu" or "hardswish").
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[int, str],
        groups: int = 1,
        if_act: bool = True,
        act: Optional[str] = None,
    ):
        super().__init__()
        self.if_act = if_act
        self.act = act
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetHead.

        Args:
            x (`torch.FloatTensor` of shape `(batch_size, in_channels, height, width)`):
                Input tensor.

        Returns:
            `torch.FloatTensor`: Output tensor of shape `(batch_size, out_channels, out_height, out_width)`.
        """
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print(f"The activation function({self.act}) is selected incorrectly.")
                exit()
        return x


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
        self, x: torch.Tensor, return_f: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the PPOCRV5ServerDetHead.

        Args:
            x (`torch.FloatTensor` of shape `(batch_size, in_channels, height, width)`):
                Input feature map.
            return_f (`bool`, *optional*, defaults to `False`):
                Whether to return the intermediate feature map before the final convolution.

        Returns:
            `torch.FloatTensor` or `tuple(torch.FloatTensor, torch.FloatTensor)`:
                - **x** (`torch.FloatTensor`): Final probability map of shape `(batch_size, 1, H*4, W*4)`.
                - **f** (`torch.FloatTensor`, *optional*): Intermediate features, returned only if `return_f` is `True`.
        """
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu2(x)
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = torch.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class PPOCRV5ServerDetDBHead(nn.Module):
    """
    Differentiable Binarization (DB) Head wrapper.

    Args:
        in_channels (`int`): Input channel depth.
        k (`int`, *optional*, defaults to 50): Amplification factor for the binarization step.
        kernel_list (`List[int]`, *optional*, defaults to `[3, 2, 2]`): Kernel sizes for the internal Head.
    """

    def __init__(self, in_channels: int, k: int = 50, kernel_list: list[int] = [3, 2, 2]):
        super().__init__()
        self.k = k
        self.binarize = PPOCRV5ServerDetHead(in_channels=in_channels, kernel_list=kernel_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate the initial shrink map.

        Args:
            x (`torch.FloatTensor`): Input feature map.

        Returns:
            `torch.FloatTensor`: Shrink probability map.
        """
        shrink_maps = self.binarize(x)
        return shrink_maps


class PPOCRV5ServerDetLocalModule(nn.Module):
    """
    Local Refinement Module that refines the initial probability map by
    concatenating it with higher-resolution features.

    Args:
        in_c (`int`): Number of channels in the feature map `x`.
        mid_c (`int`): Hidden channel size for the refinement layers.
        act (`str`): Activation function name.
    """

    def __init__(self, in_c: int, mid_c: int, act: str):
        super().__init__()
        self.last_3 = PPOCRV5ServerDetConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act=act)
        self.last_1 = nn.Conv2d(
            in_channels=mid_c,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor, init_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (`torch.FloatTensor`): Upsampled intermediate feature map.
            init_map (`torch.FloatTensor`): Initial probability map (shrink map).

        Returns:
            `torch.FloatTensor`: Refined single-channel logit map.
        """
        outf = torch.cat([init_map, x], dim=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out


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
        super().__init__(in_channels=config.neck_out_channels, k=config.k, kernel_list=config.kernel_list)

        self.up_conv = nn.Upsample(scale_factor=config.scale_factor, mode=config.interpolate_mode)
        if config.mode == "large":
            mid_ch = config.neck_out_channels // 4
        elif config.mode == "small":
            mid_ch = config.neck_out_channels // 8
        else:
            raise ValueError(f"mode must be 'large' or 'small', currently {config.mode}")
        self.cbn_layer = PPOCRV5ServerDetLocalModule(config.neck_out_channels // 4, mid_ch, config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPOCRV5ServerDetPFHeadLocal, combining base shrink maps and locally refined maps.

        Args:
            x (`torch.FloatTensor` of shape `(batch_size, neck_out_channels, H, W)`):
                Fused feature map from the neck.

        Returns:
            `torch.FloatTensor`:
                The final refined text detection probability map, calculated as the
                average of the base map and the refined local map.
        """
        shrink_maps, f = self.binarize(x, return_f=True)
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps)
        cbn_maps = torch.sigmoid(cbn_maps)

        return 0.5 * (base_maps + cbn_maps)


@dataclass
class PPOCRV5ServerDetModelOutput(ModelOutput):
    """
    Output class for the PPOCRV5ServerDetModel.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
            Binary segmentation probability maps from the head. Higher values indicate
            higher probability of text presence.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, channels, height/32, width/32)`, *optional*):
            Sequence of hidden-states from the last stage of the backbone.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of each stage) of various shapes.
            Returned if `output_hidden_states=True` is passed or `config.output_hidden_states=True`.
    """

    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


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

        if isinstance(module, PPOCRV5ServerDetHGNetV2):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0.0)


@auto_docstring(custom_intro="The PPOCRV5 Server Det model.")
class PPOCRV5ServerDetModel(PPOCRV5ServerDetPreTrainedModel):
    """
    Core PPOCRV5 Server Det model.
    Integration of PPOCRV5ServerDetHGNetV2 (Backbone), PPOCRV5ServerDetLKPAN (Neck), and PPOCRV5ServerDetPFHeadLocal (Head).
    """

    def __init__(self, config: PPOCRV5ServerDetConfig):
        """
        Initialize the PPOCRV5ServerDetModel with the specified configuration.

        Args:
            config (PPOCRV5ServerDetConfig): Configuration object containing all model hyperparameters.
        """
        super().__init__(config)

        self.backbone = PPOCRV5ServerDetHGNetV2(config)
        self.neck = PPOCRV5ServerDetLKPAN(config, in_channels=self.backbone.out_channels)
        self.head = PPOCRV5ServerDetPFHeadLocal(config)
        self.post_init()

    def forward(
        self,
        hidden_state: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPOCRV5ServerDetModelOutput]:
        """
        Forward pass of the PPOCRV5ServerDetModel.

        Args:
            hidden_state (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
                Input image pixels.
            output_hidden_states (`bool`, *optional*):
                Whether to return all intermediate features.
            return_dict (`bool`, *optional*):
                Whether to return a `PPOCRV5ServerDetModelOutput` instead of a plain tuple.

        Returns:
            `PPOCRV5ServerDetModelOutput` or `tuple(torch.FloatTensor)`:
                A `PPOCRV5ServerDetModelOutput` (if `return_dict=True` is passed or `config.use_return_dict=True`)
                containing the segmentation logits and optional hidden states.
        """
        hidden_state, last_hidden_state, all_hidden_states = self.backbone(hidden_state, output_hidden_states)
        hidden_state = self.neck(hidden_state)
        hidden_state = self.head(hidden_state)
        if not return_dict:
            output = (last_hidden_state,)
            if output_hidden_states:
                output += (all_hidden_states,)
            output += (hidden_state,)
            return output
        return PPOCRV5ServerDetModelOutput(
            logits=hidden_state,
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states if output_hidden_states else None,
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

    logits: Optional[torch.FloatTensor] = None
    shape: Optional[torch.FloatTensor] = None


@auto_docstring(custom_intro="ObjectDetection for the PPOCRV5 Server Det model.")
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
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[list[dict]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPOCRV5ServerDetForObjectDetectionOutput]:
        """
        Forward pass of the PPOCRV5 detection model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
                Pixel values of the input images.
            labels (`list[dict]`, *optional*):
                Ground truth for training (not implemented in this forward pass).
            output_hidden_states (`bool`, *optional*):
                Whether to return backbone's intermediate states.
            return_dict (`bool`, *optional*):
                Whether to return a structured output object.

        Returns:
            `PPOCRV5ServerDetForObjectDetectionOutput` or `tuple(torch.FloatTensor)`:
                The detection result containing `logits` (segmentation mask).
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        if not return_dict:
            output = (outputs[0],)
            if output_hidden_states:
                output += (outputs[1], outputs[2])
            else:
                output += (outputs[1],)

            return output

        return PPOCRV5ServerDetForObjectDetectionOutput(
            logits=outputs.logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )


__all__ = [
    "PPOCRV5ServerDetForObjectDetection",
    "PPOCRV5ServerDetImageProcessor",
    "PPOCRV5ServerDetImageProcessorFast",
    "PPOCRV5ServerDetConfig",
    "PPOCRV5ServerDetModel",
    "PPOCRV5ServerDetPreTrainedModel",
]
