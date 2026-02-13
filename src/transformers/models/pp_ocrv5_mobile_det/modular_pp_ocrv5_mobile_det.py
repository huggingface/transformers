import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF

from transformers.models.hgnet_v2.modeling_hgnet_v2 import (
    HGNetV2ConvLayer,
    HGNetV2LearnableAffineBlock,
)
from transformers.models.mobilenet_v2.modeling_mobilenet_v2 import make_divisible

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


@auto_docstring(custom_intro="Configuration for the PPOCRV5 Mobile Det model.")
class PPOCRV5MobileDetConfig(PreTrainedConfig):
    model_type = "pp_ocrv5_mobile_det"

    """
    This is the configuration class to store the configuration of a [`PPOCRV5MobileDet`]. It is used to instantiate a
    PPOCRV5 Mobile text detection model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the PPOCRV5 Mobile Det
    [PaddlePaddle/PP-OCRv5-mobile-det](https://huggingface.co/PaddlePaddle/PP-OCRv5-mobile-det) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*, defaults to `None`):
            The configuration of the backbone model. If `None`, the default backbone configuration for PPOCRV5 Mobile Det
            will be used.
        scale (`float`, *optional*, defaults to 1.0):
            The scaling factor for the model's channel dimensions, used to adjust the model size and computational cost
            without changing the overall architecture.
        conv_kxk_num (`int`, *optional*, defaults to 4):
            The number of stacked kxk convolutional layers in the backbone network, which is used to extract deep
            visual features from the input images.
        reduction (`int`, *optional*, defaults to 4):
            The reduction factor for feature channel dimensions, used to reduce the number of model parameters and
            computational complexity while maintaining feature representability.
        divisor (`int`, *optional*, defaults to 16):
            The divisor for adjusting channel dimensions, ensuring that the number of channels meets hardware
            optimization requirements (e.g., for efficient inference on mobile devices).
        backbone_out_channels (`int`, *optional*, defaults to 512):
            The number of output channels from the backbone network, which represents the dimension of the final
            feature maps extracted by the backbone.
        hidden_act (`str`, *optional*, defaults to `"hswish"`):
            The non-linear activation function used in the hidden layers of the model. Supported functions include
            `"hswish"`, `"relu"`, `"silu"`, and `"gelu"`. `"hswish"` is preferred for mobile-friendly efficient
            inference.
        neck_out_channels (`int`, *optional*, defaults to 96):
            The number of output channels from the neck network, which is responsible for feature fusion and
            refinement before passing features to the head network.
        shortcut (`bool`, *optional*, defaults to `True`):
            Whether to use shortcut connections (residual connections) in the neck network. Shortcut connections help
            alleviate the vanishing gradient problem and improve feature propagation across layers.
        interpolate_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation mode used for upsampling or downsampling feature maps in the neck network. Supported
            modes include `"nearest"` (nearest neighbor interpolation) and `"bilinear"`.
        k (`int`, *optional*, defaults to 50):
            The candidate box number threshold for the head network, which controls the maximum number of text region
            candidates generated during the text detection process.
        kernel_list (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            The list of kernel sizes for convolutional layers in the head network, used for multi-scale feature
            extraction to detect text regions of different sizes.

    Examples:
    ```python
    >>> from transformers import PPOCRV5MobileDetConfig, PPOCRV5MobileDetForTextDetection
    >>> # Initializing a PPOCRV5 Mobile Det configuration
    >>> configuration = PPOCRV5MobileDetConfig()
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = PPOCRV5MobileDetForTextDetection(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    """

    def __init__(
        self,
        backbone_config=None,
        scale=1.0,
        conv_kxk_num=4,
        reduction=4,
        divisor=16,
        backbone_out_channels=512,
        hidden_act="hswish",
        neck_out_channels=96,
        shortcut=True,
        interpolate_mode="nearest",
        k=50,
        kernel_list=[3, 2, 2],
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ---- Backbone ----
        self.backbone_config = backbone_config
        self.scale = scale
        self.conv_kxk_num = conv_kxk_num
        self.reduction = reduction
        self.divisor = divisor
        self.backbone_out_channels = backbone_out_channels
        self.hidden_act = hidden_act

        # ---- Neck ----
        self.neck_out_channels = neck_out_channels
        self.shortcut = shortcut
        self.interpolate_mode = interpolate_mode

        # ---- Head ----
        self.k = k
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


@auto_docstring(custom_intro="ImageProcessor for the PPOCRV5 Mobile Det model.")
class PPOCRV5MobileDetImageProcessor(BaseImageProcessor):
    """
    Image Processor for the PPOCRV5 Mobile Det text detection model.

    This class handles all image preprocessing (resizing, rescaling, normalization, channel flipping)
    and post-processing (converting model logits to detected text boxes) required for the PPOCRV5 Mobile Det model.
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


@auto_docstring(custom_intro="ImageProcessorFast for the PPOCRV5 Mobile Det model.")
class PPOCRV5MobileDetImageProcessorFast(BaseImageProcessorFast):
    """
    Image processor for PPOCRV5 Mobile Det model, handling preprocessing (resizing, normalization)
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


class PPOCRV5MobileDetLearnableAffineBlock(HGNetV2LearnableAffineBlock):
    pass


class PPOCRV5MobileDetAct(nn.Module):
    """
    Activation block with a trainable affine transformation applied after the non-linear activation.
    Supports two activation functions: Hardswish (hswish) for mobile-efficient inference and ReLU.
    """

    def __init__(self, act="hswish"):
        """
        Initialize the activation block with the specified non-linear activation.

        Args:
            act (str, optional): Type of activation function to use. Options are "hswish" and "relu".
                Defaults to "hswish".
        """
        super().__init__()
        if act == "hswish":
            self.act = nn.Hardswish()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Act must be hswish or relu.")
        self.lab = PPOCRV5MobileDetLearnableAffineBlock()

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.act(hidden_state)
        hidden_state = self.lab(hidden_state)
        return hidden_state


class PPOCRV5MobileDetConvBNLayer(HGNetV2ConvLayer):
    pass


class PPOCRV5MobileDetLearnableRepLayer(nn.Module):
    """
    Learnable Reparameterization Layer (RepLayer) that fuses multiple convolution branches
    (kxk and 1x1) with an optional identity branch. This layer enables structural reparameterization
    for efficient inference while maintaining training flexibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        act: str,
        stride: int,
        num_conv_branches: int,
        groups: int = 1,
    ):
        """
        Initialize the PPOCRV5MobileDetLearnableRepLayer with multiple convolution branches and optional identity connection.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the kxk convolution kernel.
            act (str): Activation function type (passed to PPOCRV5MobileDetAct block).
            stride (int): Stride of the convolution operations.
            num_conv_branches (int): Number of kxk convolution branches to stack.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        """
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = (
            nn.BatchNorm2d(num_features=in_channels, momentum=0.9)
            if out_channels == in_channels and stride == 1
            else None
        )

        self.conv_kxk = nn.ModuleList(
            [
                PPOCRV5MobileDetConvBNLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                    activation=None,
                )
                for _ in range(self.num_conv_branches)
            ]
        )

        self.conv_1x1 = (
            PPOCRV5MobileDetConvBNLayer(in_channels, out_channels, 1, stride, groups=groups, activation=None)
            if kernel_size > 1
            else None
        )

        self.lab = PPOCRV5MobileDetLearnableAffineBlock()
        self.act = PPOCRV5MobileDetAct(act=act)

    def forward(self, hidden_state: torch.Tensor):
        """
        Forward pass of the PPOCRV5MobileDetLearnableRepLayer, fusing all enabled branches and applying post-processing.

        Args:
            hidden_state (torch.Tensor): Input feature tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output feature tensor of shape (B, out_channels, H', W').
        """
        output = 0
        if self.identity is not None:
            output += self.identity(hidden_state)

        if self.conv_1x1 is not None:
            output += self.conv_1x1(hidden_state)

        for conv in self.conv_kxk:
            output += conv(hidden_state)

        hidden_state = self.lab(output)
        if self.stride != 2:
            hidden_state = self.act(hidden_state)
        return hidden_state


class PPOCRV5MobileDetSELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Layer for channel-wise feature recalibration.
    This layer adaptively scales channel features based on their importance,
    improving the model's ability to capture informative features.
    """

    def __init__(self, channel, reduction=4):
        """
        Initialize the PPOCRV5MobileDetSELayer with channel reduction factor.

        Args:
            channel (int): Number of input/output channels.
            reduction (int, optional): Reduction factor for the squeeze operation (controls the size of the hidden layer).
                Defaults to 4.
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, hidden_state: torch.Tensor):
        """
        Apply squeeze-and-excitation to the input feature tensor.

        Args:
            hidden_state (torch.Tensor): Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Recalibrated feature tensor of shape (B, C, H, W).
        """
        identity = hidden_state
        hidden_state = self.avg_pool(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.relu(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.hardsigmoid(hidden_state)
        hidden_state = torch.multiply(identity, hidden_state)
        return hidden_state


class PPOCRV5MobileDetLCNetV3Block(nn.Module):
    """
    Lightweight Convolutional Network V3 (LCNetV3) Block, the core building block of the PPOCRV5 Mobile Det backbone.
    Consists of a depthwise PPOCRV5MobileDetLearnableRepLayer, an optional SE Layer, and a pointwise PPOCRV5MobileDetLearnableRepLayer.
    Optimized for mobile devices with low computational complexity and high efficiency.
    """

    def __init__(self, in_channels, out_channels, act, dw_size, stride, use_se, conv_kxk_num, reduction):
        """
        Initialize the PPOCRV5MobileDetLCNetV3Block with specified parameters for depthwise and pointwise layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            act (str): Activation function type (passed to PPOCRV5MobileDetAct block).
            dw_size (int): Kernel size for the depthwise convolution.
            stride (int): Stride of the depthwise convolution.
            use_se (bool): Whether to enable the SE Layer for channel recalibration.
            conv_kxk_num (int): Number of kxk convolution branches in PPOCRV5MobileDetLearnableRepLayer.
            reduction (int): Reduction factor for the SE Layer (if enabled).
        """
        super().__init__()
        self.use_se = use_se
        self.dw_conv = PPOCRV5MobileDetLearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            act=act,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
        )
        if use_se:
            self.se = PPOCRV5MobileDetSELayer(in_channels, reduction=reduction)
        self.pw_conv = PPOCRV5MobileDetLearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=act,
            stride=1,
            num_conv_branches=conv_kxk_num,
        )

    def forward(self, hidden_state: torch.Tensor):
        """
        Forward pass of the PPOCRV5MobileDetLCNetV3Block, applying depthwise convolution, optional SE, and pointwise convolution.

        Args:
            hidden_state (torch.Tensor): Input feature tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output feature tensor of shape (B, out_channels, H', W').
        """
        hidden_state = self.dw_conv(hidden_state)
        if self.use_se:
            hidden_state = self.se(hidden_state)
        hidden_state = self.pw_conv(hidden_state)
        return hidden_state


class PPOCRV5MobileDetBackbone(nn.Module):
    """
    Backbone network for PPOCRV5 Mobile Det, built with LCNetV3Blocks.
    Extracts multi-scale feature maps from input images, which are passed to the neck network for further fusion.
    Optimized for mobile devices with lightweight, efficient layers and channel scaling.
    """

    def __init__(self, config: PPOCRV5MobileDetConfig):
        """
        Initialize the PPOCRV5MobileDetBackbone with the specified model configuration.

        Args:
            config (PPOCRV5MobileDetConfig): Configuration object containing backbone hyperparameters.
        """
        super().__init__()

        self.backbone_config = config.backbone_config
        self.out_channels = make_divisible(config.backbone_out_channels * config.scale, config.divisor)

        self.conv1 = PPOCRV5MobileDetConvBNLayer(
            in_channels=3,
            out_channels=make_divisible(16 * config.scale, config.divisor),
            kernel_size=3,
            stride=2,
            activation=None,
        )

        def _build_blocks(block_key):
            return nn.Sequential(
                *[
                    PPOCRV5MobileDetLCNetV3Block(
                        in_channels=make_divisible(in_c * config.scale, config.divisor),
                        out_channels=make_divisible(out_c * config.scale, config.divisor),
                        act=config.hidden_act,
                        dw_size=k,
                        stride=s,
                        use_se=se,
                        conv_kxk_num=config.conv_kxk_num,
                        reduction=config.reduction,
                    )
                    for i, (k, in_c, out_c, s, se) in enumerate(self.backbone_config[block_key])
                ]
            )

        self.blocks2 = _build_blocks("blocks2")
        self.blocks3 = _build_blocks("blocks3")
        self.blocks4 = _build_blocks("blocks4")
        self.blocks5 = _build_blocks("blocks5")
        self.blocks6 = _build_blocks("blocks6")

        mv_c = self.backbone_config["layer_list_out_channels"]

        self.out_channels = [
            make_divisible(self.backbone_config["blocks3"][-1][2] * config.scale, config.divisor),
            make_divisible(self.backbone_config["blocks4"][-1][2] * config.scale, config.divisor),
            make_divisible(self.backbone_config["blocks5"][-1][2] * config.scale, config.divisor),
            make_divisible(self.backbone_config["blocks6"][-1][2] * config.scale, config.divisor),
        ]

        self.layer_list = nn.ModuleList(
            [
                nn.Conv2d(self.out_channels[0], int(mv_c[0] * config.scale), 1, 1, 0),
                nn.Conv2d(self.out_channels[1], int(mv_c[1] * config.scale), 1, 1, 0),
                nn.Conv2d(self.out_channels[2], int(mv_c[2] * config.scale), 1, 1, 0),
                nn.Conv2d(self.out_channels[3], int(mv_c[3] * config.scale), 1, 1, 0),
            ]
        )
        self.out_channels = [
            int(mv_c[0] * config.scale),
            int(mv_c[1] * config.scale),
            int(mv_c[2] * config.scale),
            int(mv_c[3] * config.scale),
        ]

    def forward(self, hidden_state: torch.Tensor, output_hidden_states: bool, return_dict: bool = True):
        """
        Forward pass of the backbone network, extracting multi-scale feature maps and optional hidden states.

        Args:
            hidden_state (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            output_hidden_states (bool): Whether to return all intermediate hidden states for analysis.
            return_dict (bool, optional): Unused (for consistency with other modules). Defaults to True.

        Returns:
            tuple:
                - list[torch.Tensor]: Multi-scale feature maps after projection (4 feature maps).
                - torch.Tensor: Last hidden state (output of blocks6).
                - tuple[torch.Tensor, ...]: All intermediate hidden states (if output_hidden_states is True).
        """
        hidden_states = () if output_hidden_states else None

        out_list = []
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.conv1(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.blocks2(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        hidden_state = self.blocks3(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        out_list.append(hidden_state)
        hidden_state = self.blocks4(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        out_list.append(hidden_state)
        hidden_state = self.blocks5(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        out_list.append(hidden_state)
        hidden_state = self.blocks6(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        out_list.append(hidden_state)
        last_hidden_state = hidden_state
        out_list[0] = self.layer_list[0](out_list[0])
        out_list[1] = self.layer_list[1](out_list[1])
        out_list[2] = self.layer_list[2](out_list[2])
        out_list[3] = self.layer_list[3](out_list[3])

        return out_list, last_hidden_state, hidden_states


class PPOCRV5MobileDetSEModule(nn.Module):
    """
    Simplified Squeeze-and-Excitation (SE) Module for the neck network.
    Applies channel-wise recalibration with a clamped activation to stabilize training.
    """

    def __init__(self, in_channels, reduction=4):
        """
        Initialize the PPOCRV5MobileDetSEModule with channel reduction factor.

        Args:
            in_channels (int): Number of input/output channels.
            reduction (int, optional): Reduction factor for the squeeze operation. Defaults to 4.
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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

    def forward(self, inputs):
        """
        Apply simplified squeeze-and-excitation to the input tensor.

        Args:
            inputs (torch.Tensor): Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Recalibrated feature tensor of shape (B, C, H, W).
        """
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = torch.clamp(0.2 * outputs + 0.5, min=0.0, max=1.0)
        return inputs * outputs


class PPOCRV5MobileDetRSELayer(nn.Module):
    """
    Residual Squeeze-and-Excitation (RSE) Layer for the neck network.
    Combines a 1x1/3x3 convolution with an SE Module and an optional residual shortcut connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        """
        Initialize the PPOCRV5MobileDetRSELayer with convolution and residual connection parameters.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            shortcut (bool, optional): Whether to enable the residual shortcut connection. Defaults to True.
        """
        super().__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False,
        )
        self.se_block = PPOCRV5MobileDetSEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        """
        Forward pass of the PPOCRV5MobileDetRSELayer, applying convolution, SE recalibration, and optional residual connection.

        Args:
            ins (torch.Tensor): Input feature tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output feature tensor of shape (B, out_channels, H, W).
        """
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class PPOCRV5MobileDetNeck(nn.Module):
    """
    Neck network for PPOCRV5 Mobile Det, responsible for multi-scale feature fusion.
    Uses RSELayers to process backbone features and upsampling to fuse features at the same spatial scale,
    then concatenates the fused features for input to the head network.
    """

    def __init__(self, config: PPOCRV5MobileDetConfig, in_channels: list[int]):
        """
        Initialize the PPOCRV5MobileDetNeck with the specified model configuration and input channels.

        Args:
            config (PPOCRV5MobileDetConfig): Configuration object containing neck hyperparameters.
            in_channels (list[int]): List of input channels from the backbone's multi-scale feature maps.
        """
        super().__init__()
        self.interpolate_mode = config.interpolate_mode

        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                PPOCRV5MobileDetRSELayer(
                    in_channels[i], config.neck_out_channels, kernel_size=1, shortcut=config.shortcut
                )
            )
            self.inp_conv.append(
                PPOCRV5MobileDetRSELayer(
                    config.neck_out_channels, config.neck_out_channels // 4, kernel_size=3, shortcut=config.shortcut
                )
            )

    def forward(self, x):
        """
        Forward pass of the neck network, fusing multi-scale backbone features.

        Args:
            x (list[torch.Tensor]): List of multi-scale feature maps from the backbone (4 feature maps).

        Returns:
            torch.Tensor: Concatenated fused feature tensor of shape (B, C, H, W).
        """
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode=self.interpolate_mode)  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode=self.interpolate_mode)  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode=self.interpolate_mode)  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.interpolate(p5, scale_factor=8, mode=self.interpolate_mode)
        p4 = F.interpolate(p4, scale_factor=4, mode=self.interpolate_mode)
        p3 = F.interpolate(p3, scale_factor=2, mode=self.interpolate_mode)

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class PPOCRV5MobileDetHead(nn.Module):
    """
    Head sub-module for PPOCRV5 Mobile Det, responsible for generating text segmentation maps.
    Uses two transposed convolutions for upsampling to recover the original image spatial scale,
    and a sigmoid activation to produce binary segmentation logits.
    """

    def __init__(self, in_channels, kernel_list=[3, 2, 2]):
        """
        Initialize the Head sub-module with convolution and upsampling parameters.

        Args:
            in_channels (int): Number of input channels from the neck network.
            kernel_list (list[int], optional): List of kernel sizes for the three convolution layers:
                [conv1 kernel, conv2 transposed kernel, conv3 transposed kernel]. Defaults to [3, 2, 2].
        """
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

    def forward(self, x):
        """
        Forward pass of the Head sub-module, generating binary segmentation logits.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Binary segmentation logits of shape (B, 1, H', W') (original image scale).
        """
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x


class PPOCRV5MobileDetDBHead(nn.Module):
    """
    Head network for PPOCRV5 Mobile Det, wrapping the Head sub-module to generate text segmentation maps.
    Contains the `k` parameter for candidate box selection during post-processing.
    """

    def __init__(self, config: PPOCRV5MobileDetConfig):
        """
        Initialize the PPOCRV5MobileDetHead with the specified model configuration.

        Args:
            config (PPOCRV5MobileDetConfig): Configuration object containing head hyperparameters.
        """
        super().__init__()
        self.k = config.k
        self.binarize = PPOCRV5MobileDetHead(config.neck_out_channels, config.kernel_list)

    def forward(self, x):
        """
        Forward pass of the head network, generating text segmentation (shrink) maps.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C, H, W) from the neck network.

        Returns:
            torch.Tensor: Binary segmentation shrink maps of shape (B, 1, H', W').
        """
        shrink_maps = self.binarize(x)
        return shrink_maps


@dataclass
class PPOCRV5MobileDetModelOutput(ModelOutput):
    """
    Output class for the PPOCRV5MobileDetModel.

    Args:
        logits (torch.FloatTensor, optional): Binary segmentation logits from the head network,
            shape (B, 1, H, W).
        last_hidden_state (torch.FloatTensor, optional): Last hidden state from the backbone network,
            shape (B, C, H, W).
        hidden_states (tuple[torch.FloatTensor], optional): Tuple of all intermediate hidden states from the backbone,
            if `output_hidden_states` is True.
    """

    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


class PPOCRV5MobileDetPreTrainedModel(PreTrainedModel):
    """
    Base class for all PPOCRV5 Mobile Det pre-trained models. Handles model initialization,
    configuration, and loading of pre-trained weights, following the Transformers library conventions.
    """

    config: PPOCRV5MobileDetConfig
    base_model_prefix = "pp_ocrv5_mobile_det"
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, PPOCRV5MobileDetConvBNLayer):
            nn.init.kaiming_normal_(module.convolution.weight)

        if isinstance(module, PPOCRV5MobileDetHead):
            nn.init.constant_(module.conv_bn1.weight, 1.0)
            nn.init.constant_(module.conv_bn1.bias, 1e-4)
            nn.init.constant_(module.conv_bn2.weight, 1.0)
            nn.init.constant_(module.conv_bn2.bias, 1e-4)
            nn.init.kaiming_uniform_(module.conv2.weight)
            nn.init.kaiming_uniform_(module.conv3.weight)

        if isinstance(module, PPOCRV5MobileDetRSELayer):
            nn.init.kaiming_uniform_(module.in_conv.weight)


@auto_docstring(custom_intro="The PPOCRV5 Mobile Det model.")
class PPOCRV5MobileDetModel(PPOCRV5MobileDetPreTrainedModel):
    """
    Core PPOCRV5 Mobile Det model, consisting of Backbone, Neck, and Head networks.
    Generates binary text segmentation maps for text detection tasks.
    """

    def __init__(self, config: PPOCRV5MobileDetConfig):
        """
        Initialize the PPOCRV5MobileDetModel with the specified configuration.

        Args:
            config (PPOCRV5MobileDetConfig): Configuration object containing all model hyperparameters.
        """
        super().__init__(config)

        self.backbone = PPOCRV5MobileDetBackbone(config)
        self.neck = PPOCRV5MobileDetNeck(config, self.backbone.out_channels)
        self.head = PPOCRV5MobileDetDBHead(config)
        self.post_init()

    def forward(
        self,
        hidden_state: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPOCRV5MobileDetModelOutput]:
        """
        Forward pass of the PPOCRV5MobileDetModel, processing input images to generate segmentation logits.

        Args:
            hidden_state (torch.FloatTensor): Input image tensor of shape (B, 3, H, W) (pixel values).
            output_hidden_states (bool, optional): Whether to return all intermediate hidden states from the backbone.
                If None, uses the configuration's `output_hidden_states` value.
            return_dict (bool, optional): Whether to return a `PPOCRV5MobileDetModelOutput` object or a tuple.
                If None, uses the configuration's `use_return_dict` value.

        Returns:
            Union[tuple[torch.FloatTensor], PPOCRV5MobileDetModelOutput]: Model output containing segmentation logits,
                last hidden state, and optional hidden states.
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

        return PPOCRV5MobileDetModelOutput(
            logits=hidden_state,
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


@dataclass
class PPOCRV5MobileDetForObjectDetectionOutput(BaseModelOutputWithNoAttention):
    """
    Output class for PPOCRV5MobileDetForObjectDetection. Extends BaseModelOutputWithNoAttention
    to include text segmentation logits.

    Args:
        logits (torch.FloatTensor, optional): Binary segmentation logits from the head network,
            shape (B, 1, H, W).
        shape (torch.FloatTensor, optional): Unused placeholder for consistency with object detection output formats.
        last_hidden_state (torch.FloatTensor, optional): Last hidden state from the backbone network.
        hidden_states (tuple[torch.FloatTensor], optional): Tuple of all intermediate hidden states from the backbone,
            if `output_hidden_states` is True.
    """

    logits: Optional[torch.FloatTensor] = None
    shape: Optional[torch.FloatTensor] = None


@auto_docstring(custom_intro="ObjectDetection for the PPOCRV5 Mobile Det model.")
class PPOCRV5MobileDetForObjectDetection(PPOCRV5MobileDetPreTrainedModel):
    """
    PPOCRV5 Mobile Det model for object (text) detection tasks. Wraps the core PPOCRV5MobileDetModel
    and returns outputs compatible with the Transformers object detection API.
    """

    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: PPOCRV5MobileDetConfig):
        """
        Initialize the PPOCRV5MobileDetForObjectDetection with the specified configuration.

        Args:
            config (PPOCRV5MobileDetConfig): Configuration object containing all model hyperparameters.
        """
        super().__init__(config)
        self.model = PPOCRV5MobileDetModel(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[list[dict]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.FloatTensor], PPOCRV5MobileDetForObjectDetectionOutput]:
        """
        Forward pass of the PPOCRV5MobileDetForObjectDetection model, processing input images to generate
        text detection logits.

        Args:
            pixel_values (torch.FloatTensor): Input image tensor of shape (B, 3, H, W) (preprocessed pixel values).
            labels (list[dict], optional): Unused placeholder for training (object detection labels). Defaults to None.
            output_hidden_states (bool, optional): Whether to return all intermediate hidden states from the backbone.
                If None, uses the configuration's `output_hidden_states` value.
            return_dict (bool, optional): Whether to return a `PPOCRV5MobileDetForObjectDetectionOutput` object or a tuple.
                If None, uses the configuration's `use_return_dict` value.
            **kwargs: Additional unused keyword arguments for compatibility.

        Returns:
            Union[tuple[torch.FloatTensor], PPOCRV5MobileDetForObjectDetectionOutput]: Detection output containing
                segmentation logits, last hidden state, and optional hidden states.
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

        return PPOCRV5MobileDetForObjectDetectionOutput(
            logits=outputs.logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
        )


__all__ = [
    "PPOCRV5MobileDetForObjectDetection",
    "PPOCRV5MobileDetImageProcessor",
    "PPOCRV5MobileDetImageProcessorFast",
    "PPOCRV5MobileDetConfig",
    "PPOCRV5MobileDetModel",
    "PPOCRV5MobileDetPreTrainedModel",
]
