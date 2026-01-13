import math
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np
import pyclipper
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2.functional import InterpolationMode

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
from ...utils import ModelOutput, auto_docstring, filter_out_non_signature_kwargs, logging
from ...utils.generic import TensorType


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
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the input images. Defaults to 3 for RGB color images; set to 1 for grayscale
            images.
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
        fix_nan (`bool`, *optional*, defaults to `False`):
            Whether to enable the mechanism to fix NaN values that may occur during model training. Enabling this
            can stabilize training but may introduce a small amount of additional computational overhead.

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
        backbone_config: dict | None = None,
        scale: float = 1.0,
        conv_kxk_num: int = 4,
        reduction: int = 4,
        divisor: int = 16,
        num_channels: int = 3,
        backbone_out_channels: int = 512,
        hidden_act: str = "hswish",
        neck_out_channels: int = 96,
        shortcut: bool = True,
        interpolate_mode: str = "nearest",
        k: int = 50,
        kernel_list: list = [3, 2, 2],
        fix_nan: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Backbone
        self.backbone_config = backbone_config
        self.scale = scale
        self.conv_kxk_num = conv_kxk_num
        self.reduction = reduction
        self.divisor = divisor
        self.num_channels = num_channels
        self.backbone_out_channels = backbone_out_channels
        self.hidden_act = hidden_act

        # Neck
        self.neck_out_channels = neck_out_channels
        self.shortcut = shortcut
        self.interpolate_mode = interpolate_mode

        # Head
        self.k = k
        self.kernel_list = kernel_list
        self.fix_nan = fix_nan


import math
from typing import Optional, Union

import cv2
import numpy as np
import pyclipper

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import flip_channel_order, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import auto_docstring, filter_out_non_signature_kwargs
from ...utils.generic import TensorType


def unclip(box, unclip_ratio):
    """
    Expands (dilates) a detected text bounding box to recover the full text region.
    The expansion distance is computed based on the contour area and perimeter,
    and Pyclipper is used to perform smooth contour offsetting.

    Args:
        box (np.ndarray): Input contour of shape (N, 2), where N is the number of points.
        unclip_ratio (float): Expansion ratio, typically greater than 1.0.

    Returns:
        np.ndarray: Expanded contour of shape (M, 2).
    """
    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    distance = area * unclip_ratio / length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        expanded = np.array(offset.Execute(distance))
    except ValueError:
        expanded = np.array(offset.Execute(distance)[0])
    return expanded


def get_mini_boxes(contour):
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


def box_score_fast(bitmap, _box):
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
    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def box_score_slow(bitmap, contour):
    """
    Computes the mean score of a contour region in the prediction map using
    the exact polygon shape, which is slower but more accurate.

    Args:
        bitmap (np.ndarray): Binary or float prediction map of shape (H, W).
        contour (np.ndarray): Contour polygon of shape (N, 2).

    Returns:
        float: Mean score within the contour region.
    """
    h, w = bitmap.shape[:2]
    contour = contour.copy()
    contour = np.reshape(contour, (-1, 2))

    xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
    xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
    ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
    ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

    contour[:, 0] = contour[:, 0] - xmin
    contour[:, 1] = contour[:, 1] - ymin

    cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def polygons_from_bitmap(
    pred,
    _bitmap,
    dest_width,
    dest_height,
    box_thresh,
    unclip_ratio,
    min_size,
    max_candidates,
):
    """
    Extracts text polygons from a binary segmentation map.

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
            - boxes (list): List of polygons, each of shape (N, 2).
            - scores (list): List of corresponding scores.
    """

    bitmap = _bitmap
    height, width = bitmap.shape
    width_scale = dest_width / width
    height_scale = dest_height / height
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue

        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue

        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)

        if len(box) > 0:
            _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < min_size + 2:
                continue
        else:
            continue

        box = np.array(box)
        for i in range(box.shape[0]):
            box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
            box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

        boxes.append(box)
        scores.append(score)
    return boxes, scores


def boxes_from_bitmap(
    pred,
    _bitmap,
    dest_width,
    dest_height,
    box_thresh,
    unclip_ratio,
    min_size,
    max_candidates,
    score_mode,
):
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
        score_mode (str): Scoring mode, either "fast" or "slow".

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
        if score_mode == "fast":
            score = box_score_fast(pred, points.reshape(-1, 2))
        else:
            score = box_score_slow(pred, contour)
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
    pred, size, thresh, box_type, box_thresh, unclip_ratio, use_dilation, min_size, max_candidates, score_mode
):
    """
    Main post-processing function to convert model predictions into text boxes.

    Args:
        pred (torch.Tensor): Model output of shape (1, H, W).
        size (torch.Tensor): Original image size (height, width).
        thresh (float): Threshold for binarizing the prediction map.
        box_type (str): Type of boxes to extract, either "quad" or "poly".
        box_thresh (float): Score threshold for filtering boxes.
        unclip_ratio (float): Expansion ratio for unclipping.
        use_dilation (bool): Whether to apply dilation on the segmentation mask.
        min_size (int): Minimum side length of valid boxes.
        max_candidates (int): Maximum number of boxes to extract.
        score_mode (str): Scoring mode, either "fast" or "slow".

    Returns:
        tuple:
            - boxes (list or np.ndarray): Extracted text boxes.
            - scores (list): Corresponding confidence scores.
    """
    pred = pred[0, :, :].cpu().detach().numpy()
    segmentation = pred > thresh
    dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])
    src_h, src_w = size.cpu().detach().numpy()
    if dilation_kernel is not None:
        mask = cv2.dilate(
            np.array(segmentation).astype(np.uint8),
            dilation_kernel,
        )
    else:
        mask = segmentation
    if box_type == "poly":
        boxes, scores = polygons_from_bitmap(
            pred,
            mask,
            src_w,
            src_h,
            box_thresh,
            unclip_ratio,
            min_size,
            max_candidates,
        )
    elif box_type == "quad":
        boxes, scores = boxes_from_bitmap(
            pred, mask, src_w, src_h, box_thresh, unclip_ratio, min_size, max_candidates, score_mode
        )
    else:
        raise ValueError("box_type can only be one of ['quad', 'poly']")
    return boxes, scores


@auto_docstring(custom_intro="ImageProcessor for the PPOCRV5 Mobile Det model.")
class PPOCRV5MobileDetImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]
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
    def __init__(
        self,
        limit_side_len: int = 960,
        limit_type: str = "max",
        max_side_limit: int = 4000,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = [0.406, 0.456, 0.485],
        image_std: Optional[Union[float, list[float]]] = [0.225, 0.224, 0.229],
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
        threshold: float = 0.5,
        target_sizes=None,
        thresh=0.3,
        box_thresh=0.6,
        max_candidates=1000,
        min_size=3,
        unclip_ratio=1.5,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
    ):  
        """
        Converts model outputs into detected text boxes.

        Args:
            preds (torch.Tensor): Model outputs.
            threshold (float): Confidence threshold (unused).
            target_sizes (TensorType or list[tuple]): Original image sizes.
            thresh (float): Binarization threshold.
            box_thresh (float): Box score threshold.
            max_candidates (int): Maximum number of boxes.
            min_size (int): Minimum box size.
            unclip_ratio (float): Expansion ratio.
            use_dilation (bool): Whether to dilate the mask.
            score_mode (str): Scoring mode.
            box_type (str): Box type, "quad" or "poly".

        Returns:
            list[dict]: List of detection results.
        """
        assert score_mode in [
            "slow",
            "fast",
        ], f"Score mode must be in [slow, fast] but got: {score_mode}"
        return self.postprocess(
            preds=preds.logits,
            target_sizes=target_sizes,
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            min_size=min_size,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            box_type=box_type,
        )

    def postprocess(
        self,
        preds,
        target_sizes,
        thresh=0.3,
        box_thresh=0.6,
        max_candidates=1000,
        min_size=3,
        unclip_ratio=1.5,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
    ):
        """
        Post-processes model outputs to extract text boxes.

        Args:
            preds (torch.Tensor): Model logits.
            thresh (float): Binarization threshold.
            target_sizes (list[tuple]): Original image sizes.
            box_thresh (float): Box score threshold.
            max_candidates (int): Maximum number of boxes.
            min_size (int): Minimum box size.
            unclip_ratio (float): Expansion ratio.
            use_dilation (bool): Whether to dilate the mask.
            score_mode (str): Scoring mode.
            box_type (str): Box type.

        Returns:
            list[dict]: List of detection results.
        """
        results = []
        for pred, size in zip(preds, target_sizes):
            box, score = process(
                pred=pred,
                size=size,
                thresh=thresh,
                box_type=box_type,
                box_thresh=box_thresh,
                unclip_ratio=unclip_ratio,
                use_dilation=use_dilation,
                min_size=min_size,
                max_candidates=max_candidates,
                score_mode=score_mode,
            )

            results.append({"scores": score, "boxes": box})
        return results

    def get_image_size(
        self,
        img: np.ndarray,
        limit_side_len: Union[int, None],
        limit_type: Union[str, None],
        max_side_limit: Union[int, None] = None,
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
        images: list[torch.Tensor],
        size: Optional[list[dict[str, int]]],
        do_resize: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        interpolation: Optional[InterpolationMode] = None,
        resample: Optional[PILImageResampling] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocesses a list of images for input to the PPOCRV5 Mobile Det model.

        Args:
            images (list[torch.Tensor]): List of input images.
            size (list[dict[str, int]]): Target size for each image.
            do_resize (bool): Whether to resize images.
            do_rescale (bool): Whether to rescale pixel values.
            rescale_factor (float): Rescale factor.
            do_normalize (bool): Whether to normalize images.
            image_mean (list[float] or float): Mean values for normalization.
            image_std (list[float] or float): Std values for normalization.
            return_tensors (str or TensorType): Type of tensors to return.
            interpolation (InterpolationMode): Interpolation mode for resizing.
            resample (PILImageResampling): PIL resampling mode (unused).

        Returns:
            BatchFeature: Preprocessed images and additional information.
        """
        data = {}
        resize_imgs, target_sizes = [], []
        if do_resize:
            # interpolation = InterpolationMode.BILINEAR
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
        threshold: float = 0.5,
        target_sizes: Union[TensorType, list[tuple]] = None,
        thresh: float = 0.3,
        box_thresh=0.6,
        max_candidates=1000,
        min_size=3,
        unclip_ratio=1.5,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
    ):
        """
        Converts model outputs into detected text boxes.

        Args:
            preds (torch.Tensor): Model outputs.
            threshold (float): Confidence threshold (unused).
            target_sizes (TensorType or list[tuple]): Original image sizes.
            thresh (float): Binarization threshold.
            box_thresh (float): Box score threshold.
            max_candidates (int): Maximum number of boxes.
            min_size (int): Minimum box size.
            unclip_ratio (float): Expansion ratio.
            use_dilation (bool): Whether to dilate the mask.
            score_mode (str): Scoring mode.
            box_type (str): Box type, "quad" or "poly".

        Returns:
            list[dict]: List of detection results.
        """
        assert score_mode in [
            "slow",
            "fast",
        ], f"Score mode must be in [slow, fast] but got: {score_mode}"

        return self.postprocess(
            preds=preds.logits,
            thresh=thresh,
            target_sizes=target_sizes,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            min_size=min_size,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            box_type=box_type,
        )

    def postprocess(
        self,
        preds,
        thresh=0.3,
        target_sizes=None,
        box_thresh=0.6,
        max_candidates=1000,
        min_size=3,
        unclip_ratio=1.5,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
    ):
        """
        Post-processes model outputs to extract text boxes.

        Args:
            preds (torch.Tensor): Model logits.
            thresh (float): Binarization threshold.
            target_sizes (list[tuple]): Original image sizes.
            box_thresh (float): Box score threshold.
            max_candidates (int): Maximum number of boxes.
            min_size (int): Minimum box size.
            unclip_ratio (float): Expansion ratio.
            use_dilation (bool): Whether to dilate the mask.
            score_mode (str): Scoring mode.
            box_type (str): Box type.

        Returns:
            list[dict]: List of detection results.
        """
        results = []
        for pred, size in zip(preds, target_sizes):
            box, score = process(
                pred=pred,
                size=size,
                thresh=thresh,
                box_type=box_type,
                box_thresh=box_thresh,
                unclip_ratio=unclip_ratio,
                use_dilation=use_dilation,
                min_size=min_size,
                max_candidates=max_candidates,
                score_mode=score_mode,
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
        img: torch.Tensor,
        limit_side_len: Union[int, None],
        limit_type: Union[str, None],
        max_side_limit: Union[int, None] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        c, h, w = img.shape
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


def make_divisible(v, divisor: int = 16, min_value=None):
    """
    Ensures that the input value `v` is rounded to the nearest multiple of `divisor`,
    with a minimum value constraint. This is used to adjust channel dimensions for
    hardware-efficient neural network inference (especially on mobile devices).

    Args:
        v (float): Input value to be adjusted (typically channel count).
        divisor (int, optional): The divisor to align the value with. Defaults to 16.
        min_value (int, optional): Minimum allowed value after adjustment. If None,
            defaults to `divisor`.

    Returns:
        int: Adjusted value that is a multiple of `divisor` and meets the minimum value requirement.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LearnableAffineBlock(nn.Module):
    """
    Learnable affine transformation block that applies scale and bias to the input tensor.
    Both scale and bias are trainable parameters, allowing the model to learn optimal
    linear transformations for feature normalization or enhancement.
    """
    def __init__(self, scale_value=1.0, bias_value=0.0):
        """
        Initialize the LearnableAffineBlock with initial scale and bias values.

        Args:
            scale_value (float, optional): Initial value for the scale parameter. Defaults to 1.0.
            bias_value (float, optional): Initial value for the bias parameter. Defaults to 0.0.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, hidden_state: torch.Tensor):
        """
        Apply the affine transformation to the input hidden state.

        Args:
            hidden_state (torch.Tensor): Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Transformed feature tensor after applying scale and bias.
        """
        return self.scale * hidden_state + self.bias


class Act(nn.Module):
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
        else:
            assert act == "relu"
            self.act = nn.ReLU()
        self.lab = LearnableAffineBlock()

    def forward(self, hidden_state: torch.Tensor):
        """
        Apply the non-linear activation followed by the learnable affine transformation.

        Args:
            hidden_state (torch.Tensor): Input feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Activated and transformed feature tensor.
        """
        return self.lab(self.act(hidden_state))


class ConvBNLayer(nn.Module):
    """
    Convolution-Batch Normalization layer block, a fundamental building block for modern CNNs.
    Applies 2D convolution followed by batch normalization, with He Kaiming initialization for the convolution weights.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        """
        Initialize the ConvBNLayer with specified convolution and batch normalization parameters.

        Args:
            in_channels (int): Number of input channels for the convolution layer.
            out_channels (int): Number of output channels for the convolution layer.
            kernel_size (int): Size of the convolution kernel (square kernel).
            stride (int): Stride of the convolution operation.
            groups (int, optional): Number of groups for grouped convolution (used for depthwise convolutions).
                Defaults to 1 (standard convolution).
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight)

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)

    def forward(self, hidden_state: torch.Tensor):
        """
        Apply convolution followed by batch normalization to the input tensor.

        Args:
            hidden_state (torch.Tensor): Input feature tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output feature tensor of shape (B, out_channels, H', W').
        """
        hidden_state = self.conv(hidden_state)
        hidden_state = self.bn(hidden_state)
        return hidden_state


class LearnableRepLayer(nn.Module):
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
        Initialize the LearnableRepLayer with multiple convolution branches and optional identity connection.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the kxk convolution kernel.
            act (str): Activation function type (passed to Act block).
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
                ConvBNLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                )
                for _ in range(self.num_conv_branches)
            ]
        )

        self.conv_1x1 = ConvBNLayer(in_channels, out_channels, 1, stride, groups=groups) if kernel_size > 1 else None

        self.lab = LearnableAffineBlock()
        self.act = Act(act=act)

    def forward(self, hidden_state: torch.Tensor):
        """
        Forward pass of the LearnableRepLayer, fusing all enabled branches and applying post-processing.

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


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Layer for channel-wise feature recalibration.
    This layer adaptively scales channel features based on their importance,
    improving the model's ability to capture informative features.
    """
    def __init__(self, channel, reduction=4):
        """
        Initialize the SELayer with channel reduction factor.

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


class LCNetV3Block(nn.Module):
    """
    Lightweight Convolutional Network V3 (LCNetV3) Block, the core building block of the PPOCRV5 Mobile Det backbone.
    Consists of a depthwise LearnableRepLayer, an optional SE Layer, and a pointwise LearnableRepLayer.
    Optimized for mobile devices with low computational complexity and high efficiency.
    """
    def __init__(self, in_channels, out_channels, act, dw_size, stride, use_se, conv_kxk_num, reduction):
        """
        Initialize the LCNetV3Block with specified parameters for depthwise and pointwise layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            act (str): Activation function type (passed to Act block).
            dw_size (int): Kernel size for the depthwise convolution.
            stride (int): Stride of the depthwise convolution.
            use_se (bool): Whether to enable the SE Layer for channel recalibration.
            conv_kxk_num (int): Number of kxk convolution branches in LearnableRepLayer.
            reduction (int): Reduction factor for the SE Layer (if enabled).
        """
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            act=act,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
        )
        if use_se:
            self.se = SELayer(in_channels, reduction=reduction)
        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=act,
            stride=1,
            num_conv_branches=conv_kxk_num,
        )

    def forward(self, hidden_state: torch.Tensor):
        """
        Forward pass of the LCNetV3Block, applying depthwise convolution, optional SE, and pointwise convolution.

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

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=make_divisible(16 * config.scale, config.divisor),
            kernel_size=3,
            stride=2,
        )

        def _build_blocks(block_key):
            return nn.Sequential(
                *[
                    LCNetV3Block(
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


class SEModule(nn.Module):
    """
    Simplified Squeeze-and-Excitation (SE) Module for the neck network.
    Applies channel-wise recalibration with a clamped activation to stabilize training.
    """
    def __init__(self, in_channels, reduction=4):
        """
        Initialize the SEModule with channel reduction factor.

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


class RSELayer(nn.Module):
    """
    Residual Squeeze-and-Excitation (RSE) Layer for the neck network.
    Combines a 1x1/3x3 convolution with an SE Module and an optional residual shortcut connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        """
        Initialize the RSELayer with convolution and residual connection parameters.

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
        nn.init.kaiming_uniform_(self.in_conv.weight)
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        """
        Forward pass of the RSELayer, applying convolution, SE recalibration, and optional residual connection.

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
                RSELayer(in_channels[i], config.neck_out_channels, kernel_size=1, shortcut=config.shortcut)
            )
            self.inp_conv.append(
                RSELayer(
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


class Head(nn.Module):
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

        nn.init.constant_(self.conv_bn1.weight, 1.0)
        nn.init.constant_(self.conv_bn1.bias, 1e-4)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
        )
        nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4, momentum=0.9)
        self.relu2 = nn.ReLU()

        nn.init.constant_(self.conv_bn2.weight, 1.0)
        nn.init.constant_(self.conv_bn2.bias, 1e-4)

        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
        )
        nn.init.kaiming_uniform_(self.conv3.weight)

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


class PPOCRV5MobileDetHead(nn.Module):
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
        self.binarize = Head(config.neck_out_channels, config.kernel_list)

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
        self.head = PPOCRV5MobileDetHead(config)

    def forward(
        self,
        hidden_state: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
