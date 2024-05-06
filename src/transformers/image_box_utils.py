import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from .utils import (
    is_numpy_array,
    is_torch_available,
    is_torch_tensor,
)


if is_torch_available():
    import torch


# TODO: align with transformers logging
logger = logging.getLogger(__name__)

ArrayType = Union["torch.Tensor", np.ndarray]


SUPPORTED_BOX_FORMATS = [
    "absolute_xyxy",
    "absolute_xywh",
    "absolute_xcycwh",
    "relative_xyxy",
    "relative_xywh",
    "relative_xcycwh",
    "coco",
    "pascal_voc",
    "yolo",
    "albumentations",
    "xyxy",
    "xywh",
    "xcycwh",
]


BOX_FORMAT_MAPPING = {
    "coco": "absolute_xywh",
    "pascal_voc": "absolute_xyxy",
    "yolo": "relative_xcycwh",
    "albumentations": "relative_xyxy",
    "xyxy": "absolute_xyxy",
    "xywh": "absolute_xywh",
    "xcycwh": "relative_xcycwh",
    "absolute_xyxy": "absolute_xyxy",
    "absolute_xywh": "absolute_xywh",
    "absolute_xcycwh": "absolute_xcycwh",
    "relative_xyxy": "relative_xyxy",
    "relative_xywh": "relative_xywh",
    "relative_xcycwh": "relative_xcycwh",
}


class BoxOutOfBoundsError(Exception):
    pass


class NotValidBoxError(Exception):
    pass


def is_array_type(obj) -> bool:
    return is_torch_tensor(obj) or isinstance(obj, np.ndarray)


def is_numpy_scalar(obj) -> bool:
    return isinstance(obj, (np.floating, np.integer))


def is_numpy_object(obj) -> bool:
    return is_numpy_array(obj) or is_numpy_scalar(obj)


def validate_box_format(format: str) -> None:
    if format not in SUPPORTED_BOX_FORMATS:
        raise ValueError(f"Unsupported box format: {format}, supported formats are {SUPPORTED_BOX_FORMATS}")


def map_box_format(format: str) -> str:
    return BOX_FORMAT_MAPPING[format]


def get_depth_of_nested_objects(obj) -> int:
    if is_torch_tensor(obj) or is_numpy_array(obj):
        return obj.ndim
    elif isinstance(obj, (list, tuple)):
        if not obj:
            return 2  # empty list/tuple could not be a box, we treat it as image with no boxes
        return get_depth_of_nested_objects(obj[0]) + 1
    elif isinstance(obj, (int, float)) or is_numpy_scalar(obj):
        return 0
    else:
        raise NotImplementedError(f"Unsupported type of object: {type(obj)}")


# ------------------------------------------------------------------------------------------------
# Basic framework-agnostic operations
# ------------------------------------------------------------------------------------------------


def _shape(obj: ArrayType) -> List[int]:
    if is_torch_tensor(obj) or is_numpy_array(obj):
        return list(obj.shape)
    else:
        raise NotImplementedError(f"Unsupported type {type(obj)}")


def _split(objects: ArrayType, axis: int = -1) -> List[ArrayType]:
    if is_torch_tensor(objects):
        return objects.unbind(dim=axis)
    elif is_numpy_array(objects):
        n_splits = objects.shape[axis]
        return [np.take(objects, i, axis=axis) for i in range(n_splits)]
    else:
        raise NotImplementedError(f"Unsupported type {type(objects)}")


def _stack(objects: List[ArrayType], axis: int = 0) -> ArrayType:
    if all(is_torch_tensor(x) for x in objects):
        return torch.stack(objects, dim=axis)
    elif all(is_numpy_object(x) for x in objects):
        return np.stack(objects, axis=axis)
    else:
        raise NotImplementedError(f"Unsupported type in {[type(x) for x in objects]}")


def _make_like(obj: ArrayType, like: ArrayType) -> ArrayType:
    if is_torch_tensor(like):
        return torch.tensor(obj).to(like.device)
    elif is_numpy_array(like):
        return np.array(obj)
    else:
        raise NotImplementedError(f"Unsupported type {type(like)}")


def _max(obj: Union[ArrayType, int, float]) -> Union[ArrayType, int, float]:
    if is_torch_tensor(obj):
        return obj.max().item()
    elif is_numpy_array(obj):
        return obj.max()
    elif isinstance(obj, (int, float)) or is_numpy_scalar(obj):
        return obj
    else:
        raise NotImplementedError(f"Unsupported type {type(obj)}")


def _min(obj: Union[ArrayType, int, float]) -> Union[ArrayType, int, float]:
    if is_torch_tensor(obj):
        return obj.min().item()
    elif is_numpy_array(obj):
        return obj.min()
    elif isinstance(obj, (int, float)) or is_numpy_scalar(obj):
        return obj
    else:
        raise NotImplementedError(f"Unsupported type {type(obj)}")


def _expand_on_zero_dim(obj: Union[ArrayType, List, Tuple]) -> Union[ArrayType, List, Tuple]:
    if is_torch_tensor(obj):
        return obj.unsqueeze(dim=0)
    elif is_numpy_array(obj):
        return np.expand_dims(obj, axis=0)
    elif isinstance(obj, list):
        return [obj]
    elif isinstance(obj, tuple):
        return (obj,)
    else:
        raise NotImplementedError(f"Unsupported type {type(obj)}")


# ------------------------------------------------------------------------------------------------
# Format converters
# ------------------------------------------------------------------------------------------------


def _convert_xcycwh_to_xyxy(boxes: ArrayType) -> ArrayType:
    """
    Convert boxes from (center_x, center_y, width, height) to (x0, y0, x1, y1)
    Args:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    Returns:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    """
    cx, cy, w, h, *rest = _split(boxes, axis=-1)
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return _stack([x_min, y_min, x_max, y_max, *rest], axis=-1)


def _convert_xyxy_to_xcycwh(boxes: ArrayType) -> ArrayType:
    """
    Convert boxes from (x0, y0, x1, y1) to (center_x, center_y, width, height)
    Args:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    Returns:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    """
    x_min, y_min, x_max, y_max, *rest = _split(boxes, axis=-1)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return _stack([cx, cy, w, h, *rest], axis=-1)


def _convert_xywh_to_xyxy(boxes: ArrayType) -> ArrayType:
    """
    Convert boxes from (x, y, w, h) to (x0, y0, x1, y1)
    Args:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    Returns:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    """
    x_min, y_min, w, h, *rest = _split(boxes, axis=-1)
    x_max = x_min + w
    y_max = y_min + h
    return _stack([x_min, y_min, x_max, y_max, *rest], axis=-1)


def _convert_xyxy_to_xywh(boxes: ArrayType) -> ArrayType:
    """
    Convert boxes from (x0, y0, x1, y1) to (x, y, w, h)
    Args:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    Returns:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    """
    x_min, y_min, x_max, y_max, *rest = _split(boxes, axis=-1)
    w = x_max - x_min
    h = y_max - y_min
    return _stack([x_min, y_min, w, h, *rest], axis=-1)


def _convert_relative_to_absolute(boxes: ArrayType, image_size: ArrayType) -> ArrayType:
    """
    Convert boxes from relative to absolute coordinates
    Args:
        boxes: tensor of shape (N, 4) where N is the number of boxes
        image_size: tensor of shape (2,) where 2 is the image size (height, width)
    Returns:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    """
    image_size = _make_like(image_size, like=boxes)
    height, width = _split(image_size, axis=-1)
    x_rel, y_rel, x_or_w_rel, y_or_h_rel, *rest = _split(boxes, axis=-1)
    x_abs = x_rel * width
    y_abs = y_rel * height
    x_or_w_abs = x_or_w_rel * width
    y_or_h_abs = y_or_h_rel * height
    return _stack([x_abs, y_abs, x_or_w_abs, y_or_h_abs, *rest], axis=-1)


def _convert_absolute_to_relative(boxes: ArrayType, image_size: ArrayType) -> ArrayType:
    """
    Convert boxes from absolute to relative coordinates
    Args:
        boxes: tensor of shape (N, 4) where N is the number of boxes
        image_size: tensor of shape (2,) where 2 is the image size (height, width)
    Returns:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    """
    height, width = _make_like(image_size, like=boxes)
    x_abs, y_abs, x_or_w_abs, y_or_h_abs, *rest = _split(boxes, axis=-1)
    x_rel = x_abs / width
    y_rel = y_abs / height
    x_or_w_rel = x_or_w_abs / width
    y_or_h_rel = y_or_h_abs / height
    return _stack([x_rel, y_rel, x_or_w_rel, y_or_h_rel, *rest], axis=-1)


# ------------------------------------------------------------------------------------------------
# Box conversion operations
# ------------------------------------------------------------------------------------------------


def log_or_raise(message: str, check: str) -> None:
    if check == "warn":
        logger.warning(message)
    elif check == "raise":
        raise BoxOutOfBoundsError(message)


def check_relative_boxes_xyxy(boxes: ArrayType, check: str) -> None:
    # Check if boxes are in the range [0, 1]
    max_value = _max(boxes)
    min_value = _min(boxes)
    if max_value > 1 or min_value < 0:
        message = (
            "Relative boxes are expected to be in the range [0, 1], "
            "but some of your boxes are outside of this range."
        )
        log_or_raise(message, check)

    # Check if width and height are positive
    x_min, y_min, x_max, y_max, *_ = _split(boxes, axis=-1)
    w = x_max - x_min
    h = y_max - y_min
    if _min(w) <= 0 or _min(h) <= 0:
        message = (
            "Width and height of your boxes are expected to be positive, "
            "but some of your boxes have non-positive width or height."
        )
        log_or_raise(message, check)


def check_absolute_boxes_xyxy(boxes: ArrayType, check: str) -> None:
    # Check if boxes coordinates are non-negative
    max_value = _max(boxes)
    min_value = _min(boxes)
    if min_value < 0:
        message = "Some of your boxes are outside of image boundaries."
        log_or_raise(message, check)

    # Check that maximum coordinate is greater than 2,
    # otherwise it is very unlikely for absolute boxes
    if max_value < 2:
        message = (
            "Maximum coordinate value of your boxes is less than 2.0, which "
            "is unexpected for absolute boxes. You might have relative boxes."
        )
        log_or_raise(message, check)

    # Check if width and height are positive
    x_min, y_min, x_max, y_max, *_ = _split(boxes, axis=-1)
    w = x_max - x_min
    h = y_max - y_min
    if _min(w) <= 0 or _min(h) <= 0:
        message = (
            "Width and height of your boxes are expected to be positive, "
            "but some of your boxes have non-positive width or height."
        )
        log_or_raise(message, check)


def _convert_boxes_arrays(
    boxes: ArrayType,
    input_format: str,
    output_format: str,
    image_size: Optional[ArrayType | List] = None,
    check: Optional[str] = "warn",
) -> ArrayType:
    """
    Convert array/tensor boxes from one format to another

    Args:
        boxes: 1d/2d/3d array, where the last dim is 4 or more elements representing the box coordinates
        input_format: format of the input boxes
        output_format: format of the output boxes
        image_size: tensor of shape (2,) where 2 is the image size (height, width)
    Returns:
        boxes: tensor of shape (N, 4) where N is the number of boxes
    """

    input_scale, input_coords_type = input_format.split("_")
    output_scale, output_coords_type = output_format.split("_")

    if input_scale != output_scale and image_size is None:
        raise ValueError(
            f"Expected `image_size` to be provided when converting from {input_scale} to {output_scale} coordinates"
        )

    # type conversion to intermediate "xyxy" format from any input format
    if input_coords_type == "xywh":
        xyxy_boxes = _convert_xywh_to_xyxy(boxes)
    elif input_coords_type == "xcycwh":
        xyxy_boxes = _convert_xcycwh_to_xyxy(boxes)
    else:
        xyxy_boxes = boxes

    # here we can go with some basic validation for "xyxy" format
    if check is not None and input_scale == "relative":
        check_relative_boxes_xyxy(xyxy_boxes, check)
    elif check is not None and input_scale == "absolute":
        check_absolute_boxes_xyxy(xyxy_boxes, check)

    # type conversion from intermediate "xyxy" to any output format
    if output_coords_type == "xywh":
        converted_boxes = _convert_xyxy_to_xywh(xyxy_boxes)
    elif output_coords_type == "xcycwh":
        converted_boxes = _convert_xyxy_to_xcycwh(xyxy_boxes)
    else:
        converted_boxes = xyxy_boxes

    # scale conversion (relative -> absolute or absolute -> relative)
    if input_scale == "relative" and output_scale == "absolute":
        converted_boxes = _convert_relative_to_absolute(converted_boxes, image_size)
    elif input_scale == "absolute" and output_scale == "relative":
        converted_boxes = _convert_absolute_to_relative(converted_boxes, image_size)

    return converted_boxes


def _convert_boxes_recursively(
    boxes: ArrayType | List | Tuple,
    input_format: str,
    output_format: str,
    image_size: Optional[ArrayType | List] = None,
    check: Optional[str] = "warn",
):
    depth = get_depth_of_nested_objects(boxes)

    # check for "empty" boxes/images
    if len(boxes) == 0:
        return boxes
    elif is_array_type(boxes) and _shape(boxes)[-1] == 0:
        return boxes

    # check for unsupported depths
    if depth > 3:
        raise ValueError(
            f"Expected boxes to have at most 3 dimensions (n_images, n_boxes, coords), got {depth} dimensions."
        )

    # check image size is provided correctly
    if depth == 3 and image_size is not None:
        image_depth = get_depth_of_nested_objects(image_size)
        if image_depth == 1:
            raise ValueError(f"Expected get list of image_sizes but get a single image size: {image_size}.")
        elif len(image_size) != len(boxes):
            raise ValueError(
                f"Expected the same number of images and image sizes, got {len(boxes)} images and {len(image_size)} image sizes."
            )
    if depth < 3 and image_size is not None and len(image_size) != 2:
        raise ValueError(f"Expected image_size to have 2 elements (height, width), got {len(image_size)} elements.")

    # check box has enough coordinates
    if depth == 1 and len(boxes) < 4:
        raise ValueError(f"Expected boxes to have at least 4 elements, got {len(boxes)} elements.")
    if is_array_type(boxes) and _shape(boxes)[-1] < 4:
        raise ValueError(f"Expected boxes to have at least 4 elements, got {_shape(boxes)[-1]} elements.")

    # Base of recursion.
    if depth == 2 and is_array_type(boxes):
        return _convert_boxes_arrays(boxes, input_format, output_format, image_size, check)

    # Recursive approach.
    elif depth == 1 and isinstance(boxes, (list, tuple) or is_array_type(boxes)):
        boxes_2d = _expand_on_zero_dim(boxes)
        return _convert_boxes_recursively(boxes_2d, input_format, output_format, image_size, check)[0]

    elif depth == 2 and isinstance(boxes, (list, tuple)):
        np_boxes = np.array(boxes)
        np_converted_boxes = _convert_boxes_recursively(np_boxes, input_format, output_format, image_size, check)
        converted_boxes = np_converted_boxes.tolist()
        if isinstance(boxes, tuple):
            converted_boxes = tuple([tuple(box) for box in converted_boxes])
        return converted_boxes

    elif depth == 3 and isinstance(boxes, (list, tuple)):
        image_sizes = image_size if image_size is not None else [None] * len(boxes)
        return type(boxes)(
            [_convert_boxes_recursively(b, input_format, output_format, s, check) for b, s in zip(boxes, image_sizes)]
        )

    elif depth == 3 and is_array_type(boxes):
        image_sizes = image_size if image_size is not None else [None] * len(boxes)
        converted_boxes = [
            _convert_boxes_recursively(b, input_format, output_format, s, check) for b, s in zip(boxes, image_sizes)
        ]
        return _stack(converted_boxes, axis=0)

    else:
        raise ValueError(f"Unsupported type of boxes: {type(boxes)}")


def convert_boxes(
    boxes: ArrayType | List | Tuple,
    input_format: str,
    output_format: str,
    image_size: Optional[ArrayType | List | Tuple] = None,
    check: Optional[str] = "warn",
):
    """
    Convert boxes from one format to another.

    Bounding boxes can be provided as:
        - A single box (torch.Tensor | np.ndarray | List[float, ...]): Box of shape (B,), where B is 4 or more elements, 
        and the first 4 elements are the coordinates of the box. 
        - Boxes from one image (torch.Tensor | np.ndarray | List[List[float, ...]]): A set of boxes (N, B) where N is 
        the number of boxes.
        - Boxes from multiple images (torch.Tensor | np.ndarray | List[torch.Tensor | np.ndarray] | List[List[List[float, ...]]]):
        A set of images with their boxes (I, N, B), where I is number of images. Can be represented as a single 3D array/tensor 
        or most likely as a list of 2D arrays/tensors. 

    Supported input/output bounding box formats:
        - `absolute_xyxy` (aliases: `pascal_voc`, `xyxy`): [x_min, y_min, x_max, y_max]
        - `absolute_xywh` (aliases: `coco`, `xywh`): [x_min, y_min, width, height]
        - `absolute_xcycwh`: [center_x, center_y, width, height]
        - `relative_xyxy` (aliases: `albumentations`): [x_min, y_min, x_max, y_max] normalized to [0, 1] by image size
        - `relative_xywh`: [x_min, y_min, width, height] normalized to [0, 1] by image size
        - `relative_xcycwh` (aliases: `yolo`, `xcycwh`): [center_x, center_y, width, height] normalized to [0, 1] by image size

    Args:
        boxes: A single box / boxes from one image / boxes from multiple images.
        input_format (str): Format of the input boxes.
        output_format (str): Format of the output boxes.
        image_size (Optional[torch.tensor | List | Tuple]): (height, width) of the image if boxes are from one image, or list of (height, width) tuples 
            if provided boxes from multiple images.
        check (Optional[str]): Whether to check bounding boxes or not.
            - `"warn"` raise warning if bounding boxes are outside image borders or have negative width/height.
            - `"raise"` raise error if bounding boxes are outside image borders or have negative width/height.
            - `None` no checks are performed.

    Returns:
        boxes: Boxes converted to output format with the same shape and data type as the input boxes
    """

    validate_box_format(input_format)
    validate_box_format(output_format)

    # map to standard format: relative/absolute + xyxy/xywh/xcycwh
    input_format = map_box_format(input_format)
    output_format = map_box_format(output_format)

    if not (is_array_type(boxes) or isinstance(boxes, (list, tuple))):
        raise ValueError(f"Unsupported type of boxes: {type(boxes)}")

    return _convert_boxes_recursively(boxes, input_format, output_format, image_size, check)
