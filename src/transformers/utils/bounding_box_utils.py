# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

from typing import Optional, Tuple, Union

from transformers.utils import (
    ExplicitEnum,
)

from .import_utils import is_torch_available


if is_torch_available():
    import torch


class BoundingBoxFormat(ExplicitEnum):
    """Coordinate formats to represent a bounding box."""

    XYXY = "xyxy"  # absolute coordinates
    XYWH = "xywh"  # absolute coordinates
    XCYCWH = "xcycwh"  # absolute coordinates
    RELATIVE_XYWH = "relative_xywh"  # relative coordinates
    RELATIVE_XCYCWH = "relative_xcycwh"  # relative coordinates


def _is_relative_format(format):
    """
    Check if the bounding box format is relative.

    Args:
        format (str): The format of the bounding box.

    Returns:
        bool: True if the format is relative, False otherwise.
    """
    return format.startswith("relative")


def _xywh_to_xyxy(xywh: torch.Tensor, inplace: bool) -> torch.Tensor:
    """
    Convert bounding box format from XYWH to XYXY.

    Args:
        xywh (torch.Tensor): The bounding box in XYWH format.
        inplace (bool): If True, perform operation in-place.

    Returns:
        torch.Tensor: The bounding box in XYXY format.
    """
    xyxy = xywh if inplace else xywh.clone()
    xyxy[..., 2:].add_(xyxy[..., :2])
    return xyxy


def _xyxy_to_xywh(xyxy: torch.Tensor, inplace: bool) -> torch.Tensor:
    """
    Convert bounding box format from XYXY to XYWH.

    Args:
        xyxy (torch.Tensor): The bounding box in XYXY format.
        inplace (bool): If True, perform operation in-place.

    Returns:
        torch.Tensor: The bounding box in XYWH format.
    """
    xyxy = xyxy if inplace else xyxy.clone()
    xyxy[..., 2:].sub_(xyxy[..., :2])
    return xyxy


# adapted from https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_meta.py
def _xcycwh_to_xyxy(xcycwh: torch.Tensor, inplace: bool) -> torch.Tensor:
    """
    Convert bounding box format from XCYCWH to XYXY.

    Args:
        xcycwh (torch.Tensor): The bounding box in XCYCWH format.
        inplace (bool): If True, perform operation in-place.

    Returns:
        torch.Tensor: The bounding box in XYXY format.
    """
    xcycwh = xcycwh if inplace else xcycwh.clone()
    # Trick to do fast division by 2 and ceil
    rounding_mode = None if xcycwh.is_floating_point() else "floor"
    half_wh = xcycwh[..., 2:].div(-2, rounding_mode=rounding_mode).abs_()
    # (xc - width / 2) = x1 and (yc - height / 2) = x1
    xcycwh[..., :2].sub_(half_wh)
    # (x1 + width) = x2 and (y1 + height) = y2
    xcycwh[..., 2:].add_(xcycwh[..., :2])
    return xcycwh


# adapted from https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_meta.py
def _xyxy_to_xcycwh(xyxy: torch.Tensor, inplace: bool) -> torch.Tensor:
    """
    Convert bounding box format from XYXY to XCYCWH.

    Args:
        xyxy (torch.Tensor): The bounding box in XYXY format.
        inplace (bool): If True, perform operation in-place.

    Returns:
        torch.Tensor: The bounding box in XCYCWH format.
    """
    xyxy = xyxy if inplace else xyxy.clone()
    # (x2 - x1) = width and (y2 - y1) = height
    xyxy[..., 2:].sub_(xyxy[..., :2])  # => x, y, width, height
    # cx and cy can be written with different terms
    # (x1 * 2 + width)/2 = x1 + width/2 = x1 + (x2-x1)/2 = (x1 + x2)/2 = cy
    # (y1 * 2 + height)/2 = y1 + height/2 = y1 + (y2-y1)/2 = (y1 + y2)/2 = cy
    rounding_mode = None if xyxy.is_floating_point() else "floor"
    xyxy[..., :2].mul_(2).add_(xyxy[..., 2:]).div_(2, rounding_mode=rounding_mode)
    return xyxy


def _relxywh_to_xyxy(relxywh: torch.Tensor, img_shape: Tuple[int, int], inplace: bool) -> torch.Tensor:
    """
    Convert relative XYWH bounding box format to absolute XYXY format.

    Args:
        relxywh (torch.Tensor): The bounding box in relative XYWH format.
        img_shape (Tuple[int, int]): The shape of the image (height, width).
        inplace (bool): If True, perform operation in-place.

    Returns:
        torch.Tensor: The bounding box in absolute XYXY format.
    """
    relxywh = relxywh if inplace else relxywh.clone()
    # convert to relative_xyxy
    relxywh[..., 2:].add_(relxywh[..., :2])
    # convert to xywh
    relxywh.multiply_(img_shape.repeat(2).flip(0))
    return relxywh


def _relxcycwh_to_xyxy(relxcycwh: torch.Tensor, img_shape: Tuple[int, int], inplace: bool) -> torch.Tensor:
    relxcycwh = relxcycwh if inplace else relxcycwh.clone()
    # convert to relative_xyxy
    relxyxy = _xcycwh_to_xyxy(relxcycwh, inplace)
    # convert to xyxy
    relxyxy.multiply_(img_shape.repeat(2).flip(0))
    return relxyxy


def _xyxy_to_relxywh(xyxy: torch.Tensor, img_shape: Tuple[int, int], inplace: bool) -> torch.Tensor:
    xyxy = xyxy if inplace else xyxy.clone()
    # to xywh
    xyxy = _xyxy_to_xywh(xyxy, inplace)
    # divide by (height, width) to make coordinates in relative format (rel_xywh)
    xyxy.divide_(img_shape.repeat(2).flip(0))
    return xyxy


def _xyxy_to_relxcycwh(xyxy: torch.Tensor, img_shape: Tuple[int, int], inplace: bool) -> torch.Tensor:
    xyxy = xyxy if inplace else xyxy.clone()
    # to xcycwh
    xyxy = _xyxy_to_xcycwh(xyxy, inplace)
    # divide by (height, width) to make coordinates in relative format (rel_xcycwh)
    xyxy.divide_(img_shape.repeat(2).flip(0))
    return xyxy


def transform_box_format(
    bbox: torch.Tensor,
    orig_format: BoundingBoxFormat,
    dest_format: BoundingBoxFormat,
    img_shape: Optional[Union[Tuple[int, int], torch.Tensor]] = None,
    inplace: bool = False,
    do_round: bool = False,
):
    """
    Transform a bounding box from one format to another.

    Args:
        bbox (torch.Tensor): The bounding box to transform. orig_format (BoundingBoxFormat): The
        original format of the bounding box. dest_format (BoundingBoxFormat): The desired destination format of the
        bounding box. img_shape (Optional[Tuple[int, int]]): The shape of the image (height, width), required for
        relative formats. inplace (bool): If True, perform operation in-place.
        do_round (bool):
            If True, and the destination format is not a relative format, the coordinates of boxes are rounded.

    Returns:
        Union[torch.Tensor, np.ndarray]: The transformed bounding box.

    Raises:
        ValueError: If image shape is required but not provided.
    """
    # no transformation is needed
    if orig_format == dest_format:
        return bbox

    bbx_format_members = BoundingBoxFormat.__members__.values()
    if orig_format not in bbx_format_members:
        raise ValueError("orig_format is not a valid BoundingBoxFormat.")
    if dest_format not in bbx_format_members:
        raise ValueError("dest_format is not a valid BoundingBoxFormat.")

    if _is_relative_format(orig_format):
        bbox = bbox.type(torch.float32)
        if img_shape is None:
            raise ValueError(f"Image shape (height, width) is required if the input format format is {dest_format}")
        if not isinstance(img_shape, torch.Tensor):
            img_shape = torch.Tensor(img_shape)

    if _is_relative_format(dest_format):
        if img_shape is None:
            raise ValueError(
                f"Image shape (height, width) is required if the desired destination format is {dest_format}"
            )
        if not isinstance(img_shape, torch.Tensor):
            img_shape = torch.Tensor(img_shape)

    bbox = bbox.type(torch.float)
    # convert to xyxy
    if orig_format == BoundingBoxFormat.XYWH:
        bbox = _xywh_to_xyxy(bbox, inplace)
    elif orig_format == BoundingBoxFormat.XCYCWH:
        bbox = _xcycwh_to_xyxy(bbox, inplace)
    elif orig_format == BoundingBoxFormat.RELATIVE_XYWH:
        bbox = _relxywh_to_xyxy(bbox, img_shape, inplace)
    elif orig_format == BoundingBoxFormat.RELATIVE_XCYCWH:
        bbox = _relxcycwh_to_xyxy(bbox, img_shape, inplace)

    # boxes are now in xyxy format

    if dest_format == BoundingBoxFormat.XYWH:
        bbox = _xyxy_to_xywh(bbox, inplace)
    elif dest_format == BoundingBoxFormat.XCYCWH:
        bbox = _xyxy_to_xcycwh(bbox, inplace)
    elif dest_format == BoundingBoxFormat.RELATIVE_XYWH:
        bbox = _xyxy_to_relxywh(bbox, img_shape, inplace)
    elif dest_format == BoundingBoxFormat.RELATIVE_XCYCWH:
        bbox = _xyxy_to_relxcycwh(bbox, img_shape, inplace)

    # If rounded is requested and destination format is not a relative format, round coordinates.
    if do_round and not _is_relative_format(dest_format):
        bbox = torch.round(bbox, decimals=1)

    return bbox
