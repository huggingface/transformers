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

from typing import Union, Optional, Tuple
import numpy as np
import torch

from transformers.utils import (
    ExplicitEnum,
)

class BoundingBoxFormat(ExplicitEnum):
    """Coordinate formats to represent a bounding box"""
    XYXY = "xyxy"  # absolute coordinates
    XYWH = "xywh"  # absolute coordinates
    XCYCWH = "xcycwh"  # absolute coordinates
    RELATIVE_XYXY = "relative_xyxy"  # relative coordinates
    RELATIVE_XYWH  = "relative_xywh"  # relative coordinates
    RELATIVE_XCYCWH = "relative_xcycwh"  # relative coordinates

_relative_formats = (BoundingBoxFormat.RELATIVE_XYXY, BoundingBoxFormat.RELATIVE_XYWH, BoundingBoxFormat.RELATIVE_XCYCWH)

def _infer_box_format(bbox: Union[torch.Tensor, np.ndarray]) -> BoundingBoxFormat:
    # check if bbox has 4 coordinates
    pass

def _xywh_to_xyxy(xywh: torch.Tensor, inplace: bool) -> torch.Tensor:
    xyxy = xywh if inplace else xywh.clone()
    xyxy[..., 2:].add_(xyxy[..., :2])
    return xyxy

def _xyxy_to_xywh(xyxy: torch.Tensor, inplace: bool) -> torch.Tensor:
    xywh = xyxy if inplace else xyxy.clone()
    xywh[..., 2:].sub_(xywh[..., :2])
    return xywh


# adapted from https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_meta.py
def _xcycwh_to_xyxy(xcycwh: torch.Tensor, inplace: bool) -> torch.Tensor:
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
    xyxy = xyxy if inplace else xyxy.clone()
    # (x2 - x1) = width and (y2 - y1) = height
    xyxy[..., 2:].sub_(xyxy[..., :2])  # => x, y, width, height
    # cx and cy can be written with different terms
    # (x1 * 2 + width)/2 = x1 + width/2 = x1 + (x2-x1)/2 = (x1 + x2)/2 = cy
    # (y1 * 2 + height)/2 = y1 + height/2 = y1 + (y2-y1)/2 = (y1 + y2)/2 = cy
    rounding_mode=None if xyxy.is_floating_point() else "floor"
    xyxy[..., :2].mul_(2).add_(xyxy[..., 2:]).div_(2, rounding_mode=rounding_mode)
    return xyxy

def transform_box_format(bbox: Union[torch.Tensor, np.ndarray], dest_format: BoundingBoxFormat, orig_format: Optional[BoundingBoxFormat] = None, img_shape: Optional[Tuple[int, int]] = None, inplace: bool = False):
    if orig_format is None:
        orig_format = _infer_box_format(bbox)
    
    # no transformation is needed
    if orig_format == dest_format:
        return bbox
    
    if dest_format in _relative_formats and img_shape is None:
        raise ValueError(f"Image shape is required if the desired destination format is {dest_format}")

    if orig_format in _relative_formats and img_shape is None:
        raise ValueError(f"Image shape is required if the input format format is {dest_format}")

    
    # convert to xyxy
    if orig_format == BoundingBoxFormat.XYWH:
        bbox = _xywh_to_xyxy(bbox, inplace)
    elif orig_format == BoundingBoxFormat.XCYCWH:
        bbox = _xcycwh_to_xyxy(bbox, inplace)
    # boxes are now in xyxy format
    if dest_format == BoundingBoxFormat.XYWH:
        bbox = _xyxy_to_xywh(bbox, inplace)
    elif dest_format == BoundingBoxFormat.XCYCWH:
        bbox = _xyxy_to_xcycwh(bbox, inplace)

    return bbox

# General transformations
def horizontal_flip():
    pass

def vertical_flip():
    pass

def resize():
    pass

def crop():
    pass

boxes_xywh = torch.Tensor([[10, 20, 50, 60],
            [25, 35, 40, 45],
            [5, 5, 100, 100],
            [15, 30, 55, 75],
            [60, 70, 30, 20],
            [0, 0, 50, 50],
            [45, 10, 90, 80],
            [100, 200, 120, 130],
            [80, 90, 40, 60],
            [110, 120, 70, 90]])

# assert torch.equal(boxes_xywh, transform_box_format(boxes_xywh, dest_format = BoundingBoxFormat.XYWH, orig_format = BoundingBoxFormat.XYWH))

# res = transform_box_format(boxes_xywh, dest_format = BoundingBoxFormat.XYXY, orig_format = BoundingBoxFormat.XYWH)
# xywh = torch.Tensor([[10, 20, 60, 80],
#         [25, 35, 65, 80],
#         [5, 5, 105, 105],
#         [15, 30, 70, 105],
#         [60, 70, 90, 90],
#         [0, 0, 50, 50],
#         [45, 10, 135, 90],
#         [100, 200, 220, 330],
#         [80, 90, 120, 150],
#         [110, 120, 180, 210]])
# assert torch.equal(res, xywh)
