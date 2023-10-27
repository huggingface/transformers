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

from .utils import (
    ExplicitEnum,
)

class BoundingBoxFormat(ExplicitEnum):
    XYXY = "xyxy"
    XYWH = "xywh"
    XCYCWH = "XCYCWH"

def _infer_box_format(bbox: Union[torch.Tensor, np.ndarray]) -> BoundingBoxFormat:
    # check if bbox has 4 coordinates
    pass

def _to_xyxy(bbox_xyxy):
    pass

def _to_xywh(bbox_xyxy):
    pass

def _to_xcycwh(bbox, orig_format):
    pass

def transform_box_format(bbox: Union[torch.Tensor, np.ndarray], dest_format: BoundingBoxFormat, orig_format: Optional[BoundingBoxFormat] = None, img_shape: Optional[Tuple[int, int]] = None):
    if orig_format is None:
        orig_format = _infer_box_format(bbox)
    
    # no transformation is needed
    if orig_format == dest_format:
        return bbox
    
    if dest_format is BoundingBoxFormat.XCYCWH and img_shape is None:
        raise ValueError(f"Image shape is required if the desired destination format is {dest_format}")

    if orig_format is BoundingBoxFormat.XCYCWH and img_shape is None:
        raise ValueError(f"Image shape is required if the input format format is {dest_format}")

    if orig_format == BoundingBoxFormat.XYXY:
        x, y, x2,y2 = bbox
    elif orig_format == BoundingBoxFormat.XYWH:
        x, y, w, h = bbox
        x2, y2 = x+w, y+h
    elif orig_format == BoundingBoxFormat.XCYCWH:
        cx, cy, w, h = bbox
        x = cx - w/2
        y = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

    if dest_format == BoundingBoxFormat.XYXY:
        return x, y, x2, y2
    elif dest_format == BoundingBoxFormat.XYWH:
        return _to_xywh((x,y,x2,y2))
    elif dest_format == BoundingBoxFormat.XCYCWH:
        return _to_xcycwh((x,y,x2,y2))