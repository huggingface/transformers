# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""

# from typing import Tuple

import torch


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [(x), (y), (x + w), (y + h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_cxcywh(x):
    x, y, w, h = x.unbind(-1)
    b = [(x + 0.5 * w), (y + 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x, y, X, Y = x.unbind(-1)
    b = [(x), (y), (X - x), (Y - y)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Batched version of box area. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes: Tensor of shape (..., 4)

    Returns:
    - areas: Tensor of shape (...,)
    """
    x0, y0, x1, y1 = boxes.unbind(-1)
    return (x1 - x0) * (y1 - y0)
