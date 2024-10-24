#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Finetuning any 🤗 Transformers model supported by AutoModelForObjectDetection for object detection leveraging the Trainer API."""

import logging
from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

import albumentations as A
import torch
from datasets import load_dataset

from transformers import (
    AutoImageProcessor,
    HfArgumentParser,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.46.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/object-detection/requirements.txt")


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = [
        {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        for category, area, bbox in zip(categories, areas, bboxes)
    ]

    return {
        "image_id": image_id,
        "annotations": annotations,
    }

def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Convert bounding boxes from YOLO format to Pascal VOC format.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format.
        image_size (Tuple[int, int]): Image size in format (height, width).

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max).
    """
    # Convert center to corners format
    boxes = center_to_corners_format(boxes)

    # Convert to absolute coordinates
    height, width = image_size
    boxes *= torch.tensor([[width, height, width, height]])

    return boxes

def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations and format annotations in COCO format for object detection task.

    Args:
        examples (Mapping[str, Any]): Input examples for augmentation.
        transform (A.Compose): Augmentation pipeline.
        image_processor (AutoImageProcessor): Processor for image transformations.
        return_pixel_mask (bool): Flag to return pixel mask.

    Returns:
        BatchFeature: Transformed images and additional information.
    """
    # Ensure images and annotations are properly processed
    if 'images' not in examples or 'annotations' not in examples:
        logger.warning("Required keys: 'images' and 'annotations' are missing from examples.")
        return BatchFeature()  # Return empty batch feature on error

    images = examples['images']
    #annotations = examples['annotations']

    # Apply transformations
    transformed_images = [transform(image=image)['image'] for image in images]

    # Process images with image processor
    processed_images = image_processor(images=transformed_images, return_tensors="pt")

    # Return a BatchFeature object
    return BatchFeature(data=processed_images.data, pixel_mask=return_pixel_mask)

    # Further code would go here (e.g., model training, dataset preparation)
    
