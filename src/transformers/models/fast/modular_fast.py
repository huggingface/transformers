# coding=utf-8
# Copyright 2025 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for FAST."""

import math

import numpy as np

from ...utils.import_utils import is_cv2_available, is_scipy_available, is_torch_available, requires_backends


if is_cv2_available():
    import cv2

if is_scipy_available():
    import scipy.ndimage as ndi
    from scipy.spatial import ConvexHull

if is_torch_available():
    import torch
    import torch.nn.functional as F
from transformers.models.textnet.image_processing_textnet import TextNetImageProcessor


def connected_components(image, connectivity=8):
    """
    Computes connected components of a binary image using SciPy.

    Parameters:
        image (np.ndarray): Binary input image (0s and 1s)
        connectivity (int): Connectivity, 4 or 8 (default is 8)

    Returns:
        labels (np.ndarray): Labeled output image
        num_labels (int): Number of labels found
    """
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)  # 4-connectivity

    labels, num_labels = ndi.label(image, structure=structure)
    return num_labels, labels


def compute_min_area_rect(points):
    """
    Compute the minimum area rotated bounding rectangle around a set of 2D points.

    Args:
        points (np.ndarray): Nx2 array of (x, y) coordinates.

    Returns:
        tuple: ((cx, cy), (w, h), angle) where
            - (cx, cy) is the center of the rectangle,
            - (w, h) are the width and height of the rectangle,
            - angle is the rotation angle in degrees.
    """
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    edges = np.diff(hull_points, axis=0, append=hull_points[:1])
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
    edge_angles = np.unique(edge_angles)

    min_area = float("inf")
    best_box = None

    for angle in edge_angles:
        # Rotate points by -angle (clockwise)
        R = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        rotated = points @ R.T

        # Bounding box in rotated space
        min_x, min_y = rotated.min(axis=0)
        max_x, max_y = rotated.max(axis=0)
        w = max_x - min_x
        h = max_y - min_y
        area = w * h

        if area < min_area:
            min_area = area
            best_box = (min_x, min_y, max_x, max_y, angle, w, h)

    min_x, min_y, max_x, max_y, angle, w, h = best_box
    center_rotated = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    R_inv = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = center_rotated @ R_inv.T

    angle_deg = np.degrees(angle)

    # we ensure angle is in the range [-90, 0)
    while angle_deg >= 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180

    return (tuple(center), (w, h), angle_deg)


def get_box_points(rect):
    """
    Computes the four corner points of a rotated rectangle in OpenCV's order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]

    Args:
        rect (tuple): ((cx, cy), (w, h), angle)
                      - Center coordinates (cx, cy)
                      - Width and height (w, h)
                      - Rotation angle in degrees

    Returns:
        np.ndarray: (4, 2) array of corner points in OpenCV order.
    """
    (center_x, center_y), (width, height), angle_degrees = rect
    angle_radians = np.radians(angle_degrees)

    cos_angle = np.cos(angle_radians) * 0.5
    sin_angle = np.sin(angle_radians) * 0.5

    # compute top-left and top-right corners
    top_left_x = center_x - sin_angle * height - cos_angle * width
    top_left_y = center_y + cos_angle * height - sin_angle * width
    top_left = [top_left_x, top_left_y]

    top_right_x = center_x + sin_angle * height - cos_angle * width
    top_right_y = center_y - cos_angle * height - sin_angle * width
    top_right = [top_right_x, top_right_y]

    # mirror across the center to get the other two corners
    bottom_right = [2 * center_x - top_left_x, 2 * center_y - top_left_y]
    bottom_left = [2 * center_x - top_right_x, 2 * center_y - top_right_y]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


class FastImageProcessor(TextNetImageProcessor):
    r"""
    min_area (`int`, *optional*, defaults to 250):
        Minimum area (in pixels) for a region to be considered a valid detection.
        Regions smaller than this threshold will be ignored during post-processing.
    pooling_size (`int`, *optional*, defaults to 9):
        Size of the pooling window used during region proposal aggregation or feature map downsampling.
        This controls the granularity of spatial features extracted from the image.
    """

    def __init__(self, min_area: int = 250, pooling_size: int = 9, **super_kwargs):
        super().__init__(self, **super_kwargs)
        self.min_area = min_area
        self.pooling_size = pooling_size

    def post_process_text_detection(self, output, target_sizes=None, threshold=0.5, output_type="boxes"):
        """
        Post-processes the raw model output to generate bounding boxes and scores for text detection.

        Args:
            output (dict): Dictionary containing model outputs. Must include key `"logits"` (Tensor of shape [B, C, H, W]).
            target_sizes (List[Tuple[int, int]], optional): Original image sizes (height, width) for each item in the batch.
                                                            Used to scale detection results back to original image dimensions.
            threshold (float): Confidence threshold for filtering low-score text regions.
            output_type (str): "boxes" (rotated rectangles) or "polygons" (polygon).

        Returns:
            List[Dict]: Each dict contains:
                - "boxes": np.ndarray of shape (N, 5) if output_type="boxes", or (N, 8) if output_type="polygons"
                - "scores": np.ndarray of shape (N,)
        """
        if output_type not in ["boxes", "polygons"]:
            raise ValueError(f"Invalid output_type: {output_type}. Must be 'boxes' or 'polygons'.")
        out = output["logits"]
        batch_size, _, H, W = out.shape

        # generate score maps
        texts = F.interpolate(out[:, 0:1, :, :], size=(H, W), mode="nearest")
        texts = F.max_pool2d(
            texts, kernel_size=self.pooling_size // 2 + 1, stride=1, padding=(self.pooling_size // 2) // 2
        )
        score_maps = torch.sigmoid(texts)
        score_maps = score_maps.squeeze(1)

        # generate label maps
        kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B x H x W
        labels_ = []
        for kernel in kernels.cpu().numpy():
            _, label_ = connected_components(kernel)
            labels_.append(label_)
        labels_ = torch.from_numpy(np.array(labels_)).unsqueeze(1).float()
        labels = (
            F.max_pool2d(
                labels_, kernel_size=self.pooling_size // 2 + 1, stride=1, padding=(self.pooling_size // 2) // 2
            )
            .squeeze(1)
            .to(torch.int32)
        )

        results = []
        for i in range(batch_size):
            if target_sizes is not None:
                orig_h, orig_w = target_sizes[i]
                scale_x = orig_w / W
                scale_y = orig_h / H
            else:
                scale_x = scale_y = 1.0

            keys = torch.unique(labels_[i], sorted=True)
            if output_type == "boxes":
                bboxes, scores = self._get_rotated_boxes(keys, labels[i], score_maps[i], (scale_x, scale_y), threshold)
            elif output_type == "polygons":
                bboxes, scores = self._get_polygons(keys, labels[i], score_maps[i], (scale_x, scale_y), threshold)
            else:
                raise ValueError(f"Unsupported output_type: {output_type}")

            results.append({"boxes": bboxes, "scores": scores})

        return results

    def _get_rotated_boxes(
        self,
        keys: torch.Tensor,
        label: torch.Tensor,
        score: torch.Tensor,
        scales: tuple[float, float],
        threshold: float,
    ) -> tuple[list[list[tuple[int, int]]], list[float]]:
        """
        Generates rotated rectangular bounding boxes for connected components.

        Args:
            keys (Tensor): Unique instance labels.
            label (Tensor): Label map (H x W).
            score (Tensor): Confidence map (H x W).
            scales (Tuple[float, float]): Scaling factors (x, y) to match original image dimensions.
            threshold (float): Minimum average score for a region to be considered valid.

        Returns:
            Tuple[List[List[int]], List[float]]:
                - List of rotated rectangle bounding boxes as flattened coordinates.
                - List of corresponding confidence scores.
        """
        bounding_boxes = []
        scores = []
        for index in range(1, len(keys)):
            i = keys[index]
            ind = label == i
            ind_np = ind.data.cpu().numpy()
            points = np.array(np.where(ind_np)).transpose((1, 0))
            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue
            score_i = score[ind].mean().item()
            if score_i < threshold:
                label[ind] = 0
                continue

            rect = compute_min_area_rect(points[:, ::-1])
            alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
            rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
            bounding_box = get_box_points(rect) * scales

            bounding_box = bounding_box.astype("int32")
            bounding_boxes.append([tuple(point) for point in bounding_box.tolist()])
            scores.append(score_i)
        return bounding_boxes, scores

    def _get_polygons(
        self,
        keys: torch.Tensor,
        label: torch.Tensor,
        score: torch.Tensor,
        scales: tuple[float, float],
        threshold: float,
    ) -> tuple[list[list[int]], list[float]]:
        """
        Generates polygonal bounding boxes using OpenCV contours for connected components.

        Note:
            Requires OpenCV backend (`cv2`) to be available.

        Args:
            keys (Tensor): Unique labels.
            label (Tensor): Label map (H x W).
            score (Tensor): Score map (H x W).
            scales (Tuple[float, float]): Scaling factors (x, y).
            threshold (float): Minimum average score for a valid region.

        Returns:
            Tuple[List[List[int]], List[float]]:
                - List of polygon contour bounding boxes as flattened coordinates.
                - List of corresponding confidence scores.
        """
        requires_backends(self, "cv2")
        bounding_boxes = []
        scores = []
        for index in range(1, len(keys)):
            i = keys[index]
            ind = label == i
            ind_np = ind.data.cpu().numpy()
            points = np.array(np.where(ind_np)).transpose((1, 0))
            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue
            score_i = score[ind].mean().item()
            if score_i < threshold:
                label[ind] = 0
                continue

            binary = np.zeros(label.shape, dtype="uint8")
            binary[ind_np] = 1
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_box = contours[0] * scales

            bounding_box = bounding_box.astype("int32")
            bounding_boxes.append(bounding_box.reshape(-1).tolist())
            scores.append(score_i)
        return bounding_boxes, scores


__all__ = [
    "FastImageProcessor",
]
