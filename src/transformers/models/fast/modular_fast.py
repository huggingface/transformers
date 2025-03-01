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
from ...utils.import_utils import is_cv2_available, is_torch_available, is_scipy_available

if is_scipy_available():
    import scipy.spatial

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
from transformers.models.textnet.image_processing_textnet import TextNetImageProcessor

class FastImageProcessor(TextNetImageProcessor):
    def _max_pooling(self, input_tensor, scale=1):
        kernel_size = self.pooling_size // 2 + 1 if scale == 2 else self.pooling_size
        padding = (self.pooling_size // 2) // 2 if scale == 2 else (self.pooling_size - 1) // 2
        
        pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        
        pooled_output = pooling(input_tensor)
        return pooled_output

    def post_process_text_detection(self, output, target_sizes, threshold, bbox_type="rect", img_size=None):
        """Post-processes the model's segmentation maps into bounding boxes."""
        scale = 2
        img_size = img_size or self.img_size
        batch_size = output["last_hidden_state"].size(0)
        score_maps = self._compute_score_maps(output["last_hidden_state"], img_size, scale)
        labels = self._compute_instance_labels(output["last_hidden_state"], img_size, scale)
        keys = [torch.unique(labels[i], sorted=True) for i in range(batch_size)]
        results = self._generate_boxes(keys, labels, score_maps, target_sizes, img_size, threshold, bbox_type)

        return results

    def _compute_score_maps(self, out, img_size, scale):
        """Computes score maps using max pooling and sigmoid activation."""
        texts = F.interpolate(out[:, 0:1, :, :], size=(img_size[0] // scale, img_size[1] // scale), mode="nearest")
        texts = self._max_pooling(texts, scale=scale)
        score_maps = torch.sigmoid(texts)
        score_maps = F.interpolate(score_maps, size=img_size, mode="nearest")
        return score_maps.squeeze(1)

    def _compute_instance_labels(self, out, img_size, scale):
        """Generates instance segmentation labels without using cv2."""
        kernels = (out[:, 0, :, :] > 0).to(torch.uint8)

        labels = torch.zeros_like(kernels, dtype=torch.int32)
        label_counter = 1
        for i in range(kernels.shape[0]):  # iterate over batch
            y, x = torch.where(kernels[i] > 0)
            if len(y) > 0:
                labels[i, y, x] = label_counter
                label_counter += 1

        labels = labels.unsqueeze(1).float()
        labels = F.interpolate(labels, size=(img_size[0] // scale, img_size[1] // scale), mode="nearest")
        labels = self._max_pooling(labels, scale=scale)
        labels = F.interpolate(labels, size=img_size, mode="nearest").squeeze(1).int()
        return labels

    def _generate_boxes(self, keys, labels, score_maps, target_sizes, img_size, threshold, bbox_type):
        """Converts instance segmentation maps into bounding boxes."""
        results = []
        for i in range(labels.shape[0]):  # iterate over batch
            original_size = target_sizes[i]
            scales = (original_size[1] / img_size[1], original_size[0] / img_size[0])

            if bbox_type == "rect":
                boxes, scores = self._generate_rect_boxes(keys[i], labels[i], score_maps[i], scales, threshold)
            else:
                boxes, scores = self._generate_poly_boxes(keys[i], labels[i], score_maps[i], scales, threshold)

            results.append({"boxes": boxes, "scores": scores})

        return results

    def _generate_rect_boxes(self, keys, label, score_map, scales, threshold):
        """Generates rectangular bounding boxes """
        boxes, scores = [], []
        for i in keys[1:]:  # skip background label 0
            mask = (label == i).cpu().numpy()
            points = np.array(np.where(mask)).T  # (y, x)

            if points.shape[0] < self.min_area:
                label[label == i] = 0
                continue

            score = score_map[label == i].mean().item()
            if score < threshold:
                label[label == i] = 0
                continue

            ymin, xmin = points.min(axis=0)
            ymax, xmax = points.max(axis=0)
            box = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]) * scales
            boxes.append(box.astype(np.int32).flatten().tolist())
            scores.append(score)

        return boxes, scores

    def _generate_poly_boxes(self, keys, label, score_map, scales, threshold):
        """Generates polygonal bounding boxes """
        boxes, scores = [], []
        for i in keys[1:]:  # skip background label 0
            mask = (label == i).cpu().numpy()
            points = np.array(np.where(mask)).T  # (y, x)

            if points.shape[0] < self.min_area:
                label[label == i] = 0
                continue

            score = score_map[label == i].mean().item()
            if score < threshold:
                label[label == i] = 0
                continue

            # approximate convex hull using numpy
            hull = self._convex_hull(points)
            box = hull * scales
            boxes.append(box.astype(np.int32).flatten().tolist())
            scores.append(score)

        return boxes, scores

    def _convex_hull(self, points):
        """Computes a convex hull using NumPy (alternative to cv2.findContours)."""
        from scipy.spatial import ConvexHull  # use scipy only for convex hull computation

        if points.shape[0] < 3:
            return np.array([[points.min(axis=0)], [points.max(axis=0)]])  # return min-max rect if <3 points

        hull = ConvexHull(points)
        return points[hull.vertices]

    def _max_pooling(self, tensor, scale):
        """Performs max pooling on the given tensor."""
        kernel_size = scale * 2
        return F.max_pool2d(tensor, kernel_size=kernel_size, stride=scale, padding=kernel_size // 2)

__all__ = [
    "FastImageProcessor",
]
