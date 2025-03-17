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
from ...utils.import_utils import is_cv2_available, is_torch_available, is_scipy_available, requires_backends
import math
if is_cv2_available():
    import cv2

if is_scipy_available():
    import scipy.ndimage as ndi
    from scipy.spatial import ConvexHull

if is_torch_available():
    import torch
    import torch.nn as nn
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
        # compute convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # compute edge angles
        edges = np.diff(hull_points, axis=0, append=hull_points[:1])
        edge_angles = np.arctan2(edges[:, 1], edges[:, 0])  # get angles in radians
        edge_angles = np.unique(np.abs(edge_angles))  # remove duplicates

        # initialize min area variables
        min_area = float('inf')
        best_rect = None

        for angle in edge_angles:
            # rotation matrix
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated_points = points @ R.T

            # get bounding box in rotated space
            xmin, ymin = rotated_points.min(axis=0)
            xmax, ymax = rotated_points.max(axis=0)
            w, h = xmax - xmin, ymax - ymin
            area = w * h

            if area < min_area:
                min_area = area
                best_rect = (xmin, ymin, xmax, ymax, angle, w, h)

        # extract best rectangle parameters
        xmin, ymin, xmax, ymax, angle, w, h = best_rect

        # compute center in rotated space
        center_rotated = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])

        # rotate center back to original coordinates
        R_inv = np.linalg.inv(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
        center = center_rotated @ R_inv.T

        # convert angle to degrees
        angle = np.degrees(angle)

        # fix angle range to match OpenCV [-90, 0]
        if w < h:
            angle += 90
            w, h = h, w  # Swap width and height

        # we ensure angle sign matches opencv's convention
        if angle > 0:
            angle -= 180

        return ((center[0], center[1]), (w, h), -angle)
        
def get_box_points(rect):
    """
    Computes the four corner points of a rotated rectangle in OpenCV's order.

    Args:
        rect (tuple): (cx, cy, w, h, angle)
                    - Center (cx, cy)
                    - Width (w) and Height (h)
                    - Rotation angle in degrees

    Returns:
        np.ndarray: (4,2) array containing the rectangle's four corners in OpenCV order:
                    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
    (cx, cy), (w, h), angle = rect

    # convert angle from degrees to radians
    angle = np.radians(angle)

    # compute movement vectors using OpenCV's method
    b = np.cos(angle) * 0.5
    a = np.sin(angle) * 0.5

    # compute four corners
    points = np.array([
        [cx - a * h - b * w, cy + b * h - a * w],  # top-left
        [cx + a * h - b * w, cy - b * h - a * w],  # Top-right
        [2 * cx - (cx - a * h - b * w), 2 * cy - (cy + b * h - a * w)],  # bottom-right
        [2 * cx - (cx + a * h - b * w), 2 * cy - (cy - b * h - a * w)]   # bottom-left
    ], dtype=np.float32)

    return points

class FastImageProcessor(TextNetImageProcessor):
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"] , with the longest edge
        resized to keep the input aspect ratio. Both the height and width are resized to be divisible by 32.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            size_divisor (`int`, *optional*, defaults to `32`):
                Ensures height and width are rounded to a multiple of this value after resizing.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            default_to_square (`bool`, *optional*, defaults to `False`):
                The value to be passed to `get_size_dict` as `default_to_square` when computing the image size. If the
                `size` argument in `get_size_dict` is an `int`, it determines whether to default to a square image or
                not.Note that this attribute is not used in computing `crop_size` via calling `get_size_dict`.
        """
        if "shortest_edge" in size:
            size = size["shortest_edge"]
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        height, width = get_resize_output_image_size(
            image, size=size, input_data_format=input_data_format, default_to_square=False
        )
        if height % self.size_divisor != 0:
            height += self.size_divisor - (height % self.size_divisor)
        if width % self.size_divisor != 0:
            width += self.size_divisor - (width % self.size_divisor)
        #TODO: would be great to find a more efficient way in modular
        # as we're only adding this line of code
        self.img_size = (height, width)
        return resize(
            image,
            size=(height, width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    def _max_pooling(self, input_tensor, scale=1):
        kernel_size = self.pooling_size // 2 + 1 if scale == 2 else self.pooling_size
        padding = (self.pooling_size // 2) // 2 if scale == 2 else (self.pooling_size - 1) // 2
        
        pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        
        pooled_output = pooling(input_tensor)
        return pooled_output

    def post_process_text_detection(self, output, target_sizes, threshold, bbox_type="rect", image_size=None):
        """
        Post-processes the raw model output to generate bounding boxes and scores for text detection.

        Args:
            output (dict): Dictionary containing model outputs. Must include key `"last_hidden_state"` (Tensor of shape [B, C, H, W]).
            target_sizes (List[Tuple[int, int]]): Original image sizes (height, width) for each item in the batch.
                                                Used to scale detection results back to original image dimensions.
            threshold (float): Confidence threshold for filtering low-score text regions.
            bbox_type (str, optional): Type of bounding box to return. Must be one of:
                                    - "rect": rotated bounding rectangles
                                    - "poly": polygon boundaries
            image_size (Tuple[int, int], optional): Size (height, width) of the image used during inference.
                                                If not provided, defaults to `self.img_size`.

        Returns:
            List[Dict]: A list of dictionaries, each containing:
                - "bboxes" (np.ndarray): Array of detected bounding boxes in the specified format.
                - "scores" (np.ndarray): Corresponding confidence scores for each bounding box.
        """
        scale = 2
        image_size = image_size if image_size is not None else self.img_size
        out = output["last_hidden_state"]
        batch_size = out.size(0)
        final_results = {}

        texts = F.interpolate(
            out[:, 0:1, :, :], size=(image_size[0] // scale, image_size[1] // scale), mode="nearest"
        )  # B*1*320*320
        texts = self._max_pooling(texts, scale=scale)  # B*1*320*320
        score_maps = torch.sigmoid_(texts)  # B*1*320*320
        score_maps = F.interpolate(score_maps, size=(image_size[0], image_size[1]), mode="nearest")  # B*1*640*640
        score_maps = score_maps.squeeze(1)  # B*640*640

        kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B*160*160
        labels_ = []
        for kernel in kernels.numpy():

            ret, label_ = self.connected_components(kernel)
            labels_.append(label_)
        labels_ = np.array(labels_)
        labels_ = torch.from_numpy(labels_)
        labels = labels_.unsqueeze(1).to(torch.float32)  # B*1*160*160
        labels = F.interpolate(
            labels, size=(image_size[0] // scale, image_size[1] // scale), mode="nearest"
        )  # B*1*320*320
        labels = self._max_pooling(labels, scale=scale)
        labels = F.interpolate(labels, size=(image_size[0], image_size[1]), mode="nearest")  # B*1*640*640
        labels = labels.squeeze(1).to(torch.int32)  # B*640*640

        keys = [torch.unique(labels_[i], sorted=True) for i in range(batch_size)]

        final_results.update({"kernels": kernels.data.cpu()})

        results = []
        for i in range(batch_size):
            org_image_size = target_sizes[i]
            scales = (float(org_image_size[1]) / float(image_size[1]), float(org_image_size[0]) / float(image_size[0]))

            bboxes, scores = self.generate_bbox(
                keys[i], labels[i], score_maps[i], scales, threshold, bbox_type=bbox_type
            )
            results.append({"bboxes": bboxes, "scores": scores})
        final_results.update({"results": results})

        return results

    def generate_bbox(self, keys, label, score, scales, threshold, bbox_type):
        """
        Generates bounding boxes and corresponding confidence scores from instance labels and score maps.

        Args:
            keys (Tensor): Unique instance labels (1D Tensor) for connected components in the image.
            label (Tensor): Instance segmentation map (H x W), where each connected region has a unique label.
            score (Tensor): Confidence map (H x W) with values in [0, 1], representing text region probabilities.
            scales (Tuple[float, float]): Scaling factors (width_scale, height_scale) used to map results
                                        back to original image dimensions.
            threshold (float): Minimum average score required for a region to be considered a valid detection.
            bbox_type (str): Type of bounding box to generate for each instance. Options:
                            - "rect": Minimum area rotated rectangle (via `compute_min_area_rect`).
                            - "poly": Polygonal contour (via `cv2.findContours`, requires OpenCV).

        Returns:
            Tuple[List[List[int]], List[float]]:
                - List of bounding boxes (`bboxes`), each a flattened list of coordinates.
                - List of corresponding confidence scores (`scores`).
        """
        label_num = len(keys)
        bboxes = []
        scores = []
        for index in range(1, label_num):
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

            if bbox_type == "rect":
                rect = compute_min_area_rect(points[:, ::-1])
                alpha = math.sqrt(math.sqrt(points.shape[0] / (rect[1][0] * rect[1][1])))
                rect = (rect[0], (rect[1][0] * alpha, rect[1][1] * alpha), rect[2])
                bbox = get_box_points(rect) * scales

            elif bbox_type == "poly":
                requires_backend(self, "cv2")
                binary = np.zeros(label.shape, dtype="uint8")
                binary[ind_np] = 1
                # cv2.findContours is too complex to replicate :(
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scales
            bbox = bbox.astype("int32")
            bboxes.append(bbox.reshape(-1).tolist())
            scores.append(score_i)

        return bboxes, scores

__all__ = [
    "FastImageProcessor",
]
