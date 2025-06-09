# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for SAM2.
"""

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
from torchvision.transforms import Normalize, ToTensor

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_tf_available, is_torch_available, logging


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


class Sam2Processor(ProcessorMixin):
    r"""
    Constructs a SAM2 processor which wraps a SAM2 image processor and an 2D points & Bounding boxes processor into a
    single processor.

    [`Sam2Processor`] offers all the functionalities of [`Sam2ImageProcessor`]. See the docstring of
    [`~Sam2ImageProcessor.__call__`] for more information.

    Args:
        image_processor (`Sam2ImageProcessor`):
            An instance of [`Sam2ImageProcessor`]. The image processor is a required input.
    """

    attributes = ["image_processor"]
    image_processor_class = "Sam2ImageProcessor"

    def __init__(self, image_processor):
        super().__init__(image_processor)
        self.current_processor = self.image_processor
        self.point_pad_value = -10
        self.target_size = self.image_processor.size["longest_edge"]

        # Video inference state
        self.inference_state = None

    def __call__(
        self,
        images=None,
        segmentation_maps=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`SamImageProcessor.__call__`] method to prepare image(s) for the model. It also prepares 2D
        points and bounding boxes for the model if they are provided.
        """
        encoding_image_processor = self.image_processor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors=return_tensors,
            **kwargs,
        )

        # pop arguments that are not used in the foward but used nevertheless
        original_sizes = encoding_image_processor["original_sizes"]

        if hasattr(original_sizes, "numpy"):  # Checks if Torch or TF tensor
            original_sizes = original_sizes.numpy()

        input_points, input_labels, input_boxes = self._check_and_preprocess_points(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
        )

        encoding_image_processor = self._normalize_and_convert(
            encoding_image_processor,
            original_sizes,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors=return_tensors,
        )

        return encoding_image_processor

    def _normalize_and_convert(
        self,
        encoding_image_processor,
        original_sizes,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors="pt",
    ):
        if input_points is not None:
            if len(original_sizes) != len(input_points):
                input_points = [
                    self._normalize_coordinates(self.target_size, point, original_sizes[0]) for point in input_points
                ]
            else:
                input_points = [
                    self._normalize_coordinates(self.target_size, point, original_size)
                    for point, original_size in zip(input_points, original_sizes)
                ]
            # check that all arrays have the same shape
            if not all(point.shape == input_points[0].shape for point in input_points):
                if input_labels is not None:
                    input_points, input_labels = self._pad_points_and_labels(input_points, input_labels)

            input_points = np.array(input_points)

        if input_labels is not None:
            input_labels = np.array(input_labels)

        if input_boxes is not None:
            if len(original_sizes) != len(input_boxes):
                input_boxes = [
                    self._normalize_coordinates(self.target_size, box, original_sizes[0], is_bounding_box=True)
                    for box in input_boxes
                ]
            else:
                input_boxes = [
                    self._normalize_coordinates(self.target_size, box, original_size, is_bounding_box=True)
                    for box, original_size in zip(input_boxes, original_sizes)
                ]
            input_boxes = np.array(input_boxes)

        if input_boxes is not None:
            if return_tensors == "pt":
                input_boxes = torch.from_numpy(input_boxes)
                # boxes batch size of 1 by default
                input_boxes = input_boxes.unsqueeze(1) if len(input_boxes.shape) != 3 else input_boxes
            elif return_tensors == "tf":
                input_boxes = tf.convert_to_tensor(input_boxes)
                # boxes batch size of 1 by default
                input_boxes = tf.expand_dims(input_boxes, 1) if len(input_boxes.shape) != 3 else input_boxes
            encoding_image_processor.update({"input_boxes": input_boxes})
        if input_points is not None:
            if return_tensors == "pt":
                input_points = torch.from_numpy(input_points)
                # point batch size of 1 by default
                input_points = input_points.unsqueeze(1) if len(input_points.shape) != 4 else input_points
            elif return_tensors == "tf":
                input_points = tf.convert_to_tensor(input_points)
                # point batch size of 1 by default
                input_points = tf.expand_dims(input_points, 1) if len(input_points.shape) != 4 else input_points
            encoding_image_processor.update({"input_points": input_points})
        if input_labels is not None:
            if return_tensors == "pt":
                input_labels = torch.from_numpy(input_labels)
                # point batch size of 1 by default
                input_labels = input_labels.unsqueeze(1) if len(input_labels.shape) != 3 else input_labels
            elif return_tensors == "tf":
                input_labels = tf.convert_to_tensor(input_labels)
                # point batch size of 1 by default
                input_labels = tf.expand_dims(input_labels, 1) if len(input_labels.shape) != 3 else input_labels
            encoding_image_processor.update({"input_labels": input_labels})

        return encoding_image_processor

    def _pad_points_and_labels(self, input_points, input_labels):
        r"""
        The method pads the 2D points and labels to the maximum number of points in the batch.
        """
        expected_nb_points = max([point.shape[0] for point in input_points])
        processed_input_points = []
        for i, point in enumerate(input_points):
            if point.shape[0] != expected_nb_points:
                point = np.concatenate(
                    [point, np.zeros((expected_nb_points - point.shape[0], 2)) + self.point_pad_value], axis=0
                )
                input_labels[i] = np.append(input_labels[i], [self.point_pad_value])
            processed_input_points.append(point)
        input_points = processed_input_points
        return input_points, input_labels

    def _normalize_coordinates(
        self, target_size: int, coords: np.ndarray, original_size, is_bounding_box=False
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = target_size, target_size
        coords = deepcopy(coords).astype(float)

        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)

        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        return coords

    def _check_and_preprocess_points(
        self,
        input_points=None,
        input_labels=None,
        input_boxes=None,
    ):
        r"""
        Check and preprocesses the 2D points, labels and bounding boxes. It checks if the input is valid and if they
        are, it converts the coordinates of the points and bounding boxes. If a user passes directly a `torch.Tensor`,
        it is converted to a `numpy.ndarray` and then to a `list`.
        """
        if input_points is not None:
            if hasattr(input_points, "numpy"):  # Checks for TF or Torch tensor
                input_points = input_points.numpy().tolist()

            if not isinstance(input_points, list) or not isinstance(input_points[0], list):
                raise ValueError("Input points must be a list of list of floating points.")
            input_points = [np.array(input_point) for input_point in input_points]
        else:
            input_points = None

        if input_labels is not None:
            if hasattr(input_labels, "numpy"):
                input_labels = input_labels.numpy().tolist()

            if not isinstance(input_labels, list) or not isinstance(input_labels[0], list):
                raise ValueError("Input labels must be a list of list integers.")
            input_labels = [np.array(label) for label in input_labels]
        else:
            input_labels = None

        if input_boxes is not None:
            if hasattr(input_boxes, "numpy"):
                input_boxes = input_boxes.numpy().tolist()

            if (
                not isinstance(input_boxes, list)
                or not isinstance(input_boxes[0], list)
                or not isinstance(input_boxes[0][0], list)
            ):
                raise ValueError("Input boxes must be a list of list of list of floating points.")
            input_boxes = [np.array(box).astype(np.float32) for box in input_boxes]
        else:
            input_boxes = None

        return input_points, input_labels, input_boxes

    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names))

    def post_process_masks(self, *args, **kwargs):
        return self.image_processor.post_process_masks(*args, **kwargs)

    def init_state(
        self,
        video_path: Union[str, Path],
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize video inference state."""
        if not is_torch_available():
            raise ImportError("Video inference requires PyTorch to be installed")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load video frames
        images, video_height, video_width = self._load_video_frames(
            video_path=video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            device=device,
        )

        # Initialize inference state
        self.inference_state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else device,
            # Input tracking
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            # Visual features cache
            "cached_features": {},
            "constants": {},
            # Object management
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            # Output tracking
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }

        logger.info(f"Initialized video state with {len(images)} frames at resolution {video_height}x{video_width}")

    def reset_state(self) -> None:
        """Reset the video inference state."""
        if self.inference_state is not None:
            # Clear all state
            self.inference_state["point_inputs_per_obj"].clear()
            self.inference_state["mask_inputs_per_obj"].clear()
            self.inference_state["cached_features"].clear()
            self.inference_state["constants"].clear()
            self.inference_state["obj_id_to_idx"].clear()
            self.inference_state["obj_idx_to_id"].clear()
            self.inference_state["obj_ids"].clear()
            self.inference_state["output_dict_per_obj"].clear()
            self.inference_state["temp_output_dict_per_obj"].clear()
            self.inference_state["frames_tracked_per_obj"].clear()

        self.inference_state = None
        logger.info("Reset video inference state")

    def _load_video_frames(
        self,
        video_path: Union[str, Path],
        offload_video_to_cpu: bool = False,
        async_loading_frames: bool = False,
        device: torch.device = None,
    ) -> Tuple[List[torch.Tensor], int, int]:
        """Load video frames from a directory of images."""
        video_path = Path(video_path)

        if not video_path.exists():
            raise ValueError(f"Video path {video_path} does not exist")

        # Get image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [f for f in video_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            raise ValueError(f"No image files found in {video_path}")

        # Sort files by name (assuming frame order)
        image_files.sort(key=lambda x: x.name)

        # Load first image to get dimensions
        from PIL import Image

        first_image = Image.open(image_files[0])
        video_width, video_height = first_image.size

        # Process images using image processor
        images = []
        for img_path in image_files:
            image = Image.open(img_path)
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Process image
            image = image.resize((1024, 1024))
            IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
            to_tensor = ToTensor()
            transforms = torch.jit.script(
                nn.Sequential(
                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                )
            )
            # processed = self.image_processor(image, return_tensors="pt")
            # image_tensor = processed["pixel_values"].squeeze(0)  # Remove batch dim
            image_tensor = transforms(to_tensor(image))
            if not offload_video_to_cpu and device is not None:
                image_tensor = image_tensor.to(device)

            images.append(image_tensor)

        return images, video_height, video_width

    def _obj_id_to_idx(self, obj_id: int) -> int:
        """Map client-side object id to model-side object index."""
        if self.inference_state is None:
            raise ValueError("Video state not initialized. Call init_state() first.")

        obj_idx = self.inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # Add new object
        obj_idx = len(self.inference_state["obj_id_to_idx"])
        self.inference_state["obj_id_to_idx"][obj_id] = obj_idx
        self.inference_state["obj_idx_to_id"][obj_idx] = obj_id
        self.inference_state["obj_ids"] = list(self.inference_state["obj_id_to_idx"])

        # Set up input and output structures for this object
        self.inference_state["point_inputs_per_obj"][obj_idx] = {}
        self.inference_state["mask_inputs_per_obj"][obj_idx] = {}
        self.inference_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        self.inference_state["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        self.inference_state["frames_tracked_per_obj"][obj_idx] = {}

        return obj_idx

    def add_new_points_or_box(
        self,
        frame_idx: int,
        obj_id: int,
        points: Optional[List[List[float]]] = None,
        labels: Optional[List[int]] = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Add new points or box to a frame and return preprocessed inputs for model."""
        if self.inference_state is None:
            raise ValueError("Video state not initialized. Call init_state() first.")

        if not is_torch_available():
            raise ImportError("Video inference requires PyTorch to be installed")

        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        # Validate inputs
        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        device = self.inference_state["device"]

        # Process points
        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0).unsqueeze(0)  # add batch dimension and object dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0).unsqueeze(0)  # add batch dimension and object dimension
        if points.dim() == 3:
            points = points.unsqueeze(0)  # add batch dimension or object dimension
        if labels.dim() == 2:
            labels = labels.unsqueeze(0)  # add batch dimension or object dimension

        # Process box if provided
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 1, 2)
            points = torch.cat([box_coords, points], dim=2)
            labels = torch.cat([box_labels, labels], dim=2)

        # Normalize coordinates
        if normalize_coords:
            video_H = self.inference_state["video_height"]
            video_W = self.inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)

        # Scale by model's internal image size
        target_size = self.target_size
        points = points * target_size
        points = points.to(device)
        labels = labels.to(device)

        # Handle existing points
        if not clear_old_points:
            existing_points = point_inputs_per_frame.get(frame_idx, None)
            if existing_points is not None:
                # Concatenate with existing points
                points = torch.cat([existing_points["point_coords"], points], dim=2)
                labels = torch.cat([existing_points["point_labels"], labels], dim=2)

        point_inputs = {
            "point_coords": points,
            "point_labels": labels,
        }

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)  # Clear any mask inputs

        # Determine frame type and tracking direction
        obj_frames_tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked

        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]

        # Return preprocessed inputs for the model
        return {
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "obj_idx": obj_idx,
            "point_inputs": point_inputs,
            "mask_inputs": None,
            "is_init_cond_frame": is_init_cond_frame,
            "reverse": reverse,
        }

    def add_new_mask(
        self,
        frame_idx: int,
        obj_id: int,
        mask: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """Add new mask to a frame and return preprocessed inputs for model."""
        if self.inference_state is None:
            raise ValueError("Video state not initialized. Call init_state() first.")

        if not is_torch_available():
            raise ImportError("Video inference requires PyTorch to be installed")

        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        device = self.inference_state["device"]

        # Process mask
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(device)

        # Resize mask if needed
        if mask_H != self.target_size or mask_W != self.target_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.target_size, self.target_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)  # Clear any point inputs

        # Determine frame type and tracking direction
        obj_frames_tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked

        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]

        # Return preprocessed inputs for the model
        return {
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "obj_idx": obj_idx,
            "point_inputs": None,
            "mask_inputs": mask_inputs,
            "is_init_cond_frame": is_init_cond_frame,
            "reverse": reverse,
        }


__all__ = ["Sam2Processor"]
