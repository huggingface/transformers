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

from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_tf_available, is_torch_available, logging
from ...video_utils import VideoInput
from .modeling_sam2 import Sam2VideoSessionState


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_tf_available():
    pass


class Sam2Processor(ProcessorMixin):
    r"""
    Constructs a SAM2 processor which wraps a SAM2 image processor and an 2D points & Bounding boxes processor into a
    single processor.

    [`Sam2Processor`] offers all the functionalities of [`Sam2ImageProcessor`] and [`Sam2VideoProcessor`]. See the docstring of
    [`~Sam2ImageProcessor.__call__`] and [`~Sam2VideoProcessor.__call__`] for more information.

    Args:
        image_processor (`Sam2ImageProcessor`):
            An instance of [`Sam2ImageProcessor`].
        video_processor (`Sam2VideoProcessor`):
            An instance of [`Sam2VideoProcessor`].
        target_size (`int`, *optional*):
            The target size (target_size, target_size) to which the image will be resized.
        point_pad_value (`int`, *optional*, defaults to -10):
            The value used for padding input points.
    """

    attributes = ["image_processor", "video_processor"]
    image_processor_class = "Sam2ImageProcessorFast"
    video_processor_class = "Sam2VideoProcessor"

    def __init__(
        self, image_processor, video_processor, target_size: Optional[int] = None, point_pad_value: int = -10, **kwargs
    ):
        super().__init__(image_processor, video_processor, **kwargs)
        self.point_pad_value = point_pad_value
        self.target_size = target_size if target_size is not None else self.image_processor.size["height"]

    def __call__(
        self,
        images=None,
        segmentation_maps=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        original_sizes=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`SamImageProcessor.__call__`] method to prepare image(s) for the model. It also prepares 2D
        points and bounding boxes for the model if they are provided.
        """
        if images is not None:
            encoding_image_processor = self.image_processor(
                images,
                segmentation_maps=segmentation_maps,
                return_tensors=return_tensors,
                **kwargs,
            )
        elif original_sizes is not None:
            if isinstance(original_sizes, torch.Tensor):
                original_sizes = original_sizes.cpu().tolist()
            encoding_image_processor = BatchEncoding({"original_sizes": original_sizes}, tensor_type=return_tensors)
        else:
            raise ValueError("Either images or original_sizes must be provided")

        # pop arguments that are not used in the foward but used nevertheless
        original_sizes = encoding_image_processor["original_sizes"]
        # Check original_sizes is of length 1 or len(images)
        if images is not None and len(original_sizes) != 1 and len(original_sizes) != len(images):
            raise ValueError(
                "original_sizes must be of length 1 or len(images). If you are passing a single image, you must pass a single original_size."
            )

        # Process input points, labels, and boxes if provided
        if input_points is not None or input_labels is not None or input_boxes is not None:
            # Validate and convert inputs to standardized format
            processed_points = self._process_single_input(
                input_points,
                expected_depth=4,
                input_name="points",
                expected_format="[image_idx, object_idx, point_idx, point_coords]",
                expected_coord_size=2,
            )
            processed_labels = self._process_single_input(
                input_labels,
                expected_depth=3,
                input_name="labels",
                expected_format="[image_idx, object_idx, point_idx]",
            )
            processed_boxes = self._process_single_input(
                input_boxes,
                expected_depth=3,
                input_name="boxes",
                expected_format="[image_idx, box_idx, box_coords]",
                expected_coord_size=4,
            )

            # Get padding requirements for all inputs
            if processed_points is not None:
                points_max_dims = self._get_nested_dimensions(processed_points)[:3]
            if processed_labels is not None:
                labels_max_dims = self._get_nested_dimensions(processed_labels)[:3]
            if processed_boxes is not None:
                boxes_max_dims = self._get_nested_dimensions(processed_boxes)[:2]

            # Ensure points and labels have consistent dimensions
            if processed_points is not None and processed_labels is not None:
                if points_max_dims != labels_max_dims:
                    raise ValueError(
                        "Input points and labels have inconsistent dimensions. Please ensure they have the same dimensions."
                    )

            # Check that boxes don't need padding (model limitation)
            if processed_boxes is not None and len(processed_boxes) >= 2:
                if any(len(img_boxes) < boxes_max_dims[1] for img_boxes in processed_boxes):
                    raise ValueError(
                        "Input boxes have inconsistent dimensions that would require padding, "
                        "but boxes cannot be padded due to model limitations. "
                        "Please ensure all images have the same number of boxes."
                    )

            # Pad and normalize all inputs to final tensor format
            if processed_points is not None:
                padded_points = self._pad_nested_list(processed_points, points_max_dims + [2])
                final_points = torch.tensor(padded_points, dtype=torch.float32)
                self._normalize_tensor_coordinates(final_points, original_sizes, preserve_padding=True)
                encoding_image_processor.update({"input_points": final_points})

            if processed_labels is not None:
                padded_labels = self._pad_nested_list(processed_labels, labels_max_dims)
                final_labels = torch.tensor(padded_labels, dtype=torch.int64)
                encoding_image_processor.update({"input_labels": final_labels})

            if processed_boxes is not None:
                final_boxes = torch.tensor(processed_boxes, dtype=torch.float32)
                self._normalize_tensor_coordinates(final_boxes, original_sizes, is_bounding_box=True)
                encoding_image_processor.update({"input_boxes": final_boxes})

        return encoding_image_processor

    def _normalize_coordinates(
        self, target_size: int, coords: "torch.Tensor", original_size, is_bounding_box=False
    ) -> "torch.Tensor":
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.

        Args:
            target_size (`int`):
                The target size of the image.
            coords (`torch.Tensor`):
                The coordinates to be normalized.
            original_size (`tuple`):
                The original size of the image.
            is_bounding_box (`bool`, *optional*, defaults to `False`):
                Whether the coordinates are bounding boxes.
        """
        old_h, old_w = original_size
        new_h, new_w = target_size, target_size
        coords = deepcopy(coords).float()

        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        return coords

    def _convert_to_nested_list(self, data, expected_depth, current_depth=0):
        """
        Recursively convert various input formats (tensors, numpy arrays, lists) to nested lists.

        Args:
            data: Input data in any format
            expected_depth: Expected nesting depth
            current_depth: Current depth in recursion

        Returns:
            Nested list representation of the data
        """
        if data is None:
            return None

        # Convert tensor/numpy to list if we're at a leaf level or if it's a multi-dimensional array
        if isinstance(data, torch.Tensor):  # PyTorch tensor
            if current_depth == expected_depth - 2 or len(data.shape) <= 2:  # At coordinate level or small tensor
                return data.numpy().tolist()
            else:
                return [self._convert_to_nested_list(item, expected_depth, current_depth + 1) for item in data]
        elif isinstance(data, np.ndarray):  # NumPy array
            if current_depth == expected_depth - 2 or len(data.shape) <= 2:  # At coordinate level or small array
                return data.tolist()
            else:
                return [self._convert_to_nested_list(item, expected_depth, current_depth + 1) for item in data]
        elif isinstance(data, list):
            if current_depth == expected_depth:
                # We've reached the expected depth, return as is
                return data
            else:
                # Continue recursion
                return [self._convert_to_nested_list(item, expected_depth, current_depth + 1) for item in data]
        elif isinstance(data, (int, float)):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _get_nested_dimensions(self, nested_list, max_dims=None):
        """
        Get the maximum dimensions at each level of nesting.

        Args:
            nested_list (`list`):
                Nested list structure.
            max_dims (`list`, *optional*):
                Current maximum dimensions (for recursion).

        Returns:
            `list`: A list of maximum dimensions for each nesting level.
        """
        if max_dims is None:
            max_dims = []

        if not isinstance(nested_list, list):
            return max_dims

        if len(max_dims) == 0:
            max_dims.append(len(nested_list))
        else:
            max_dims[0] = max(max_dims[0], len(nested_list))

        if len(nested_list) > 0:
            for item in nested_list:
                if isinstance(item, list):
                    sub_dims = self._get_nested_dimensions(item)
                    # Merge sub_dims into max_dims
                    for i, dim in enumerate(sub_dims):
                        if i + 1 >= len(max_dims):
                            max_dims.append(dim)
                        else:
                            max_dims[i + 1] = max(max_dims[i + 1], dim)

        return max_dims

    def _pad_nested_list(self, nested_list, target_dims, current_level=0, pad_value=None):
        """
        Recursively pad a nested list to match target dimensions.

        Args:
            nested_list (`list`):
                Nested list to pad.
            target_dims (`list`):
                Target dimensions for each level.
            current_level (`int`, *optional*, defaults to 0):
                Current nesting level.
            pad_value (`int`, *optional*):
                Value to use for padding.

        Returns:
            `list`: The padded nested list.
        """
        if pad_value is None:
            pad_value = self.point_pad_value

        if current_level >= len(target_dims):
            return nested_list

        # Ensure we have a list
        if not isinstance(nested_list, list):
            nested_list = [nested_list]

        # Pad current level
        current_size = len(nested_list)
        target_size = target_dims[current_level]

        # Pad with appropriate values
        if current_level == len(target_dims) - 1:
            # At the coordinate level, pad with pad_value
            nested_list.extend([pad_value] * (target_size - current_size))
        else:
            # At higher levels, pad with nested structures
            if current_size > 0:
                # Create appropriately sized template
                if current_level < len(target_dims) - 2:
                    # For non-coordinate levels, create empty nested structure
                    template_dims = target_dims[current_level + 1 :]
                    template = self._create_empty_nested_structure(template_dims, pad_value)
                else:
                    # For coordinate level, create list of pad_values
                    template = [pad_value] * target_dims[current_level + 1]

                nested_list.extend([deepcopy(template) for _ in range(target_size - current_size)])
            else:
                # Create from scratch
                template_dims = target_dims[current_level + 1 :]
                template = self._create_empty_nested_structure(template_dims, pad_value)
                nested_list.extend([deepcopy(template) for _ in range(target_size)])

        # Recursively pad sublists
        if current_level < len(target_dims) - 1:
            for i in range(len(nested_list)):
                if isinstance(nested_list[i], list):
                    nested_list[i] = self._pad_nested_list(nested_list[i], target_dims, current_level + 1, pad_value)

        return nested_list

    def _create_empty_nested_structure(self, dims, pad_value):
        """
        Create an empty nested structure with given dimensions filled with pad_value.

        Args:
            dims (`list`):
                The dimensions of the nested structure.
            pad_value (`int`):
                The value to fill the structure with.
        """
        if len(dims) == 1:
            return [pad_value] * dims[0]
        else:
            return [self._create_empty_nested_structure(dims[1:], pad_value) for _ in range(dims[0])]

    def _get_nesting_level(self, input_list):
        """
        Get the nesting level of a list structure.

        Args:
            input_list (`list`):
                The list to get the nesting level of.
        """
        if isinstance(input_list, list):
            if len(input_list) == 0:
                return 1
            return 1 + self._get_nesting_level(input_list[0])
        elif isinstance(input_list, (np.ndarray, torch.Tensor)):
            # For arrays/tensors, the nesting level is the number of dimensions
            return len(input_list.shape)
        return 0

    def _ensure_proper_nesting(self, data, expected_depth):
        """
        Ensure data has the proper nesting level by unsqueezing from the first dimensions if needed.

        Args:
            data (`torch.Tensor`, `np.ndarray`, or `list`):
                Input data.
            expected_depth (`int`):
                Expected nesting depth.

        Returns:
            The data with proper nesting level.
        """
        if data is None:
            return None

        # Handle tensors and numpy arrays first
        if isinstance(data, (torch.Tensor, np.ndarray)):
            # For tensors/arrays, we can directly check the number of dimensions
            current_depth = len(data.shape)
            # Unsqueeze from the beginning if needed
            while current_depth < expected_depth:
                if isinstance(data, torch.Tensor):  # PyTorch tensor
                    data = data.unsqueeze(0)
                else:  # NumPy array
                    data = np.expand_dims(data, axis=0)
                current_depth += 1
            return data

        # Handle nested lists
        if isinstance(data, list):
            current_depth = self._get_nesting_level(data)
            # Unsqueeze from the beginning if needed
            while current_depth < expected_depth:
                data = [data]
                current_depth += 1
            return data

        # Handle scalar values (wrap in appropriate nesting)
        else:
            # Create the appropriate nesting level
            result = data
            for _ in range(expected_depth):
                result = [result]
            return result

    def _process_single_input(self, data, expected_depth, input_name, expected_format, expected_coord_size=None):
        """
        Process a single input by ensuring proper nesting and converting to nested list format.

        Args:
            data (`torch.Tensor`, `np.ndarray`, or `list`):
                Input data to process.
            expected_depth (`int`):
                Expected nesting depth.
            input_name (`str`):
                Name of the input for error messages.
            expected_format (`str`):
                The expected format of the input.
            expected_coord_size (`int`, *optional*):
                Expected coordinate size (2 for points, 4 for boxes, None for labels).

        Returns:
            Processed nested list or `None` if data is `None`.
        """
        if data is None:
            return None

        try:
            data = self._ensure_proper_nesting(data, expected_depth)
            return self._convert_to_nested_list(data, expected_depth)
        except ValueError as e:
            coord_info = f" Coordinates must be length {expected_coord_size}." if expected_coord_size else ""
            raise ValueError(
                f"Input {input_name} must be a nested list with the specified dimensions and format {expected_format}.{coord_info} "
                f"Missing dimensions are automatically unsqueezed from the beginning. Error: {e}"
            )

    def _normalize_tensor_coordinates(self, tensor, original_sizes, is_bounding_box=False, preserve_padding=False):
        """
        Helper method to normalize coordinates in a tensor across multiple images.

        Args:
            tensor (`torch.Tensor`):
                Input tensor with coordinates.
            original_sizes (`list`):
                Original image sizes.
            is_bounding_box (`bool`, *optional*, defaults to `False`):
                Whether coordinates are bounding boxes.
            preserve_padding (`bool`, *optional*, defaults to `False`):
                Whether to preserve padding values (for points).
        """
        if preserve_padding:
            # For points: avoid normalizing pad values
            mask = tensor != self.point_pad_value
            coord_mask = mask.all(dim=-1, keepdim=True)

        for img_idx in range(len(original_sizes)):
            if img_idx < tensor.shape[0]:
                original_size = original_sizes[img_idx] if img_idx < len(original_sizes) else original_sizes[0]
                normalized_coords = self._normalize_coordinates(
                    self.target_size, tensor[img_idx], original_size, is_bounding_box=is_bounding_box
                )

                if preserve_padding:
                    # Only update non-padded values
                    img_mask = coord_mask[img_idx]
                    tensor[img_idx] = torch.where(
                        img_mask.expand_as(tensor[img_idx]), normalized_coords, tensor[img_idx]
                    )
                else:
                    tensor[img_idx] = normalized_coords

    def post_process_masks(self, *args, **kwargs):
        return self.image_processor.post_process_masks(*args, **kwargs)

    def init_video_session(
        self,
        video: VideoInput,
        inference_device: Union[str, "torch.device"] = "cpu",
        inference_state_device: Union[str, "torch.device"] = None,
        processing_device: Union[str, "torch.device"] = None,
        video_storage_device: Union[str, "torch.device"] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes a video session for inference.

        Args:
            video (`VideoInput`):
                The video to process.
            inference_device (`str` or `torch.device`, *optional*, defaults to "cpu"):
                The device to use for inference.
            inference_state_device (`str` or `torch.device`, *optional*):
                The device to store the inference state on.
            processing_device (`str` or `torch.device`, *optional*):
                The device to use for video processing.
            video_storage_device (`str` or `torch.device`, *optional*):
                The device to store the processed video frames on.
            torch_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The torch dtype to use for the whole session.
        """
        video_storage_device = video_storage_device if video_storage_device is not None else inference_device
        inference_state_device = inference_state_device if inference_state_device is not None else inference_device
        processing_device = processing_device if processing_device is not None else inference_device
        processed_video = self.video_processor(videos=video, device=processing_device, return_tensors="pt").to(
            torch_dtype
        )
        if video_storage_device != inference_device:
            processed_video.pixel_values_videos = processed_video.pixel_values_videos.to(video_storage_device)
        elif processing_device != inference_device:
            processed_video.pixel_values_videos = processed_video.pixel_values_videos.to(inference_device)
        inference_state = Sam2VideoSessionState(
            processed_video.pixel_values_videos[0],
            video_height=processed_video.original_sizes[0][0],
            video_width=processed_video.original_sizes[0][1],
            inference_device=inference_device,
            video_storage_device=video_storage_device,
            inference_state_device=inference_state_device,
            torch_dtype=torch_dtype,
        )
        return inference_state

    def process_new_points_or_box_for_video_frame(
        self,
        inference_state: Sam2VideoSessionState,
        frame_idx: int,
        obj_ids: Union[list[int], int],
        input_points: Optional[list[list[float]]] = None,
        input_labels: Optional[list[int]] = None,
        input_boxes: Optional[list[list[float]]] = None,
        clear_old_inputs: bool = True,
    ) -> dict[str, Any]:
        """
        Process new points or box for a video frame and return preprocessed inputs for model.

        Args:
            inference_state (`Sam2VideoSessionState`):
                The inference state for the video session.
            frame_idx (`int`):
                The index of the frame to process.
            obj_ids (`list[int]` or `int`):
                The object ID(s) to associate with the points or box.
                These can be any integers and can be reused later on to specify an object.
            input_points (`list[list[float]]`, *optional*):
                The points to add to the frame.
            input_labels (`list[int]`, *optional*):
                The labels for the points.
            input_boxes (`list[list[float]]`, *optional*):
                The bounding boxes to add to the frame.
            clear_old_inputs (`bool`, *optional*, defaults to `True`):
                Whether to clear old inputs for the object.
        """

        if isinstance(obj_ids, int):
            obj_ids = [obj_ids]

        # Validate inputs
        if (input_points is not None) != (input_labels is not None):
            raise ValueError("points and labels must be provided together")
        if input_points is None and input_boxes is None:
            raise ValueError("at least one of points or box must be provided as input")

        device = inference_state.inference_device
        original_sizes = [[inference_state.video_height, inference_state.video_width]]

        encoded_inputs = self(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            original_sizes=original_sizes,
            return_tensors="pt",
        ).to(device)
        input_points = encoded_inputs.get("input_points", None)
        input_labels = encoded_inputs.get("input_labels", None)
        input_boxes = encoded_inputs.get("input_boxes", None)

        if input_points is not None:
            if input_points.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of points ({input_points.shape[1]})"
                )
        else:
            input_points = torch.zeros(1, len(obj_ids), 0, 2, dtype=torch.float32, device=device)
        if input_labels is not None:
            if input_labels.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of labels ({input_labels.shape[1]})"
                )
        else:
            input_labels = torch.zeros(1, len(obj_ids), 0, dtype=torch.int32, device=device)
        if input_boxes is not None:
            if input_boxes.shape[1] != len(obj_ids):
                raise ValueError(
                    f"Number of object ids ({len(obj_ids)}) does not match number of boxes ({input_boxes.shape[1]})"
                )

        if input_boxes is not None:
            if not clear_old_inputs:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            box_coords = input_boxes.reshape(1, -1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=input_labels.device)
            box_labels = box_labels.reshape(1, -1, 2)
            input_points = torch.cat([box_coords, input_points], dim=2)
            input_labels = torch.cat([box_labels, input_labels], dim=2)

        for obj_id, idx in zip(obj_ids, range(len(obj_ids))):
            obj_idx = inference_state._obj_id_to_idx(obj_id)
            input_points_for_obj = input_points[:, idx, :, :].unsqueeze(1)
            input_labels_for_obj = input_labels[:, idx, :].unsqueeze(1)
            # Handle existing points
            if not clear_old_inputs:
                existing_points = inference_state.point_inputs_per_obj[obj_idx].get(frame_idx, None)
                if existing_points is not None:
                    # Concatenate with existing points
                    input_points_for_obj = torch.cat([existing_points["point_coords"], input_points_for_obj], dim=2)
                    input_labels_for_obj = torch.cat([existing_points["point_labels"], input_labels_for_obj], dim=2)
            point_inputs = {
                "point_coords": input_points_for_obj,
                "point_labels": input_labels_for_obj,
            }

            inference_state.point_inputs_per_obj[obj_idx][frame_idx] = point_inputs
            inference_state.mask_inputs_per_obj[obj_idx].pop(frame_idx, None)  # Clear any mask inputs

        return inference_state

    def process_new_mask_for_video_frame(
        self,
        inference_state: Sam2VideoSessionState,
        frame_idx: int,
        obj_ids: Union[list[int], int],
        input_masks: Union[np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor]],
    ) -> dict[str, Any]:
        """
        Add new mask to a frame and return preprocessed inputs for model.

        Args:
            inference_state (`Sam2VideoSessionState`):
                The inference state for the video session.
            frame_idx (`int`):
                The index of the frame to process.
            obj_ids (`list[int]` or `int`):
                The object ID(s) to associate with the mask.
                These can be any integers and can be reused later on to specify an object.
            input_masks (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, or `list[torch.Tensor]`):
                The mask(s) to add to the frame.
        """
        if isinstance(obj_ids, int):
            obj_ids = [obj_ids]
        if not isinstance(input_masks, list):
            input_masks = [input_masks]
        if len(input_masks) != len(obj_ids):
            raise ValueError(
                f"Number of object ids ({len(obj_ids)}) does not match number of masks ({len(input_masks)})"
            )

        for obj_id, mask in zip(obj_ids, input_masks):
            obj_idx = inference_state._obj_id_to_idx(obj_id)

            device = inference_state.inference_device

            # Process mask
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)
            nb_dim = mask.dim()
            if nb_dim > 4 or nb_dim < 2:
                raise ValueError(f"Mask has an unsupported number of dimensions: {nb_dim}")
            for i in range(4 - nb_dim):
                mask = mask.unsqueeze(0)

            mask_H, mask_W = mask.shape[-2:]
            mask_inputs_orig = mask.to(device)
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

            inference_state.mask_inputs_per_obj[obj_idx][frame_idx] = mask_inputs.to(inference_state.torch_dtype)
            inference_state.point_inputs_per_obj[obj_idx].pop(frame_idx, None)  # Clear any point inputs

        return inference_state


__all__ = ["Sam2Processor"]
