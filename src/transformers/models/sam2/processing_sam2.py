# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
from typing import Optional, Union

import numpy as np

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_torch_available, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch


@requires(backends=("torch",))
class Sam2Processor(ProcessorMixin):
    r"""
    Constructs a SAM2 processor which wraps a SAM2 image processor and an 2D points & Bounding boxes processor into a
    single processor.

    [`Sam2Processor`] offers all the functionalities of [`Sam2ImageProcessorFast`] and [`Sam2VideoProcessor`]. See the docstring of
    [`~Sam2ImageProcessorFast.__call__`] and [`~Sam2VideoProcessor.__call__`] for more information.

    Args:
        image_processor (`Sam2ImageProcessorFast`):
            An instance of [`Sam2ImageProcessorFast`].
        target_size (`int`, *optional*):
            The target size (target_size, target_size) to which the image will be resized.
        point_pad_value (`int`, *optional*, defaults to -10):
            The value used for padding input points.
    """

    attributes = ["image_processor"]
    image_processor_class = "Sam2ImageProcessorFast"

    def __init__(self, image_processor, target_size: Optional[int] = None, point_pad_value: int = -10, **kwargs):
        super().__init__(image_processor, **kwargs)
        self.point_pad_value = point_pad_value
        self.target_size = target_size if target_size is not None else self.image_processor.size["height"]

    def __call__(
        self,
        images: ImageInput = None,
        segmentation_maps: ImageInput = None,
        input_points: Optional[Union[list[list[list[list[float]]]], torch.Tensor]] = None,
        input_labels: Optional[Union[list[list[list[int]]], torch.Tensor]] = None,
        input_boxes: Optional[Union[list[list[list[float]]], torch.Tensor]] = None,
        original_sizes: Optional[Union[list[list[float]], torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        This method uses [`Sam2ImageProcessorFast.__call__`] method to prepare image(s) for the model. It also prepares 2D
        points and bounding boxes for the model if they are provided.

        Args:
            images (`ImageInput`, *optional*):
                The image(s) to process.
            segmentation_maps (`ImageInput`, *optional*):
                The segmentation maps to process.
            input_points (`list[list[list[list[float]]]]`, `torch.Tensor`, *optional*):
                The points to add to the frame.
            input_labels (`list[list[list[int]]]`, `torch.Tensor`, *optional*):
                The labels for the points.
            input_boxes (`list[list[list[float]]]`, `torch.Tensor`, *optional*):
                The bounding boxes to add to the frame.
            original_sizes (`list[list[float]]`, `torch.Tensor`, *optional*):
                The original sizes of the images.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return.
            **kwargs:
                Additional keyword arguments to pass to the image processor.

        Returns:
            A [`BatchEncoding`] with the following fields:
            - `pixel_values` (`torch.Tensor`): The processed image(s).
            - `original_sizes` (`list[list[float]]`): The original sizes of the images.
            - `reshaped_input_sizes` (`torch.Tensor`): The reshaped input sizes of the images.
            - `labels` (`torch.Tensor`): The processed segmentation maps (if provided).
            - `input_points` (`torch.Tensor`): The processed points.
            - `input_labels` (`torch.Tensor`): The processed labels.
            - `input_boxes` (`torch.Tensor`): The processed bounding boxes.
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
            processed_points = self._validate_single_input(
                input_points,
                expected_depth=4,
                input_name="points",
                expected_format="[image level, object level, point level, point coordinates]",
                expected_coord_size=2,
            )
            processed_labels = self._validate_single_input(
                input_labels,
                expected_depth=3,
                input_name="labels",
                expected_format="[image level, object level, point level]",
            )
            processed_boxes = self._validate_single_input(
                input_boxes,
                expected_depth=3,
                input_name="boxes",
                expected_format="[image level, box level, box coordinates]",
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

    def _validate_single_input(
        self,
        data: Union[torch.Tensor, np.ndarray, list],
        expected_depth: int,
        input_name: str,
        expected_format: str,
        expected_coord_size: Optional[int] = None,
    ) -> list:
        """
                Validate a single input by ensuring proper nesting and raising an error if the input is not valid.

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
        .
        """
        if data is None:
            return None

        # Handle tensors and numpy arrays first
        if isinstance(data, (torch.Tensor, np.ndarray)):
            # For tensors/arrays, we can directly check the number of dimensions
            if data.ndim != expected_depth:
                raise ValueError(
                    f"Input {input_name} must be a tensor/array with {expected_depth} dimensions. The expected nesting format is {expected_format}. Got {data.ndim} dimensions."
                )
            elif expected_coord_size is not None:
                if data.shape[-1] != expected_coord_size:
                    raise ValueError(
                        f"Input {input_name} must be a tensor/array with {expected_coord_size} as the last dimension, got {data.shape[-1]}."
                    )
            return self._convert_to_nested_list(data, expected_depth)

        # Handle nested lists
        if isinstance(data, list):
            current_depth = self._get_nesting_level(data)
            if current_depth != expected_depth:
                raise ValueError(
                    f"Input {input_name} must be a nested list with {expected_depth} levels. The expected nesting format is {expected_format}. Got {current_depth} levels."
                )
            return self._convert_to_nested_list(data, expected_depth)

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

    def post_process_masks(
        self,
        masks,
        original_sizes,
        mask_threshold=0.0,
        binarize=True,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        apply_non_overlapping_constraints=False,
        **kwargs,
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[torch.Tensor], List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold for binarization and post-processing operations.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            max_hole_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a hole to fill.
            max_sprinkle_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a sprinkle to fill.
            apply_non_overlapping_constraints (`bool`, *optional*, defaults to `False`):
                Whether to apply non-overlapping constraints to the masks.

        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        return self.image_processor.post_process_masks(
            masks,
            original_sizes,
            mask_threshold,
            binarize,
            max_hole_area,
            max_sprinkle_area,
            apply_non_overlapping_constraints,
            **kwargs,
        )


__all__ = ["Sam2Processor"]
