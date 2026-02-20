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
Processor class for SAM3.
"""

from copy import deepcopy

import numpy as np

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import TensorType, auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

if is_torch_available():
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


@requires(backends=("torch",))
@auto_docstring
class Sam3Processor(ProcessorMixin):
    def __init__(
        self, image_processor, tokenizer, target_size: int | None = None, point_pad_value: int = -10, **kwargs
    ):
        r"""
        target_size (`int`, *optional*):
            The target size (target_size, target_size) to which the image will be resized.
        point_pad_value (`int`, *optional*, defaults to -10):
            The value used for padding input boxes.
        """
        super().__init__(image_processor, tokenizer, **kwargs)
        self.point_pad_value = point_pad_value
        self.target_size = target_size if target_size is not None else self.image_processor.size["height"]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        segmentation_maps: ImageInput | None = None,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = None,
        input_boxes_labels: list[list[list[int]]] | torch.Tensor | None = None,
        original_sizes: list[list[float]] | torch.Tensor | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        images (`ImageInput`, *optional*):
            The image(s) to process.
        text (`str`, `list[str]`, `list[list[str]]`, *optional*):
            The text to process.
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to process.
        input_boxes (`list[list[list[float]]]`, `torch.Tensor`, *optional*):
            The bounding boxes to process.
        input_boxes_labels (`list[list[int]]`, `torch.Tensor`, *optional*):
            The labels for the bounding boxes.
        original_sizes (`list[list[float]]`, `torch.Tensor`, *optional*):
            The original sizes of the images.

        Returns:
            A [`BatchEncoding`] with the following fields:
            - `pixel_values` (`torch.Tensor`): The processed image(s).
            - `original_sizes` (`list[list[float]]`): The original sizes of the images.
            - `labels` (`torch.Tensor`): The processed segmentation maps (if provided).
            - `input_boxes_labels` (`torch.Tensor`): The processed labels for the bounding boxes.
            - `input_boxes` (`torch.Tensor`): The processed bounding boxes.
        """
        encoding = None
        if images is not None:
            encoding = self.image_processor(
                images,
                segmentation_maps=segmentation_maps,
                return_tensors=return_tensors,
                **kwargs,
            )
        elif original_sizes is not None:
            if isinstance(original_sizes, torch.Tensor):
                original_sizes = original_sizes.cpu().tolist()
            encoding = BatchEncoding({"original_sizes": original_sizes}, tensor_type=return_tensors)
        elif input_boxes is not None:
            raise ValueError("Either images or original_sizes must be provided if input_boxes is not None")

        text = self._resolve_text_prompts(text, input_boxes)
        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, padding="max_length", max_length=32)
            if encoding is not None:
                encoding.update(text_inputs)
            else:
                encoding = text_inputs

        # Process input boxes if provided
        if input_boxes is not None:
            original_sizes = encoding["original_sizes"]
            # Validate and convert inputs to standardized format
            processed_boxes = self._validate_single_input(
                input_boxes,
                expected_depth=3,
                input_name="boxes",
                expected_format="[image level, box level, box coordinates]",
                expected_coord_size=4,
            )
            processed_boxes_labels = self._validate_single_input(
                input_boxes_labels,
                expected_depth=2,
                input_name="labels",
                expected_format="[image level, box level]",
            )

            # Get padding requirements for all inputs
            if processed_boxes is not None:
                boxes_max_dims = self._get_nested_dimensions(processed_boxes)[:2]
            if processed_boxes_labels is not None:
                boxes_labels_max_dims = self._get_nested_dimensions(processed_boxes_labels)[:2]

            # Ensure boxes and labels have consistent dimensions
            if processed_boxes is not None and processed_boxes_labels is not None:
                if boxes_max_dims != boxes_labels_max_dims:
                    raise ValueError(
                        "Input boxes and labels have inconsistent dimensions. Please ensure they have the same dimensions."
                    )

            # Pad and normalize all inputs to final tensor format
            if processed_boxes is not None:
                padded_boxes = self._pad_nested_list(processed_boxes, boxes_max_dims + [4])
                final_boxes = torch.tensor(padded_boxes, dtype=torch.float32)
                self._normalize_tensor_coordinates(
                    final_boxes, original_sizes, is_bounding_box=True, preserve_padding=True
                )
                final_boxes = box_xyxy_to_cxcywh(final_boxes)
                encoding.update({"input_boxes": final_boxes})

            if processed_boxes_labels is not None:
                padded_boxes_labels = self._pad_nested_list(processed_boxes_labels, boxes_labels_max_dims)
                final_boxes_labels = torch.tensor(padded_boxes_labels, dtype=torch.int64)
                encoding.update({"input_boxes_labels": final_boxes_labels})

        return encoding

    def _normalize_coordinates(self, coords: "torch.Tensor", original_size, is_bounding_box=False) -> "torch.Tensor":
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
        coords = deepcopy(coords).float()

        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)
        coords[..., 0] = coords[..., 0] / old_w
        coords[..., 1] = coords[..., 1] / old_h

        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        return coords

    def _convert_to_nested_list(self, data, expected_depth, current_depth=0):
        """
        Recursively convert various input formats (tensors, numpy arrays, lists) to nested lists.
        Preserves None values within lists.

        Args:
            data: Input data in any format (may be None or contain None values)
            expected_depth: Expected nesting depth
            current_depth: Current depth in recursion

        Returns:
            Nested list representation of the data (or None)
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
                # Continue recursion, preserving None values
                return [
                    self._convert_to_nested_list(item, expected_depth, current_depth + 1) if item is not None else None
                    for item in data
                ]
        elif isinstance(data, (int, float)):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _resolve_text_prompts(self, text, input_boxes):
        """
        Resolve text prompts by setting defaults based on prompt types.
        """
        # If no text provided, infer default based on prompt type
        if text is None:
            return "visual" if input_boxes else None

        if not isinstance(text, (list, tuple)):
            return text

        # Validate list/tuple length matches both prompt types if provided
        text = list(text)  # Convert to list to allow modification

        if input_boxes and len(text) != len(input_boxes):
            raise ValueError(
                f"The number of text prompts must match the number of input boxes. "
                f"Got {len(text)} text prompts and {len(input_boxes)} input boxes."
            )

        # Fill in None values with defaults based on corresponding prompt
        for i, text_value in enumerate(text):
            if text_value is None and input_boxes and input_boxes[i] is not None:
                text[i] = "visual"

        return text

    def _get_nested_dimensions(self, nested_list, max_dims=None):
        """
        Get the maximum dimensions at each level of nesting, skipping None values.

        Args:
            nested_list (`list`):
                Nested list structure (may contain None values).
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
                # Skip None values
                if item is None:
                    continue
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
        Recursively pad a nested list to match target dimensions. Replaces None values with padded structures.

        Args:
            nested_list (`list`):
                Nested list to pad (may contain None values).
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

        # Recursively pad sublists, replacing None with padded structures
        if current_level < len(target_dims) - 1:
            for i in range(len(nested_list)):
                if nested_list[i] is None:
                    # Replace None with fully padded structure
                    template_dims = target_dims[current_level + 1 :]
                    nested_list[i] = self._create_empty_nested_structure(template_dims, pad_value)
                elif isinstance(nested_list[i], list):
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
        Get the nesting level of a list structure, skipping None values.

        Args:
            input_list (`list`):
                The list to get the nesting level of.
        """
        if isinstance(input_list, list):
            if len(input_list) == 0:
                return 1
            # Find first non-None element to determine nesting level
            for item in input_list:
                if item is not None:
                    return 1 + self._get_nesting_level(item)
            # All elements are None, treat as single level
            return 1
        elif isinstance(input_list, (np.ndarray, torch.Tensor)):
            # For arrays/tensors, the nesting level is the number of dimensions
            return len(input_list.shape)
        return 0

    def _validate_single_input(
        self,
        data: torch.Tensor | np.ndarray | list,
        expected_depth: int,
        input_name: str,
        expected_format: str,
        expected_coord_size: int | None = None,
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
                        Expected coordinate size (4 for boxes, None for labels).
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
                Whether to preserve padding values (for boxes).
        """
        if preserve_padding:
            # For boxes: avoid normalizing pad values
            mask = tensor != self.point_pad_value
            coord_mask = mask.all(dim=-1, keepdim=True)

        for img_idx in range(len(original_sizes)):
            if img_idx < tensor.shape[0]:
                original_size = original_sizes[img_idx] if img_idx < len(original_sizes) else original_sizes[0]
                normalized_coords = self._normalize_coordinates(
                    tensor[img_idx], original_size, is_bounding_box=is_bounding_box
                )

                if preserve_padding:
                    # Only update non-padded values
                    img_mask = coord_mask[img_idx]
                    tensor[img_idx] = torch.where(
                        img_mask.expand_as(tensor[img_idx]), normalized_coords, tensor[img_idx]
                    )
                else:
                    tensor[img_idx] = normalized_coords

    def post_process_semantic_segmentation(self, outputs, target_sizes=None, threshold=0.5):
        """
        Converts the output of [`Sam3Model`] into semantic segmentation maps.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing semantic_seg.
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the semantic segmentation masks.

        Returns:
            semantic_segmentation: `list[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry is a binary mask (0 or 1).
        """
        return self.image_processor.post_process_semantic_segmentation(outputs, target_sizes, threshold)

    def post_process_object_detection(self, outputs, threshold=0.3, target_sizes=None):
        """
        Converts the raw output of [`Sam3Model`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. This is a convenience wrapper around the image processor method.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing pred_boxes, pred_logits, and optionally presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep object detection predictions.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted box on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.

        Example:

        ```python
        >>> from transformers import AutoModel, AutoProcessor
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> model = AutoModel.from_pretrained("facebook/sam3-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam3-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))
        >>> inputs = processor(images=image, text="cat", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # Post-process to get bounding boxes
        >>> results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=[image.size[::-1]])
        >>> boxes = results[0]["boxes"]
        >>> scores = results[0]["scores"]
        ```
        """
        return self.image_processor.post_process_object_detection(outputs, threshold, target_sizes)

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold=0.3,
        mask_threshold=0.5,
        target_sizes=None,
    ):
        """
        Converts the raw output of [`Sam3Model`] into instance segmentation predictions with bounding boxes and masks.
        This is a convenience wrapper around the image processor method.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing pred_boxes, pred_logits, pred_masks, and optionally
                presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep instance predictions.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the predicted masks.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted instance on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.
                - **masks** (`torch.Tensor`): Binary segmentation masks for each instance, shape (num_instances,
                  height, width).

        Example:

        ```python
        >>> from transformers import AutoModel, AutoProcessor
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> model = AutoModel.from_pretrained("facebook/sam3-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam3-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))
        >>> inputs = processor(images=image, text="cat", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # Post-process to get instance segmentation
        >>> results = processor.post_process_instance_segmentation(
        ...     outputs, threshold=0.3, target_sizes=[image.size[::-1]]
        ... )
        >>> masks = results[0]["masks"]
        >>> boxes = results[0]["boxes"]
        >>> scores = results[0]["scores"]
        ```
        """
        return self.image_processor.post_process_instance_segmentation(
            outputs, threshold, mask_threshold, target_sizes
        )


__all__ = ["Sam3Processor"]
