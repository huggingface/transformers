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
Processor class for SAMHQ.
"""

from copy import deepcopy
from typing import Optional, Union

import numpy as np

from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, BatchEncoding, PreTokenizedInput, TextInput
from ...utils import is_torch_available
from ...video_utils import VideoInput


if is_torch_available():
    import torch


class SamHQImagesKwargs(ImagesKwargs):
    segmentation_maps: Optional[ImageInput]
    input_points: Optional[list[list[float]]]
    input_labels: Optional[list[list[int]]]
    input_boxes: Optional[list[list[list[float]]]]
    point_pad_value: Optional[int]


class SamHQProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SamHQImagesKwargs
    _defaults = {
        "images_kwargs": {
            "point_pad_value": None,
        }
    }


class SamHQProcessor(ProcessorMixin):
    r"""
    Constructs a SAM HQ processor which wraps a SAM  image processor and an 2D points & Bounding boxes processor into a
    single processor.

    [`SamHQProcessor`] offers all the functionalities of [`SamImageProcessor`]. See the docstring of
    [`~SamImageProcessor.__call__`] for more information.

    Args:
        image_processor (`SamImageProcessor`):
            An instance of [`SamImageProcessor`]. The image processor is a required input.
    """

    attributes = ["image_processor"]
    image_processor_class = "SamImageProcessor"

    def __init__(self, image_processor):
        super().__init__(image_processor)
        # Ensure image_processor is properly initialized
        if not hasattr(self, "image_processor"):
            raise ValueError("image_processor was not properly initialized")
        if not hasattr(self.image_processor, "size"):
            raise ValueError("image_processor.size is not set")
        self.target_size = self.image_processor.size["longest_edge"]

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        audio: Optional[AudioInput] = None,
        video: Optional[VideoInput] = None,
        **kwargs: Unpack[SamHQProcessorKwargs],
    ) -> BatchEncoding:
        """
        This method uses [`SamImageProcessor.__call__`] method to prepare image(s) for the model. It also prepares 2D
        points and bounding boxes for the model if they are provided.
        """
        output_kwargs = self._merge_kwargs(
            SamHQProcessorKwargs,
            tokenizer_init_kwargs={},
            **kwargs,
        )

        input_points = output_kwargs["images_kwargs"].pop("input_points", None)
        input_labels = output_kwargs["images_kwargs"].pop("input_labels", None)
        input_boxes = output_kwargs["images_kwargs"].pop("input_boxes", None)

        encoding_image_processor = self.image_processor(
            images,
            **output_kwargs["images_kwargs"],
        )

        original_sizes = encoding_image_processor["original_sizes"]

        if hasattr(original_sizes, "numpy"):
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
            return_tensors=output_kwargs["common_kwargs"].get("return_tensors"),
            point_pad_value=output_kwargs["images_kwargs"].get("point_pad_value"),
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
        point_pad_value=-10,
    ):
        """
        Normalize and convert the image processor output to the expected format.
        """
        # Process input points
        if input_points is not None:
            input_points = self._normalize_batch_coordinates(input_points, original_sizes)

            if not all(point.shape == input_points[0].shape for point in input_points):
                if input_labels is not None:
                    input_points, input_labels = self._pad_points_and_labels(
                        input_points, input_labels, point_pad_value
                    )

            input_points = np.array(input_points)

        # Process input labels
        if input_labels is not None:
            input_labels = np.array(input_labels)

        # Process input boxes
        if input_boxes is not None:
            input_boxes = self._normalize_batch_coordinates(input_boxes, original_sizes, is_bounding_box=True)
            input_boxes = np.array(input_boxes)

        # Update processor with converted inputs
        if input_boxes is not None:
            encoding_image_processor["input_boxes"] = self._to_tensor(input_boxes, 3, return_tensors)
        if input_points is not None:
            encoding_image_processor["input_points"] = self._to_tensor(input_points, 4, return_tensors)
        if input_labels is not None:
            encoding_image_processor["input_labels"] = self._to_tensor(input_labels, 3, return_tensors)

        return encoding_image_processor

    def _pad_points_and_labels(self, input_points, input_labels, point_pad_value):
        r"""
        The method pads the 2D points and labels to the maximum number of points in the batch.
        """
        expected_nb_points = max([point.shape[0] for point in input_points])
        processed_input_points = []
        for i, point in enumerate(input_points):
            if point.shape[0] != expected_nb_points:
                point = np.concatenate(
                    [point, np.zeros((expected_nb_points - point.shape[0], 2)) + point_pad_value], axis=0
                )
                input_labels[i] = np.append(input_labels[i], [point_pad_value])
            processed_input_points.append(point)
        input_points = processed_input_points
        return input_points, input_labels

    def _normalize_coordinates(
        self, target_size: int, coords: np.ndarray, original_size, is_bounding_box=False
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H,W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.image_processor._get_preprocess_shape(original_size, longest_edge=target_size)
        coords = deepcopy(coords).astype(float)

        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)

        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        return coords

    def _preprocess_input(self, inputs, error_message, expected_nesting=1, dtype=None):
        """
        Preprocess input by converting torch tensors to numpy arrays and validating structure.

        Args:
            inputs: The input to process
            error_message: Error message if validation fails
            expected_nesting: Expected nesting level (1 for points/labels, 2 for boxes)
            dtype: Optional data type for numpy array conversion

        Returns:
            Processed input as list of numpy arrays or None
        """
        if inputs is None:
            return None

        # Convert torch tensor to list if applicable
        if hasattr(inputs, "numpy"):
            inputs = inputs.numpy().tolist()

        # Validate structure based on expected nesting
        valid = isinstance(inputs, list)
        current = inputs

        for _ in range(expected_nesting):
            if not valid or not current:
                break
            valid = valid and isinstance(current[0], list)
            current = current[0] if current else None

        if not valid:
            raise ValueError(error_message)

        # Convert to numpy arrays
        return [np.array(item, dtype=dtype) for item in inputs]

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
        # Process each input type
        input_points = self._preprocess_input(input_points, "Input points must be a list of list of floating points.")

        input_labels = self._preprocess_input(input_labels, "Input labels must be a list of list integers.")

        input_boxes = self._preprocess_input(
            input_boxes,
            "Input boxes must be a list of list of list of floating points.",
            expected_nesting=2,
            dtype=np.float32,
        )

        return input_points, input_labels, input_boxes

    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names))

    def post_process_masks(self, *args, **kwargs):
        return self.image_processor.post_process_masks(*args, **kwargs)

    def _to_tensor(self, array, min_dim, return_tensors):
        """
        Convert numpy array to tensor and ensure proper dimensionality.
        Args:
            array: The numpy array to convert
            min_dim: The minimum number of dimensions the result should have
            return_tensors: The type of tensors to return (e.g., "pt" for PyTorch tensors)
        Returns:
            The converted array or tensor with proper dimensions
        """
        if return_tensors == "pt":
            array = torch.from_numpy(array)
            return array.unsqueeze(1) if array.ndim < min_dim else array
        return array

    def _normalize_batch_coordinates(self, inputs, original_sizes, is_bounding_box=False):
        """
        Normalize coordinates based on original sizes.
        Args:
            inputs: List of coordinate arrays
            original_sizes: Original sizes of the images
            is_bounding_box: Whether inputs are bounding boxes
        Returns:
            Normalized coordinates as list
        """
        if len(original_sizes) != len(inputs):
            # Use first original size for all inputs
            return [
                self._normalize_coordinates(self.target_size, item, original_sizes[0], is_bounding_box=is_bounding_box)
                for item in inputs
            ]
        else:
            # Use paired original sizes for each input
            return [
                self._normalize_coordinates(self.target_size, item, size, is_bounding_box=is_bounding_box)
                for item, size in zip(inputs, original_sizes)
            ]


__all__ = ["SamHQProcessor"]
