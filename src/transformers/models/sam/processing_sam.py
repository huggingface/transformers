# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for SAM.
"""
from typing import Optional, Union

import numpy as np

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_torch_available


if is_torch_available():
    import torch
    import torch.nn.functional as F


class SamProcessor(ProcessorMixin):
    r"""
    Constructs a SAM processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.

    [`BlipProcessor`] offers all the functionalities of [`SamImageProcessor`] and [`AutoTokenizer`]. See the docstring
    of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`SamImageProcessor`):
            An instance of [`SamImageProcessor`]. The image processor is a required input.
    """
    attributes = ["image_processor"]
    image_processor_class = "SamImageProcessor"

    def __init__(self, image_processor):
        super().__init__(image_processor)
        self.current_processor = self.image_processor
        self.point_pad_value = -10

    def __call__(
        self,
        images=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`SamImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        encoding_image_processor = self.image_processor(
            images,
            return_tensors=return_tensors,
            **kwargs,
        )
        
        original_sizes = encoding_image_processor.pop("original_sizes")

        input_points, input_labels, input_boxes = self._check_and_preprocess_points(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
        )

        if input_points is not None:
            if len(original_sizes) != len(input_points):
                # TODO deal better with this case
                input_points = [self.normalize_coordinates(point, original_sizes[0]) for point in input_points]
            else:
                input_points = [
                    self.normalize_coordinates(point, original_size)
                    for point, original_size in zip(input_points, original_sizes)
                ]

            # check that all arrays have the same shape
            if not all([point.shape == input_points[0].shape for point in input_points]):
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
            input_points = np.array(input_points)

        if input_labels is not None:
            input_labels = np.array(input_labels)

        if input_boxes is not None:
            if len(original_sizes) != len(input_boxes):
                # TODO deal better with this case
                input_boxes = [
                    self.normalize_coordinates(box, original_sizes[0], is_bounding_box=True) for box in input_boxes
                ]
            else:
                input_boxes = [
                    self.normalize_coordinates(box, original_size, is_bounding_box=True)
                    for box, original_size in zip(input_boxes, original_sizes)
                ]
            input_boxes = np.array(input_boxes)

        if input_boxes is not None:
            input_boxes = torch.from_numpy(input_boxes) if return_tensors == "pt" else input_boxes
            encoding_image_processor.update({"input_boxes": input_boxes})
        if input_points is not None:
            input_points = torch.from_numpy(input_points) if return_tensors == "pt" else input_points
            encoding_image_processor.update({"input_points": input_points})
        if input_labels is not None:
            input_labels = torch.from_numpy(input_labels) if return_tensors == "pt" else input_labels
            encoding_image_processor.update({"input_labels": input_labels})

        return encoding_image_processor

    def _check_and_preprocess_points(
        self,
        input_points=None,
        input_labels=None,
        input_boxes=None,
    ):
        if input_points is not None:
            if not isinstance(input_points, list) and not isinstance(input_points[0], list):
                raise ValueError("Input points must be a list of list of floating integers.")
            input_points = [np.array(input_point) for input_point in input_points]
        else:
            input_points = None

        if input_labels is not None:
            if not isinstance(input_labels, list) and not isinstance(input_labels[0], list):
                raise ValueError("Input labels must be a list of list integers.")
            input_labels = [np.array(label) for label in input_labels]
        else:
            input_labels = None

        if input_boxes is not None:
            if not isinstance(input_boxes, tuple):
                raise ValueError("Input boxes must be a tuple of tuple of floating integers.")
            input_boxes = [np.array(box).astype(np.float32) for box in input_boxes]
        else:
            input_boxes = None

        return input_points, input_labels, input_boxes

    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names))

    def pad_to_target_size(
        self,
        image: np.ndarray,
        target_size: int = None,
    ):
        target_size = target_size if target_size is not None else self.target_size
        image = torch.from_numpy(image).permute(2, 0, 1)

        height, width = image.shape[-2:]
        padh = target_size - height
        padw = target_size - width
        image = F.pad(image, (0, padw, 0, padh))

        return image.numpy()
