# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

from typing import Dict, Iterable, Optional, Union

from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .feature_extraction_utils import FeatureExtractionMixin
from .utils import logging


logger = logging.get_logger(__name__)


# TODO: Move BatchFeature to be imported by both feature_extraction_utils and image_processing_utils
# We override the class string here, but logic is the same.
class BatchFeature(BaseBatchFeature):
    r"""
    Holds the output of the image processor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """


# We use aliasing whilst we phase out the old API. Once feature extractors for vision models
# are deprecated, ImageProcessor mixin will be implemented. Any shared logic will be abstracted out.
ImageProcessorMixin = FeatureExtractionMixin


class BaseImageProcessor(ImageProcessorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, images, **kwargs) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs) -> BatchFeature:
        raise NotImplementedError("Each image processor must implement its own preprocess method")


def get_size_dict(
    size: Union[int, Iterable[int], Dict[str, int]] = None,
    max_size: Optional[int] = None,
    height_width_order: bool = True,
    default_to_square: bool = True,
) -> dict:
    """
    Converts the old size parameter in the config into the new dict expected in the config. This is to ensure backwards
    compatibility with the old feature extractor configs and removes ambiguity over whether the tuple is in (height,
    width) or (width, height) format.

    - If `size` is tuple, it is converted to `{"height": size[0], "width": size[1]}` or `{"height": size[1], "width":
    size[0]}` if `height_width_order` is `False`.
    - If `size` is an int, and `default_to_square` is `True`, it is converted to `{"height": size, "width": size}`.
    - If `size` is an int and `default_to_square` is False, it is converted to `{"shortest_edge": size}`. If `max_size`
      is set, it is added to the dict as `{"longest_edge": max_size}`.

    Args:
        size (`Union[int, Iterable[int], Dict[str, int]]`, *optional*):
            The `size` parameter to be cast into a size dictionary.
        max_size (`Optional[int]`, *optional*):
            The `max_size` parameter to be cast into a size dictionary.
        height_width_order (`bool`, *optional*, defaults to `True`):
            If `size` is a tuple, whether it's in (height, width) or (width, height) order.
        default_to_square (`bool`, *optional*, defaults to `True`):
            If `size` is an int, whether to default to a square image or not.
    """
    # If a dict is passed, we check if it's a valid size dict and then return it.
    if isinstance(size, dict):
        size_keys = set(size.keys())
        if (
            size_keys != set(["height", "width"])
            and size_keys != set(["shortest_edge"])
            and size_keys != set(["shortest_edge", "longest_edge"])
        ):
            raise ValueError(
                "The size dict must contain either the keys ('height', 'width') or ('shortest_edge')"
                f"or ('shortest_edge', 'longest_edge') but got {size_keys}"
            )
        return size

    # By default, if size is an int we assume it represents a tuple of (size, size).
    elif isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError("Cannot specify both size as an int, with default_to_square=True and max_size")
        size_dict = {"height": size, "width": size}
    # In other configs, if size is an int and default_to_square is False, size represents the length of the shortest edge after resizing.
    elif isinstance(size, int) and not default_to_square:
        if max_size is not None:
            size_dict = {"shortest_edge": size, "longest_edge": max_size}
        else:
            size_dict = {"shortest_edge": size}
    elif isinstance(size, (tuple, list)) and height_width_order:
        size_dict = {"height": size[0], "width": size[1]}
    elif isinstance(size, (tuple, list)) and not height_width_order:
        size_dict = {"height": size[1], "width": size[0]}

    logger.warning(
        "The size parameter should be a dictionary with keys ('height', 'width'), ('shortest_edge', 'longest_edge')"
        f" or ('shortest_edge',) got {size}. Setting as {size_dict}.",
    )
    return size_dict
