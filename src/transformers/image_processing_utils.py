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
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs) -> BatchFeature:
        raise NotImplementedError("Each image processor must implement its own preprocess method")
