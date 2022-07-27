# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for GLPN."""

from tkinter import Image
from typing import Union

from numpy import np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, rescale
from ...image_utils import ImageType, is_batched, to_numpy_array, valid_images, get_image_size
from ...utils import logging

logger = logging.get_logger(__name__)


class GLPNImageProcessor(BaseImageProcessor):
    def __init__(self, do_resize=True, do_rescale=True, size_divisor=32, resample=Image.Resampling.BILINEAR, **kwargs) -> None:
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.size_divisor = size_divisor
        self.resample = resample
        super().__init__(**kwargs)

    def resize(self, image: np.ndarray, size_divisor: Union[int, float], resample: Image.Resampling, **kwargs) -> np.ndarray:
        height, width = get_image_size(image)
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        image = resize(image, (new_h, new_w), resample=resample, **kwargs)
        return image

    def rescale(self, image: np.ndarray, scale: Union[int, float], **kwargs) -> np.ndarray:
        return rescale(image, scale, **kwargs)

    def preprocess(self, images, do_resize=None, do_rescale=None, size_divisor=None, resample=None, return_tensors=None, **kwargs) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample

        # If a return type isn't specified, default to numpy arrays.
        return_tensors = ImageType.NUMPY if return_tensors is None else return_tensors

        if do_resize and size_divisor is None:
            raise ValueError("size_divisor is required for resizing")

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError("Invalid image(s)")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(img) for img in images]

        if do_resize:
            images = [self.resize(image, size_divisor=size_divisor, resample=resample) for image in images]

        if do_rescale:
            images = [self.rescale(image) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(**data, return_tensors=return_tensors)
