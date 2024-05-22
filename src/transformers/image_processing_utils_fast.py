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

import functools

from .image_processing_utils import BaseImageProcessor
from .utils import is_torchvision_available


if is_torchvision_available():
    from torchvision.transforms import functional as F


class BaseImageProcessorFast(BaseImageProcessor):
    _transform_params = None
    _transform_settings = None

    def _set_transform_settings(self, **kwargs):
        settings = {}
        for k, v in kwargs.items():
            if k not in self._transform_params:
                raise ValueError(f"Invalid transform parameter {k}={v}.")
            settings[k] = v
        self._transform_settings = settings

    def _same_transforms_settings(self, **kwargs):
        """
        Check if the current settings are the same as the current transforms.
        """
        if self._transform_settings is None:
            return False

        for key, value in kwargs.items():
            if value not in self._transform_settings or value != self._transform_settings[key]:
                return False
        return True

    def _build_transforms(self, **kwargs):
        """
        Given the input settings e.g. do_resize, build the image transforms.
        """
        raise NotImplementedError

    def set_transforms(self, **kwargs):
        """
        Set the image transforms based on the given settings.
        If the settings are the same as the current ones, do nothing.
        """
        if self._same_transforms_settings(**kwargs):
            return self._transforms

        transforms = self._build_transforms(**kwargs)
        self._set_transform_settings(**kwargs)
        self._transforms = transforms

    @functools.lru_cache(maxsize=1)
    def _maybe_update_transforms(self, **kwargs):
        """
        If settings are different from those stored in `self._transform_settings`, update
        the image transforms to apply
        """
        if self._same_transforms_settings(**kwargs):
            return
        self.set_transforms(**kwargs)


def _cast_tensor_to_float(x):
    if x.is_floating_point():
        return x
    return x.float()


class FusedRescaleNormalize:
    """
    Rescale and normalize the input image in one step.
    """

    def __init__(self, mean, std, rescale_factor: float = 1.0, inplace: bool = False):
        self.mean = mean * (1.0 / rescale_factor)
        self.std = std * (1.0 / rescale_factor)

    def __call__(self, image):
        image = _cast_tensor_to_float(image)
        return F.normalize(image, self.mean, self.std, inplace=self.inplace)


class Rescale:
    """
    Rescale the input image by rescale factor: image *= rescale_factor.
    """

    def __init__(self, rescale_factor: float = 1.0):
        self.rescale_factor = rescale_factor

    def __call__(self, image):
        return image.mul(self.rescale_factor)
