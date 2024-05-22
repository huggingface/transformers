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
from dataclasses import dataclass

from .image_processing_utils import BaseImageProcessor
from .utils.import_utils import is_torchvision_available


if is_torchvision_available():
    from torchvision.transforms import Compose


@dataclass(frozen=True)
class SizeDict:
    """
    Hashable dictionary to store image size information.
    """

    height: int = None
    width: int = None
    longest_edge: int = None
    shortest_edge: int = None
    max_height: int = None
    max_width: int = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in SizeDict.")


class BaseImageProcessorFast(BaseImageProcessor):
    _transform_params = None
    _transform_settings = None

    def _set_transform_settings(self, **kwargs) -> None:
        settings = {}
        for k, v in kwargs.items():
            if k not in self._transform_params:
                raise ValueError(f"Invalid transform parameter {k}={v}.")
            settings[k] = v
        self._transform_settings = settings

    def _same_transforms_settings(self, **kwargs) -> bool:
        """
        Check if the current settings are the same as the current transforms.
        """
        if self._transform_settings is None:
            return False

        for key, value in kwargs.items():
            if value not in self._transform_settings or value != self._transform_settings[key]:
                return False
        return True

    def _build_transforms(self, **kwargs) -> Compose:
        """
        Given the input settings e.g. do_resize, build the image transforms.
        """
        raise NotImplementedError

    def set_transforms(self, **kwargs) -> Compose:
        """
        Set the image transforms based on the given settings.
        If the settings are the same as the current ones, do nothing.
        """
        if self._same_transforms_settings(**kwargs):
            return self._transforms

        transforms = self._build_transforms(**kwargs)
        self._set_transform_settings(**kwargs)
        self._transforms = transforms
        return transforms

    @functools.lru_cache(maxsize=1)
    def _maybe_update_transforms(self, **kwargs) -> Compose:
        """
        If settings are different from those stored in `self._transform_settings`, update
        the image transforms to apply
        """
        if self._same_transforms_settings(**kwargs):
            return self._transforms
        return self.set_transforms(**kwargs)
