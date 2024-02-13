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


class BaseImageProcessorFast(BaseImageProcessor):
    _transform_params = None

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
        for key, value in kwargs.items():
            if value not in self._transform_settings or value != self._transform_settings[key]:
                return False
        return True

    def _build_transforms(self, **kwargs):
        raise NotImplementedError

    def set_transforms(self, **kwargs):
        if self._same_transforms_settings(**kwargs):
            return self._transforms

        transforms = self._build_transforms(**kwargs)
        self._set_transform_settings(**kwargs)
        self._transforms = transforms

    @functools.lru_cache(maxsize=1)
    def _maybe_update_transforms(self, **kwargs):
        if self._same_transforms_settings(**kwargs):
            return
        self.set_transforms(**kwargs)
