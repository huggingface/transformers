# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

    def _build_transforms(self, **kwargs) -> "Compose":
        """
        Given the input settings e.g. do_resize, build the image transforms.
        """
        raise NotImplementedError

    def _validate_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k not in self._transform_params:
                raise ValueError(f"Invalid transform parameter {k}={v}.")

    @functools.lru_cache(maxsize=1)
    def get_transforms(self, **kwargs) -> "Compose":
        self._validate_params(**kwargs)
        return self._build_transforms(**kwargs)

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_transform_params", None)
        return encoder_dict
