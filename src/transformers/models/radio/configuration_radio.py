# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
"""RADIO vision encoder configuration."""

from ...configuration_utils import PretrainedConfig
from .common import DEFAULT_VERSION, RESOURCE_MAP, Resolution


__all__ = ["RADIOConfig"]


class RADIOConfig(PretrainedConfig):
    """Configuration for RADIO vision encoder models."""

    model_type = "radio"

    def __init__(
        self,
        args: dict | None = None,
        version: str | None = DEFAULT_VERSION,
        patch_size: int | None = None,
        max_resolution: int | None = None,
        preferred_resolution: Resolution | None = None,
        adaptor_names: str | list[str] = None,
        adaptor_configs: dict[str, dict[str, int]] = None,
        vitdet_window_size: int | None = None,
        feature_normalizer_config: dict | None = None,
        inter_feature_normalizer_config: dict | None = None,
        **kwargs,
    ):
        self.args = args
        for field in ["dtype", "amp_dtype"]:
            if self.args is not None and field in self.args:
                self.args[field] = str(args[field]).split(".")[-1]
        self.version = version
        resource = RESOURCE_MAP[version]
        self.patch_size = patch_size or resource.patch_size
        self.max_resolution = max_resolution or resource.max_resolution
        self.preferred_resolution = preferred_resolution or resource.preferred_resolution
        self.adaptor_names = adaptor_names
        self.adaptor_configs = adaptor_configs
        self.vitdet_window_size = vitdet_window_size
        self.feature_normalizer_config = feature_normalizer_config
        self.inter_feature_normalizer_config = inter_feature_normalizer_config
        super().__init__(**kwargs)
