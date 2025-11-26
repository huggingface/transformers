# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Chatterbox model configuration"""

from ...configuration_utils import PretrainedConfig


class ChatterboxConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ChatterboxModel`]. It is used to instantiate a
    Chatterbox model according to the specified arguments, defining the model architecture.

    Chatterbox is a complete TTS pipeline that combines T3, S3Gen, and HiFTNet models.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.

    Args:
        t3_config (`dict` or `T3Config`, *optional*):
            Dictionary or config object for T3 model. If not provided, uses English-only defaults.
        s3gen_config (`dict` or `S3GenConfig`, *optional*):
            Dictionary or config object for S3Gen model. If not provided, uses defaults.
        hiftnet_config (`dict` or `HiFTNetConfig`, *optional*):
            Dictionary or config object for HiFTNet model. If not provided, uses defaults.
        is_multilingual (`bool`, *optional*, defaults to False):
            Whether to use multilingual configuration.

    Example:

    ```python
    >>> from transformers import ChatterboxConfig, ChatterboxModel

    >>> # Initializing a Chatterbox configuration
    >>> configuration = ChatterboxConfig()

    >>> # Initializing a model from the configuration
    >>> model = ChatterboxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "chatterbox"
    is_composition = True

    def __init__(
        self,
        t3_config=None,
        s3gen_config=None,
        hiftnet_config=None,
        is_multilingual=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        from ...models.t3.configuration_t3 import T3Config
        from ...models.s3gen.configuration_s3gen import S3GenConfig
        from ...models.hiftnet.configuration_hiftnet import HiFTNetConfig

        # Initialize sub-model configs
        # Handle both dict and Config object inputs
        if t3_config is None:
            if is_multilingual:
                self.t3_config = T3Config.multilingual()
            else:
                self.t3_config = T3Config.english_only()
        elif isinstance(t3_config, dict):
            self.t3_config = T3Config(**t3_config)
        else:
            self.t3_config = t3_config

        if s3gen_config is None:
            self.s3gen_config = S3GenConfig()
        elif isinstance(s3gen_config, dict):
            self.s3gen_config = S3GenConfig(**s3gen_config)
        else:
            self.s3gen_config = s3gen_config

        if hiftnet_config is None:
            self.hiftnet_config = HiFTNetConfig()
        elif isinstance(hiftnet_config, dict):
            self.hiftnet_config = HiFTNetConfig(**hiftnet_config)
        else:
            self.hiftnet_config = hiftnet_config

        self.is_multilingual = is_multilingual

    @classmethod
    def english_only(cls):
        """Create English-only configuration."""
        return cls(is_multilingual=False)

    @classmethod
    def multilingual(cls):
        """Create multilingual configuration."""
        return cls(is_multilingual=True)

    def to_dict(self):
        """Serialize to dict."""
        output = super().to_dict()
        output["t3_config"] = self.t3_config.to_dict()
        output["s3gen_config"] = self.s3gen_config.to_dict()
        output["hiftnet_config"] = self.hiftnet_config.to_dict()
        return output
