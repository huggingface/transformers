# coding=utf-8
# Copyright 2025 The DeepseekAI and HuggingFace Team. All rights reserved.
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
"""DeepseekVL model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..sam.configuration_sam import SamVisionConfig
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..llama.configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)


class DeepseekVLAlignerConfig(PretrainedConfig):
    model_type = "deepseek_vl_align_model"
    base_config_key = "aligner_config"

    def __init__(
        self,
        depth=2,
        input_dim=1024,
        n_embed=4096,
        projector_type="low_high_hybrid_split_mlp_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.projector_type = projector_type


class DeepseekVLVisionConfig(PretrainedConfig):
    model_type = "deepseek_vl_vision_model"
    base_config_key = "vision_config"
    sub_configs = {"low_res_config": SiglipVisionConfig, "high_res_config": SamVisionConfig}

    def __init__(self, concat_type="tuple", use_high_res=False, low_res_config=None, high_res_config=None, **kwargs):
        super().__init__(**kwargs)

        if low_res_config is None:
            low_res_config = {}
            logger.info("`low_res_config` is `None`. Initializing the `SiglipVisionConfig` with default values.")

        if high_res_config is None:
            high_res_config = {}
            logger.info("`high_res_config` is `None`. Initializing the `SamVisionConfig` with default values.")

        self.concat_type = concat_type
        self.use_high_res = use_high_res
        self.low_res_config = SiglipVisionConfig(**low_res_config)
        self.high_res_config = SamVisionConfig(**high_res_config)


class DeepseekVLConfig(PretrainedConfig):
    model_type = "deepseek_vl"
    sub_configs = {
        "text_config": LlamaConfig,
        "aligner_config": DeepseekVLAlignerConfig,
        "vision_config": DeepseekVLVisionConfig,
    }

    def __init__(self, text_config=None, aligner_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `LlamaConfig` with default values.")

        if aligner_config is None:
            aligner_config = {}
            logger.info("`aligner_config` is `None`. Initializing the `DeepseekVLAlignerConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `DeepseekVLVisionConfig` with default values.")

        self.text_config = LlamaConfig(**text_config)
        self.aligner_config = DeepseekVLAlignerConfig(**aligner_config)
        self.vision_config = DeepseekVLVisionConfig(**vision_config)

    @classmethod
    def from_text_vision_configs(cls, text_config: LlamaConfig, aligner_config: DeepseekVLAlignerConfig, vision_config: DeepseekVLVisionConfig, **kwargs):
        r"""
        Instantiate a [`SiglipConfig`] (or a derived class) from siglip text model configuration and siglip vision
        model configuration.

        Returns:
            [`DeepseekVLConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), aligner_config=aligner_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


__all__ = ["DeepseekVLAlignerConfig", "DeepseekVLVisionConfig", "DeepseekVLConfig"]
