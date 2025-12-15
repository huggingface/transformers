# coding=utf-8
# Copyright 2025 bzantium and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeekV3 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V3)

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
"""DeepSeekV3.2 model configuration"""

from typing import Optional

from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


DEEPSEEK_V32_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DeepseekV32Config(DeepseekV3Config):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV32Model`]. It is used to instantiate a DeepSeek
    V3.2 model according to the specified arguments, defining the model architecture. 
    
    DeepSeek V3.2 extends DeepSeek V3 with native sparse attention mechanism using an indexer for efficient
    attention computation on long sequences.

    Configuration objects inherit from [`DeepseekV3Config`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        index_topk (`int`, *optional*, defaults to 2048):
            Number of top-k tokens to select for sparse attention. This enables the native sparse attention
            mechanism in DeepSeek V3.2.
        **kwargs:
            All other arguments from DeepseekV3Config.

    ```python
    >>> from transformers import DeepseekV32Model, DeepseekV32Config

    >>> # Initializing a Deepseek-V3.2 style configuration
    >>> configuration = DeepseekV32Config()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v32"

    def __init__(
        self,
        index_topk: Optional[int] = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_topk = index_topk


__all__ = ["DeepseekV32Config"]
