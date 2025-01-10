# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""X-ALMA model configuration"""

from transformers.models.llama.configuration_llama import LlamaConfig


class XALMAConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`XALMAModel`]. It is used to instantiate an X-ALMA
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`LlamaConfig`] and can be used to control the model outputs. Read the
    documentation from [`LlamaConfig`] for more information.


    Args:
        lora_size (:obj:`int`, *optional*, defaults to 512):
            The size of the LoRA layer.
        lora_alpha (:obj:`int`, *optional*, defaults to 2):
            The alpha hyper-parameter for the LoRA layer.

    ```python
    >>> from transformers import XALMAModel, XALMAConfig

    >>> # Initializing an X-ALMA style configuration
    >>> configuration = XALMAConfig()

    >>> # Initializing a model from the X-ALMA style configuration
    >>> model = XALMAModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xalma"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        lora_size=512,
        lora_alpha=2,
        **kwargs,
    ):
        self.lora_size = lora_size
        self.lora_alpha = lora_alpha

        super().__init__(**kwargs)
