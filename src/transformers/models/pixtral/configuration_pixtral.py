# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
"""Pixtral model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from ...utils import logging


logger = logging.get_logger(__name__)


class PixtralVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PixtralVisionModel`]. It is used to instantiate an
    Pixtral vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to the vision encoder used by Pixtral-12B.

    e.g. [pixtral-hf/pixtral-9b](https://huggingface.co/pixtral-hf/pixtral-9b)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input channels in the input images.
        image_size (`int`, *optional*, defaults to 1024):
            Max dimension of the input images.
        patch_size (`int`, *optional*, defaults to 16):
            Size of the image patches.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function used in the hidden layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the attention layers.
        rope_parameters (`RopeParameters`, *optional*):
            The RopeParameters
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import PixtralVisionModel, PixtralVisionConfig

    >>> # Initializing a Pixtral-12B style configuration
    >>> config = PixtralVisionConfig()

    >>> # Initializing a model (with randomly initialized weights) from the configuration
    >>> model = PixtralVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pixtral"

    def __init__(
        self,
        hidden_size: Optional[int] = 1024,
        intermediate_size: Optional[int] = 4096,
        num_hidden_layers: Optional[int] = 24,
        num_attention_heads: Optional[int] = 16,
        num_channels: Optional[int] = 3,
        image_size: Optional[int] = 1024,
        patch_size: Optional[int] = 16,
        hidden_act: Optional[str] = "gelu",
        attention_dropout: Optional[float] = 0.0,
        rope_parameters: Optional[RopeParameters | dict[RopeParameters]] = None,
        initializer_range: Optional[float] = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.head_dim = hidden_size // num_attention_heads
        self.initializer_range = initializer_range
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 10000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)


__all__ = ["PixtralVisionConfig"]
