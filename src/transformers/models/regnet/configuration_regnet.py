# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""RegNet model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/regnet-y-040")
@strict
class RegNetConfig(PreTrainedConfig):
    r"""
    groups_width (`int`, *optional*, defaults to 64):
        Width of group for each stage.
    layer_type (`str`, *optional*, defaults to `"y"`):
        The layer to use, it can be either `"x" or `"y"`. An `x` layer is a ResNet's BottleNeck layer with
        `reduction` fixed to `1`. While a `y` layer is a `x` but with squeeze and excitation. Please refer to the
        paper for a detailed explanation of how these layers were constructed.
    downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
        If `True`, the first stage will downsample the inputs using a `stride` of 2.

    Example:
    ```python
    >>> from transformers import RegNetConfig, RegNetModel

    >>> # Initializing a RegNet regnet-y-40 style configuration
    >>> configuration = RegNetConfig()
    >>> # Initializing a model from the regnet-y-40 style configuration
    >>> model = RegNetModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "regnet"
    layer_types = ["x", "y"]

    num_channels: int = 3
    embedding_size: int = 32
    hidden_sizes: list[int] | tuple[int, ...] = (128, 192, 512, 1088)
    depths: list[int] | tuple[int, ...] = (2, 6, 12, 2)
    groups_width: int = 64
    layer_type: str = "y"
    hidden_act: str = "relu"
    downsample_in_first_stage: bool = True

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.layer_type not in self.layer_types:
            raise ValueError(f"layer_type={self.layer_type} is not one of {','.join(self.layer_types)}")


__all__ = ["RegNetConfig"]
