# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""VitMatte model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="hustvl/vitmatte-small-composition-1k")
@strict
class VitMatteConfig(PreTrainedConfig):
    r"""
    batch_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the batch norm layers.
    convstream_hidden_sizes (`list[int]`, *optional*, defaults to `[48, 96, 192]`):
        The output channels of the ConvStream module.
    fusion_hidden_sizes (`list[int]`, *optional*, defaults to `[256, 128, 64, 32]`):
        The output channels of the Fusion blocks.

    Example:

    ```python
    >>> from transformers import VitMatteConfig, VitMatteForImageMatting

    >>> # Initializing a ViTMatte hustvl/vitmatte-small-composition-1k style configuration
    >>> configuration = VitMatteConfig()

    >>> # Initializing a model (with random weights) from the hustvl/vitmatte-small-composition-1k style configuration
    >>> model = VitMatteForImageMatting(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vitmatte"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 384
    batch_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    convstream_hidden_sizes: list[int] | tuple[int, ...] = (48, 96, 192)
    fusion_hidden_sizes: list[int] | tuple[int, ...] = (256, 128, 64, 32)

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="vitdet",
            default_config_kwargs={"out_features": ["stage4"]},
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["VitMatteConfig"]
