# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""UperNet model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="openmmlab/upernet-convnext-tiny")
@strict
class UperNetConfig(PreTrainedConfig):
    r"""
    pool_scales (`tuple[int]`, *optional*, defaults to `[1, 2, 3, 6]`):
        Pooling scales used in Pooling Pyramid Module applied on the last feature map.
    use_auxiliary_head (`bool`, *optional*, defaults to `True`):
        Whether to use an auxiliary head during training.
    auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):
        Weight of the cross-entropy loss of the auxiliary head.
    auxiliary_in_channels (`int`, *optional*, defaults to 256):
        Number of input channels in the auxiliary head.
    auxiliary_channels (`int`, *optional*, defaults to 256):
        Number of channels to use in the auxiliary head.
    auxiliary_num_convs (`int`, *optional*, defaults to 1):
        Number of convolutional layers to use in the auxiliary head.
    auxiliary_concat_input (`bool`, *optional*, defaults to `False`):
        Whether to concatenate the output of the auxiliary head with the input before the classification layer.
    loss_ignore_index (`int`, *optional*, defaults to 255):
        The index that is ignored by the loss function.

    Examples:

    ```python
    >>> from transformers import UperNetConfig, UperNetForSemanticSegmentation

    >>> # Initializing a configuration
    >>> configuration = UperNetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = UperNetForSemanticSegmentation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "upernet"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 512
    initializer_range: float = 0.02
    pool_scales: list[int] | tuple[int, ...] = (1, 2, 3, 6)
    use_auxiliary_head: bool = True
    auxiliary_loss_weight: float = 0.4
    auxiliary_in_channels: int | None = None
    auxiliary_channels: int = 256
    auxiliary_num_convs: int = 1
    auxiliary_concat_input: bool = False
    loss_ignore_index: int = 255

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="resnet",
            default_config_kwargs={
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["UperNetConfig"]
