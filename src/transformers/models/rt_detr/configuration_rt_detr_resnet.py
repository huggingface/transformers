# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""RT-DETR ResNet model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/resnet-50")
@strict
class RTDetrResNetConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    depths (`list[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
        Depth (number of layers) for each stage.
    layer_type (`str`, *optional*, defaults to `"bottleneck"`):
        The layer to use, it can be either `"basic"` (used for smaller models, like resnet-18 or resnet-34) or
        `"bottleneck"` (used for larger models like resnet-50 and above).
    hidden_act (`str`, *optional*, defaults to `"relu"`):
        The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
        are supported.
    downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
        If `True`, the first stage will downsample the inputs using a `stride` of 2.
    downsample_in_bottleneck (`bool`, *optional*, defaults to `False`):
        If `True`, the first conv 1x1 in ResNetBottleNeckLayer will downsample the inputs using a `stride` of 2.

    Example:
    ```python
    >>> from transformers import RTDetrResNetConfig, RTDetrResnetBackbone

    >>> # Initializing a ResNet resnet-50 style configuration
    >>> configuration = RTDetrResNetConfig()

    >>> # Initializing a model (with random weights) from the resnet-50 style configuration
    >>> model = RTDetrResnetBackbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "rt_detr_resnet"
    layer_types = ["basic", "bottleneck"]

    num_channels: int = 3
    embedding_size: int = 64
    hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048)
    depths: list[int] | tuple[int, ...] = (3, 4, 6, 3)
    layer_type: str = "bottleneck"
    hidden_act: str = "relu"
    downsample_in_first_stage: bool = False
    downsample_in_bottleneck: bool = False
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        self.hidden_sizes = list(self.hidden_sizes)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.layer_type not in self.layer_types:
            raise ValueError(f"layer_type={self.layer_type} is not one of {','.join(self.layer_types)}")


__all__ = ["RTDetrResNetConfig"]
