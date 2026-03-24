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
"""VitPose backbone configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="usyd-community/vitpose-base-simple")
@strict
class VitPoseBackboneConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    part_features (`int`, *optional*):
        The number of part features to output. Only used in case `num_experts` is greater than 1.

    Example:

    ```python
    >>> from transformers import VitPoseBackboneConfig, VitPoseBackbone

    >>> # Initializing a VitPose configuration
    >>> configuration = VitPoseBackboneConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = VitPoseBackbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vitpose_backbone"

    image_size: int | list[int] | tuple[int, ...] = (256, 192)
    patch_size: int | list[int] | tuple[int, ...] = (16, 16)
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    num_experts: int = 1
    part_features: int = 256
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["VitPoseBackboneConfig"]
