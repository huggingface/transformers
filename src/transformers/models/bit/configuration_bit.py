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
"""BiT model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/bit-50")
@strict
class BitConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    layer_type (`str`, *optional*, defaults to `"preactivation"`):
        The layer to use, it can be either `"preactivation"` or `"bottleneck"`.
    global_padding (`str`, *optional*):
        Padding strategy to use for the convolutional layers. Can be either `"valid"`, `"same"`, or `None`.
    num_groups (`int`, *optional*, defaults to 32):
        Number of groups used for the `BitGroupNormActivation` layers.
    embedding_dynamic_padding (`bool`, *optional*, defaults to `False`):
        Whether or not to make use of dynamic padding for the embedding layer.
    width_factor (`int`, *optional*, defaults to 1):
        The width factor for the model.

    Example:
    ```python
    >>> from transformers import BitConfig, BitModel

    >>> # Initializing a BiT bit-50 style configuration
    >>> configuration = BitConfig()

    >>> # Initializing a model (with random weights) from the bit-50 style configuration
    >>> model = BitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "bit"
    layer_types = ["preactivation", "bottleneck"]
    supported_padding = [None, "SAME", "VALID"]

    num_channels: int = 3
    embedding_size: int = 64
    hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048)
    depths: list[int] | tuple[int, ...] = (3, 4, 6, 3)
    layer_type: str = "preactivation"
    hidden_act: str = "relu"
    global_padding: str | None = None
    num_groups: int = 32
    drop_path_rate: float | int = 0.0
    embedding_dynamic_padding: bool = False
    output_stride: int = 32
    width_factor: int = 1
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.hidden_sizes = list(self.hidden_sizes)
        self.depths = list(self.depths)

        if self.global_padding is not None:
            self.global_padding = self.global_padding.upper()

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.layer_type not in self.layer_types:
            raise ValueError(f"layer_type={self.layer_type} is not one of {','.join(self.layer_types)}")

        if self.global_padding not in self.supported_padding:
            raise ValueError(f"Padding strategy {self.global_padding} not supported")


__all__ = ["BitConfig"]
