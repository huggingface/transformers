# Copyright 2023 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""ConvNeXTV2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/convnextv2-tiny-1k-224")
@strict
class ConvNextV2Config(BackboneConfigMixin, PreTrainedConfig):
    r"""
    num_stages (`int`, *optional*, defaults to 4):
        The number of stages in the model.

    Example:
    ```python
    >>> from transformers import ConvNeXTV2Config, ConvNextV2Model

    >>> # Initializing a ConvNeXTV2 convnextv2-tiny-1k-224 style configuration
    >>> configuration = ConvNeXTV2Config()

    >>> # Initializing a model (with random weights) from the convnextv2-tiny-1k-224 style configuration
    >>> model = ConvNextV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "convnextv2"

    num_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 4
    num_stages: int = 4
    hidden_sizes: list[int] | tuple[int, ...] | None = (96, 192, 384, 768)
    depths: list[int] | tuple[int, ...] | None = (3, 3, 9, 3)
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    drop_path_rate: float = 0.0
    image_size: int | list[int] | tuple[int, int] = 224
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["ConvNextV2Config"]
