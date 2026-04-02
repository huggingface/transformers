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
"""ViViT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/vivit-b-16x2-kinetics400")
@strict
class VivitConfig(PreTrainedConfig):
    r"""
    num_frames (`int`, *optional*, defaults to 32):
        The number of frames in each video.
    tubelet_size (`list[int]`, *optional*, defaults to `[2, 16, 16]`):
        The size (resolution) of each tubelet.

    Example:

    ```python
    >>> from transformers import VivitConfig, VivitModel

    >>> # Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
    >>> configuration = VivitConfig()

    >>> # Initializing a model (with random weights) from the google/vivit-b-16x2-kinetics400 style configuration
    >>> model = VivitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vivit"

    image_size: int | list[int] | tuple[int, int] = 224
    num_frames: int = 32
    tubelet_size: list[int] | tuple[int, ...] = (2, 16, 16)
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_fast"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-06
    qkv_bias: bool = True


__all__ = ["VivitConfig"]
