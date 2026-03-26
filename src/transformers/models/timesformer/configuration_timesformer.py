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
"""TimeSformer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/timesformer-base-finetuned-k600")
@strict
class TimesformerConfig(PreTrainedConfig):
    r"""
    num_frames (`int`, *optional*, defaults to 8):
        The number of frames in each video.
    attention_type (`str`, *optional*, defaults to `"divided_space_time"`):
        The attention type to use. Must be one of `"divided_space_time"`, `"space_only"`, `"joint_space_time"`.

    Example:

    ```python
    >>> from transformers import TimesformerConfig, TimesformerModel

    >>> # Initializing a TimeSformer timesformer-base style configuration
    >>> configuration = TimesformerConfig()

    >>> # Initializing a model from the configuration
    >>> model = TimesformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "timesformer"

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    num_frames: int = 8
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    attention_type: str = "divided_space_time"
    drop_path_rate: int = 0


__all__ = ["TimesformerConfig"]
