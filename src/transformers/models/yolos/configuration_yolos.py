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
"""YOLOS model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="hustvl/yolos-base")
@strict
class YolosConfig(PreTrainedConfig):
    r"""
    num_detection_tokens (`int`, *optional*, defaults to 100):
        The number of detection tokens.
    use_mid_position_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to use the mid-layer position encodings.

    Example:

    ```python
    >>> from transformers import YolosConfig, YolosModel

    >>> # Initializing a YOLOS hustvl/yolos-base style configuration
    >>> configuration = YolosConfig()

    >>> # Initializing a model (with random weights) from the hustvl/yolos-base style configuration
    >>> model = YolosModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "yolos"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: list[int] | tuple[int, ...] = (512, 864)
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    num_detection_tokens: int = 100
    use_mid_position_embeddings: bool = True
    auxiliary_loss: bool = False
    class_cost: int = 1
    bbox_cost: int = 5
    giou_cost: int = 2
    bbox_loss_coefficient: int = 5
    giou_loss_coefficient: int = 2
    eos_coefficient: float = 0.1


__all__ = ["YolosConfig"]
