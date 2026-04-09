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
"""MGP-STR model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="alibaba-damo/mgp-str-base")
@strict
class MgpstrConfig(PreTrainedConfig):
    r"""
    max_token_length (`int`, *optional*, defaults to 27):
        The max number of output tokens.
    num_character_labels (`int`, *optional*, defaults to 38):
        The number of classes for character head .
    num_bpe_labels (`int`, *optional*, defaults to 50257):
        The number of classes for bpe head .
    num_wordpiece_labels (`int`, *optional*, defaults to 30522):
        The number of classes for wordpiece head .
    distilled (`bool`, *optional*, defaults to `False`):
        Model includes a distillation token and head as in DeiT models.
    drop_rate (`float`, *optional*, defaults to 0.0):
        The dropout probability for all fully connected layers in the embeddings, encoder.
    attn_drop_rate (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    output_a3_attentions (`bool`, *optional*, defaults to `False`):
        Whether or not the model should returns A^3 module attentions.

    Example:

    ```python
    >>> from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

    >>> # Initializing a Mgpstr mgp-str-base style configuration
    >>> configuration = MgpstrConfig()

    >>> # Initializing a model (with random weights) from the mgp-str-base style configuration
    >>> model = MgpstrForSceneTextRecognition(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mgp-str"

    image_size: list[int] | tuple[int, ...] = (32, 128)
    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    max_token_length: int = 27
    num_character_labels: int = 38
    num_bpe_labels: int = 50257
    num_wordpiece_labels: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: float | int = 4.0
    qkv_bias: bool = True
    distilled: bool = False
    layer_norm_eps: float = 1e-5
    drop_rate: float | int = 0.0
    attn_drop_rate: float | int = 0.0
    drop_path_rate: float | int = 0.0
    output_a3_attentions: bool = False
    initializer_range: float = 0.02


__all__ = ["MgpstrConfig"]
