# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""VJEPA 2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/vjepa2-vitl-fpc64-256")
@strict
class VJEPA2Config(PreTrainedConfig):
    r"""
    crop_size (`int`, *optional*, defaults to 256):
        Input resolution of the model
    frames_per_clip (`int`, *optional*, defaults to 64):
        The number of frames the model has been pretrained with. Does not impact inference.
    tubelet_size (`int`, *optional*, defaults to 2):
        The number of temporal frames used for a single rastor, check paper for more information.
    num_pooler_layers (`int`, *optional*, defaults to 3):
        The number of self-attention layers in the pooler.
    pred_hidden_size (`int`, *optional*, defaults to 384):
        Dimensionality of the predictor layers
    pred_num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Predictor
    pred_num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Predictor
    pred_num_mask_tokens (`int`, *optional*, defaults to 10):
        Define the number of mask tokens to use in the Predictor
    pred_zero_init_mask_tokens (`bool`, *optional*, defaults to `True`):
        Initialize the mask tokens in the predictor with 0.
    pred_mlp_ratio (`float`, *optional*, defaults to 4.0):
        Ratio of the hidden size of the MLPs used in Predictor relative to the `pred_hidden_size`.

    Example:

    ```python
    >>> from transformers import VJEPA2Config, VJEPA2Model

    >>> # Initializing a VJEPA2 vjepa2-vitl-fpc64-256 style configuration
    >>> configuration = VJEPA2Config()

    >>> # Initializing a model (with random weights) from the vjepa2-vitl-fpc64-256  style configuration
    >>> model = VJEPA2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vjepa2"

    patch_size: int | list[int] | tuple[int, int] = 16
    crop_size: int = 256
    frames_per_clip: int = 64
    tubelet_size: int = 2
    hidden_size: int = 1024
    in_chans: int = 3
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    drop_path_rate: float = 0.0
    mlp_ratio: int | float = 4.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    attention_probs_dropout_prob: float = 0.0
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0
    num_pooler_layers: int = 3
    pred_hidden_size: int = 384
    pred_num_attention_heads: int = 12
    pred_num_hidden_layers: int = 12
    pred_num_mask_tokens: int = 10
    pred_zero_init_mask_tokens: bool = True
    pred_mlp_ratio: int | float = 4.0


__all__ = ["VJEPA2Config"]
