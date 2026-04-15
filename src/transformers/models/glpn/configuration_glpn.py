# Copyright 2022 KAIST and The HuggingFace Inc. team. All rights reserved.
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
"""GLPN model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="vinvino02/glpn-kitti")
@strict
class GLPNConfig(PreTrainedConfig):
    r"""
    num_encoder_blocks (`int`, *optional*, defaults to 4):
        The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
    depths (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
        The number of layers in each encoder block.
    sr_ratios (`list[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
        Sequence reduction ratios in each encoder block.
    patch_sizes (`list[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
        Patch size before each encoder block.
    strides (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
        Stride before each encoder block.
    num_attention_heads (`list[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
        Number of attention heads for each attention layer in each block of the Transformer encoder.
    mlp_ratios (`list[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
        Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
        encoder blocks.
    decoder_hidden_size (`int`, *optional*, defaults to 64):
        The dimension of the decoder.
    max_depth (`int`, *optional*, defaults to 10):
        The maximum depth of the decoder.
    head_in_index (`int`, *optional*, defaults to -1):
        The index of the features to use in the head.

    Example:

    ```python
    >>> from transformers import GLPNModel, GLPNConfig

    >>> # Initializing a GLPN vinvino02/glpn-kitti style configuration
    >>> configuration = GLPNConfig()

    >>> # Initializing a model from the vinvino02/glpn-kitti style configuration
    >>> model = GLPNModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glpn"

    num_channels: int = 3
    num_encoder_blocks: int = 4
    depths: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    sr_ratios: list[int] | tuple[int, ...] = (8, 4, 2, 1)
    hidden_sizes: list[int] | tuple[int, ...] = (32, 64, 160, 256)
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3, 3)
    strides: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    num_attention_heads: list[int] | tuple[int, ...] = (1, 2, 5, 8)
    mlp_ratios: list[int] | tuple[int, ...] = (4, 4, 4, 4)
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    drop_path_rate: float | int = 0.1
    layer_norm_eps: float = 1e-6
    decoder_hidden_size: int = 64
    max_depth: int = 10
    head_in_index: int = -1


__all__ = ["GLPNConfig"]
