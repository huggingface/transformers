# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# All rights reserved.
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
"""Pvt model configuration"""

from collections.abc import Callable, Mapping

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Xrenya/pvt-tiny-224")
class PvtConfig(PreTrainedConfig):
    r"""
    num_encoder_blocks (`int`, *optional*, defaults to 4):
        The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
    depths (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
        The number of layers in each encoder block.
    sequence_reduction_ratios (`list[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
        Sequence reduction ratios in each encoder block.
    patch_sizes (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
        Patch size before each encoder block.
    strides (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
        Stride before each encoder block.
    num_attention_heads (`list[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
        Number of attention heads for each attention layer in each block of the Transformer encoder.
    mlp_ratios (`list[int]`, *optional*, defaults to `[8, 8, 4, 4]`):
        Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
        encoder blocks.
    num_labels ('int', *optional*, defaults to 1000):
        The number of classes.

    Example:

    ```python
    >>> from transformers import PvtModel, PvtConfig

    >>> # Initializing a PVT Xrenya/pvt-tiny-224 style configuration
    >>> configuration = PvtConfig()

    >>> # Initializing a model from the Xrenya/pvt-tiny-224 style configuration
    >>> model = PvtModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pvt"

    def __init__(
        self,
        image_size: int = 224,
        num_channels: int = 3,
        num_encoder_blocks: int = 4,
        depths: list[int] = [2, 2, 2, 2],
        sequence_reduction_ratios: list[int] = [8, 4, 2, 1],
        hidden_sizes: list[int] = [64, 128, 320, 512],
        patch_sizes: list[int] = [4, 2, 2, 2],
        strides: list[int] = [4, 2, 2, 2],
        num_attention_heads: list[int] = [1, 2, 5, 8],
        mlp_ratios: list[int] = [8, 8, 4, 4],
        hidden_act: Mapping[str, Callable] = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        num_labels: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sequence_reduction_ratios = sequence_reduction_ratios
        self.hidden_sizes = hidden_sizes
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.num_labels = num_labels
        self.qkv_bias = qkv_bias


__all__ = ["PvtConfig"]
