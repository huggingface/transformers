# Copyright 2024 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
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
"""Pvt V2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="OpenGVLab/pvt_v2_b0")
@strict
class PvtV2Config(BackboneConfigMixin, PreTrainedConfig):
    r"""
    num_encoder_blocks (`[int]`, *optional*, defaults to 4):
        The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
    sr_ratios (`list[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
        Spatial reduction ratios in each encoder block.
    patch_sizes (`list[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
        Patch size for overlapping patch embedding before each encoder block.
    strides (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
        Stride for overlapping patch embedding before each encoder block.
    num_attention_heads (`list[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
        Number of attention heads for each attention layer in each block of the Transformer encoder.
    mlp_ratios (`list[int]`, *optional*, defaults to `[8, 8, 4, 4]`):
        Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
        encoder blocks.
    linear_attention (`bool`, *optional*, defaults to `False`):
        Use linear attention complexity. If set to True, `sr_ratio` is ignored and average pooling is used for
        dimensionality reduction in the attention layers rather than strided convolution.

    Example:

    ```python
    >>> from transformers import PvtV2Model, PvtV2Config

    >>> # Initializing a pvt_v2_b0 style configuration
    >>> configuration = PvtV2Config()

    >>> # Initializing a model from the OpenGVLab/pvt_v2_b0 style configuration
    >>> model = PvtV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pvt_v2"

    image_size: int | list[int] | tuple[int, int] | dict = 224
    num_channels: int = 3
    num_encoder_blocks: int = 4
    depths: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    sr_ratios: list[int] | tuple[int, ...] = (8, 4, 2, 1)
    hidden_sizes: list[int] | tuple[int, ...] = (32, 64, 160, 256)
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3, 3)
    strides: list[int] | tuple[int, ...] = (4, 2, 2, 2)
    num_attention_heads: list[int] | tuple[int, ...] = (1, 2, 5, 8)
    mlp_ratios: list[int] | tuple[int, ...] = (8, 8, 4, 4)
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    drop_path_rate: float = 0.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    linear_attention: bool = False
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.image_size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
        self.stage_names = [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["PvtV2Config"]
