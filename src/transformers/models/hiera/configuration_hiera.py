# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
""" hiera  model configuration"""


from typing import Tuple

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

HIERA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "namangarg110/hiera_base_224": "https://huggingface.co/namangarg110/hiera_base_224/blob/main/config.json",
    # See all Hiera models at https://huggingface.co/models?filter=hiera
}


class HieraConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HieraModel`]. It is used to instantiate a Hiera model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the HieraModel
    [namangarg110/hiera_base_224](https://huggingface.co/namangarg110/hiera_base_224/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        input_size (`Tuple[int, int]` or `int`, *optional*, defaults to `(224, 224)`):
            Dimensions of the input image (height, width).
        in_chans (`int`, *optional*, defaults to 3):
            Number of input channels.
        embedding_dimension (`int`, *optional*, defaults to 96):
            Dimension of the initial embedding.
        num_attention_heads (`int`, *optional*, defaults to 1):
            Initial number of attention heads.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of output classes.
        stages (`Tuple[int, ...]`, *optional*, defaults to `(2, 3, 16, 3)`):
            Defines the number of blocks at each stage of the model.
        q_pool (`int`, *optional*, defaults to 3):
            Number of pooling stages for queries.
        q_stride (`Tuple[int, ...]`, *optional*, defaults to `(2, 2)`):
            Stride size for pooling.
        mask_unit_size (`Tuple[int, ...]`, *optional*, defaults to `(8, 8)`):
            Dimensions for the mask unit. Must be compatible with q_stride.
        mask_unit_attn (`Tuple[bool, ...]`, *optional*, defaults to `(True, True, False, False)`):
            Specifies which stages use mask unit attention.
        dim_mul (`float`, *optional*, defaults to 2.0):
            Factor for increasing the dimensionality through the network.
        head_mul (`float`, *optional*, defaults to 2.0):
            Factor for increasing the number of heads through the network.
        patch_kernel (`Tuple[int, ...]`, *optional*, defaults to `(7, 7)`):
            Kernel size for patch embedding.
        patch_stride (`Tuple[int, ...]`, *optional*, defaults to `(4, 4)`):
            Stride for patch embedding.
        patch_padding (`Tuple[int, ...]`, *optional*, defaults to `(3, 3)`):
            Padding for patch embedding.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of hidden size to feed-forward layer size.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for stochastic depth.
        head_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for attention heads.
        head_init_scale (`float`, *optional*, defaults to 0.001):
            Initial scaling factor for attention head weights.
        sep_position_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to use separate position embeddings.
    """

    model_type = "hiera"

    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        embedding_dimension: int = 96,  # initial embedding input_dim
        num_attention_heads: int = 1,  # initial number of num_attention_heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_position_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.in_chans = in_chans
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes
        self.stages = stages
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.mask_unit_size = mask_unit_size
        self.mask_unit_attn = mask_unit_attn
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.patch_kernel = patch_kernel
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.head_dropout = head_dropout
        self.head_init_scale = head_init_scale
        self.sep_position_embeddings = sep_position_embeddings
