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
"""CvT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/cvt-13")
@strict
class CvtConfig(PreTrainedConfig):
    r"""
    patch_stride (`list[int]`, *optional*, defaults to `[4, 2, 2]`):
        The stride size of each encoder's patch embedding.
    patch_padding (`list[int]`, *optional*, defaults to `[2, 1, 1]`):
        The padding size of each encoder's patch embedding.
    depth (`list[int]`, *optional*, defaults to `[1, 2, 10]`):
        The number of layers in each encoder block.
    attention_drop_rate (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`):
        The dropout ratio for the attention probabilities.
    drop_rate (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`):
        The dropout ratio for the patch embeddings probabilities.
    cls_token (`list[bool]`, *optional*, defaults to `[False, False, True]`):
        Whether or not to add a classification token to the output of each of the last 3 stages.
    qkv_projection_method (`list[string]`, *optional*, defaults to ["dw_bn", "dw_bn", "dw_bn"]`):
        The projection method for query, key and value Default is depth-wise convolutions with batch norm. For
        Linear projection use "avg".
    kernel_qkv (`list[int]`, *optional*, defaults to `[3, 3, 3]`):
        The kernel size for query, key and value in attention layer
    padding_kv (`list[int]`, *optional*, defaults to `[1, 1, 1]`):
        The padding size for key and value in attention layer
    stride_kv (`list[int]`, *optional*, defaults to `[2, 2, 2]`):
        The stride size for key and value in attention layer
    padding_q (`list[int]`, *optional*, defaults to `[1, 1, 1]`):
        The padding size for query in attention layer
    stride_q (`list[int]`, *optional*, defaults to `[1, 1, 1]`):
        The stride size for query in attention layer

    Example:

    ```python
    >>> from transformers import CvtConfig, CvtModel

    >>> # Initializing a Cvt msft/cvt style configuration
    >>> configuration = CvtConfig()

    >>> # Initializing a model (with random weights) from the msft/cvt style configuration
    >>> model = CvtModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cvt"

    num_channels: int = 3
    patch_sizes: list[int] | tuple[int, ...] = (7, 3, 3)
    patch_stride: list[int] | tuple[int, ...] = (4, 2, 2)
    patch_padding: list[int] | tuple[int, ...] = (2, 1, 1)
    embed_dim: list[int] | tuple[int, ...] = (64, 192, 384)
    num_heads: list[int] | tuple[int, ...] = (1, 3, 6)
    depth: list[int] | tuple[int, ...] = (1, 2, 10)
    mlp_ratio: list[float] | tuple[float, ...] = (4.0, 4.0, 4.0)
    attention_drop_rate: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0)
    drop_rate: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0)
    drop_path_rate: list[float] | tuple[float, ...] = (0.0, 0.0, 0.1)
    qkv_bias: list[bool] | tuple[bool, ...] = (True, True, True)
    cls_token: list[bool] | tuple[bool, ...] = (False, False, True)
    qkv_projection_method: list[str] | tuple[str, ...] = ("dw_bn", "dw_bn", "dw_bn")
    kernel_qkv: list[int] | tuple[int, ...] = (3, 3, 3)
    padding_kv: list[int] | tuple[int, ...] = (1, 1, 1)
    stride_kv: list[int] | tuple[int, ...] = (2, 2, 2)
    padding_q: list[int] | tuple[int, ...] = (1, 1, 1)
    stride_q: list[int] | tuple[int, ...] = (1, 1, 1)
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12


__all__ = ["CvtConfig"]
