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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/cvt-13")
class CvtConfig(PreTrainedConfig):
    r"""
    patch_padding (`list[int]`, *optional*, defaults to `[2, 1, 1]`):
        The padding size of each encoder's patch embedding.
    depth (`list[int]`, *optional*, defaults to `[1, 2, 10]`):
        The number of layers in each encoder block.
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
    attention_drop_rate (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`):
        The dropout ratio for the attention probabilities.
    drop_rate (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`):
        The dropout ratio for the patch embeddings probabilities.
    patch_stride (`list[int]`, *optional*, defaults to `[4, 2, 2]`):
        The stride size of each encoder's patch embedding.

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

    def __init__(
        self,
        num_channels=3,
        patch_sizes=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embed_dim=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 2, 10],
        mlp_ratio=[4.0, 4.0, 4.0],
        attention_drop_rate=[0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        qkv_bias=[True, True, True],
        cls_token=[False, False, True],
        qkv_projection_method=["dw_bn", "dw_bn", "dw_bn"],
        kernel_qkv=[3, 3, 3],
        padding_kv=[1, 1, 1],
        stride_kv=[2, 2, 2],
        padding_q=[1, 1, 1],
        stride_q=[1, 1, 1],
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.patch_sizes = patch_sizes
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.attention_drop_rate = attention_drop_rate
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.cls_token = cls_token
        self.qkv_projection_method = qkv_projection_method
        self.kernel_qkv = kernel_qkv
        self.padding_kv = padding_kv
        self.stride_kv = stride_kv
        self.padding_q = padding_q
        self.stride_q = stride_q
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


__all__ = ["CvtConfig"]
