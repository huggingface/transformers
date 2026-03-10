# Copyright 2022 Sea AI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""PoolFormer model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="sail/poolformer_s12")
class PoolFormerConfig(PreTrainedConfig):
    r"""
    stride (`int`, *optional*, defaults to 16):
        The stride of the input patch.
    pool_size (`int`, *optional*, defaults to 3):
        The size of the pooling window.
    patch_sizes (`list`, *optional*, defaults to `[7, 3, 3, 3]`):
        The size of the input patch for each encoder block.
    strides (`list`, *optional*, defaults to `[4, 2, 2, 2]`):
        The stride of the input patch for each encoder block.
    padding (`list`, *optional*, defaults to `[2, 1, 1, 1]`):
        The padding of the input patch for each encoder block.
    num_encoder_blocks (`int`, *optional*, defaults to 4):
        The number of encoder blocks.
    use_layer_scale (`bool`, *optional*, defaults to `True`):
        Whether to use layer scale.

    Example:

    ```python
    >>> from transformers import PoolFormerConfig, PoolFormerModel

    >>> # Initializing a PoolFormer sail/poolformer_s12 style configuration
    >>> configuration = PoolFormerConfig()

    >>> # Initializing a model (with random weights) from the sail/poolformer_s12 style configuration
    >>> model = PoolFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "poolformer"

    def __init__(
        self,
        num_channels=3,
        patch_size=16,
        stride=16,
        pool_size=3,
        mlp_ratio=4.0,
        depths=[2, 2, 6, 2],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        padding=[2, 1, 1, 1],
        num_encoder_blocks=4,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        initializer_range=0.02,
        **kwargs,
    ):
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.hidden_sizes = hidden_sizes
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_encoder_blocks = num_encoder_blocks
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


__all__ = ["PoolFormerConfig"]
