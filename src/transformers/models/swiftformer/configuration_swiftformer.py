# Copyright 2023 MBZUAI and The HuggingFace Inc. team. All rights reserved.
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
"""SwiftFormer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="MBZUAI/swiftformer-xs")
@strict
class SwiftFormerConfig(PreTrainedConfig):
    r"""
    embed_dims (`list[int]`, *optional*, defaults to `[48, 56, 112, 220]`):
        The embedding dimension at each stage
    downsamples (`list[bool]`, *optional*, defaults to `[True, True, True, True]`):
        Whether or not to downsample inputs between two stages.
    down_patch_size (`int`, *optional*, defaults to 3):
        The size of patches in downsampling layers.
    down_stride (`int`, *optional*, defaults to 2):
        The stride of convolution kernels in downsampling layers.
    down_pad (`int`, *optional*, defaults to 1):
        Padding in downsampling layers.
    drop_mlp_rate (`float`, *optional*, defaults to 0.0):
        Dropout rate for the MLP component of SwiftFormer.
    drop_conv_encoder_rate (`float`, *optional*, defaults to 0.0):
        Dropout rate for the ConvEncoder component of SwiftFormer.
    use_layer_scale (`bool`, *optional*, defaults to `True`):
        Whether to scale outputs from token mixers.

    Example:

    ```python
    >>> from transformers import SwiftFormerConfig, SwiftFormerModel

    >>> # Initializing a SwiftFormer swiftformer-base-patch16-224 style configuration
    >>> configuration = SwiftFormerConfig()

    >>> # Initializing a model (with random weights) from the swiftformer-base-patch16-224 style configuration
    >>> model = SwiftFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "swiftformer"

    image_size: int | list[int] | tuple[int, int] = 224
    num_channels: int = 3
    depths: list[int] | tuple[int, ...] = (3, 3, 6, 4)
    embed_dims: list[int] | tuple[int, ...] = (48, 56, 112, 220)
    mlp_ratio: int = 4
    downsamples: list[bool] | tuple[bool, ...] = (True, True, True, True)
    hidden_act: str = "gelu"
    down_patch_size: int | list[int] | tuple[int, int] = 3
    down_stride: int = 2
    down_pad: int = 1
    drop_path_rate: float | int = 0.0
    drop_mlp_rate: float | int = 0.0
    drop_conv_encoder_rate: float | int = 0.0
    use_layer_scale: bool = True
    layer_scale_init_value: float = 1e-5
    batch_norm_eps: float = 1e-5


__all__ = ["SwiftFormerConfig"]
