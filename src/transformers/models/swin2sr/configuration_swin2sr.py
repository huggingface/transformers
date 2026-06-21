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
"""Swin2SR Transformer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="caidas/swin2sr-classicalsr-x2-64")
@strict
class Swin2SRConfig(PreTrainedConfig):
    r"""
    num_channels_out (`int`, *optional*, defaults to `num_channels`):
        The number of output channels. If not set, it will be set to `num_channels`.
    depths (`list(int)`, *optional*, defaults to `[6, 6, 6, 6, 6, 6]`):
        Depth of each layer in the Transformer encoder.
    num_heads (`list(int)`, *optional*, defaults to `[6, 6, 6, 6, 6, 6]`):
        Number of attention heads in each layer of the Transformer encoder.
    window_size (`int`, *optional*, defaults to 8):
        Size of windows.
    upscale (`int`, *optional*, defaults to 2):
        The upscale factor for the image. 2/3/4/8 for image super resolution, 1 for denoising and compress artifact
        reduction
    img_range (`float`, *optional*, defaults to 1.0):
        The range of the values of the input image.
    resi_connection (`str`, *optional*, defaults to `"1conv"`):
        The convolutional block to use before the residual connection in each stage.
    upsampler (`str`, *optional*, defaults to `"pixelshuffle"`):
        The reconstruction reconstruction module. Can be 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None.

    Example:

    ```python
    >>> from transformers import Swin2SRConfig, Swin2SRModel

    >>> # Initializing a Swin2SR caidas/swin2sr-classicalsr-x2-64 style configuration
    >>> configuration = Swin2SRConfig()

    >>> # Initializing a model (with random weights) from the caidas/swin2sr-classicalsr-x2-64 style configuration
    >>> model = Swin2SRModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "swin2sr"

    attribute_map = {
        "hidden_size": "embed_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    image_size: int | list[int] | tuple[int, int] = 64
    patch_size: int | list[int] | tuple[int, int] = 1
    num_channels: int = 3
    num_channels_out: int | None = None
    embed_dim: int = 180
    depths: list[int] | tuple[int, ...] = (6, 6, 6, 6, 6, 6)
    num_heads: list[int] | tuple[int, ...] = (6, 6, 6, 6, 6, 6)
    window_size: int = 8
    mlp_ratio: float = 2.0
    qkv_bias: bool = True
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.1
    hidden_act: str = "gelu"
    use_absolute_embeddings: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    upscale: int = 2
    img_range: float = 1.0
    resi_connection: str = "1conv"
    upsampler: str = "pixelshuffle"

    def __post_init__(self, **kwargs):
        self.num_channels_out = self.num_channels if self.num_channels_out is None else self.num_channels_out
        self.num_layers = len(self.depths)
        super().__post_init__(**kwargs)


__all__ = ["Swin2SRConfig"]
