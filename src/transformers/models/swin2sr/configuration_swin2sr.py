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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="caidas/swin2sr-classicalsr-x2-64")
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

    def __init__(
        self,
        image_size=64,
        patch_size=1,
        num_channels=3,
        num_channels_out=None,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        upscale=2,
        img_range=1.0,
        resi_connection="1conv",
        upsampler="pixelshuffle",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_channels_out = num_channels if num_channels_out is None else num_channels_out
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.upscale = upscale
        self.img_range = img_range
        self.resi_connection = resi_connection
        self.upsampler = upsampler


__all__ = ["Swin2SRConfig"]
