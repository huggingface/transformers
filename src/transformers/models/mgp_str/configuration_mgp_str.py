# coding=utf-8
# Copyright 2022 Google AI and The HuggingFace Inc. team. All rights reserved.
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
""" MGPSTR model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "alibaba-damo/mgp-str-base": "https://huggingface.co/alibaba-damo/mgp-str-base/resolve/main/config.json",
}


class MGPSTRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MGPSTRModel`]. It is used to instantiate an
    MGPSTR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MGPSTR
    [alibaba-damo/mgp-str-base](https://huggingface.co/alibaba-damo/mgp-str-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        img_size (`int`, *optional*, defaults to `32x128`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to `4`):
            The size (resolution) of each patch.
        in_chans (`int`, *optional*, defaults to `3`):
            The number of input channels.
        max_token_length (`int`, *optional*, defaults to `27`):
            The max number of output tokens.
        char_num_classes (`int`, *optional*, defaults to `27`):
            The number of classes for char head .
        bpe_num_classes (`int`, *optional*, defaults to `27`):
            The number of classes for bpe head .
        wp_num_classes (`int`, *optional*, defaults to `27`):
            The number of classes for wp head .
        embed_dim (`int`, *optional*, defaults to 768):
            The embedding dimension.
        depth (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`int`, *optional*, defaults to 12):
            The ratio of mlp hidden dim to embedding dim.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        representation_size (`int`, *optional*, defaults to `None`):
            Enable and set representation layer (pre-logits) to this value if set.
        distilled (`bool`, *optional*, defaults to `False`):
            Model includes a distillation token and head as in DeiT models.
        drop_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        attn_drop_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The stochastic depth rate.
        norm_layer (`str` or `function`, *optional*, defaults to `None`):
            The normalization layer function (function or string) in the encoder. for example, `"LayerNorm"`.
        act_layer (`str` or `function`, *optional*, defaults to `"None"`):
            The non-linear activation function (function or string) in the encoder. for example, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"`.
        weight_init (`str`, *optional*, defaults to `''`):
            The weight init scheme.

    Example:

    ```python
    >>> from transformers import MGPSTRConfig, MGPSTRModel

    >>> # Initializing a MGPSTR mgp-str-base style configuration
    >>> configuration = MGPSTRConfig()

    >>> # Initializing a model (with random weights) from the mgp-str-base style configuration
    >>> model = MGPSTRModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mgp-str"

    def __init__(
        self,
        img_size=(32, 128),
        patch_size=4,
        in_chans=3,
        max_token_length=27,
        char_num_classes=38,
        bpe_num_classes=50257,
        wp_num_classes=30522,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.max_token_length = max_token_length
        self.char_num_classes = char_num_classes
        self.bpe_num_classes = bpe_num_classes
        self.wp_num_classes = wp_num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.representation_size = representation_size
        self.distilled = distilled
        self.patch_size = patch_size
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.weight_init = weight_init
