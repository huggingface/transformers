# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" MGP-STR model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SEGGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Raghavan/seggpt_semantic_segmentation": (
        "https://huggingface.co/Raghavan/seggpt_semantic_segmentation/raw/main/config.json"
    ),
    # See all BEiT models at https://huggingface.co/models?filter=beit
}


class SegGPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MgpstrModel`]. It is used to instantiate an
    MGP-STR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MGP-STR
    [alibaba-damo/mgp-str-base](https://huggingface.co/alibaba-damo/mgp-str-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`List[int]`, *optional*, defaults to `[896, 448]`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        embed_dim (`<fill_type>`, *optional*, defaults to 1024): <fill_docstring>
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The stochastic depth rate.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of mlp hidden dim to embedding dim.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_group_blocks (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
        use_rel_pos (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        out_feature (`<fill_type>`, *optional*, defaults to `"last_feat"`): <fill_docstring>
        decoder_embed_dim (`<fill_type>`, *optional*, defaults to 64): <fill_docstring>
        pretrain_img_size (`<fill_type>`, *optional*, defaults to 224): <fill_docstring>
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        merge_index (`<fill_type>`, *optional*, defaults to 2): <fill_docstring>

    Example:

    ```python
    >>> from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

    >>> # Initializing a Mgpstr mgp-str-base style configuration
    >>> configuration = MgpstrConfig()

    >>> # Initializing a model (with random weights) from the mgp-str-base style configuration
    >>> model = MgpstrForSceneTextRecognition(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mgp-str"

    def __init__(
        self,
        num_channels=3,
        image_size=[896, 448],
        patch_size=16,
        embed_dim=1024,
        num_attention_heads=16,
        drop_path_rate=0.1,
        qkv_bias=True,
        mlp_ratio=4.0,
        layer_norm_eps=1e-6,
        num_hidden_layers=24,
        num_group_blocks=4,
        use_rel_pos=True,
        out_feature="last_feat",
        decoder_embed_dim=64,
        pretrain_img_size=224,
        initializer_range=0.02,
        merge_index=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # self.depth = depth
        self.num_group_blocks = num_group_blocks
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.use_rel_pos = use_rel_pos
        self.out_feature = out_feature
        self.decoder_embed_dim = decoder_embed_dim
        self.initializer_range = initializer_range
        self.pretrain_img_size = pretrain_img_size
        self.merge_index = merge_index
