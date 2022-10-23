# coding=utf-8
# Copyright 2022 kiansierra90@gmail.com and The HuggingFace Inc. team. All rights reserved.
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
""" FAN model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

FAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ksmcg/fan_tiny_12_p16_224": "https://huggingface.co/ksmcg/fan_tiny_12_p16_224/resolve/main/config.json",
    "ksmcg/fan_small_12_p16_224_se_attn": "https://huggingface.co/ksmcg/fan_small_12_p16_224_se_attn/resolve/main/config.json",
    "ksmcg/fan_small_12_p16_224": "https://huggingface.co/ksmcg/fan_small_12_p16_224/resolve/main/config.json",
    "ksmcg/fan_base_18_p16_224": "https://huggingface.co/ksmcg/fan_base_18_p16_224/resolve/main/config.json",
    "ksmcg/fan_large_24_p16_224": "https://huggingface.co/ksmcg/fan_large_24_p16_224/resolve/main/config.json",
}


class FANConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~FANModel`].
    It is used to instantiate an FAN model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the FAN [nvidia/fan](https://huggingface.co/nvidia/fan) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the FAN model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~FANModel`] or
            [`~TFFANModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~FANModel`] or
            [`~TFFANModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import FANModel, FANConfig

    >>> # Initializing a FAN nvidia/fan style configuration
    >>> configuration = FANConfig()

    >>> # Initializing a model from the nvidia/fan style configuration
    >>> model = FANModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "fan"

    def __init__(
        self,
        patch_size=16,
        embed_dim=384,
        depth=12,
        depths=None,
        num_heads=8,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        se_mlp=False,
        sr_ratio=None,
        initializer_range=1.0,
        img_size=(224, 224),
        in_chans=3,
        num_classes=1000,
        backbone=None,
        use_checkpoint=False,
        use_pos_embed=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=None,
        norm_layer=None,
        cls_attn_layers=2,
        c_head_num=None,
        hybrid_patch_size=2,
        head_init_scale=1.0,
        channel_dims=None,
        feat_downsample=False,
        out_index=-1,
        rounding_mode="floor",
        in_channels=[128, 256, 480, 480],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        decoder_hidden_size=768,
        reshape_last_stage=False,
        semantic_loss_ignore_index=-100,
        **kwargs,
    ):

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        head_init_scale = head_init_scale
        self.depth = depth
        self.depths = depths
        self.num_heads = num_heads
        self.eta = eta
        self.tokens_norm = tokens_norm
        self.sharpen_attn = sharpen_attn
        self.se_mlp = se_mlp
        self.sr_ratio = sr_ratio if sr_ratio else [1] * (depth // 2) + [1] * (depth // 2)
        self.initializer_range = initializer_range
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.backbone = backbone
        self.use_checkpoint = use_checkpoint
        self.use_pos_embed = use_pos_embed
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        # TODO: Clean Different Dropout Rates
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.dropout_ratio = dropout_ratio  # TODO: Decoder Dropout
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.cls_attn_layers = cls_attn_layers
        self.norm_layer = norm_layer
        self.c_head_num = c_head_num
        self.hybrid_patch_size = hybrid_patch_size
        self.channel_dims = channel_dims
        self.out_index = out_index
        self.feat_downsample = feat_downsample
        self.rounding_mode = rounding_mode
        self.in_channels = in_channels
        self.in_index = in_index
        self.feature_strides = feature_strides
        self.channels = channels

        self.decoder_hidden_size = decoder_hidden_size
        self.reshape_last_stage = reshape_last_stage
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        super().__init__(**kwargs)
