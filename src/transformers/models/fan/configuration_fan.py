# coding=utf-8
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
""" FAN model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

FAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ksmcg/fan_tiny_12_p16_224": "https://huggingface.co/ksmcg/fan_tiny_12_p16_224/resolve/main/config.json",
    "ksmcg/fan_small_12_p16_224_se_attn": (
        "https://huggingface.co/ksmcg/fan_small_12_p16_224_se_attn/resolve/main/config.json"
    ),
    "ksmcg/fan_small_12_p16_224": "https://huggingface.co/ksmcg/fan_small_12_p16_224/resolve/main/config.json",
    "ksmcg/fan_base_18_p16_224": "https://huggingface.co/ksmcg/fan_base_18_p16_224/resolve/main/config.json",
    "ksmcg/fan_large_24_p16_224": "https://huggingface.co/ksmcg/fan_large_24_p16_224/resolve/main/config.json",
}


# ISSUE: Move configuration to nvidia/fan
class FANConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~FANModel`]. It is used to instantiate an FAN
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FAN
    [ksmcg/fan_base_18_p16_224](https://huggingface.co/ksmcg/fan_base_18_p16_224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        patch_size (int, defaults to 16):
            Size of each patch to generated the embedding tokens from the image
        hidden_size (`int`, *optional*, defaults to 384):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        depths (`List[int]`, *optional*, defaults to None):
            The number of layers in each encoder block, only applicable when using hybrid backbone (ConvNeXt).
        eta (`float` defatults to 1.0):
            Weight Initialization value for channel importance.
        tokens_norm (`bool`, defaults to True):
            Whether or not to apply normalization in the Class Attention block.
        se_mlp (`bool`, defaults to False):
            Wheter or not to use Squeeze-Excite in the FANEncoder layers MLP.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        img_size (`List[int]`, defaults to (224,224)):
            The size of the images being passed to the model.
        num_channels (`int`, defaults to 3):
            Number of channels the input image has.
        backbone (`string`, *optional*, defaults to None):
            Wheter or not to use 'hybrid' backbone.
        use_pos_embed ( `bool`, defaults to True):
            Wheter or not to use positional_encoding in the embeddings.
        mlp_ratio (`float`, defaults to 4.0):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        qkv_bias (`bool`, defaults to True):
            Whether or not to use bias in Query, Key and Value in attention layers.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, defaults to 0):
            Dropout used in Toxen Mixing after MLP.
        decoder_dropout (`float`, defaults to 0.1):
            Dropout used in Decode Head for SemanticSegmentation tasks.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        cls_attn_layers (`int`, defaults to 2):
            Number of ClassAttentionBlock used. Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239.
        hybrid_patch_size (`int`, defaults to 2):
            The patch size used in the hybrid embeddings, when using default backbone.
        channel_dims (`tuple(int)`, *optional*, defaults to None):
            List of Input channels for each of the encoder layers. If None it defaults to [config.hidden_size] *
            config.num_hidden_layers.
        feat_downsample (`bool`, defaults to ):
            Whether or not to use a learnable downsample convolution to obtain hidden states for SemanticSegmentation
            tasks. Only appliable with hybrid backbone.
        out_index (`int`, *optional*, defaults to -1):
            Additional Hidden state index position to add to the backbone hidden states and the last hidden state. Only
            applicable when using hybrid backbone.
        rounding_mode (`string`, *optional*, defaults to 'floor'):
            Torch Divison rounding mode used for positional encoding. Should be set to None in Semantic Segmentation
            tasks to be compatible with original paper implementation.
        segmentation_in_channels (tuple(int), defaults to (128, 256, 480, 480)):
            Number of channels in each of the hidden features used for Semantic Segmentation.
        decoder_hidden_size (`int`, *optional*, defaults to 768):
            The dimension of the all-MLP decode head for Semantic Segmenatation.
        semantic_loss_ignore_index (`int`, *optional*, defaults to -100):
            The index that is ignored by the loss function of the semantic segmentation model.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.

        Example:

        ```python
        >>> from transformers import FANModel, FANConfig

        >>> # Initializing a FAN ksmcg/fan_base_18_p16_224 style configuration
        >>> configuration = FANConfig()

        >>> # Initializing a model from the ksmcg/fan_base_18_p16_224 style configuration
        >>> model = FANModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```"""
    model_type = "fan"

    def __init__(
        self,
        patch_size=16,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=8,
        depths=None,
        eta=1.0,
        tokens_norm=True,
        se_mlp=False,
        initializer_range=1.0,
        img_size=[224, 224],
        num_channels=3,
        backbone=None,
        use_pos_embed=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        decoder_dropout=0.1,
        hidden_act="gelu",
        cls_attn_layers=2,
        hybrid_patch_size=2,
        channel_dims=None,
        feat_downsample=False,
        out_index=-1,
        rounding_mode="floor",
        segmentation_in_channels=[128, 256, 480, 480],
        decoder_hidden_size=768,
        semantic_loss_ignore_index=-100,
        layer_norm_eps=1e-6,
        **kwargs,
    ):

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.depths = depths
        self.num_attention_heads = num_attention_heads
        self.eta = eta
        self.tokens_norm = tokens_norm
        self.se_mlp = se_mlp
        self.initializer_range = initializer_range
        self.img_size = img_size
        self.num_channels = num_channels
        self.backbone = backbone
        self.use_pos_embed = use_pos_embed
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.decoder_dropout = decoder_dropout
        self.hidden_act = hidden_act
        self.cls_attn_layers = cls_attn_layers
        self.hybrid_patch_size = hybrid_patch_size
        self.channel_dims = channel_dims
        self.out_index = out_index
        self.feat_downsample = feat_downsample
        self.rounding_mode = rounding_mode
        self.segmentation_in_channels = segmentation_in_channels
        self.decoder_hidden_size = decoder_hidden_size
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.layer_norm_eps = layer_norm_eps
        super().__init__(**kwargs)
