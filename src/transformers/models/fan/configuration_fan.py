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

original_feature_mapping = {
    "num_heads": "num_attention_heads",
    "dropout_ratio": "decoder_dropout",
    "depth": "num_hidden_layers",
    "in_channels": "segmentation_in_channels",
    "in_chans": "num_channels",
    "num_classes": "num_labels",
    "embed_dim": "hidden_size",
}
# TODO: Rename embed_dim to hidden_size
# DONE: Rename num_classes to num_labels
# TODO: FANConfig Attributes rewrite
# TODO: FANConfig features rename
# ISSUE: Move configuration to nvidia/fan
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
        patch_size (int, defaults to 16):
            Size of each patch to generated the embedding tokens from the image
        hidden_size (`int`, *optional*, defaults to 384):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        se_mlp (`bool`, defaults to False):
            Wheter or not to use Squeeze-Excite in the FANEncoder layers MLP.
        mlp_ratio (`float`, defaults to 4.0):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        qkv_bias (`bool`, defaults to True):
            Whether or not to use bias in Query, Key and Value in attention layers.
        depths (`List[int]`, *optional*, defaults to None):
            The number of layers in each encoder block, only applicable when using hybrid backbone (ConvNeXt).
        eta (`float` defatults to 1.0):
            Weight Initialization value for channel importance.
        use_pos_embed ( `bool`, defaults to True):
            Wheter or not to use positional_encoding in the embeddings.
        img_size (`List[int]`, defaults to (224,224)):
            The size of the images being passed to the model.
        num_channels (`int`, defaults to 3):
            Number of channels the input image has.
        num_labels (`int`, defaults to 1000):
            Number of classes used to predict, used for ImageClassification and  SemanticSegmentation tasks.
        backbone (`string`, *optional*, defaults to None):
            Wheter or not to use 'hybrid' backbone.
        segmentation_in_channels (tuple(int), defaults to (128, 256, 480, 480)):
            Number of channels in each of the hidden features used for Semantic Segmentation.
        drop_rate (`float`, defaults to 0):
            Dropout used in MLP.
        attn_drop_rate (`float`, defaults to 0):
            Dropout used in Toxen Mixing after attention.
        drop_path_rate (`float`, defaults to 0):
            Dropout used in Toxen Mixing after MLP.
        decoder_dropout (`float`, defaults to 0.1):
            Dropout used in Decode Head for SemanticSegmentation tasks.
        tokens_norm (`bool`, defaults to True):
            Whether or not to apply normalization in the Class Attention block.
        feat_downsample (`bool`, defaults to ):
            Whether or not to use a learnable downsample convolution to obtain hidden states for SemanticSegmentation tasks.
            Only appliable with hybrid backbone.
        channel_dims (`tuple(int)`, *optional*, defaults to None):
            List of Input channels for each of the encoder layers.
            If None it defaults to [config.hidden_size] * config.num_hidden_layers.
        rounding_mode (`string`, *optional*, defaults to 'floor'):
            Torch Divison rounding mode used for positional encoding.
            Should be set to None in Semantic Segmentation tasks to be compatible with original paper implementation.
        semantic_loss_ignore_index (`int`, *optional*, defaults to -100):
            The index that is ignored by the loss function of the semantic segmentation model.
        cls_attn_layers (`int`, defaults to 2):
            Number of ClassAttentionBlock used.
            Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239.
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
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
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
        patch_size=16,  # HASCOMMENTS
        hidden_size=384,  # HASCOMMENTS
        num_hidden_layers=12,  # HASCOMMENTS
        num_attention_heads=8,  # HASCOMMENTS
        depths=None,  # HASCOMMENTS
        eta=1.0,  # HASCOMMENTS
        tokens_norm=True,  # HASCOMMENTS
        se_mlp=False,  # HASCOMMENTS
        initializer_range=1.0,  # HASCOMMENTS
        img_size=[224, 224],  # HASCOMMENTS
        num_channels=3,  # HASCOMMENTS
        # num_labels=1000,  # HASCOMMENTS
        backbone=None,  # HASCOMMENTS
        use_checkpoint=False,  # TODO: Rename for HF Consistency
        use_pos_embed=True,  # HASCOMMENTS
        mlp_ratio=4.0,  # HASCOMMENTS
        qkv_bias=True,  # HASCOMMENTS
        drop_rate=0.0,  # HASCOMMENTS
        attn_drop_rate=0.0,  # HASCOMMENTS
        drop_path_rate=0.0,  # HASCOMMENTS
        decoder_dropout=0.1,  # HASCOMMENTS
        act_layer=None,  # TODO: Add Documentation Refactor modeling to include ACT2CLS/ACT2FN from activations
        norm_layer=None,
        cls_attn_layers=2,  # HASCOMMENTS
        hybrid_patch_size=2,
        channel_dims=None,  # HASCOMMENTS
        feat_downsample=False,  # HASCOMMENTS
        out_index=-1,
        rounding_mode="floor",  # HASCOMMENTS
        segmentation_in_channels=[128, 256, 480, 480],  # HASCOMMENTS
        feature_strides=[4, 8, 16, 32],
        channels=256,
        decoder_hidden_size=768,
        reshape_last_stage=False,
        semantic_loss_ignore_index=-100,  # HASCOMMENTS
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
        # self.num_labels = num_labels
        self.backbone = backbone
        self.use_checkpoint = use_checkpoint
        self.use_pos_embed = use_pos_embed
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        # TODO: Clean Different Dropout Rates
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.decoder_dropout = decoder_dropout
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.cls_attn_layers = cls_attn_layers
        self.hybrid_patch_size = hybrid_patch_size
        self.channel_dims = channel_dims
        self.out_index = out_index
        self.feat_downsample = feat_downsample
        self.rounding_mode = rounding_mode
        self.segmentation_in_channels = segmentation_in_channels
        self.feature_strides = feature_strides
        self.channels = channels

        self.decoder_hidden_size = decoder_hidden_size
        self.reshape_last_stage = reshape_last_stage
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        super().__init__(**kwargs)
