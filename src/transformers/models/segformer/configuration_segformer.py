# coding=utf-8
# Copyright 2021 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
""" SegFormer model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nvidia/segformer-b0-finetuned-ade-512-512": "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/config.json",
    # See all SegFormer models at https://huggingface.co/models?filter=segformer
}


class SegformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.SegformerModel`. It is used
    to instantiate an SegFormer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the SegFormer
    `nvidia/segformer-b0-finetuned-ade-512-512 <https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512>`__
    architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        image_size (:obj:`int`, `optional`, defaults to 512):
            The size (resolution) of each image.
        num_channels (:obj:`int`, `optional`, defaults to 3):
            The number of input channels.
        num_encoder_blocks (:obj:`int`, `optional`, defaults to 4):
            The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
        depths (:obj:`List[int]`, `optional`, defaults to [2, 2, 2, 2]):
            The number of layers in each encoder block.
        sr_ratios (:obj:`List[int]`, `optional`, defaults to [8, 4, 2, 1]):
            Sequence reduction ratios in each encoder block.
        hidden_sizes (:obj:`List[int]`, `optional`, defaults to [32, 64, 160, 256]):
            Dimension of each of the encoder blocks.
        downsampling_rates (:obj:`List[int]`, `optional`, defaults to [1, 4, 8, 16]):
            Downsample rate of the image resolution compared to the original image size before each encoder block.
        patch_sizes (:obj:`List[int]`, `optional`, defaults to [7, 3, 3, 3]):
            Patch size before each encoder block.
        strides (:obj:`List[int]`, `optional`, defaults to [4, 2, 2, 2]):
            Stride before each encoder block.
        num_attention_heads (:obj:`List[int]`, `optional`, defaults to [1, 2, 4, 8]):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratios (:obj:`List[int]`, `optional`, defaults to [4, 4, 4, 4]):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        classifier_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability before the classification head.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        drop_path_rate (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        decoder_hidden_size (:obj:`int`, `optional`, defaults to 256):
            The dimension of the all-MLP decode head.
        reshape_last_stage (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to reshape the features of the last stage back to :obj:`(batch_size, num_channels, height, width)`.
            Only required for the semantic segmentation model.
        semantic_loss_ignore_index (:obj:`int`, `optional`, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    Example::

        >>> from transformers import SegformerModel, SegformerConfig

        >>> # Initializing a SegFormer nvidia/segformer-b0-finetuned-ade-512-512 style configuration
        >>> configuration = SegformerConfig()

        >>> # Initializing a model from the nvidia/segformer-b0-finetuned-ade-512-512 style configuration
        >>> model = SegformerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "segformer"

    def __init__(
        self,
        image_size=224,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        downsampling_rates=[1, 4, 8, 16],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256,
        is_encoder_decoder=False,
        reshape_last_stage=True,
        semantic_loss_ignore_index=255,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.downsampling_rates = downsampling_rates
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.reshape_last_stage = reshape_last_stage
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
