# coding=utf-8
# Copyright 2024 Facebook AI and The HuggingFace Inc. team. All rights reserved.
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
""" Audio MAE model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


from ..deprecated._archive_maps import VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP  # noqa: F401, E402


class AudioMAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AudioMAEModel`]. It is used to instantiate an ViT
    MAE model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the ViT
    [facebook/audiomae-base](https://huggingface.co/facebook/audiomae-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        decoder_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        mask_ratio (`float`, *optional*, defaults to 0.75):
            The ratio of the number of masked tokens in the input sequence.
        norm_pix_loss (`bool`, *optional*, defaults to `False`):
            Whether or not to train with normalized pixels (see Table 3 in the paper). Using normalized pixels improved
            representation quality in the experiments of the authors.

    Example:

    ```python
    >>> from transformers import AudioMAEConfig, AudioMAEModel

    >>> # Initializing a ViT MAE vit-mae-base style configuration
    >>> configuration = AudioMAEConfig()

    >>> # Initializing a model (with random weights) from the vit-mae-base style configuration
    >>> model = AudioMAEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audio_mae"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        max_length = 1024,
        num_mel_bins = 128,
        patch_size=16,
        num_channels=1,
        mlp_ratio=4,
        contextual_depth=8,
        audio_exp=True,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=16,
        drop_path_rate=0.0,
        mask_ratio=0.8,
        norm_pix_loss=True,
        mask_t_prob=0.6, 
        mask_f_prob=0.5,
        mask_2d=False,
        window_size=(4, 4),
        decoder_input_resolution=(64, 8),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.mlp_ratio = mlp_ratio
        self.contextual_depth = contextual_depth
        self.audio_exp = audio_exp
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob
        self.mask_2d = mask_2d
        self.window_size = window_size
        self.decoder_input_resolution = decoder_input_resolution
        self.hidden_act = hidden_act
