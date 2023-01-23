# coding=utf-8
# Copyright 2022 MURGe-Lab and The HuggingFace Inc. team. All rights reserved.
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
""" TVLT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "TVLT/tvlt-base": "https://huggingface.co/TVLT/tvlt-base/blob/main/config.json",
}


class TvltConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TvltModel`]. It is used to instantiate a TVLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TVLT
    [TVLT/tvlt-base](https://huggingface.co/TVLT/tvlt-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        audio_size (`int`, *optional*, defaults to 2048):
            The time length of each audio spectrogram.
        feature_size (`int`, *optional*, defaults to 128):
            The frequency length of audio spectrogram.
        image_patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each image patch.
        audio_patch_size (`List[int]`, *optional*, defaults to `[16, 16]`):
            The size (resolution) of each audio patch.
        num_image_channels (`int`, *optional*, defaults to 3):
            The number of input image channels.
        num_audio_channels (`int`, *optional*, defaults to 1):
            The number of input audio channels.
        num_frames (`int`, *optional*, defaults to 16):
            The maximum number of frames for an input video.
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
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to mean pool the final hidden states instead of using the final hidden state of the [CLS] token.
        decoder_num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 384):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        pixel_mask_ratio (`float`, *optional*, defaults to 0.75):
            Pixel patch masking ratio.
        audio_mask_ratio (`float`, *optional*, defaults to 0.15):
            Audio patch masking ratio.
        task_matching (`bool`, *optional*, defaults to `True`):
            Whether to use vision audio matching task in pretraining.
        task_mae (`bool`, *optional*, defaults to `True`):
            Whether to use mae task in pretraining.
        loss_type (`str`, *optional*, defaults to `classification`):
            Loss types including regression and classification.

    Example:

    ```python
    >>> from transformers import TvltConfig, TvltModel

    >>> # # Initializing a TVLT TVLT/tvlt-base style configuration
    >>> configuration = TvltConfig()

    >>> # # Initializing a model (with random weights) from the TVLT/tvlt-base style configuration
    >>> model = TvltModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "tvlt"

    def __init__(
        self,
        pixel_size=224,
        audio_size=2048,
        feature_size=128,
        pixel_patch_size=16,
        audio_patch_size=[16, 16],
        num_pixel_channels=3,
        num_audio_channels=1,
        num_frames=8,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        use_mean_pooling=False,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        pixel_mask_ratio=0.75,
        audio_mask_ratio=0.15,
        task_matching=True,
        task_mae=True,
        loss_type="classification",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.pixel_size = pixel_size
        self.audio_size = audio_size
        self.feature_size = feature_size
        self.pixel_patch_size = pixel_patch_size
        self.audio_patch_size = audio_patch_size
        self.num_pixel_channels = num_pixel_channels
        self.num_audio_channels = num_audio_channels
        self.num_frames = num_frames

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_mean_pooling = use_mean_pooling

        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.pixel_mask_ratio = pixel_mask_ratio
        self.audio_mask_ratio = audio_mask_ratio

        self.task_matching = task_matching
        self.task_mae = task_mae
        self.loss_type = loss_type
