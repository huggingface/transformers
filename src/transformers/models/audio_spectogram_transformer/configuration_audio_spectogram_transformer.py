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
""" AudioSpectogramTransformer model configuration"""


from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

AUDIO_SPECTOGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # TODO update to appropriate organization
    "nielsr/audio-spectogram-transformer-finetuned-audioset-10-10-0.4593": "https://huggingface.co/nielsr/audio-spectogram-transformer-finetuned-audioset-10-10-0.4593/resolve/main/config.json",
}


class AudioSpectogramTransformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AudioSpectogramTransformerModel`]. It is used to
    instantiate an AudioSpectogramTransformer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    AudioSpectogramTransformer
    [nielsr/audio-spectogram-transformer-finetuned-audioset-10-10-0.4593](https://huggingface.co/nielsr/audio-spectogram-transformer-finetuned-audioset-10-10-0.4593)
    architecture.

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
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        patch_size (`int`, *optional*, defaults to `16`):
            The size (resolution) of each patch.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        frequency_stride (`int`, *optional*, defaults to 10):
            Frequency stride to use when patchifying the spectograms.
        time_stride (`int`, *optional*, defaults to 10):
            Temporal stride to use when patchifying the spectograms.
        time_dimension (`int`, *optional*, defaults to 1024):
            Temporal dimension of the spectrograms.
        frequency_dimension (`int`, *optional*, defaults to 128):
            Frequency dimension of the spectrograms.

    Example:

    ```python
    >>> from transformers import AudioSpectogramTransformerModel, AudioSpectogramTransformerConfig

    >>> # Initializing a AudioSpectogramTransformer audio_spectogram_transformer-base-patch16-224 style configuration
    >>> configuration = AudioSpectogramTransformerConfig()

    >>> # Initializing a model from the audio_spectogram_transformer-base-patch16-224 style configuration
    >>> model = AudioSpectogramTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "audio-spectogram-transformer"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        patch_size=16,
        qkv_bias=True,
        frequency_stride=10,
        time_stride=10,
        time_dimension=1024,
        frequency_dimension=128,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.frequency_stride = frequency_stride
        self.time_stride = time_stride
        self.time_dimension = time_dimension
        self.frequency_dimension = frequency_dimension
