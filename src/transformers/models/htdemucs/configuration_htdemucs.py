# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
""" HtDemucs model configuration"""
import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)

MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/htdemucs": "https://huggingface.co/facebook/htdemucs/resolve/main/config.json",
    # See all Htdemucs models at https://huggingface.co/models?filter=htdemucs
}


class HtdemucsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HtdemucsModel`]. It is used to instantiate a
    HtDemucs model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the HtDemucs
    [facebook/htdemucs](https://huggingface.co/facebook/htdemucs/resolve/main/config.json) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-4):
            Initialization values for the post-attention block scaling weights.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(hidden_size).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        num_stems (`int`, *optional*, defaults to 8):
            The number of stems the audio is split into.
    """
    model_type = "htdemucs"

    def __init__(
        self,
        max_position_embeddings=2048,
        num_hidden_layers=6,
        ffn_dim=2048,
        num_attention_heads=8,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=512,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_factor=0.02,
        layer_scale_init_value=1e-4,
        scale_embedding=False,
        num_stems=8,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.initializer_factor = initializer_factor
        self.layer_scale_init_value = layer_scale_init_value
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.num_stems = num_stems

        head_dim = self.hidden_size // self.num_attention_heads
        if (head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"`hidden_size` must be divisible by `num_attention_heads`. Got `hidden_size`: {self.hidden_size}"
                f" and `num_attention_heads`: {self.num_attention_heads}."
            )

        super().__init__(
            **kwargs,
        )