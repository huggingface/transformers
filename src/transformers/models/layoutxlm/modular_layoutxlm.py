# coding=utf-8
# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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

from ..layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config


class LayoutXLMConfig(LayoutLMv2Config):
    r"""
    This is the configuration class to store the configuration of a [`LayoutXLMModel`]. It is used to instantiate an
    LayoutXLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutXLM
    [microsoft/layoutxlm-base](https://huggingface.co/microsoft/layoutxlm-base) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the LayoutXLM model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`LayoutXLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`LayoutXLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the 2D position embedding might ever be used with. Typically set this to something
            large just in case (e.g., 1024).
        max_rel_pos (`int`, *optional*, defaults to 128):
            The maximum number of relative positions to be used in the self-attention mechanism.
        rel_pos_bins (`int`, *optional*, defaults to 32):
            The number of relative position bins to be used in the self-attention mechanism.
        fast_qkv (`bool`, *optional*, defaults to `True`):
            Whether or not to use a single matrix for the queries, keys, values in the self-attention layers.
        max_rel_2d_pos (`int`, *optional*, defaults to 256):
            The maximum number of relative 2D positions in the self-attention mechanism.
        rel_2d_pos_bins (`int`, *optional*, defaults to 64):
            The number of 2D relative position bins in the self-attention mechanism.
        convert_sync_batchnorm (`bool`, *optional*, defaults to `True`):
            Whether or not to convert batch normalization layers to synchronized batch normalization layers.
        image_feature_pool_shape (`list[int]`, *optional*, defaults to `[7, 7, 256]`):
            The shape of the average-pooled feature map.
        coordinate_size (`int`, *optional*, defaults to 128):
            Dimension of the coordinate embeddings.
        shape_size (`int`, *optional*, defaults to 128):
            Dimension of the width and height embeddings.
        has_relative_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a relative attention bias in the self-attention mechanism.
        has_spatial_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a spatial attention bias in the self-attention mechanism.
        has_visual_segment_embedding (`bool`, *optional*, defaults to `False`):
            Whether or not to add visual segment embeddings.
        detectron2_config_args (`dict`, *optional*):
            Dictionary containing the configuration arguments of the Detectron2 visual backbone. Refer to [this
            file](https://github.com/microsoft/unilm/blob/master/layoutlmft/layoutlmft/models/layoutxlm/detectron2_config.py)
            for details regarding default values.


    Example:

    ```python
    >>> from transformers import LayoutXLMConfig, LayoutXLMModel

    >>> # Initializing a LayoutXLM microsoft/layoutxlm-base style configuration
    >>> configuration = LayoutXLMConfig()

    >>> # Initializing a model (with random weights) from the microsoft/layoutxlm-base style configuration
    >>> model = LayoutXLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    pass


__all__ = ["LayoutXLMConfig"]
