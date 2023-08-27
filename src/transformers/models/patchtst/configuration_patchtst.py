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
"""PatchTST model configuration"""

from typing import List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/patchtst-base": "https://huggingface.co/ibm/patchtst-base/resolve/main/config.json",
}


class PatchTSTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`PatchTSTModel`]. It is used to instantiate an
    PatchTST model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        prediction_length (`int`):
            The prediction length for the decoder. In other words, the prediction horizon of the model. This value is
            typically dictated by the dataset and we recommend to set it appropriately.
        context_length (`int`, *optional*, defaults to `prediction_length`):
            The context length for the encoder. If `None`, the context length will be the same as the
            `prediction_length`.
        input_size (`int`, *optional*, defaults to 1):
            The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
            multivariate targets.
        scaling (`string` or `bool`, *optional* defaults to `"mean"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        num_time_features (`int`, *optional*, defaults to 0):
            The number of time features in the input time series.
        num_dynamic_real_features (`int`, *optional*, defaults to 0):
            The number of dynamic real valued features.
        num_static_categorical_features (`int`, *optional*, defaults to 0):
            The number of static categorical features.
        num_static_real_features (`int`, *optional*, defaults to 0):
            The number of static real valued features.
        embedding_dimension (`list[int]`, *optional*):
            The dimension of the embedding for each of the static categorical features. Should be a list of integers,
            having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        d_model (`int`, *optional*, defaults to 64):
            Dimensionality of the transformer layers.
        encoder_layers (`int`, *optional*, defaults to 2):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 2):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and decoder. If string, `"gelu"` and
            `"relu"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder, and decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each encoder layer.
        decoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each decoder layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability used between the two layers of the feed-forward networks.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for each time step of inference.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.
        attention_type (`str`, *optional*, defaults to "prob"):
            Attention used in encoder. This can be set to "prob" (PatchTST's ProbAttention) or "full" (vanilla
            transformer's canonical self-attention).
        sampling_factor (`int`, *optional*, defaults to 5):
            ProbSparse sampling factor (only makes affect when `attention_type`="prob"). It is used to control the
            reduced query matrix (Q_reduce) input length.
        distil (`bool`, *optional*, defaults to `True`):
            Whether to use distilling in encoder.

    Example:

    ```python
    >>> from transformers import PatchTSTConfig, PatchTSTModel

    >>> # Initializing an PatchTST configuration with 12 time steps for prediction
    >>> configuration = PatchTSTConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "patchtst"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    def __init__(
        self,
        input_size: int = 1,
        context_length: int = 32,
        patch_length: int = 8,
        stride: int = 8,
        encoder_layers: int = 3,
        d_model: int = 128,
        encoder_attention_heads: int = 16,
        shared_embedding: bool = True,
        channel_attention: bool = False,
        encoder_ffn_dim: int = 256,
        norm: str = "BatchNorm",
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        positional_dropout: float = 0.0,
        dropout_path: float = 0.0,
        ff_dropout: float = 0.0,
        bias: bool = True,
        activation_function: str = "gelu",
        pre_norm: bool = False,
        store_attn: bool = False,
        positional_encoding: str = "sincos",
        learn_pe: bool = False,
        use_cls_token: bool = False,
        patch_last: bool = True,
        individual: bool = False,
        seed_number= None,
        mask_input: Optional[bool] = None,
        mask_type: str = "random",
        mask_ratio=0.5,
        mask_patches: list = [2, 3],
        mask_patch_ratios: list = [1, 1],
        channel_consistent_masking: bool = True,
        d_size: str = "4D",
        unmasked_channel_indices: list = None,
        mask_value=0,
        pooling: str = 'mean',
        num_classes: int = 1,
        head_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
        num_dynamic_real_features: int = 0,
        num_static_real_features: int = 0,
        num_static_categorical_features: int = 0,
        num_time_features: int = 0,
        is_encoder_decoder: bool = False,
        encoder_layerdrop: float = 0.1,
        prediction_length: int = 24,

        # PatchTST arguments
        attention_type: str = "prob",
        sampling_factor: int = 5,
        distil: bool = True,
        **kwargs,
    ):

        # time series specific configuration
        self.context_length = context_length
        self.input_size = input_size # n_vars
        self.num_time_features = num_time_features
        self.num_dynamic_real_features = num_dynamic_real_features
        self.num_static_real_features = num_static_real_features
        self.num_static_categorical_features = num_static_categorical_features

        # Transformer architecture configuration
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.shared_embedding = shared_embedding
        self.channel_attention = channel_attention
        self.norm = norm
        self.positional_dropout = positional_dropout
        self.dropout_path = dropout_path
        self.ff_dropout = ff_dropout
        self.bias = bias
        self.activation_function = activation_function
        self.pre_norm = pre_norm
        self.store_attention = store_attn
        self.positional_encoding = positional_encoding
        self.learn_pe = learn_pe
        self.use_cls_token = use_cls_token
        self.patch_last = patch_last
        self.individual = individual

        # PatchTST
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = self._num_patches()
        self.attention_type = attention_type
        self.sampling_factor = sampling_factor
        self.distil = distil

        # Masking
        self.seed_number = seed_number
        self.mask_input = mask_input
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.mask_patches = mask_patches
        self.mask_patch_ratios = mask_patch_ratios
        self.channel_consistent_masking = channel_consistent_masking
        self.d_size = d_size
        self.unmasked_channel_indices = unmasked_channel_indices
        self.mask_value = mask_value

        # Classification
        self.pooling = pooling
        self.num_classes = num_classes
        self.head_dropout = head_dropout
        self.proj_dropout = proj_dropout
        self.qkv_bias = qkv_bias

        # Forcasting
        self.prediction_length = prediction_length

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    def _num_patches(self):
        return (max(self.context_length, self.patch_length) - self.patch_length) // self.stride + 1

