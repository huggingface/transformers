# coding=utf-8
# Copyright 2022 Chan Woo Kim and The HuggingFace Inc. team. All rights reserved.
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
""" mCTC model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MCTC_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mctc-large": "https://huggingface.co/mctc-large/resolve/main/config.json",
    # See all mCTC models at https://huggingface.co/models?filter=mctc
}


class MCTCConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~MCTCModel`]. It is used to instantiate an mCTC
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the mCTC [mctc-large](https://huggingface.co/mctc-large)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 8065):
            Vocabulary size of the mCTC model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~MCTCModel`] or [`~TFMCTCModel`].
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
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~MCTCModel`] or [`~TFMCTCModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import MCTCModel, MCTCConfig

    >>> # Initializing a mCTC mctc-large style configuration
    >>> configuration = MCTCConfig()

    >>> # Initializing a model from the mctc-large style configuration
    >>> model = MCTCModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mctc"

    def __init__(
        self,
        vocab_size=8065,
        hidden_size=1536,
        num_hidden_layers=36,
        intermediate_size=6144,
        num_attention_heads=4,
        attention_head_dim=384,
        position_embedding_type="relative_key",
        max_position_embeddings=920,
        layer_norm_eps=1e-5,
        layerdrop=0.3,
        hidden_act="relu",
        initializer_range=0.02,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
        use_cache=True,
        is_encoder_decoder=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        conv_glu_dim=1,
        conv_dropout=0.3,
        num_conv_layers=1,
        conv_kernel=[7],
        conv_stride=[3],
        input_feat_per_channel=80,
        input_channels=1,
        conv_channels=None,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.layerdrop = layerdrop
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_cache = use_cache
        self.is_encoder_decoder = is_encoder_decoder
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.conv_glu_dim = conv_glu_dim
        self.conv_dropout = conv_dropout
        self.num_conv_layers = num_conv_layers
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        if len(self.conv_kernel) != self.num_conv_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.conv_kernel)` == `config.num_conv_layers` "
                f"but is `len(config.conv_kernel) = {len(self.conv_kernel)}`, "
                f"`config.num_conv_layers = {self.num_conv_layers}`."
            )
