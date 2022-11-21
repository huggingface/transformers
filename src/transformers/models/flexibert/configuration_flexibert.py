# coding=utf-8
# Copyright 2022 Shikhar Tuli. All rights reserved.
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
""" FlexiBERT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

FLEXIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "flexibert-mini": "https://huggingface.co/flexibert-mini/resolve/main/config.json",
    # See all FlexiBERT models at https://huggingface.co/models?filter=flexibert
}


class FlexiBERTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~FlexiBERTModel`]. It is used to instantiate an
    FlexiBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FlexiBERT
    [flexibert-mini](https://huggingface.co/flexibert-mini) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the FlexiBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~FlexiBERTModel`]
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
            The vocabulary size of the `token_type_ids` passed when calling [`~FlexiBERTModel`]
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import FlexiBERTModel, FlexiBERTConfig

    >>> # Initializing a FlexiBERT flexibert-mini style configuration
    >>> configuration = FlexiBERTConfig()

    >>> # Initializing a model from the flexibert-mini style configuration
    >>> model = FlexiBERTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "flexibert"

    def __init__(
        self,
        vocab_size=30522,
        num_hidden_layers=4,
        hidden_size=768,
        num_attention_heads=12,
        is_encoder_decoder=False,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,  # As per RoBERTa's config (old - 512)
        type_vocab_size=1,  # As per RoBERTa's config (old - 2)
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=2,
        head_ratio=2,
        conv_kernel_size=9,
        num_groups=1,
        gradient_checkpointing=False,
        position_embedding_type="relative_key",
        use_cache=True,
        attention_type=None,
        hidden_dim_list=None,
        attention_heads_list=None,
        ff_dim_list=None,
        similarity_list=None,
        from_model_dict_hetero=False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            is_encoder_decoder=is_encoder_decoder,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.head_ratio = head_ratio
        self.conv_kernel_size = conv_kernel_size
        self.num_groups = num_groups

        # Set default values for BERT-like model
        if attention_type is not None:
            self.attention_type = attention_type
        else:
            self.attention_type = ["sa"] * self.num_hidden_layers
        if hidden_dim_list is not None:
            self.hidden_dim_list = hidden_dim_list
        else:
            self.hidden_dim_list = [hidden_size] * self.num_hidden_layers
        if attention_heads_list is not None:
            self.attention_heads_list = attention_heads_list
        else:
            self.attention_heads_list = [num_attention_heads] * self.num_hidden_layers
        if ff_dim_list is not None:
            self.ff_dim_list = ff_dim_list
        else:
            self.ff_dim_list = [
                [intermediate_size],
            ] * self.num_hidden_layers
        if similarity_list is not None:
            self.similarity_list = similarity_list
        else:
            self.similarity_list = ["sdp"] * self.num_hidden_layers

        self.from_model_dict_hetero = from_model_dict_hetero

    def from_model_dict(self, model_dict):
        self.num_hidden_layers = model_dict["l"]
        self.attention_type = model_dict["o"]  # options = 'l','sa','c'
        self.hidden_dim_list = model_dict["h"]
        self.attention_heads_list = model_dict["n"]
        self.ff_dim_list = model_dict["f"]
        self.similarity_list = model_dict[
            "p"
        ]  # options = if 'sa'-->'sdp'/'wma' , elif 'l'-->'dft'/'dct', elif 'c' --> 5,9,13

        self.from_model_dict_hetero = False

    def from_model_dict_hetero(self, model_dict):
        self.num_hidden_layers = model_dict["l"]
        self.attention_heads_list = model_dict[
            "o"
        ]  # options = 'l_dft_{attention_head_size}', 'l_dct_{}', 'sa_sdp_{}', 'sa_wma_{}', 'c_5_{}', 'c_9_{}', 'c_13_{}'
        self.hidden_dim_list = model_dict["h"]
        self.ff_dim_list = model_dict["f"]

        self.from_model_dict_hetero = True
