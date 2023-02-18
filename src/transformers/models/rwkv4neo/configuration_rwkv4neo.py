# coding=utf-8
# Copyright 2022 arensc and The HuggingFace Inc. team. All rights reserved.
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
""" RWKV4Neo model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

RWKV4NEO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "rwkv-4": "https://huggingface.co/rwkv-4/resolve/main/config.json",
    # See all RWKV4Neo models at https://huggingface.co/models?filter=rwkv4neo
}


class Rwkv4NeoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~Rwkv4NeoModel`].
    It is used to instantiate an RWKV4Neo model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the RWKV4Neo [rwkv-4](https://huggingface.co/rwkv-4) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the RWKV4Neo model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~Rwkv4NeoModel`] or
            [`~TFRwkv4NeoModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
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
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~Rwkv4NeoModel`] or
            [`~TFRwkv4NeoModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import Rwkv4NeoModel, Rwkv4NeoConfig

    >>> # Initializing a RWKV4Neo rwkv-4 style configuration
    >>> configuration = Rwkv4NeoConfig()

    >>> # Initializing a model from the rwkv-4 style configuration
    >>> model = Rwkv4NeoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "rwkv4neo"
    
    # "eos_token_id":0,
    # "pad_token_id":1, 
    # "d_model":5120, 
    # "is_encoder_decoder":false, 
    # "num_decoder_layers":40,
    # "vocab_size": 50277,
    # "n_positions":1024

    # Training parameters
    #return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
    #return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    # Config Params in a nutshell if anyone needs for their own project for RWKV4a-Neo
    # Training
    # ```json 

#     {
#     "use_tiny_attention":False,
#     "use_two_pos_embed":False,
#     "DeepSpeedCPUAdam": {
#         "optim_groups": "",
#         "lr": "self.args.lr_init",
#         "betas": "self.args.betas",
#         "eps": "self.args.adam_eps",
#         "bias_correction": true,
#         "adamw_mode": false,
#         "weight_decay": 0,
#         "amsgrad": false
#     },
#     "FusedAdam": {
#         "optim_groups": "",
#         "lr": "self.args.lr_init",
#         "betas": "self.args.betas",
#         "eps": "self.args.adam_eps",
#         "bias_correction": true,
#         "adam_w_mode": false,
#         "weight_decay": 0,
#         "amsgrad": false
#     }
# }

    # ```
    def __init__(
        self,
        vocab_size=50277,
        hidden_size=768, # 1024 ? 
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=None,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

    