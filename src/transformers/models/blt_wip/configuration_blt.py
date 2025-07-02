# coding=utf-8
# Copyright 2024 Facebook Research and The HuggingFace Inc. team. All rights reserved.
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
"""BLT model configuration"""

from enum import Enum
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

class BLTLocalEncoderConfig(PretrainedConfig):
    """
    Configuration class for the BLT Local Encoder component.
    """
    
    model_type = "blt_local_encoder"
    
    def __init__(
          self,
        vocab_size=256,
        cross_attn_all_layers=True,
        cross_attn_k=2,
        hidden_size_global=2048,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=None,
        num_hidden_layers=8,
        norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
        intermediate_size=None,
        _attn_implementation="sdpa",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_k = cross_attn_k
        self.hidden_size_global = hidden_size_global
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {"rope_type": "default"}
        self.hidden_act = hidden_act
        self._attn_implementation = _attn_implementation
        
        super().__init__(**kwargs)
    
class BLTLocalDecoderConfig(PretrainedConfig):
    """
    Configuration class for the BLT Local Decoder component.
    """
    
    model_type = "blt_local_decoder"
    
    def __init__(
        self,
        vocab_size=256,
        cross_attn_all_layers=True,
        cross_attn_k=2,
        hidden_size_global=2048,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=None,
        num_hidden_layers=8,
        norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
        intermediate_size=None,
        _attn_implementation="sdpa",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_k = cross_attn_k
        self.hidden_size_global = hidden_size_global
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {"rope_type": "default"}
        self.hidden_act = hidden_act
        self._attn_implementation = _attn_implementation

        super().__init__(**kwargs)


class BLTGlobalTransformerConfig(PretrainedConfig):
    """
    Configuration class for the BLT Global Transformer component.
    """
    
    model_type = "blt_global_transformer"
    
    def __init__(
        self,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=None,
        num_hidden_layers=8,
        norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
        intermediate_size=None,
        _attn_implementation="sdpa",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {"rope_type": "default"}
        self.hidden_act = hidden_act
        self._attn_implementation = _attn_implementation
        
        super().__init__(**kwargs)


class BLTPatcherConfig(PretrainedConfig):
    r"""
    Configuration class for the BLT Patcher/Entropy model component.
    
    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size for the entropy model used in patching.
        hidden_size (`int`, *optional*, defaults to 512):
            Hidden dimension for the entropy model.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of layers in the entropy model.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the entropy model.
        head_dim (`int`, *optional*):
            Dimension of each attention head in the entropy model.
        num_key_value_heads (`int`, *optional*):
            Number of key-value heads in the entropy model.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum sequence length for the entropy model.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            Layer normalization epsilon for the entropy model.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the entropy model.
        ffn_dim_multiplier (`float`, *optional*):
            Feedforward dimension multiplier for the entropy model.
        multiple_of (`int`, *optional*, defaults to 256):
            Make feedforward dimension multiple of this for the entropy model.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE theta parameter for the entropy model.
        attn_impl (`str`, *optional*, defaults to "sdpa"):
            Attention implementation for the entropy model.
        attn_bias_type (`str`, *optional*, defaults to "causal"):
            Attention bias type for the entropy model.
    """
    
    model_type = "blt_patcher"
    
    def __init__(
        self,
        vocab_size=256,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=None,
        max_position_embeddings=1024,
        norm_eps=1e-5,
        dropout=0.0,
        rope_theta=10000.0,
        attn_impl="sdpa",
        attn_bias_type="causal",
        intermediate_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.attn_impl = attn_impl
        self.attn_bias_type = attn_bias_type
        self.hidden_act = "silu"  # BLT uses silu activation
        self.intermediate_size = intermediate_size or int(8 * self.hidden_size / 3)
        self.rope_scaling = {"rope_type": "default"}
        super().__init__(**kwargs)


class BLTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BLTModel`]. It is used to instantiate a
    BLT model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the BLT model. Defines the number of different tokens (bytes) that can be represented.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model can handle.

        # Patching configuration
        patch_in_forward (`bool`, *optional*, defaults to False):
            Whether to perform patching during forward pass.
        patch_size (`float`, *optional*):
            Size of patches for static patching.
        patching_mode (`str`, *optional*):
            Mode for patching ("entropy", "static", etc.).
        patching_threshold (`float`, *optional*):
            Threshold for entropy-based patching.
        patching_batch_size (`int`, *optional*, defaults to 1):
            Batch size for patching operations.
        patching_device (`str`, *optional*, defaults to "cuda"):
            Device to use for patching operations.
        max_patch_length (`int`, *optional*):
            Maximum length of patches.

        # Cross attention configurations
        cross_attn_k (`int`, *optional*):
            Number of cross attention components.

        # Encoder configurations
        encoder_hash_byte_group_size (`Any`, *optional*):
            Hash byte group size for encoder.
        encoder_hash_byte_group_vocab (`int`, *optional*, defaults to 30000):
            Vocabulary size for hash byte groups.
        encoder_hash_byte_group_nb_functions (`int`, *optional*, defaults to 3):
            Number of hash functions for byte groups.

        # Component configurations
        patcher_config (`Union[BLTPatcherConfig, dict]`, *optional*):
            Configuration for the BLT patcher/entropy model component.
        encoder_config (`Union[BLTLocalEncoderConfig, dict]`, *optional*):
            Configuration for the BLT local encoder component.
        decoder_config (`Union[BLTLocalDecoderConfig, dict]`, *optional*):
            Configuration for the BLT local decoder component.
        global_config (`Union[BLTGlobalTransformerConfig, dict]`, *optional*):
            Configuration for the BLT global transformer component.

    ```python
    >>> from transformers import BLTModel, BLTConfig

    >>> # Initializing a BLT configuration
    >>> configuration = BLTConfig()

    >>> # Initializing a model from the configuration
    >>> model = BLTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blt"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "patcher_config": BLTPatcherConfig, 
        "encoder_config": BLTLocalEncoderConfig, 
        "decoder_config": BLTLocalDecoderConfig, 
        "global_config": BLTGlobalTransformerConfig
    }

    def __init__(
        self,
        vocab_size=256,
        max_position_embeddings=1024,
        patch_in_forward=False,
        patch_size=None,
        patching_mode=None,
        patching_threshold=None,
        patching_batch_size=1,
        max_patch_length=None,
        cross_attn_k=2,
        encoder_hash_byte_group_size=None,
        encoder_hash_byte_group_vocab=30000,
        encoder_hash_byte_group_nb_functions=3,
        patcher_config=None,
        encoder_config=None,
        decoder_config=None,
        global_config=None,
        # Generation configuration
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=-1,
        **kwargs,
    ):
        
        # Basic model configuration
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        # Generation configuration
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.return_dict = True

        # Patching configuration
        self.patch_in_forward = patch_in_forward
        self.patch_size = patch_size
        self.patching_mode = patching_mode
        self.patching_threshold = patching_threshold
        self.patching_batch_size = patching_batch_size
        self.max_patch_length = max_patch_length

        # Cross attention configurations
        self.cross_attn_k = cross_attn_k

        # Encoder configurations
        self.encoder_hash_byte_group_size = encoder_hash_byte_group_size or [2, 3, 4]
        self.encoder_hash_byte_group_vocab = encoder_hash_byte_group_vocab
        self.encoder_hash_byte_group_nb_functions = encoder_hash_byte_group_nb_functions

        # Initialize component configurations
        if patcher_config is None:
            self.patcher_config = BLTPatcherConfig()
            logger.info("patcher_config is None, using default BLT patcher config")
        elif isinstance(patcher_config, dict):
            self.patcher_config = BLTPatcherConfig(**patcher_config)
        elif isinstance(patcher_config, BLTPatcherConfig):
            self.patcher_config = patcher_config

        if encoder_config is None:
            self.encoder_config = BLTLocalEncoderConfig()
            logger.info("encoder_config is None, using default BLT encoder config")
        elif isinstance(encoder_config, dict):
            self.encoder_config = BLTLocalEncoderConfig(**encoder_config)
        elif isinstance(encoder_config, BLTLocalEncoderConfig):
            self.encoder_config = encoder_config

        if decoder_config is None:
            self.decoder_config = BLTLocalDecoderConfig()
            logger.info("decoder_config is None, using default BLT decoder config")
        elif isinstance(decoder_config, dict):
            self.decoder_config = BLTLocalDecoderConfig(**decoder_config)
        elif isinstance(decoder_config, BLTLocalDecoderConfig):
            self.decoder_config = decoder_config

        if global_config is None:
            self.global_config = BLTGlobalTransformerConfig()
            logger.info("global_config is None, using default BLT global config")
        elif isinstance(global_config, dict):
            self.global_config = BLTGlobalTransformerConfig(**global_config)
        elif isinstance(global_config, BLTGlobalTransformerConfig):
            self.global_config = global_config

        super().__init__(**kwargs)

__all__ = [
    "BLTConfig", 
    "BLTPatcherConfig", 
    "BLTLocalEncoderConfig", 
    "BLTLocalDecoderConfig", 
    "BLTGlobalTransformerConfig", 
]
