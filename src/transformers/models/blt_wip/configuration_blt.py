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

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class InitStdFactor(str, Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)


class PatchingModeEnum(str, Enum):
    entropy = "entropy"
    bpe = "bpe"
    bpe_patcher = "bpe_patcher"
    space = "space"
    static = "static"
    byte = "byte"


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
        pm_size=0,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=None,
        head_dim=None,
        intermediate_size=None,
        num_hidden_layers=8,
        norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
        multiple_of=256,
        _attn_implementation="sdpa",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_k = cross_attn_k
        self.hidden_size_global = hidden_size_global
        self.pm_size = pm_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.intermediate_size = intermediate_size or multiple_of * ((int(8 * hidden_size / 3) + multiple_of - 1) // multiple_of)
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {"rope_type": "default"}
        self.hidden_act = hidden_act
        self.multiple_of = multiple_of
        self._attn_implementation = _attn_implementation
        self.decoder_dim_token_emb = 1024
        self.encoder_dim_token_emb = 1024
        self.encoder_dim_patch_emb = self.hidden_size
        
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
        head_dim=None,
        intermediate_size=None,
        num_hidden_layers=8,
        norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
        multiple_of=256,
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
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.intermediate_size = intermediate_size or multiple_of * ((int(8 * hidden_size / 3) + multiple_of - 1) // multiple_of)
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {"rope_type": "default"}
        self.hidden_act = hidden_act
        self.multiple_of = multiple_of
        self._attn_implementation = _attn_implementation
        self.decoder_dim_token_emb = 1024
        self.encoder_dim_token_emb = 1024

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
        head_dim=None,
        intermediate_size=None,
        num_hidden_layers=8,
        norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rope_scaling=None,
        hidden_act="silu",
        multiple_of=256,
        _attn_implementation="sdpa",
        global_dim_patch_emb=None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.intermediate_size = intermediate_size or multiple_of * ((int(8 * hidden_size / 3) + multiple_of - 1) // multiple_of)
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {"rope_type": "default"}
        self.hidden_act = hidden_act
        self.multiple_of = multiple_of
        self._attn_implementation = _attn_implementation
        self.global_dim_patch_emb = global_dim_patch_emb
        
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
        head_dim=None,
        num_key_value_heads=None,
        max_position_embeddings=1024,
        norm_eps=1e-5,
        dropout=0.0,
        ffn_dim_multiplier=None,
        multiple_of=256,
        rope_theta=10000.0,
        rope_use_fp32_in_outer_product=False,
        attn_impl="sdpa",
        attn_bias_type="causal",
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else (hidden_size // num_attention_heads)
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.intermediate_size = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.rope_theta = rope_theta
        self.attn_impl = attn_impl
        self.attn_bias_type = attn_bias_type
        
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        
        # Add attributes needed for compatibility with transformer models
        self.hidden_act = "silu"  # BLT uses silu activation
        
        # Calculate intermediate_size using BLTMLP logic based on actual hidden_size
        self.intermediate_size = multiple_of * ((int(8 * self.hidden_size / 3) + multiple_of - 1) // multiple_of)
        
        # Set simple rope scaling for patcher (no complex dynamic rope)
        self.rope_scaling = {"rope_type": "default"}


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

        # Main architecture dimensions
        hidden_size (`int`, *optional*, defaults to 512):
            Main dimension of the model.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of layers in the main transformer.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the main transformer.
        head_dim (`int`, *optional*):
            Dimension of each attention head. If not specified, computed as hidden_size // num_attention_heads.
        num_key_value_heads (`int`, *optional*):
            Number of key-value heads for grouped query attention. If not specified, defaults to num_attention_heads.

        # Component-specific dimensions
        hidden_size_global (`int`, *optional*, defaults to 512):
            Dimension of the global transformer component.
        hidden_size_local_decoder (`int`, *optional*, defaults to 512):
            Dimension of the local decoder component.
        hidden_size_local_encoder (`int`, *optional*, defaults to 512):
            Dimension of the local encoder component.
        num_hidden_layers_global (`int`, *optional*, defaults to 8):
            Number of layers in the global transformer.
        num_hidden_layers_local_decoder (`int`, *optional*, defaults to 8):
            Number of layers in the local decoder.
        num_hidden_layers_local_encoder (`int`, *optional*, defaults to 8):
            Number of layers in the local encoder.
        num_attention_heads_global (`int`, *optional*, defaults to 8):
            Number of attention heads in the global transformer.
        num_attention_heads_local_decoder (`int`, *optional*, defaults to 8):
            Number of attention heads in the local decoder.
        num_attention_heads_local_encoder (`int`, *optional*, defaults to 8):
            Number of attention heads in the local encoder.
        num_key_value_heads_global (`int`, *optional*):
            Number of key-value heads in the global transformer.

        # Transformer configuration
        norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers.
        ffn_dim_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier for the feedforward network dimension.
        multiple_of (`int`, *optional*, defaults to 256):
            Make feedforward network dimension multiple of this value.

        # Positional encoding
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.

        # Attention configuration
        attn_impl (`str`, *optional*, defaults to "sdpa"):
            Attention implementation to use ("sdpa" or "flex_attention").

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
        cross_attn_all_layers_decoder (`bool`, *optional*, defaults to False):
            Whether to apply cross attention to all decoder layers.
        cross_attn_all_layers_encoder (`bool`, *optional*, defaults to False):
            Whether to apply cross attention to all encoder layers.

        # Encoder configurations
        max_encoder_seq_length (`int`, *optional*):
            Maximum sequence length for encoder.
        encoder_hash_byte_group_size (`Any`, *optional*):
            Hash byte group size for encoder.
        encoder_hash_byte_group_vocab (`int`, *optional*, defaults to 30000):
            Vocabulary size for hash byte groups.
        encoder_hash_byte_group_nb_functions (`int`, *optional*, defaults to 3):
            Number of hash functions for byte groups.

        # Parameter mixing
        pm_size (`int`, *optional*, defaults to 0):
            Parameter mixing size.

        # Special tokens
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        pad_token_id (`int`, *optional*, defaults to -1):
            The id of the padding token.

        # Patcher configuration
        patcher_args (`dict`, *optional*):
            Dictionary containing configuration arguments for the BLT patcher/entropy model.
            If provided, these will be used to initialize a BLTPatcherConfig instance.

    ```python
    >>> from transformers import ByteLatentTransformer, BLTConfig

    >>> # Initializing a BLT configuration
    >>> configuration = BLTConfig()

    >>> # Initializing a model from the configuration
    >>> model = ByteLatentTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256,
        max_position_embeddings=1024,
        # Main architecture dimensions
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        head_dim=None,
        num_key_value_heads=None,
        # Component-specific dimensions
        hidden_size_global=512,
        hidden_size_local_decoder=512,
        hidden_size_local_encoder=512,
        num_hidden_layers_global=8,
        num_hidden_layers_local_decoder=8,
        num_hidden_layers_local_encoder=8,
        num_attention_heads_global=8,
        num_attention_heads_local_decoder=8,
        num_attention_heads_local_encoder=8,
        num_key_value_heads_global=None,
        # Transformer configuration
        norm_eps=1e-5,
        dropout=0.0,
        ffn_dim_multiplier=1.0,
        multiple_of=256,
        hidden_act="silu",
        # Positional encoding
        rope_theta=10000.0,
        # Attention configuration
        attn_impl="sdpa",
        _attn_implementation="sdpa",
        # Patching configuration
        patch_in_forward=False,
        patch_size=None,
        patching_mode=None,
        patching_threshold=None,
        patching_batch_size=1,
        patching_device="cuda",
        max_patch_length=None,
        # Cross attention configurations
        cross_attn_k=2,
        cross_attn_all_layers_decoder=False,
        cross_attn_all_layers_encoder=False,
        # Encoder configurations
        max_encoder_seq_length=None,
        encoder_hash_byte_group_size=None,
        encoder_hash_byte_group_vocab=30000,
        encoder_hash_byte_group_nb_functions=3,
        # Parameter mixing
        pm_size=0,
        # Patcher configuration
        patcher_args={},
        # Inherited
        **kwargs,
    ):
        
        # Basic model configuration
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        # Main architecture dimensions
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else (hidden_size // num_attention_heads)
        self.num_key_value_heads = num_key_value_heads

        # Component-specific dimensions
        self.hidden_size_global = hidden_size_global
        self.hidden_size_local_decoder = hidden_size_local_decoder
        self.hidden_size_local_encoder = hidden_size_local_encoder
        self.num_hidden_layers_global = num_hidden_layers_global
        self.num_hidden_layers_local_decoder = num_hidden_layers_local_decoder
        self.num_hidden_layers_local_encoder = num_hidden_layers_local_encoder
        self.num_attention_heads_global = num_attention_heads_global
        self.num_attention_heads_local_decoder = num_attention_heads_local_decoder
        self.num_attention_heads_local_encoder = num_attention_heads_local_encoder
        self.num_key_value_heads_global = num_key_value_heads_global

        # Transformer configuration
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.intermediate_size = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.hidden_act = hidden_act

        # Positional encoding
        self.rope_theta = rope_theta

        # Attention configuration
        self.attn_impl = attn_impl
        self._attn_implementation = _attn_implementation

        # Patching configuration
        self.patch_in_forward = patch_in_forward
        self.patch_size = patch_size
        self.patching_mode = patching_mode
        self.patching_threshold = patching_threshold
        self.patching_batch_size = patching_batch_size
        self.patching_device = patching_device
        self.max_patch_length = max_patch_length

        # Cross attention configurations
        self.cross_attn_k = cross_attn_k
        self.cross_attn_all_layers_decoder = cross_attn_all_layers_decoder
        self.cross_attn_all_layers_encoder = cross_attn_all_layers_encoder

        # Encoder configurations
        self.max_encoder_seq_length = max_encoder_seq_length
        self.encoder_hash_byte_group_size = encoder_hash_byte_group_size
        self.encoder_hash_byte_group_vocab = encoder_hash_byte_group_vocab
        self.encoder_hash_byte_group_nb_functions = encoder_hash_byte_group_nb_functions

        # Parameter mixing
        self.pm_size = pm_size
    
        # Initialize component configurations
        self.encoder_config = BLTLocalEncoderConfig(
            vocab_size=vocab_size,
            cross_attn_all_layers=cross_attn_all_layers_encoder,
            cross_attn_k=cross_attn_k,
            hidden_size_global=hidden_size_global,
            pm_size=pm_size,
            hidden_size=hidden_size_local_encoder,
            num_attention_heads=num_attention_heads_local_encoder,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers_local_encoder,
            norm_eps=norm_eps,
            dropout=dropout,
            max_position_embeddings=max_encoder_seq_length or max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling={"rope_type": "default"},
            hidden_act=hidden_act,
            multiple_of=multiple_of,
        )

        self.decoder_config = BLTLocalDecoderConfig(
            vocab_size=vocab_size,
            cross_attn_all_layers=cross_attn_all_layers_decoder,
            cross_attn_k=cross_attn_k,
            hidden_size_global=hidden_size_global,
            hidden_size=hidden_size_local_decoder,
            num_attention_heads=num_attention_heads_local_decoder,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers_local_decoder,
            norm_eps=norm_eps,
            dropout=dropout,
            max_position_embeddings=max_encoder_seq_length or max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling={"rope_type": "default"},
            hidden_act=hidden_act,
            multiple_of=multiple_of,
        )

        self.global_config = BLTGlobalTransformerConfig(
            hidden_size=hidden_size_global,
            num_attention_heads=num_attention_heads_global,
            num_key_value_heads=num_key_value_heads_global,
            num_hidden_layers=num_hidden_layers_global,
            norm_eps=norm_eps,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling={"rope_type": "default"},
            hidden_act=hidden_act,
            multiple_of=multiple_of,
            global_dim_patch_emb=hidden_size_local_encoder * cross_attn_k,
        )

        self.patcher_config = BLTPatcherConfig(**patcher_args)

        # Handle hash byte group size validation
        if self.encoder_hash_byte_group_size is not None and type(self.encoder_hash_byte_group_size) == str:
            self.encoder_hash_byte_group_size = [
                int(x) for x in self.encoder_hash_byte_group_size.split(",") if len(x) > 0
            ]

        # Rope scaling configuration
        self.rope_scaling = {"rope_type": "default"}

        # Set compatibility attributes for transformers
        self.num_key_value_heads = num_attention_heads_local_encoder
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size_local_encoder
        self.num_attention_heads = num_attention_heads_local_encoder
        
        # Calculate intermediate_size using BLTMLP logic for each component
        # Note: Each component uses its own hidden dimension, not the main dim
        self.intermediate_size = None  # Will be calculated per component


__all__ = [
    "BLTConfig", 
    "BLTPatcherConfig", 
    "BLTLocalEncoderConfig", 
    "BLTLocalDecoderConfig", 
    "BLTGlobalTransformerConfig", 
    "InitStdFactor", 
    "PatchingModeEnum"
]

