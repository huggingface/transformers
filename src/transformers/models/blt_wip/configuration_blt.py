# new config

# coding=utf-8
# Copyright 2024 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""BLT (Byte Latent Transformer) model configuration"""

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


class BLTPatcherConfig(PretrainedConfig):
    r"""
    Configuration class for the BLT Patcher/Entropy model component.
    
    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size for the entropy model used in patching.
        dim (`int`, *optional*, defaults to 512):
            Hidden dimension for the entropy model.
        n_layers (`int`, *optional*, defaults to 8):
            Number of layers in the entropy model.
        n_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the entropy model.
        head_dim (`int`, *optional*):
            Dimension of each attention head in the entropy model.
        n_kv_heads (`int`, *optional*):
            Number of key-value heads in the entropy model.
        max_seqlen (`int`, *optional*, defaults to 1024):
            Maximum sequence length for the entropy model.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            Layer normalization epsilon for the entropy model.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the entropy model.
        sliding_window (`int`, *optional*):
            Sliding window size for the entropy model attention.
        ffn_dim_multiplier (`float`, *optional*):
            Feedforward dimension multiplier for the entropy model.
        multiple_of (`int`, *optional*, defaults to 256):
            Make feedforward dimension multiple of this for the entropy model.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE theta parameter for the entropy model.
        rope_use_fp32_in_outer_product (`bool`, *optional*, defaults to False):
            Whether to use fp32 in RoPE outer product for the entropy model.
        attn_impl (`str`, *optional*, defaults to "sdpa"):
            Attention implementation for the entropy model.
        attn_bias_type (`str`, *optional*, defaults to "causal"):
            Attention bias type for the entropy model.
        init_base_std (`float`, *optional*):
            Base initialization standard deviation for the entropy model.
        init_std_factor (`str`, *optional*, defaults to "disabled"):
            Initialization std factor for the entropy model.
        dim_token_emb (`int`, *optional*):
            Token embedding dimension for the entropy model.
        weight_tying (`bool`, *optional*, defaults to False):
            Whether to tie embeddings in the entropy model.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of sequence token id for the entropy model.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of sequence token id for the entropy model.
    """
    
    model_type = "blt_patcher"
    
    def __init__(
        self,
        vocab_size=256,
        dim=512,
        n_layers=8,
        n_heads=8,
        head_dim=None,
        n_kv_heads=None,
        max_seqlen=1024,
        norm_eps=1e-5,
        dropout=0.0,
        sliding_window=None,
        ffn_dim_multiplier=None,
        multiple_of=256,
        rope_theta=10000.0,
        rope_use_fp32_in_outer_product=False,
        attn_impl="sdpa",
        attn_bias_type="causal",
        init_base_std=None,
        init_std_factor="disabled",
        dim_token_emb=None,
        weight_tying=False,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else (dim // n_heads)
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.max_seqlen = max_seqlen
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.sliding_window = sliding_window
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.rope_theta = rope_theta
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product
        self.attn_impl = attn_impl
        self.attn_bias_type = attn_bias_type
        self.init_base_std = init_base_std
        self.init_std_factor = InitStdFactor(init_std_factor)
        self.dim_token_emb = dim_token_emb
        self.weight_tying = weight_tying
        
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        
        # Add attributes needed for compatibility with transformer models
        self.hidden_size = dim
        self.num_attention_heads = n_heads
        self.num_key_value_heads = self.n_kv_heads  # Use the computed n_kv_heads
        self.max_position_embeddings = max_seqlen
        
        # Set simple rope scaling for patcher (no complex dynamic rope)
        self.rope_scaling = {"rope_type": "default"}


class BLTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ByteLatentTransformer`]. It is used to instantiate a
    BLT model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the BLT model. Defines the number of different tokens (bytes) that can be represented.
        max_seqlen (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model can handle.

        # Main architecture dimensions
        dim (`int`, *optional*, defaults to 512):
            Main dimension of the model.
        n_layers (`int`, *optional*, defaults to 8):
            Number of layers in the main transformer.
        n_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the main transformer.
        head_dim (`int`, *optional*):
            Dimension of each attention head. If not specified, computed as dim // n_heads.
        n_kv_heads (`int`, *optional*):
            Number of key-value heads for grouped query attention. If not specified, defaults to n_heads.

        # Component-specific dimensions
        dim_global (`int`, *optional*, defaults to 512):
            Dimension of the global transformer component.
        dim_local_decoder (`int`, *optional*, defaults to 512):
            Dimension of the local decoder component.
        dim_local_encoder (`int`, *optional*, defaults to 512):
            Dimension of the local encoder component.
        n_layers_global (`int`, *optional*, defaults to 8):
            Number of layers in the global transformer.
        n_layers_local_decoder (`int`, *optional*, defaults to 8):
            Number of layers in the local decoder.
        n_layers_local_encoder (`int`, *optional*, defaults to 8):
            Number of layers in the local encoder.
        n_heads_global (`int`, *optional*, defaults to 8):
            Number of attention heads in the global transformer.
        n_heads_local_decoder (`int`, *optional*, defaults to 8):
            Number of attention heads in the local decoder.
        n_heads_local_encoder (`int`, *optional*, defaults to 8):
            Number of attention heads in the local encoder.
        n_kv_heads_global (`int`, *optional*):
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
        rope_use_fp32_in_outer_product (`bool`, *optional*, defaults to False):
            Whether to use fp32 in RoPE outer product computation.

        # Attention configuration
        attn_impl (`str`, *optional*, defaults to "sdpa"):
            Attention implementation to use ("sdpa" or "flex_attention").
        attn_bias_type (`str`, *optional*, defaults to "causal"):
            Type of attention bias to apply.
        local_attention_window_len (`int`, *optional*):
            Window length for local attention.
        use_rope (`bool`, *optional*, defaults to True):
            Whether to use rotary position embeddings.

        # Initialization
        init_base_std (`float`, *optional*):
            Base standard deviation for weight initialization.
        init_std_factor (`str`, *optional*, defaults to "disabled"):
            Factor for adjusting initialization standard deviation.

        # Embedding dimensions
        dim_token_emb (`int`, *optional*):
            Token embedding dimension.
        dim_token (`int`, *optional*):
            Token dimension.

        # Patching configuration
        patch_in_forward (`bool`, *optional*, defaults to False):
            Whether to perform patching during forward pass.
        realtime_patching (`bool`, *optional*, defaults to True):
            Whether to use realtime patching.
        patch_size (`float`, *optional*):
            Size of patches for static patching.
        patching_mode (`str`, *optional*):
            Mode for patching ("entropy", "static", etc.).
        patching_threshold (`float`, *optional*):
            Threshold for entropy-based patching.
        patching_threshold_add (`float`, *optional*):
            Additional threshold parameter for patching.
        monotonicity (`bool`, *optional*, defaults to False):
            Whether to enforce monotonicity in patching.
        patching_batch_size (`int`, *optional*, defaults to 1):
            Batch size for patching operations.
        patching_device (`str`, *optional*, defaults to "cuda"):
            Device to use for patching operations.
        max_patch_length (`int`, *optional*):
            Maximum length of patches.
        entropy_model_checkpoint_dir (`str`, *optional*):
            Directory containing entropy model checkpoint.

        # Cross attention configurations
        cross_attn_encoder (`bool`, *optional*, defaults to False):
            Whether to use cross attention in encoder.
        cross_attn_decoder (`bool`, *optional*, defaults to False):
            Whether to use cross attention in decoder.
        cross_attn_window_encoder (`int`, *optional*):
            Cross attention window for encoder.
        cross_attn_window_decoder (`int`, *optional*):
            Cross attention window for decoder.
        cross_attn_k (`int`, *optional*):
            Number of cross attention components.
        cross_attn_nheads (`int`, *optional*):
            Number of heads for cross attention.
        cross_attn_all_layers_decoder (`bool`, *optional*, defaults to False):
            Whether to apply cross attention to all decoder layers.
        cross_attn_all_layers_encoder (`bool`, *optional*, defaults to False):
            Whether to apply cross attention to all encoder layers.
        cross_attn_use_flex_attention (`bool`, *optional*, defaults to True):
            Whether to use flexible attention for cross attention.
        cross_attn_init_by_pooling (`bool`, *optional*, defaults to False):
            Whether to initialize cross attention by pooling.

        # Encoder configurations
        use_local_encoder_transformer (`bool`, *optional*, defaults to False):
            Whether to use transformer in local encoder.
        max_encoder_seq_length (`int`, *optional*):
            Maximum sequence length for encoder.
        encoder_hash_byte_group_size (`Any`, *optional*):
            Hash byte group size for encoder.
        encoder_hash_byte_group_vocab (`int`, *optional*, defaults to 30000):
            Vocabulary size for hash byte groups.
        encoder_hash_byte_group_nb_functions (`int`, *optional*, defaults to 3):
            Number of hash functions for byte groups.
        encoder_enable_byte_ngrams (`bool`, *optional*, defaults to False):
            Whether to enable byte n-grams in encoder.
        encoder_ngram_to_size_str (`str`, *optional*):
            String defining n-gram sizes.
        downsampling_by_pooling (`str`, *optional*):
            Type of pooling for downsampling.

        # Model behavior
        share_encoder_decoder_emb (`bool`, *optional*, defaults to True):
            Whether to share encoder and decoder embeddings.
        weight_tying (`bool`, *optional*, defaults to False):
            Whether to tie input and output embeddings.

        # Performance optimization
        sequence_parallel (`bool`, *optional*, defaults to False):
            Whether to use sequence parallelism.
        loss_parallel (`bool`, *optional*, defaults to False):
            Whether to use loss parallelism.
        fuse_sequence_parallel (`bool`, *optional*, defaults to False):
            Whether to fuse sequence parallel operations.
        use_fsdp (`bool`, *optional*, defaults to True):
            Whether to use fully sharded data parallel.

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
        max_seqlen=1024,
        # Main architecture dimensions
        dim=512,
        n_layers=8,
        n_heads=8,
        head_dim=None,
        n_kv_heads=None,
        # Component-specific dimensions
        dim_global=512,
        dim_local_decoder=512,
        dim_local_encoder=512,
        n_layers_global=8,
        n_layers_local_decoder=8,
        n_layers_local_encoder=8,
        n_heads_global=8,
        n_heads_local_decoder=8,
        n_heads_local_encoder=8,
        n_kv_heads_global=None,
        # Transformer configuration
        norm_eps=1e-5,
        dropout=0.0,
        ffn_dim_multiplier=1.0,
        multiple_of=256,
        # Positional encoding
        rope_theta=10000.0,
        rope_use_fp32_in_outer_product=False,
        # Attention configuration
        attn_impl="sdpa",
        attn_bias_type="causal",
        local_attention_window_len=None,
        use_rope=True,
        # Initialization
        init_base_std=None,
        init_std_factor="disabled",
        # Embedding dimensions
        dim_token_emb=None,
        dim_token=None,
        # Patching configuration
        patch_in_forward=False,
        realtime_patching=True,
        patch_size=None,
        patching_mode=None,
        patching_threshold=None,
        patching_threshold_add=None,
        monotonicity=False,
        patching_batch_size=1,
        patching_device="cuda",
        max_patch_length=None,
        entropy_model_checkpoint_dir=None,
        # Cross attention configurations
        cross_attn_encoder=False,
        cross_attn_decoder=False,
        cross_attn_window_encoder=None,
        cross_attn_window_decoder=None,
        cross_attn_k=None,
        cross_attn_nheads=None,
        cross_attn_all_layers_decoder=False,
        cross_attn_all_layers_encoder=False,
        cross_attn_use_flex_attention=True,
        cross_attn_init_by_pooling=False,
        # Encoder configurations
        use_local_encoder_transformer=False,
        max_encoder_seq_length=None,
        encoder_hash_byte_group_size=None,
        encoder_hash_byte_group_vocab=30000,
        encoder_hash_byte_group_nb_functions=3,
        encoder_enable_byte_ngrams=False,
        encoder_ngram_to_size_str=None,
        downsampling_by_pooling=None,
        # Model behavior
        share_encoder_decoder_emb=True,
        weight_tying=False,
        # Performance optimization
        sequence_parallel=False,
        loss_parallel=False,
        fuse_sequence_parallel=False,
        use_fsdp=True,
        # Parameter mixing
        pm_size=0,
        # Special tokens
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=-1,
        boe_id=0,
        # Patcher configuration
        patcher_args=None,
        # Inherited
        **kwargs,
    ):
        
        self.sliding_window = None
        # Basic model configuration
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen

        # Main architecture dimensions
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else (dim // n_heads)
        self.n_kv_heads = n_kv_heads

        # Component-specific dimensions
        self.dim_global = dim_global
        self.dim_local_decoder = dim_local_decoder
        self.dim_local_encoder = dim_local_encoder
        self.n_layers_global = n_layers_global
        self.n_layers_local_decoder = n_layers_local_decoder
        self.n_layers_local_encoder = n_layers_local_encoder
        self.n_heads_global = n_heads_global
        self.n_heads_local_decoder = n_heads_local_decoder
        self.n_heads_local_encoder = n_heads_local_encoder
        self.n_kv_heads_global = n_kv_heads_global

        # Transformer configuration
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of

        # Positional encoding
        self.rope_theta = rope_theta
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product

        # Attention configuration
        self.attn_impl = attn_impl
        self.attn_bias_type = attn_bias_type
        self.local_attention_window_len = local_attention_window_len
        self.use_rope = use_rope

        # Initialization
        self.init_base_std = init_base_std
        self.init_std_factor = InitStdFactor(init_std_factor)

        # Embedding dimensions
        self.dim_token_emb = dim_token_emb
        self.dim_token = dim_token

        # Patching configuration
        self.patch_in_forward = patch_in_forward
        self.realtime_patching = realtime_patching
        self.patch_size = patch_size
        self.patching_mode = patching_mode
        self.patching_threshold = patching_threshold
        self.patching_threshold_add = patching_threshold_add
        self.monotonicity = monotonicity
        self.patching_batch_size = patching_batch_size
        self.patching_device = patching_device
        self.max_patch_length = max_patch_length
        self.entropy_model_checkpoint_dir = entropy_model_checkpoint_dir

        # Cross attention configurations
        self.cross_attn_encoder = cross_attn_encoder
        self.cross_attn_decoder = cross_attn_decoder
        self.cross_attn_window_encoder = cross_attn_window_encoder
        self.cross_attn_window_decoder = cross_attn_window_decoder
        self.cross_attn_k = cross_attn_k
        self.cross_attn_nheads = cross_attn_nheads
        self.cross_attn_all_layers_decoder = cross_attn_all_layers_decoder
        self.cross_attn_all_layers_encoder = cross_attn_all_layers_encoder
        self.cross_attn_use_flex_attention = cross_attn_use_flex_attention
        self.cross_attn_init_by_pooling = cross_attn_init_by_pooling

        # Encoder configurations
        self.use_local_encoder_transformer = use_local_encoder_transformer
        self.max_encoder_seq_length = max_encoder_seq_length
        self.encoder_hash_byte_group_size = encoder_hash_byte_group_size
        self.encoder_hash_byte_group_vocab = encoder_hash_byte_group_vocab
        self.encoder_hash_byte_group_nb_functions = encoder_hash_byte_group_nb_functions
        self.encoder_enable_byte_ngrams = encoder_enable_byte_ngrams
        self.encoder_ngram_to_size_str = encoder_ngram_to_size_str
        self.downsampling_by_pooling = downsampling_by_pooling

        # Model behavior
        self.share_encoder_decoder_emb = share_encoder_decoder_emb
        self.weight_tying = weight_tying

        # Performance optimization
        self.sequence_parallel = sequence_parallel
        self.loss_parallel = loss_parallel
        self.fuse_sequence_parallel = fuse_sequence_parallel
        self.use_fsdp = use_fsdp

        # Parameter mixing
        self.pm_size = pm_size
        
        # Special token IDs
        self.boe_id = boe_id

        # Initialize patcher configuration
        if patcher_args is not None:
            self.patcher_config = BLTPatcherConfig(**patcher_args)
        else:
            # Use default values if no patcher_args provided
            self.patcher_config = BLTPatcherConfig()

        # Handle hash byte group size validation
        if self.encoder_hash_byte_group_size is not None and type(self.encoder_hash_byte_group_size) == str:
            self.encoder_hash_byte_group_size = [
                int(x) for x in self.encoder_hash_byte_group_size.split(",") if len(x) > 0
            ]

        # Rope
        self.rope_scaling={
            "type": "default",
            "rope_type": "default"
        }

        self.num_key_value_heads=n_heads_local_encoder
        self.max_position_embeddings=max_seqlen
        self.hidden_size=dim_local_encoder
        self.num_attention_heads=n_heads_local_encoder
     #   self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )



    @property
    def encoder_dim_token_emb(self):
        """Compute encoder token embedding dimension."""
        if self.dim_token is not None:
            return self.dim_token
        elif self.use_local_encoder_transformer:
            return self.dim_local_encoder
        else:
            # Use default patch_size of 8 if not set
            patch_size = self.patch_size if self.patch_size is not None else 8
            return self.dim_global // patch_size

    @property
    def encoder_dim_patch_emb(self):
        """Compute encoder patch embedding dimension."""
        if self.cross_attn_encoder:
            if self.cross_attn_init_by_pooling:
                return self.dim_local_encoder
            else:
                return self.dim_global
        return None

    @property
    def global_dim_patch_emb(self):
        """Compute global patch embedding dimension."""
        dim_token_emb = self.encoder_dim_token_emb
        if self.cross_attn_encoder:
            cross_attn_k = self.cross_attn_k if self.cross_attn_k is not None else 1
            return dim_token_emb * cross_attn_k
        elif (
            self.downsampling_by_pooling is None
            or not self.downsampling_by_pooling
            or len(self.downsampling_by_pooling) == 0
        ):
            # Use default patch_size of 8 if not set
            patch_size = self.patch_size if self.patch_size is not None else 8
            return dim_token_emb * patch_size
        else:
            return dim_token_emb * sum([pooling in self.downsampling_by_pooling for pooling in ["avg", "min", "max"]])

    @property
    def decoder_dim_token_emb(self):
        """Compute decoder token embedding dimension."""
        if self.share_encoder_decoder_emb:
            return self.encoder_dim_token_emb
        elif self.dim_token is not None:
            return self.dim_token
        else:
            return self.dim_local_decoder

    def get_init_std_factor(self, depth: int) -> float:
        """
        Calculate the initialization standard deviation scaling factor for a given layer depth.

        Args:
            depth: Current layer depth (0-indexed)

        Returns:
            Scaling factor to divide the base initialization std by
        """
        if self.init_std_factor == InitStdFactor.CURRENT_DEPTH:
            return (2 * (depth + 1)) ** 0.5
        else:  # DISABLED
            return 1.0


__all__ = ["BLTConfig", "BLTPatcherConfig", "InitStdFactor", "PatchingModeEnum"]

