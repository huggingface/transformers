from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

EOS_ID: int = 2


class InitStdFactor(str, Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


class PatchingModeEnum(str, Enum):
    entropy = "entropy"
    bpe = "bpe"
    bpe_patcher = "bpe_patcher"
    space = "space"
    static = "static"
    byte = "byte"


class LMTransformerArgs(BaseModel):
    """Arguments for the Language Model Transformer (used as entropy model for patching)"""
    model_config = ConfigDict()
    
    # Basic architecture
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    
    # Transformer configuration
    max_seqlen: int = 1024
    norm_eps: float = 1e-5
    dropout: float = 0
    vocab_size: int = -1
    sliding_window: int | None = None
    
    # Feedforward
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256
    
    # Positional encoding
    rope_theta: float = 10000.0
    rope_use_fp32_in_outer_product: bool = False
    
    # Attention
    attn_impl: str = "sdpa"
    attn_bias_type: str = "causal"
    
    # Initialization
    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED
    
    # Embedding dimensions
    dim_token_emb: int | None = None
    
    # Model behavior
    weight_tying: bool = False
    seed: int = 42
    
    # Special token config
    eos_id: int = EOS_ID


class ByteLatentTransformerArgs(BaseModel):
    """Arguments for the Byte Latent Transformer (main BLT model)"""
    model_config = ConfigDict()
    
    # Basic model configuration
    seed: int = 42
    vocab_size: int = -1
    
    # Main architecture dimensions (these will be used for creating transformer args)
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    
    # Component-specific dimensions
    dim_global: int = 512
    dim_local_decoder: int = 512
    dim_local_encoder: int = 512
    n_layers_global: int = 8
    n_layers_local_decoder: int = 8
    n_layers_local_encoder: int = 8
    n_heads_global: int = 8
    n_heads_local_decoder: int = 8
    n_heads_local_encoder: int = 8
    n_kv_heads_global: int | None = None
    
    # Transformer configuration (needed by transformer components)
    max_seqlen: int = 1024
    norm_eps: float = 1e-5
    dropout: float = 0
    
    # Feedforward (needed by transformer components)
    ffn_dim_multiplier: float = 1.0
    multiple_of: int = 256
    
    # Positional encoding (needed by transformer components)
    rope_theta: float = 10000.0
    rope_use_fp32_in_outer_product: bool = False
    
    # Attention (needed by transformer components)
    attn_impl: str = "sdpa"
    attn_bias_type: str = "causal"
    
    # Initialization (needed by transformer components)
    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED
    
    # Embedding dimensions (needed by transformer components)
    dim_token_emb: int | None = None
    
    # Patching configuration
    patch_in_forward: bool = False
    realtime_patching: bool = True
    patch_size: float | None = None
    patching_mode: str | None = None
    patching_threshold: float | None = None
    patching_threshold_add: float | None = None
    monotonicity: bool = False
    patching_batch_size: int = 1
    patching_device: str = "cuda"
    max_patch_length: int | None = None
    entropy_model_checkpoint_dir: str | None = None
    
    # Cross attention configurations
    cross_attn_encoder: bool = False
    cross_attn_decoder: bool = False
    cross_attn_window_encoder: int | None = None
    cross_attn_window_decoder: int | None = None
    cross_attn_k: int | None = None
    cross_attn_nheads: int | None = None
    cross_attn_all_layers_decoder: bool = False
    cross_attn_all_layers_encoder: bool = False
    cross_attn_use_flex_attention: bool = True
    cross_attn_init_by_pooling: bool = False
    
    # Encoder configurations
    use_local_encoder_transformer: bool = False
    max_encoder_seq_length: int | None = None
    encoder_hash_byte_group_size: Any | None = None
    encoder_hash_byte_group_vocab: int = 30000
    encoder_hash_byte_group_nb_functions: int = 3
    encoder_enable_byte_ngrams: bool = False
    encoder_ngram_to_size_str: str | None = None
    downsampling_by_pooling: str | None = None
    
    # Architecture and dimensions
    dim_token: int | None = None
    share_encoder_decoder_emb: bool = True
    weight_tying: bool = False
    
    # Attention configuration
    local_attention_window_len: int | None = None
    use_rope: bool = True
    
    # Performance optimization
    sequence_parallel: bool = False
    loss_parallel: bool = False
    fuse_sequence_parallel: bool = False
    use_fsdp: bool = True
    
    # Parameter mixing
    pm_size: int = 0
    
    # Special token config
    eos_id: int = EOS_ID

    @model_validator(mode="after")
    def check_hash_byte_sizes(self) -> Self:
        if (
            self.encoder_hash_byte_group_size is not None
            and type(self.encoder_hash_byte_group_size) == str
        ):
            self.encoder_hash_byte_group_size = [
                int(x)
                for x in self.encoder_hash_byte_group_size.split(",")
                if len(x) > 0
            ]
        return self

