from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

EOS_ID: int = 2


class InitStdFactor(str, Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict()
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None

    ffn_dim_multiplier: float | None = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0
    rope_use_fp32_in_outer_product: bool = False

    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED

    max_seqlen: int = 1024

    attn_impl: str | None = "sdpa"
    attn_bias_type: str | None = None
    # Special token config
    eos_id: int | None = EOS_ID

class ByteLatentTransformerArgs(BaseTransformerArgs):
    # Basic model configuration
    seed: int = 42
    vocab_size: int = -1
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    # TODO: What is the purpose of this parameter?
    weight_tying: bool = False
    patch_in_forward: bool = False

    realtime_patching: bool = True

    # Architecture and dimensions
    dim_token: int | None = None
    dim_global: int = 512
    dim_local_decoder: int = 512
    dim_local_encoder: int = 512
    n_layers_global: int = 8
    n_layers_local_decoder: int = 8
    n_layers_local_encoder: int = 8

    # Tokenization and patching
    patch_size: float | None = None
    patching_mode: str | None = None
    patching_threshold: float | None = None
    patching_threshold_add: float | None = None
    monotonicity: bool = False
    patching_batch_size: int = 1
    patching_device: str = "cuda"
    max_patch_length: int | None = None

    # Encoder/Decoder configuration
    tie_local_encoder_decoder_logits: bool = False
    use_local_encoder_transformer: bool = False
    encoder_lm_loss: bool = False
    max_encoder_seq_length: int | None = None
    pad_to_max_length: bool = False
    encoder_enable_byte_ngrams: bool = False
    encoder_enable_byte_group_hash: bool = False
    ngram_vocab_sizes: int | None = None

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

    # Encoder hash configurations
    encoder_hash_byte_group_size: Any | None = None
    encoder_hash_byte_group_vocab: int = 30000
    encoder_hash_byte_group_nb_functions: int = 3

    # Model behavior and optimization
    log_patch_lengths: bool = False
    non_linearity: str = "swiglu"
    use_rope: bool = True
    recompute_fc1_out: bool = False
    recompute_fc3_out: bool = False
    recompute_attn: bool = True
    custom_bwd: bool = False
    layer_ckpt: str = "all"

    # Initialization and attention
    init_use_gaussian: bool = True
    init_use_depth: str = "current"
    attn_bias_type: str = "causal"
    alpha_depth: str = "disabled"
    max_length: int = 2048

    # Norm configuration
    norm_eps: float = 1e-5
    norm_affine: bool = True
    pre_norm: bool = True
    norm_type: str = "rmsnorm"

    # Additional configurations
    multiple_of: int = 256
    ffn_dim_multiplier: float = 1.0
    dropout: float = 0
    output_size: int = -1

    # Additional parameters from ModelArgs
    architecture: str = "vanilla"
    share_encoder_decoder_emb: bool = True
    global_local_decoder_residual_layer: str | None = None

    tokenize_with_bpe_delimiter: bool = False
    patching_thresholds_str: str | None = None
    tie_local_encoder_decoder: bool = False
    encoder_preds_low_entropy_toks: float | None = None
    encoder_preds_random_toks: float | None = None
    dim_token_emb: int | None = None
    dim_patch_emb: int | None = None

    encoder_ngram_table_dir: str | None = None
    encoder_ngram_to_size_str: str | None = None

    # Model architecture params
    entropy_model_checkpoint_dir: str | None = None
    entropy_model_is_ngram_model: bool = False
    downsampling_by_pooling: str | None = None
    n_heads_global: int = 8
    n_heads_local_decoder: int = 8
    n_heads_local_encoder: int = 8
    n_kv_heads: int | None = None
    n_kv_heads_global: int | None = None
    conv_kernel_size: int | None = None
    local_attention_window_len: int | None = None

    # Performance optimization
    sequence_parallel: bool = False
    loss_parallel: bool = False
    fuse_sequence_parallel: bool = False
    use_fsdp: bool = True
    attn_to_keep: str = "all"

    # Parameter mixing
    pm_size: int = 0

    # Logging
    full_logging_n_layers: int = 4

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


class GlobalTransformerArgs(ByteLatentTransformerArgs):
    # Global encoder specific dimensions
    dim_token_emb: int | None = None
    dim_patch_emb: int | None = None

    def __post_init__(self):
        # Override base args with global encoder specific values
        self.dim = self.dim_global
        self.n_layers = self.n_layers_global
        self.n_heads = self.n_heads_global
        self.n_kv_heads = self.n_kv_heads_global
        self.local_attention_window_len = None
        self.cross_attn_encoder = False
        self.cross_attn_decoder = False


class LocalDecoderArgs(ByteLatentTransformerArgs):
    # Local decoder specific dimensions
    dim_token_emb: int | None = None
    dim_patch_emb: int | None = None

    def __post_init__(self):
        # Override base args with local decoder specific values
        self.dim = self.dim_local_decoder
        self.n_layers = self.n_layers_local_decoder
        self.n_heads = self.n_heads_local_decoder
        self.cross_attn_encoder = False
        self.cross_attn_init_by_pooling = False
        self.attn_bias_type = "local_block_causal"


class LocalModelArgs(BaseTransformerArgs):
    model_config = ConfigDict()
    # Override defaults
    attn_impl: str | None = "xformers"
    attn_bias_type: str | None = "local_block_causal"

    # Local encoder specific dimensions
    dropout: float
    vocab_size: int
    patch_size: float
    sliding_window: int | None
    use_rope: bool
    cross_attn_encoder: bool | None
    cross_attn_decoder: bool | None
    cross_attn_k: int | None
    cross_attn_init_by_pooling: bool
    patching_mode: str
    use_local_encoder_transformer: bool
    downsampling_by_pooling: str | None
    encoder_hash_byte_group_size: Any | None = None
    cross_attn_all_layers_encoder: bool = False
    cross_attn_all_layers_decoder: bool = False
    cross_attn_nheads: int | None

    dim_token_emb: int
    dim_patch_emb: int | None


class LMTransformerArgs(BaseTransformerArgs):
    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    sliding_window: int | None = None

