# src/transformers/models/evo2/configuration_evo2.py

from __future__ import annotations

from typing import List, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Evo2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an :class:`~transformers.Evo2ForCausalLM` model.

    It is inspired by the StripedHyena2-based Evo 2 DNA foundation model.

    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 1920):
            Dimension of the hidden representations.
        num_layers (`int`, *optional*, defaults to 25):
            Number of layers (Hyena / attention blocks).
        num_attention_heads (`int`, *optional*, defaults to 15):
            Number of attention heads in attention layers.
        inner_mlp_size (`int`, *optional*, defaults to 5120):
            Size of the intermediate (MLP) layer in the feed-forward network.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            Maximum sequence length that this model might ever be used with.
        rotary_emb_base (`int`, *optional*, defaults to 10000):
            Base for rotary position embeddings.

        attn_layer_idxs (`List[int]`, *optional*):
            Indices of layers that use attention.
        hcl_layer_idxs (`List[int]`, *optional*):
            Indices of "HCL" Hyena layers.
        hcm_layer_idxs (`List[int]`, *optional*):
            Indices of "HCM" Hyena layers.
        hcs_layer_idxs (`List[int]`, *optional*):
            Indices of "HCS" Hyena layers.

        num_filters (`int`, *optional*, defaults to 1920):
            Number of independent filters in Hyena-LI.
        hcm_filter_length (`int`, *optional*, defaults to 128):
            Length of HCM filters.
        hcl_filter_groups (`int`, *optional*, defaults to 1920):
            Number of filter groups for HCL.
        hcm_filter_groups (`int`, *optional*, defaults to 128):
            Number of filter groups for HCM.
        hcs_filter_groups (`int`, *optional*, defaults = 128):
            Number of filter groups for HCS.
        hcs_filter_length (`int`, *optional*, defaults = 7):
            Length of HCS filters.
        short_filter_length (`int`, *optional*, defaults = 3):
            Length of short depthwise FIR filters.
        short_filter_bias (`bool`, *optional*, defaults = False):
            Whether to add a bias to FIR filters.

        state_size (`int`, *optional*, defaults = 16):
            Size of the Hyena state.
        eps (`float`, *optional*, defaults = 1e-6):
            Epsilon used for numerical stability in layer norms etc.

        proj_groups (`int`, *optional*, defaults = 1):
            Number of groups for grouped query/key/value projections.
        hyena_filter_groups (`int`, *optional*, defaults = 1):
            Number of groups for Hyena filters.

        column_split_hyena (`bool`, *optional*, defaults = False):
            Whether to column-split Hyena channels (for tensor parallelism).
        column_split (`bool`, *optional*, defaults = True):
            Whether to column-split projections.
        interleave (`bool`, *optional*, defaults = True):
            Whether to interleave channels.

        evo2_style_activations (`bool`, *optional*, defaults = True):
            Use Evo2-style activations (identity for some layers).
        mlp_activation (`str`, *optional*, defaults = "gelu"):
            Activation function in the MLP.

        make_vocab_size_divisible_by (`int`, *optional*, defaults = 8):
            Pad vocab size to be divisible by this value.
        inner_size_multiple_of (`int`, *optional*, defaults = 16):
            Force MLP inner size to be a multiple of this value.

        tie_embeddings (`bool`, *optional*, defaults = True):
            Whether to tie input and output embeddings.
        mha_out_proj_bias (`bool`, *optional*, defaults = True):
            Whether to use bias in attention output projections.
        hyena_out_proj_bias (`bool`, *optional*, defaults = True):
            Whether to use bias in Hyena output projections.
        qkv_proj_bias (`bool`, *optional*, defaults = False):
            Whether to use bias in QKV projections.
        final_norm (`bool`, *optional*, defaults = True):
            Whether to apply a final normalization layer.

        use_flash_attn (`bool`, *optional*, defaults = True):
            Whether to use FlashAttention when available.
        use_flash_rmsnorm (`bool`, *optional*, defaults = False):
            Whether to use a fused Flash RMSNorm implementation.
        use_flash_depthwise (`bool`, *optional*, defaults = False):
            Whether to use fused depthwise convolution kernels.
        use_flashfft (`bool`, *optional*, defaults = False):
            Whether to use FFT-based kernels for long convolutions.
        use_laughing_hyena (`bool`, *optional*, defaults = False):
            Experimental variant toggle.

        max_batch_size (`int`, *optional*, defaults = 1):
            Max batch size used in the original config (not enforced by HF).
        inference_mode (`bool`, *optional*, defaults = True):
            Indicates original config was built for inference.

        tokenizer_type (`str`, *optional*, defaults = "CharLevelTokenizer"):
            Name of the tokenizer expected by the original implementation.
        prefill_style (`str`, *optional*, defaults = "fft"):
            Prefill strategy used in original Evo2.

        print_activations (`bool`, *optional*, defaults = False):
            Log intermediate activations (debugging).
        log_intermediate_values (`bool`, *optional*, defaults = False):
            Log intermediate values in original code (debugging).

        model_parallel_size (`int`, *optional*, defaults = 1):
            Original MP size; informational only here.
        pipe_parallel_size (`int`, *optional*, defaults = 1):
            Original PP size; informational only here.

        hyena_flip_x1x2 (`bool`, *optional*, defaults = False):
            Flip Hyena kernel inputs (compat option).
        use_fp8_input_projections (`bool`, *optional*, defaults = True):
            Whether the original model used FP8 input projections.

        **kwargs:
            Additional keyword arguments passed to `PretrainedConfig`.
    """

    model_type = "evo2"

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 1920,
        num_layers: int = 25,
        num_attention_heads: int = 15,
        inner_mlp_size: int = 5120,
        max_position_embeddings: int = 8192,
        rotary_emb_base: int = 10000,
        attn_layer_idxs: Optional[List[int]] = None,
        hcl_layer_idxs: Optional[List[int]] = None,
        hcm_layer_idxs: Optional[List[int]] = None,
        hcs_layer_idxs: Optional[List[int]] = None,
        num_filters: int = 1920,
        hcm_filter_length: int = 128,
        hcl_filter_groups: int = 1920,
        hcm_filter_groups: int = 128,
        hcs_filter_groups: int = 128,
        hcs_filter_length: int = 7,
        short_filter_length: int = 3,
        short_filter_bias: bool = False,
        state_size: int = 16,
        eps: float = 1e-6,
        proj_groups: int = 1,
        hyena_filter_groups: int = 1,
        column_split_hyena: bool = False,
        column_split: bool = True,
        interleave: bool = True,
        evo2_style_activations: bool = True,
        mlp_activation: str = "gelu",
        make_vocab_size_divisible_by: int = 8,
        inner_size_multiple_of: int = 16,
        tie_embeddings: bool = True,
        mha_out_proj_bias: bool = True,
        hyena_out_proj_bias: bool = True,
        qkv_proj_bias: bool = False,
        final_norm: bool = True,
        use_flash_attn: bool = True,
        use_flash_rmsnorm: bool = False,
        use_flash_depthwise: bool = False,
        use_flashfft: bool = False,
        use_laughing_hyena: bool = False,
        max_batch_size: int = 1,
        inference_mode: bool = True,
        tokenizer_type: str = "CharLevelTokenizer",
        prefill_style: str = "fft",
        print_activations: bool = False,
        log_intermediate_values: bool = False,
        model_parallel_size: int = 1,
        pipe_parallel_size: int = 1,
        hyena_flip_x1x2: bool = False,
        use_fp8_input_projections: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Core HF-style fields
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = inner_mlp_size  # HF naming
        self.inner_mlp_size = inner_mlp_size     # original naming
        self.max_position_embeddings = max_position_embeddings

        # Rotary embeddings
        self.rotary_emb_base = rotary_emb_base

        # Layer index layout
        self.attn_layer_idxs = attn_layer_idxs or [3, 10, 17, 24]
        self.hcl_layer_idxs = hcl_layer_idxs or [2, 6, 9, 13, 16, 20, 23]
        self.hcm_layer_idxs = hcm_layer_idxs or [1, 5, 8, 12, 15, 19, 22]
        self.hcs_layer_idxs = hcs_layer_idxs or [0, 4, 7, 11, 14, 18, 21]

        # Hyena / filter hyperparameters
        self.num_filters = num_filters
        self.hcm_filter_length = hcm_filter_length
        self.hcl_filter_groups = hcl_filter_groups
        self.hcm_filter_groups = hcm_filter_groups
        self.hcs_filter_groups = hcs_filter_groups
        self.hcs_filter_length = hcs_filter_length
        self.short_filter_length = short_filter_length
        self.short_filter_bias = short_filter_bias

        # State & numerics
        self.state_size = state_size
        self.eps = eps

        # Grouping & splitting
        self.proj_groups = proj_groups
        self.hyena_filter_groups = hyena_filter_groups
        self.column_split_hyena = column_split_hyena
        self.column_split = column_split
        self.interleave = interleave

        # Activations / MLP
        self.evo2_style_activations = evo2_style_activations
        self.mlp_activation = mlp_activation
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.inner_size_multiple_of = inner_size_multiple_of

        # Projection / embedding knobs
        self.tie_embeddings = tie_embeddings
        self.mha_out_proj_bias = mha_out_proj_bias
        self.hyena_out_proj_bias = hyena_out_proj_bias
        self.qkv_proj_bias = qkv_proj_bias
        self.final_norm = final_norm

        # Flash / fused kernels (may be ignored in pure PyTorch version)
        self.use_flash_attn = use_flash_attn
        self.use_flash_rmsnorm = use_flash_rmsnorm
        self.use_flash_depthwise = use_flash_depthwise
        self.use_flashfft = use_flashfft
        self.use_laughing_hyena = use_laughing_hyena

        # Original inference-related fields (kept for compatibility, not enforced)
        self.max_batch_size = max_batch_size
        self.inference_mode = inference_mode

        # Tokenizer / prefill / logging metadata
        self.tokenizer_type = tokenizer_type
        self.prefill_style = prefill_style
        self.print_activations = print_activations
        self.log_intermediate_values = log_intermediate_values

        # Parallelism & numeric tricks (informational)
        self.model_parallel_size = model_parallel_size
        self.pipe_parallel_size = pipe_parallel_size
        self.hyena_flip_x1x2 = hyena_flip_x1x2
        self.use_fp8_input_projections = use_fp8_input_projections

        # For backward compatibility with original config name
        self.max_seqlen = max_position_embeddings


__all__ = ["Evo2Config"]
