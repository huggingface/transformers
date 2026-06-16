# Copyright 2025 The Susono Team. All rights reserved.
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
"""Susono model configuration.

Susono extends the Qwen3-Next architecture (hybrid full-attention + GatedDeltaNet
linear attention, MoE feed-forward) with two additional axes of capability:

  1. **Engram** (arXiv:2601.07372): Deterministic N-gram hash-based conditional
     memory inserted at selected transformer layers. Provides O(1) static pattern
     recall without routing overhead.

  2. **mHC / MHC-Lite** (arXiv:2512.24880, arXiv:2601.05732): Manifold-Constrained
     Hyper-Connections. Replaces standard single-stream residual connections with n
     parallel residual streams. H_res is parameterised as a convex combination of n!
     permutation matrices (MHC-Lite), which spans the same Birkhoff polytope as
     Sinkhorn-Knopp projection but requires no iterative computation. alpha (H_pre)
     and beta (H_post) are input-dependent via lightweight linear projections.
"""

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


class SusonoConfig(PreTrainedConfig):
    r"""
    Configuration class for the Susono model.

    Susono is a hybrid transformer language model combining:
    - Full (softmax) attention layers and GatedDeltaNet linear attention layers
    - Sparse Mixture-of-Experts (MoE) feed-forward blocks
    - Engram conditional memory (N-gram hash lookup) at selected layers
    - Manifold-Constrained Hyper-Connections (mHC) for multi-stream residuals

    Args:
        vocab_size (`int`, *optional*, defaults to 151680):
            Vocabulary size.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimension of dense MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads per full-attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads (GQA). Equals `num_attention_heads` for MHA.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Std of truncated normal weight initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for RMSNorm layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return past key/values.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input/output embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            RoPE configuration dict (must contain `rope_theta`; optionally
            `rope_type` and scaling parameters).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to add bias to Q/K/V/O projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout probability.
        head_dim (`int`, *optional*, defaults to 256):
            Per-head dimension in full-attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Convolution kernel size in GatedDeltaNet.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Key head dimension in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Value head dimension in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads in linear attention.
        linear_num_value_heads (`int`, *optional*, defaults to 16):
            Number of value heads in linear attention.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            Frequency of MoE layers (every N layers).
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of each routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert in each MoE block.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of experts selected per token.
        num_experts (`int`, *optional*, defaults to 96):
            Total number of routed experts.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Normalize top-k routing probabilities to sum to 1.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to return router logits (needed for auxiliary loss).
        router_aux_loss_coef (`float`, *optional*, defaults to 0.002):
            Coefficient for MoE load-balancing auxiliary loss.
        moe_shared_expert_gate_bias_init (`float`, *optional*, defaults to 2.0):
            Initial value for the shared expert gate bias. Positive values (e.g., 2.0)
            make the shared expert dominant at training start (sigmoid(2)≈0.88), helping
            expert balance. Default 0.0 preserves legacy behavior (sigmoid(0)=0.5).
        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
            Layer indices that use dense MLP instead of MoE.
        layer_types (`list[str]`, *optional*):
            Type of each layer: `"full_attention"` or `"linear_attention"`.
            If `None`, constructed using `full_attention_interval`.
        full_attention_interval (`int`, *optional*, defaults to 4):
            When `layer_types` is `None`, insert a full-attention layer every
            this many layers (i.e. at positions where `(i+1) % full_attention_interval == 0`).
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 151643):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 151645):
            End of stream token id.
        use_engram (`bool`, *optional*, defaults to `True`):
            Enable Engram conditional memory modules.
        engram_max_ngram_size (`int`, *optional*, defaults to 3):
            Maximum N-gram order for Engram (supports 2 … max_ngram_size).
        engram_n_embed_per_ngram (`int`, *optional*, defaults to 99991):
            Hash table size (prime) per N-gram order. Used as the modulus in
            NgramHashMapping. Should be a prime number.
        engram_embed_dim (`int`, *optional*, defaults to 672):
            Embedding dimension per row in the MultiHeadEmbedding table.
        engram_n_head_per_ngram (`int`, *optional*, defaults to 8):
            Number of independent hash heads per N-gram for collision resistance.
        engram_layer_ids (`list[int]`, *optional*):
            0-indexed layer IDs where Engram is applied. Defaults to
            `[0, num_hidden_layers // 2]` when `None`.
        engram_seed (`int`, *optional*, defaults to 0):
            Seed for deterministic Engram hash multiplier generation.
        engram_base_vocab_size (`int`, *optional*):
            Vocabulary size used by Engram tokenizer compression. Defaults to
            `vocab_size` when `None`.
        use_mhc (`bool`, *optional*, defaults to `True`):
            Enable Manifold-Constrained Hyper-Connections (MHC-Lite).
        mhc_num_streams (`int`, *optional*, defaults to 4):
            Number of parallel residual streams for mHC.
        mhc_sinkhorn_iterations (`int`, *optional*, defaults to 20):
            **Deprecated / unused.** Kept for checkpoint backward compatibility.
            MHC-Lite uses a convex combination of permutation matrices instead
            of iterative Sinkhorn-Knopp projection, so this value is ignored.
        qk_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply RMSNorm to the query and key projections in full attention.
        attention_output_gate (`bool`, *optional*, defaults to `True`):
            Whether to apply a Qwen3-Next-style sigmoid output gate to attention outputs.

    Example:
    ```python
    >>> from transformers import SusonoConfig, SusonoModel

    >>> config = SusonoConfig()
    >>> model = SusonoModel(config)
    >>> config = model.config
    ```
    """

    model_type = "susono"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        # Base architecture parameters
        vocab_size: int | None = 151680,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 5120,
        num_hidden_layers: int | None = 24,
        num_attention_heads: int | None = 8,
        num_key_value_heads: int | None = 2,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 262144,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        head_dim: int | None = 256,
        # Linear attention (GatedDeltaNet)
        linear_conv_kernel_dim: int | None = 4,
        linear_key_head_dim: int | None = 128,
        linear_value_head_dim: int | None = 128,
        linear_num_key_heads: int | None = 16,
        linear_num_value_heads: int | None = 16,
        # MoE parameters
        decoder_sparse_step: int | None = 1,
        moe_intermediate_size: int | None = 512,
        shared_expert_intermediate_size: int | None = 512,
        num_experts_per_tok: int | None = 4,
        num_experts: int | None = 96,
        norm_topk_prob: bool | None = True,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.002,
        moe_shared_expert_gate_bias_init: float | None = 2.0,
        mlp_only_layers: list[int] | None = None,
        layer_types: list[str] | None = None,
        full_attention_interval: int = 4,
        # Special tokens
        pad_token_id: int | None = None,
        bos_token_id: int | None = 151643,
        eos_token_id: int | None = 151645,
        # Engram
        use_engram: bool = True,
        engram_max_ngram_size: int = 3,
        engram_n_embed_per_ngram: int = 99991,
        engram_embed_dim: int = 672,
        engram_n_head_per_ngram: int = 8,
        engram_layer_ids: list[int] | None = None,
        engram_seed: int = 0,
        engram_base_vocab_size: int | None = None,
        # mHC
        use_mhc: bool = True,
        mhc_num_streams: int = 4,
        mhc_sinkhorn_iterations: int = 20,
        # Attention QK LayerNorm (Megatron qk_layernorm arg)
        qk_layernorm: bool = True,
        # Attention output gate (Qwen3-Next style: attn_out * sigmoid(gate))
        attention_output_gate: bool = True,
        **kwargs,
    ):
        # Special tokens
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # Base architecture
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = (
            rope_parameters
            if rope_parameters is not None
            else {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            }
        )
        kwargs.setdefault("partial_rotary_factor", 0.25)

        # Layer types
        self.full_attention_interval = full_attention_interval
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # Linear attention
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_shared_expert_gate_bias_init = moe_shared_expert_gate_bias_init
        self.mlp_only_layers = mlp_only_layers if mlp_only_layers is not None else []

        # Engram
        self.use_engram = use_engram
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_n_embed_per_ngram = engram_n_embed_per_ngram
        self.engram_embed_dim = engram_embed_dim
        self.engram_n_head_per_ngram = engram_n_head_per_ngram
        # Default: first and last full-attention layers (derived from layer_types)
        if engram_layer_ids is None:
            full_attn = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
            self.engram_layer_ids = [full_attn[0], full_attn[-1]] if len(full_attn) >= 2 else list(full_attn)
        else:
            self.engram_layer_ids = list(engram_layer_ids)
        self.engram_seed = engram_seed
        # Default to model vocab size if not specified
        self.engram_base_vocab_size = engram_base_vocab_size if engram_base_vocab_size is not None else vocab_size

        # mHC-Lite
        self.use_mhc = use_mhc
        self.mhc_num_streams = mhc_num_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations

        # Attention QK LayerNorm
        self.qk_layernorm = qk_layernorm
        # Attention output gate (Qwen3-Next)
        self.attention_output_gate = attention_output_gate

        super().__init__(**kwargs)


__all__ = ["SusonoConfig"]
