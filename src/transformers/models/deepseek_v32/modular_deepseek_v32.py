# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""DeepSeek-V3.2-Exp: DeepSeek-V3 plus DeepSeek Sparse Attention (DSA).

This is DeepSeek-V3 with a lightning indexer added to each attention layer: the indexer scores every
query against the cached keys and keeps the top-`index_topk` tokens, which are the only ones the MLA
attention then attends to. Everything else (MoE, MLA projections, RoPE, the decoder / model /
causal-LM scaffolding) is inherited unchanged from DeepSeek-V3.

The cross-layer top-k *sharing* variant is a GLM-MoE-DSA innovation and lives in that model, which
inherits from this one (see `models/glm_moe_dsa/modular_glm_moe_dsa.py`).
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...distributed.utils import is_dtensor
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer


logger = logging.get_logger(__name__)

# Upper bound, in elements, on the fp32 `[B, chunk, H_idx, T]` score tensor the indexer materializes.
# Queries are scored in chunks along the sequence so peak memory stays bounded at long context.
_INDEXER_SCORE_BUDGET = 2**29
# Upper bound, in elements, on the `[B, chunk, K, kv_lora_rank + qk_rope_head_dim]` latents the sparse
# attention gathers per query chunk.
_SPARSE_ATTENTION_BUDGET = 2**27


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V3.2-Exp")
@strict
class DeepseekV32Config(Glm4MoeLiteConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 64):
        Number of heads for the indexer projections (DSA).
    first_k_dense_replace (`int`, *optional*, defaults to 3):
        Number of leading layers that use a dense MLP; the rest use the MoE block.

    ```python
    >>> from transformers import DeepseekV32Config, DeepseekV32Model

    >>> # Initializing a DeepSeek-V3.2 configuration
    >>> configuration = DeepseekV32Config()

    >>> # Initializing a model from the configuration
    >>> model = DeepseekV32Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 163840
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    mlp_layer_types: list[str] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 64
    mlp_bias: bool = False
    head_dim: int = 64
    first_k_dense_replace: int = 3
    pretraining_tp = AttributeError()
    rope_interleave = AttributeError()
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so point `head_dim` at it: the inherited (Llama) rotary
        # embedding reads `config.head_dim` and then computes the right frequencies with no override needed.
        self.head_dim = self.qk_rope_head_dim
        # MLP layer types: the first `first_k_dense_replace` layers are dense, the rest are MoE.
        if self.mlp_layer_types is None:
            n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.num_hidden_layers - n_dense)
        # Every layer is DSA — drives cache-class dispatch.
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        # BC: re-route `num_experts` to `n_routed_experts`
        if (num_experts := kwargs.get("num_experts")) is not None:
            self.n_routed_experts = num_experts

        super().__post_init__(**kwargs)


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Normalized Walsh-Hadamard transform along the last dimension (which must be a power of two), computed
    as `log2(n)` butterfly stages. This is the reference's `rotate_activation`, i.e.
    `fast_hadamard_transform.hadamard_transform(x, scale=n**-0.5)` with the natural (Sylvester) ordering.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"The Hadamard transform requires a power-of-two last dimension, but got {n}.")
    # Butterflies accumulate in fp32 like the reference kernel, which casts back once after normalizing
    y = x.reshape(-1, n) if x.dtype in (torch.float32, torch.float64) else x.float().reshape(-1, n)
    half = 1
    while half < n:
        y = y.reshape(-1, n // (2 * half), 2, half)
        a, b = y[:, :, 0], y[:, :, 1]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        half *= 2
    return y.mul(n**-0.5).reshape(x.shape).to(x.dtype)


def fake_quant_fp8_block(x: torch.Tensor, block_size: int = 128, scale_fmt: str | None = None) -> torch.Tensor:
    """
    The reference's `act_quant` followed by dequantization: per-`block_size` block FP8 (e4m3) quantization
    along the last dimension, returned in `x.dtype`. Last dimensions that the reference cannot block (tiny
    test configs) are returned unchanged. The result is a straight-through estimator, so quantizing an
    activation does not cut the gradient of the projections that produced it.
    """
    n = x.shape[-1]
    if n % block_size:
        return x
    blocks = x.float().reshape(*x.shape[:-1], n // block_size, block_size)
    scale = blocks.abs().amax(-1).clamp_min(1e-4) / 448.0
    if scale_fmt == "ue8m0":
        # Round the scale up to a power of two, with the IEEE-754 bit manipulation of the reference kernel
        bits = scale.contiguous().view(torch.int32)
        exponent, mantissa = (bits >> 23) & 0xFF, bits & 0x7FFFFF
        scale = torch.exp2((exponent - 127 + (mantissa != 0).to(torch.int32)).to(torch.float32))
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale.unsqueeze(-1)).reshape(x.shape).to(x.dtype)
    return x + (dequantized - x).detach()


def _dsa_scale_fmt(config) -> str | None:
    """
    FP8 scale format of the KV-latent fake quantization: a top-level `scale_fmt` config field (GLM-5.x ships it
    as `null`, i.e. plain fp32 per-block scales, which is also the layout of the fp8 MLA KV cache vLLM serves
    GLM-5.x with), or, for the released DeepSeek-V3.2-Exp which only carries
    it inside `quantization_config`, that entry -- the value the reference reads from its own model config.
    """
    quant_config = getattr(config, "quantization_config", None)
    quant_scale_fmt = (
        quant_config.get("scale_fmt") if isinstance(quant_config, dict) else getattr(quant_config, "scale_fmt", None)
    )
    return getattr(config, "scale_fmt", None) or quant_scale_fmt


class DeepseekV32Indexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) indexer for selecting top-k tokens.

    The Indexer has its own lightweight projections (wq_b, wk) separate from the main MLA attention,
    and returns the additive top-k sparse mask directly (`0` at the selected tokens, `-inf` elsewhere);
    the raw top-k indices are only ever scattered into that mask, so they are not surfaced.

    **Cache strategy**: the indexer key cache lives on the per-layer `DynamicIndexedLayer` (or the
    `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `past_key_values.update_indexer()`.
    """

    def __init__(self, config: "DeepseekV32Config", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,  # Kept for BC
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

        The scoring logic computes:
            index_score[b,s,t] = Σ_h (weight[b,s,h] · softmax_scale · q[b,s,h,:] · k[b,t,:])

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            past_key_values: Cache object containing the indexer key cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]`. The eager / SDPA paths
                attend to exactly those tokens; the `flash-mla` kernel consumes them directly.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # The indexer uses NON-interleaved (half-split) RoPE — unlike the main MLA attention
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, D]

        # Quantized activations are model semantics here, not an optimization: the reference rotates and
        # FP8-quantizes q / k, and its key cache holds those fp8 values (we cache them dequantized). The indexer
        # always uses `ue8m0` power-of-two scales, as vLLM and SGLang do for every DSA model regardless of the
        # config; the dequantized values are then exactly representable in bf16. vLLM and SGLang's fused GLM
        # path skip the (logit-preserving) rotation and quantize directly, so engine-level top-k parity is approximate.
        q = fake_quant_fp8_block(hadamard_transform(q), scale_fmt="ue8m0")
        k = fake_quant_fp8_block(hadamard_transform(k), scale_fmt="ue8m0")

        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        # Queries are scored in chunks along the sequence so that the fp32 `[B, chunk, H, T]` score tensor
        # stays under `_INDEXER_SCORE_BUDGET` elements. Each query's top-k only depends on its own scores,
        # so chunking is exact — a single chunk reproduces the unchunked computation.
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        topk = min(self.index_topk, k.shape[-2])
        chunk = max(1, min(seq_len, _INDEXER_SCORE_BUDGET // (batch_size * self.n_heads * k.shape[-2])))

        topk_indices = []
        for start in range(0, seq_len, chunk):
            q_chunk = q[:, start : start + chunk].float()
            scores = torch.matmul(q_chunk, k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
            scores = F.relu(scores)

            # Weight per head and sum across heads: [B, chunk, 1, H] @ [B, chunk, H, T] → [B, chunk, T]
            index_scores = torch.matmul(weights[:, start : start + chunk].unsqueeze(-2), scores).squeeze(-2)

            # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
            mask_chunk = attention_mask[:, start : start + chunk]
            if attention_mask.dtype == torch.bool:
                index_scores = index_scores.masked_fill(~mask_chunk, float("-inf"))
            else:
                index_scores = index_scores + mask_chunk

            topk_indices.append(index_scores.topk(topk, dim=-1).indices.to(torch.int32))

        return torch.cat(topk_indices, dim=1)  # [B, S, topk]


def _absorbable_kv_b_weight(kv_b_proj: nn.Module) -> torch.Tensor | None:
    """The dense `[H * (qk_nope_head_dim + v_head_dim), kv_lora_rank]` weight `kv_b_proj` applies, or `None`.

    `sparse_attention_forward` absorbs this weight into the queries instead of calling `kv_b_proj`, so it
    must see the weight the module's `forward` would apply. That is only known for a plain `nn.Linear` (the
    weight may be a tensor-parallel DTensor, returned as is) and for an `FP8Linear`, whose float8 weight is
    dequantized here with its `weight_scale_inv`. Anything else (PEFT / quantization wrappers, accelerate
    offload hooks materializing the weight inside `forward`, FP4-packed weights) returns `None`.
    """
    weight = kv_b_proj.weight
    if hasattr(kv_b_proj, "_hf_hook") or weight.device.type == "meta":
        return None
    if type(kv_b_proj) is nn.Linear:
        return weight
    if type(kv_b_proj).__name__ == "FP8Linear":
        if weight.element_size() > 1:  # dequantized at load time
            return weight
        if weight.dtype != torch.float8_e4m3fn:
            return None
        scale = kv_b_proj.weight_scale_inv.float()  # per tensor, or one scale per `block_size` block
        if kv_b_proj.block_size is not None:
            block_n, block_k = kv_b_proj.block_size
            scale = scale.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)[
                : weight.shape[0], : weight.shape[1]
            ]
        return weight.float() * scale
    return None


def sparse_attention_forward(
    module: nn.Module,
    q_pass: torch.Tensor,
    q_rot: torch.Tensor,
    k_pass: torch.Tensor,
    k_rot: torch.Tensor,
    kv_b_weight: torch.Tensor,
    topk_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout: float = 0.0,
    scaling: float = 1.0,
):
    """MLA attention restricted to the tokens the DSA indexer selected, in weight-absorbed form.

    Rather than expanding every cached latent into per-head keys / values (`expand_kv`) and masking the
    unselected ones out, the queries are absorbed into the `kv_b_proj` weights and scored against the
    `K = topk_indices.shape[-1]` gathered latents only, as in the DeepSeek-V3.2 reference. Queries are
    processed in chunks so that the gathered latents stay under `_SPARSE_ATTENTION_BUDGET` elements: peak
    memory scales with `chunk * K * (kv_lora_rank + qk_rope_head_dim)` rather than with the cache length `T`.

    The attention modules only take this path for the `eager` / `sdpa` implementations, and only when
    `_absorbable_kv_b_weight` can resolve the `kv_b_proj` weight; otherwise they fall back to the dense
    `expand_kv` path with the unselected keys masked out, which computes the same probabilities.

    Args:
        q_pass / q_rot: queries, `[B, H, S, qk_nope_head_dim]` / `[B, H, S, qk_rope_head_dim]`.
        k_pass / k_rot: the cached latents, `[B, 1, T, kv_lora_rank]` / `[B, 1, T, qk_rope_head_dim]`.
        kv_b_weight: the `kv_b_proj` weight, `[H * (qk_nope_head_dim + v_head_dim), kv_lora_rank]`.
        topk_indices: the selected key positions, `[B, S, K]`.
        attention_mask: the 4D `[B, 1, S, T]` causal mask, gathered at the selected positions.

    Returns the attention output `[B, S, H, v_head_dim]` and the probabilities `[B, H, S, K]`, i.e. over the
    selected keys only (in the order of `topk_indices`), not over the `T` cached positions.
    """
    if is_dtensor(kv_b_weight):
        # Tensor parallelism: `kv_b_proj` is sharded over heads, so score against this rank's shard
        # (`q_pass` is sharded the same way). That bypasses the colwise backward which would otherwise
        # sum the latents' gradient over the mesh, so synchronize that partial gradient explicitly.
        from ...distributed.tensor_parallel import _AllReduceBackward

        mesh = kv_b_weight.device_mesh
        kv_b_weight = kv_b_weight.to_local()
        if torch.is_grad_enabled() and k_pass.requires_grad:
            k_pass = _AllReduceBackward.apply(k_pass, mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp"))

    # The head count follows from the (possibly sharded) weight rather than `module.num_heads`
    wkv_b = kv_b_weight.to(q_pass.dtype).view(-1, module.qk_nope_head_dim + module.v_head_dim, module.kv_lora_rank)
    w_uk, w_uv = wkv_b[:, : module.qk_nope_head_dim], wkv_b[:, -module.v_head_dim :]

    # Absorbing `w_uk` into the query scores the latents directly: q·(w_uk·c) == (q·w_uk)·c
    q_absorbed = torch.einsum("bhsd,hdc->bhsc", q_pass, w_uk)  # [B, H, S, kv_lora_rank]

    indices = topk_indices.long()  # [B, S, K]
    batch_size, seq_len, topk = indices.shape
    batch_indices = torch.arange(batch_size, device=indices.device).view(-1, 1, 1)
    kv_cache, pe_cache, mask = k_pass[:, 0], k_rot[:, 0], attention_mask[:, 0]
    chunk = max(
        1, min(seq_len, _SPARSE_ATTENTION_BUDGET // (batch_size * topk * (kv_cache.shape[-1] + pe_cache.shape[-1])))
    )

    attn_outputs, attn_weights = [], []
    for start in range(0, seq_len, chunk):
        chunk_slice = slice(start, start + chunk)
        chunk_indices = indices[:, chunk_slice]
        kv_selected = kv_cache[batch_indices, chunk_indices]  # [B, chunk, K, kv_lora_rank]
        pe_selected = pe_cache[batch_indices, chunk_indices]  # [B, chunk, K, qk_rope_head_dim]

        chunk_weights = (
            torch.einsum("bhsc,bskc->bhsk", q_absorbed[:, :, chunk_slice], kv_selected)
            + torch.einsum("bhsr,bskr->bhsk", q_rot[:, :, chunk_slice], pe_selected)
        ) * scaling

        # Selected keys can still be masked (causality, padding, or fewer than K valid keys). Masking with
        # `finfo.min` rather than `-inf` matches the additive masks: a fully masked query (a padding token
        # attending to nothing) then keeps finite, uniform weights instead of poisoning the batch with NaNs.
        mask_selected = mask[:, chunk_slice].gather(-1, chunk_indices).unsqueeze(1)  # [B, 1, chunk, K]
        if mask.dtype == torch.bool:
            chunk_weights = chunk_weights.masked_fill(~mask_selected, torch.finfo(chunk_weights.dtype).min)
        else:
            chunk_weights = chunk_weights + mask_selected

        chunk_weights = nn.functional.softmax(chunk_weights, dim=-1, dtype=torch.float32).to(q_pass.dtype)
        if mask.dtype == torch.bool:
            # SDPA outputs zeros for a query whose boolean mask row is all `False`; match it
            chunk_weights = chunk_weights.masked_fill(~mask_selected.any(-1, keepdim=True), 0.0)
        chunk_weights = nn.functional.dropout(chunk_weights, p=dropout, training=module.training)

        attn_outputs.append(torch.einsum("bhsk,bskc->bhsc", chunk_weights, kv_selected))  # [B, H, chunk, kv_lora_rank]
        attn_weights.append(chunk_weights)

    # A single chunk (the common case) hands back the tensors themselves rather than copies
    attn_output = attn_outputs[0] if len(attn_outputs) == 1 else torch.cat(attn_outputs, dim=2)
    attn_weights = attn_weights[0] if len(attn_weights) == 1 else torch.cat(attn_weights, dim=2)
    attn_output = torch.einsum("bhsc,hdc->bhsd", attn_output, w_uv)  # [B, H, S, v_head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DeepseekV32Attention(DeepseekV3Attention):
    """
    DeepSeek-V3 MLA, with a DSA indexer selecting the tokens each query attends to.
    Qlora rank formulation is dropped as it is never used in released models.
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # Both latents are viewed as single-head, 4D tensors, as expected by `expand_kv`
        k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
        # The reference deploys an fp8 KV cache, so it quantizes this latent (but not `k_rot`) before caching
        k_pass = fake_quant_fp8_block(k_pass, scale_fmt=_dsa_scale_fmt(self.config))
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        # Cache read / write is performed while latent KV is still compressed
        if past_key_values is not None:
            k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)

        # The indexer scores against a 3D `[B, S, T]` mask; the attention mask is 4D `[B, 1, S, T]`.
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            attention_mask[:, 0, :, :],
            position_ids,  # Kept for BC
            past_key_values=past_key_values,
        )  # [B, S, topk]

        eager_or_sdpa = self.config._attn_implementation in ("eager", "sdpa")
        kv_b_weight = _absorbable_kv_b_weight(self.kv_b_proj) if eager_or_sdpa else None
        if kv_b_weight is not None:
            attn_output, attn_weights = sparse_attention_forward(
                self,
                q_pass,
                q_rot,
                k_pass,
                k_rot,
                kv_b_weight,
                topk_indices,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )
        else:
            # Dense path: the latents are expanded into per-head keys / values, and the unselected keys are
            # masked out (eager / SDPA) or skipped by the kernel itself, which consumes the top-k indices.
            query_states = torch.cat((q_pass, q_rot), dim=-1)
            key_states, value_states = self.expand_kv(k_pass, k_rot)

            sparse_indices = None
            if eager_or_sdpa:
                # Boolean mask: `True` at keys *not* selected by the indexer (to be masked out).
                index_mask = (
                    topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                    .scatter(-1, topk_indices.long(), False)
                    .unsqueeze(1)
                )  # [B, 1, S, T]; True = masked

                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask & ~index_mask
                else:
                    attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
            else:
                sparse_indices = topk_indices

            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                indices=sparse_indices,
                **kwargs,
            )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_sdpa = True
    _supports_flex_attn = False


class DeepseekV32Model(DeepseekV3Model):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "allow_is_causal_skip": False,  # Always force creation to account for causality in the indexer
            }
            causal_mask_mapping = {"deepseek_sparse_attention": create_causal_mask(**mask_kwargs)}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
]
