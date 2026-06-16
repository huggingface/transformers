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
"""PyTorch Susono model.

Susono extends Qwen3-Next with two additional components:
  - Engram: conditional N-gram hash-based memory.
  - mHC-Lite: Manifold-Constrained Hyper-Connections, a multi-stream residual.

When ``use_mhc=False`` and ``use_engram=False`` the model is functionally
equivalent to Qwen3-Next.
"""

import itertools
import math
import unicodedata
from collections.abc import Callable

import torch
from torch import Tensor, nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextExperts,
    Qwen3NextForCausalLM,
    Qwen3NextForQuestionAnswering,
    Qwen3NextForSequenceClassification,
    Qwen3NextForTokenClassification,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
    Qwen3NextPreTrainedModel,
    Qwen3NextRMSNorm,
    Qwen3NextRMSNormGated,
    Qwen3NextRotaryEmbedding,
    Qwen3NextSparseMoeBlock,
    Qwen3NextTopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_susono import SusonoConfig


logger = logging.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Engram: conditional memory via scalable N-gram lookup, in batch-first format.
# ──────────────────────────────────────────────────────────────────────────────


class SusonoVocabCompressor(nn.Module):
    """Maps raw token IDs to a compressed vocabulary via NFKC normalisation.

    The lookup table is a fixed integer buffer (not a learnable parameter).
    Call `from_vocab` to build a non-trivial mapping from a token string dict.

    Args:
        base_vocab_size: Total number of tokens in the original vocabulary.
        seed: Unused, kept for interface symmetry.
    """

    def __init__(self, base_vocab_size: int, seed: int = 0) -> None:
        super().__init__()
        self.base_vocab_size = base_vocab_size
        # Default: identity mapping (overridden by from_vocab for real compression)
        mapping = torch.arange(base_vocab_size, dtype=torch.long)
        self.register_buffer("mapping", mapping)

    @classmethod
    def from_vocab(
        cls,
        vocab: dict,
        base_vocab_size: int,
        seed: int = 0,
    ) -> "SusonoVocabCompressor":
        """Build a SusonoVocabCompressor from a {token_id: token_string} dict.

        Normalisation pipeline: NFKC → NFD → strip combining marks → lower-case.
        Two token IDs that map to the same normalised form share a compressed ID.
        """
        obj = cls(base_vocab_size, seed)
        norm_to_cid: dict = {}
        mapping = torch.arange(base_vocab_size, dtype=torch.long)
        for tid, text in vocab.items():
            if not isinstance(text, str) or tid >= base_vocab_size:
                continue
            norm = unicodedata.normalize("NFKC", text)
            norm = unicodedata.normalize("NFD", norm)
            norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
            norm = norm.lower()
            if norm not in norm_to_cid:
                norm_to_cid[norm] = tid
            mapping[tid] = norm_to_cid[norm]
        obj.mapping = mapping
        return obj

    def forward(self, token_ids: Tensor) -> Tensor:
        """Compress token IDs using the normalisation lookup table.

        Args:
            token_ids: Integer tensor of shape `(batch_size, sequence_length)`.

        Returns:
            Compressed token IDs of shape `(batch_size, sequence_length)`.
        """
        return self.mapping[token_ids]


class SusonoNgramHashMapping(nn.Module):
    """Deterministic N-gram hashing using XOR-mix hash functions.

    For each N-gram order k (2 ≤ k ≤ max_ngram_size) and each hash head h:
        hash = XOR_i( t_i * m_{k,h,i} )
        index = (hash % prime_k) + head_offset_k_h

    Multipliers are seeded from (layer_id, k, h) for cross-layer independence.

    Args:
        config: SusonoConfig instance.
        layer_id: 0-indexed transformer layer ID (used to seed hashes).
    """

    _DEFAULT_PRIMES = {2: 999983, 3: 1999993, 4: 3999971}

    def __init__(self, config: SusonoConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.primes: list[int] = [config.engram_n_embed_per_ngram for k in range(2, config.engram_max_ngram_size + 1)]
        # Total rows per order = prime_k × n_head_per_ngram
        self.vocab_sizes: list[int] = [p * config.engram_n_head_per_ngram for p in self.primes]
        offsets = [0]
        for vs in self.vocab_sizes[:-1]:
            offsets.append(offsets[-1] + vs)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        multipliers = self._build_multipliers(config, layer_id)
        self.register_buffer("multipliers", multipliers)

    @staticmethod
    def _build_multipliers(config: SusonoConfig, layer_id: int) -> Tensor:
        """Generate deterministic odd multipliers for each (ngram_order, head, position)."""
        num_orders = config.engram_max_ngram_size - 1
        shape = (num_orders, config.engram_n_head_per_ngram, config.engram_max_ngram_size)
        gen = torch.Generator()
        gen.manual_seed(config.engram_seed * 10007 + layer_id * 1009)
        mults = torch.randint(1, 2**31, shape, generator=gen, dtype=torch.long)
        mults = mults * 2 + 1  # ensure odd (invertible mod 2^32)
        return mults

    def forward(self, compressed_ids: Tensor) -> Tensor:
        """Compute N-gram hash indices for all orders and heads.

        Args:
            compressed_ids: Compressed token IDs of shape `(batch_size, sequence_length)`.

        Returns:
            Hash indices of shape `(batch_size, sequence_length, total_heads)` where
            `total_heads = (max_ngram_size - 1) * n_head_per_ngram`.
        """
        B, S = compressed_ids.shape
        device = compressed_ids.device
        all_indices = []

        for order_idx, k in enumerate(range(2, self.config.engram_max_ngram_size + 1)):
            prime = self.primes[order_idx]
            offset = self.offsets[order_idx]

            # Build k-gram sequences: [B, S, k] (pad with last token for causal alignment)
            last_tok = compressed_ids[:, -1:]  # [B, 1]
            ngrams = torch.stack(
                [torch.cat([compressed_ids[:, i:], last_tok.expand(-1, i)], dim=1)[:, :S] for i in range(k)],
                dim=-1,
            )  # [B, S, k]

            # Hash multipliers for this order: [n_head, k]
            mults = self.multipliers[order_idx, :, :k]

            # XOR-mix: [B, S, n_head, k] → XOR cascade → [B, S, n_head]
            products = ngrams.unsqueeze(2) * mults.unsqueeze(0).unsqueeze(0)
            hash_val = products[..., 0]
            for pos in range(1, k):
                hash_val = hash_val ^ products[..., pos]

            # Offset into flat MultiHeadEmbedding table
            head_offset = offset + torch.arange(self.config.engram_n_head_per_ngram, device=device) * prime
            indices = (hash_val % prime) + head_offset  # [B, S, n_head]
            all_indices.append(indices)

        return torch.cat(all_indices, dim=-1)  # [B, S, total_heads]


class SusonoMultiHeadEmbedding(nn.Module):
    """Flat embedding table covering all N-gram orders and hash heads.

    Layout (row dimension):
        [head_0_ngram2 | head_1_ngram2 | … | head_0_ngram3 | …]

    Args:
        total_rows: Sum of (prime_k × n_head) for each N-gram order k.
        embed_dim: Embedding dimension per row.
    """

    def __init__(self, total_rows: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.table = nn.Embedding(total_rows, embed_dim)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, indices: Tensor) -> Tensor:
        """Look up embeddings.

        Args:
            indices: Shape `(batch_size, sequence_length, total_heads)`.

        Returns:
            Shape `(batch_size, sequence_length, total_heads, embed_dim)`.
        """
        return self.table(indices)


class SusonoShortConv(nn.Module):
    """1-D depthwise convolution along the sequence dimension.

    Fuses adjacent N-gram embeddings to capture local sequential patterns.

    Args:
        channels: Number of channels (= embed_dim after head averaging).
        kernel_size: Convolution window (default 4, matching the Engram paper).
    """

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=channels,  # depthwise
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal short convolution.

        Args:
            x: Shape `(batch_size, sequence_length, channels)`.

        Returns:
            Shape `(batch_size, sequence_length, channels)`.
        """
        x = x.permute(0, 2, 1)  # [B, C, S]
        x = self.conv(x)
        ks = self.conv.kernel_size[0]
        if ks > 1:
            x = x[..., : -(ks - 1)]  # trim right padding to preserve length S
        return x.permute(0, 2, 1)  # [B, S, C]


class SusonoEngramModule(nn.Module):
    """Engram conditional memory module (batch-first variant).

    Retrieves static N-gram memory and fuses it with current hidden states
    via context-aware gating. Designed to be called from `SusonoModel.forward`
    at selected layers before the standard decoder layer.

    Args:
        config: SusonoConfig.
        layer_id: 0-indexed transformer layer where this module is inserted.
    """

    def __init__(self, config: SusonoConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size

        self.tokenizer = SusonoVocabCompressor(config.engram_base_vocab_size, config.engram_seed)
        self.ngram_hash = SusonoNgramHashMapping(config, layer_id)

        total_rows = sum(self.ngram_hash.vocab_sizes)
        self.multi_head_emb = SusonoMultiHeadEmbedding(total_rows, config.engram_embed_dim)

        num_total_heads = (config.engram_max_ngram_size - 1) * config.engram_n_head_per_ngram
        self.short_conv = SusonoShortConv(channels=config.engram_embed_dim, kernel_size=4)

        # Collapse all heads into a single embedding vector
        self.head_proj = nn.Linear(
            num_total_heads * config.engram_embed_dim,
            config.engram_embed_dim,
            bias=False,
        )

        # Context-aware gate: hidden_states → gate weights
        self.gate_proj = nn.Linear(hidden_size, config.engram_embed_dim, bias=False)

        # Output projection; zero-init so Engram starts as identity residual
        self.out_proj = nn.Linear(config.engram_embed_dim, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, input_ids: Tensor, hidden_states: Tensor) -> Tensor:
        """Compute the Engram memory increment.

        Args:
            input_ids: Token IDs `(batch_size, sequence_length)`.
            hidden_states: Current hidden states `(batch_size, sequence_length, hidden_size)`.

        Returns:
            Memory increment `(batch_size, sequence_length, hidden_size)` to add to hidden_states.
        """
        B, S, _ = hidden_states.shape

        # 1. Compress vocabulary
        compressed = self.tokenizer(input_ids)  # [B, S]

        # 2. N-gram hash indices
        indices = self.ngram_hash(compressed)  # [B, S, total_heads]

        # 3. Multi-head embedding lookup
        emb = self.multi_head_emb(indices)  # [B, S, total_heads, n_embed]

        # 4. Flatten heads and project to single embedding
        emb = self.head_proj(emb.view(B, S, -1))  # [B, S, n_embed]

        # 5. Short convolution for local sequential fusion
        emb = self.short_conv(emb)  # [B, S, n_embed]

        # 6. Context-aware gating from hidden states
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # [B, S, n_embed]
        emb = gate * emb  # [B, S, n_embed]

        # 7. Project back to hidden size
        return self.out_proj(emb)  # [B, S, D]


# ──────────────────────────────────────────────────────────────────────────────
# mHC-Lite: Manifold-Constrained Hyper-Connections, batch-first.
# X tensor convention: [num_streams, batch_size, sequence_length, hidden_size].
# ──────────────────────────────────────────────────────────────────────────────


def _get_susono_perm_mats(n: int, device: torch.device) -> Tensor:
    """Return all n! permutation matrices [n!, n, n], cached per (n, device).

    The identity permutation is always first (index 0). The cache is stored as a
    function attribute so the helper remains self-contained.
    """
    cache = getattr(_get_susono_perm_mats, "_cache", None)
    if cache is None:
        cache = {}
        _get_susono_perm_mats._cache = cache
    key = (n, str(device))
    if key not in cache:
        perms = list(itertools.permutations(range(n)))
        idx = torch.tensor(perms, dtype=torch.long)
        eye = torch.eye(n, dtype=torch.float32)
        cache[key] = eye[idx].to(device)  # [n!, n, n]
    return cache[key]


class SusonoMHC(nn.Module):
    """mHC-Lite: Manifold-Constrained Hyper-Connections via permutation matrices.

    More memory- and compute-efficient than Sinkhorn-Knopp based mHC:
    - `H_res` is a learnable convex combination of all n! permutation matrices,
      which spans the Birkhoff polytope without iterative projection.
    - `alpha` (H_pre) and `beta` (H_post) are input-dependent (static + dynamic).

    The multi-stream state tensor `X` has shape
    `(num_streams, batch_size, sequence_length, hidden_size)`.

    Args:
        hidden_size: Dimension of each residual stream.
        num_streams: Number of parallel residual streams.
        layer_index: Layer index for the initialisation heuristic.
        sinkhorn_iterations: Unused; retained for API backward compatibility.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int,
        layer_index: int = 0,
        sinkhorn_iterations: int = 20,  # unused, kept for API compat
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.layer_index = layer_index
        n = num_streams
        num_perms = math.factorial(n)
        self.num_perms = num_perms

        # Which stream to favour at initialisation
        init_idx = layer_index % n

        # Normalise all streams concatenated: [B, S, n*D] → [B, S, n*D]
        self.norm = SusonoRMSNorm(hidden_size * n)

        # Width-connection parameters:
        #   static_alpha[:n]   → H_pre   (sigmoid gate per stream)
        #   static_alpha[n:]   → H_res   (softmax over n! permutation weights)
        init_alpha_pre = torch.ones(n) * -1.0
        init_alpha_pre[init_idx] = 1.0
        init_alpha_res = torch.ones(num_perms) * -8.0
        init_alpha_res[0] = 0.0  # identity permutation dominates at init
        self.static_alpha = nn.Parameter(torch.cat([init_alpha_pre, init_alpha_res]))

        # Dynamic (input-dependent) component: [n*D] → [n + n!]
        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(hidden_size * n, n + num_perms))
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        # Depth-connection parameters (beta / H_post)
        init_beta = torch.ones(n) * -1.0
        init_beta[init_idx] = 1.0
        self.static_beta = nn.Parameter(init_beta)

        # Dynamic component: [n*D] → [n]
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(hidden_size * n, n))
        self.h_post_scale = nn.Parameter(torch.ones(1) * 1e-2)

    def _get_norm(self, device):
        return self.norm.to(device)

    def forward(self, X: Tensor):
        """Width connection: returns `(branch_input, add_residual)`.

        Args:
            X: Multi-stream hidden states `(num_streams, batch_size, sequence_length, hidden_size)`.

        Returns:
            branch_input: Aggregated layer input `(batch_size, sequence_length, hidden_size)`.
            add_residual: Closure mapping a layer output back to the multi-stream state.
        """
        n, B, S, D = X.shape

        # Rearrange to [B, S, n, D] and flatten streams for normalisation
        X_bsd = X.permute(1, 2, 0, 3)  # [B, S, n, D]
        normed = X_bsd.reshape(B, S, n * D)  # [B, S, n*D]
        normed = self._get_norm(X.device)(normed)  # [B, S, n*D]

        # ---- Width weights (alpha) ----------------------------------------
        wc = normed @ self.dynamic_alpha_fn  # [B, S, n + n!]
        dynamic_pre = wc[..., :n]  # [B, S, n]
        dynamic_res = wc[..., n:]  # [B, S, n!]

        # H_pre: input-gated per-stream weights (sigmoid ∈ (0, 1))
        alpha_pre = torch.sigmoid(self.pre_branch_scale * dynamic_pre + self.static_alpha[:n])  # [B, S, n]

        # H_res: convex combination of permutation matrices (no Sinkhorn)
        res_coeff = torch.softmax(self.residual_scale * dynamic_res + self.static_alpha[n:], dim=-1)  # [B, S, n!]
        perms = _get_susono_perm_mats(n, X.device)  # [n!, n, n]
        H_res = torch.einsum("...r, rij -> ...ij", res_coeff, perms.to(res_coeff.dtype))  # [B, S, n, n]

        # Apply H_res: new_residuals[b,s,i,:] = Σ_j H_res[b,s,i,j] * X[b,s,j,:]
        new_residuals = torch.einsum("...ij, ...jd -> ...id", H_res, X_bsd)  # [B, S, n, D]

        # Branch input: gated weighted sum over streams
        branch_input = (alpha_pre.unsqueeze(-1) * X_bsd).sum(dim=-2)  # [B, S, D]

        # ---- Depth weights (beta) ----------------------------------------
        dc = normed @ self.dynamic_beta_fn  # [B, S, n]
        beta = torch.sigmoid(self.h_post_scale * dc + self.static_beta) * 2  # [B, S, n]  (range [0, 2])

        def add_residual(x_out: Tensor) -> Tensor:
            # x_out: [B, S, D]; beta: [B, S, n]; new_residuals: [B, S, n, D]
            output = beta.unsqueeze(-1) * x_out.unsqueeze(-2) + new_residuals  # [B, S, n, D]
            return output.permute(2, 0, 1, 3)  # [n, B, S, D]

        return branch_input, add_residual

    def aggregate_streams(self, X: Tensor) -> Tensor:
        """[Legacy] Aggregate n streams → single branch input."""
        branch_input, add_residual = self.forward(X)
        self._cached_add_residual = add_residual
        return branch_input

    def distribute_output(self, X: Tensor, x_out: Tensor) -> Tensor:
        """[Legacy] Update n streams using cached depth-connection closure."""
        add_residual = self._cached_add_residual
        self._cached_add_residual = None
        return add_residual(x_out)


# ──────────────────────────────────────────────────────────────────────────────
# Base transformer components (adapted from Qwen3-Next for SusonoConfig).
# ──────────────────────────────────────────────────────────────────────────────


class SusonoRMSNormGated(Qwen3NextRMSNormGated):
    pass


class SusonoRMSNorm(Qwen3NextRMSNorm):
    pass


class SusonoRotaryEmbedding(Qwen3NextRotaryEmbedding):
    pass


class SusonoMLP(Qwen3NextMLP):
    pass


class SusonoGELUMLP(nn.Module):
    """Dense GELU MLP for full-attention layers. Matches Megatron swiglu=False (fc1/fc2)."""

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))


class SusonoExperts(Qwen3NextExperts):
    pass


class SusonoTopKRouter(Qwen3NextTopKRouter):
    pass


class SusonoGatedDeltaNet(Qwen3NextGatedDeltaNet):
    pass


class SusonoAttention(nn.Module):
    """Multi-headed attention with an optional gated query projection and optional QK-norm.

    Both the output gate (`config.attention_output_gate`) and QK-LayerNorm
    (`config.qk_layernorm`) are configurable, unlike Qwen3-Next where both are
    always applied.
    """

    def __init__(self, config: SusonoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_output_gate = getattr(config, "attention_output_gate", False)
        q_out_dim = config.num_attention_heads * self.head_dim * (2 if self.use_output_gate else 1)
        self.q_proj = nn.Linear(config.hidden_size, q_out_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        if getattr(config, "qk_layernorm", False):
            self.q_norm = SusonoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = SusonoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.use_output_gate:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            gate = None
        if self.q_norm is not None:
            query_states = self.q_norm(query_states.view(hidden_shape))
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class SusonoSparseMoeBlock(Qwen3NextSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        # Susono adds a learnable bias on the shared-expert gate.
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=True)


class SusonoDecoderLayer(Qwen3NextDecoderLayer):
    pass


class SusonoPreTrainedModel(Qwen3NextPreTrainedModel):
    _no_split_modules = ["SusonoDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, SusonoSparseMoeBlock):
            if module.shared_expert_gate.bias is not None:
                bias_init = getattr(self.config, "moe_shared_expert_gate_bias_init", 0.0)
                init.constant_(module.shared_expert_gate.bias, bias_init)
        elif isinstance(module, SusonoMHC):
            n = module.num_streams
            init_idx = module.layer_index % n
            alpha_pre = torch.full((n,), -1.0)
            alpha_pre[init_idx] = 1.0
            alpha_res = torch.full((module.num_perms,), -8.0)
            alpha_res[0] = 0.0  # identity permutation dominates at init
            init.copy_(module.static_alpha, torch.cat([alpha_pre, alpha_res]))
            init.zeros_(module.dynamic_alpha_fn)
            init.constant_(module.pre_branch_scale, 1e-2)
            init.constant_(module.residual_scale, 1e-2)
            beta = torch.full((n,), -1.0)
            beta[init_idx] = 1.0
            init.copy_(module.static_beta, beta)
            init.zeros_(module.dynamic_beta_fn)
            init.constant_(module.h_post_scale, 1e-2)
        elif isinstance(module, SusonoNgramHashMapping):
            init.copy_(module.multipliers, module._build_multipliers(module.config, module.layer_id))
            offsets = [0]
            for vs in module.vocab_sizes[:-1]:
                offsets.append(offsets[-1] + vs)
            init.copy_(module.offsets, torch.tensor(offsets, dtype=torch.long))
        elif isinstance(module, SusonoVocabCompressor):
            init.copy_(module.mapping, torch.arange(module.base_vocab_size, dtype=torch.long))
        elif isinstance(module, SusonoEngramModule):
            # Zero-init the output projection so Engram starts as an identity residual.
            init.zeros_(module.out_proj.weight)
        elif isinstance(module, SusonoModel) and getattr(module, "stream_proj", None) is not None:
            # Final mHC stream aggregation initialised as an equal-weight average.
            n = module.config.mhc_num_streams
            d = module.config.hidden_size
            w = torch.zeros(d, n * d)
            for s in range(n):
                w[:, s * d : (s + 1) * d] = torch.eye(d) / n
            init.copy_(module.stream_proj.weight, w)


class SusonoModel(SusonoPreTrainedModel):
    """Susono transformer model with optional Engram memory and mHC multi-stream residuals.

    mHC multi-stream flow (when `use_mhc=True`): n parallel residual streams `X`
    of shape `(num_streams, batch_size, sequence_length, hidden_size)` are all
    initialised from the token embeddings. For each layer the streams are
    aggregated into a single branch input, run through the decoder layer, then
    redistributed. When `use_mhc=False`, the model is functionally equivalent to
    Qwen3-Next (with Engram optionally added to selected layers).
    """

    def __init__(self, config: SusonoConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [SusonoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = SusonoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SusonoRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # ── mHC: one connection module per layer ──────────────────────
        if config.use_mhc:
            self.mhc_modules = nn.ModuleList(
                [
                    SusonoMHC(
                        hidden_size=config.hidden_size,
                        num_streams=config.mhc_num_streams,
                        layer_index=layer_idx,
                        sinkhorn_iterations=config.mhc_sinkhorn_iterations,
                    )
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
            # Final stream aggregation, initialised as an equal-weight average.
            n = config.mhc_num_streams
            D = config.hidden_size
            self.stream_proj = nn.Linear(n * D, D, bias=False)
            with torch.no_grad():
                w = torch.zeros(D, n * D)
                for s in range(n):
                    w[:, s * D : (s + 1) * D] = torch.eye(D) / n
                self.stream_proj.weight.copy_(w)

        # ── Engram: one module per selected layer ─────────────────────
        if config.use_engram:
            self.engram_modules = nn.ModuleList(
                [SusonoEngramModule(config, layer_id=layer_id) for layer_id in config.engram_layer_ids]
            )
            self._engram_layer_map: dict[int, int] = {
                layer_id: idx for idx, layer_id in enumerate(config.engram_layer_ids)
            }

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Engram requires input_ids; disable silently when only embeds are given
        engram_active = self.config.use_engram and input_ids is not None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # ── Initialise hidden states ──────────────────────────────────
        hidden_states = inputs_embeds  # [B, S, D]

        if self.config.use_mhc:
            # n parallel residual streams, all starting from token embeddings: [n, B, S, D]
            X = hidden_states.unsqueeze(0).expand(self.config.mhc_num_streams, -1, -1, -1).clone()

        # ── Layer loop ────────────────────────────────────────────────
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            # mHC: width connection — get branch input and depth-connection closure
            if self.config.use_mhc:
                x_in, add_residual = self.mhc_modules[layer_idx](X)  # [B,S,D], closure
            else:
                x_in = hidden_states  # [B, S, D]

            # Engram: inject conditional N-gram memory at selected layers
            if engram_active and layer_idx in self._engram_layer_map:
                engram_idx = self._engram_layer_map[layer_idx]
                x_in = x_in + self.engram_modules[engram_idx](input_ids, x_in)

            # Standard transformer layer
            x_out = decoder_layer(
                x_in,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

            # mHC: depth connection — update n streams via closure
            if self.config.use_mhc:
                X = add_residual(x_out)  # [n, B, S, D]
            else:
                hidden_states = x_out

        # ── Final aggregation (mHC) ───────────────────────────────────
        if self.config.use_mhc:
            n_s, B_s, S_s, D_s = X.shape
            X_flat = X.permute(1, 2, 0, 3).reshape(B_s, S_s, n_s * D_s)
            hidden_states = self.stream_proj(X_flat)

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        """
        NOTE: Left-padding is used for the linear attention mask. No need to zero
        states when (1) doing a cached forward, or (2) attending to all inputs.
        """
        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask


class SusonoForCausalLM(Qwen3NextForCausalLM):
    pass


class SusonoForSequenceClassification(Qwen3NextForSequenceClassification):
    pass


class SusonoForTokenClassification(Qwen3NextForTokenClassification):
    pass


class SusonoForQuestionAnswering(Qwen3NextForQuestionAnswering):
    pass


__all__ = [
    "SusonoForCausalLM",
    "SusonoForQuestionAnswering",
    "SusonoForSequenceClassification",
    "SusonoForTokenClassification",
    "SusonoModel",
    "SusonoPreTrainedModel",
]
