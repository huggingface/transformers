# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
HELIX: Hierarchical Episodic Linear IndeX.

A decoder-only architecture that braids three sequence mixers in every block -- exact multi-scale local
attention, a gated delta-rule recurrence with a matrix state, and a hierarchical landmark index over the
whole past. Together they give linear-in-context training cost, a bounded amount of hot state at decode
time, and the ability to pull an exact value back out of the distant past.

Read :class:`HelixBraid` for the fusion, :class:`HelixRecurrentStrand` for the recurrence, and
``HelixBraid._select_memory_blocks`` for the index descent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .cache import HelixCache
from .config import HelixConfig

ACT2FN = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu, "tanh": torch.tanh}


class HelixRotaryEmbedding(nn.Module):
    """Standard rotary position embedding. Strands L and I share one set of rotated queries and keys."""

    def __init__(self, config: HelixConfig) -> None:
        super().__init__()
        self.config = config
        inverse_frequency = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.int64).float() / config.head_dim)
        )
        self.inv_freq = nn.Buffer(inverse_frequency, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inverse_frequency = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        # Force float32: rotary phases lose too much in half precision, and autocast would do just that.
        with torch.autocast(device_type=x.device.type, enabled=False):
            frequencies = (inverse_frequency.to(x.device) @ positions).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cos, sin = embedding.cos(), embedding.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class HelixRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        HelixRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# NOTE: the FLA package does not re-cast to `input_dtype` in its implementation, maybe we should do the same
class HelixRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)


class HelixMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


# NOTE: the FLA package computes `x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)` instead, so if we align
# with the GatedRMSNorm, maybe we can make that change as well.
def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes the gated delta rule, by chunking along the sequence dimension.
    Args:
        query: Query tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        key: Key tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        value: Value tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]. num_v_heads can be equal
            to num_k_heads, same for v_head_dim and k_head_dim.
        g: Decay (in log space) tensor of shape [batch_size, sequence_length, num_v_heads]: the recurrent state is
            multiplied by exp(g) at each step, so entries must be <= 0.
        beta: Beta tensor of shape [batch_size, sequence_length, num_v_heads]
        chunk_size: Size of the chunks along the sequence dimension.
        initial_state: The recurrent state, an optional tensor of shape
            [batch_size, num_v_heads, k_head_dim, v_head_dim]
        output_final_state: Whether to output the new recurrent state along with the output.
        use_qk_l2norm_in_kernel: If this flag is set to True, query and key vectors are L2-normalized.
    Returns:
        - The output tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]
        - Either None or the new recurrent state tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)  # -1 is the padded sequence length
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (in fp32 to match the FLA library)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # And always normalize queries by the head dimension
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    # Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Apply beta to K and V, which is the "learning rate" of the recurrent state for a given token, ie.
    # how much the new state influences the old one. Beta is normalized to (0, 1): 0 = no update,
    # 1 = overwrite the old state.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape all tensors to chunk the sequence dimension (adds a new dimension of size chunk_size)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # Create a chunk-sized strictly upper triangular mask, ie. the mask of what a causal chunk may not attend to
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

    # Cumulative decay within each chunk (dim 3 is the position inside the chunk). Since decay is in log space,
    # cum_decay[..., t] is the log of a product of decays between the start of the chunk and position t
    cum_decay = decay.cumsum(dim=3)

    # First phase: compute intra-chunk quantities.
    # The pairwise decays: pairwise_decay[..., i, j] = exp(cum_decay_i - cum_decay_j) is the decay accumulated between
    # positions j and i of a chunk. Positive values are masked to -inf before exp to avoid overflow
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()  # with the exp, we exit log space, so we can apply this decay to the states

    # Compute auxiliary tensors: the Upper Triangular (ut) transform system and the intra-chunk attn (QK dot product)
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Gated delta attention uses a UT transform to condense several delta rule updates into a few matmuls. After the UT
    # system is solved, we can then compute the new_values (called "u" in the DeltaNet paper) and the decayed keys
    # reading the old state (k_cumdecay). In the update, the part of new_values that the old state already predicts is
    # subtracted out, so that only the correction is written to the recurrent state: this is the delta rule.

    # Not all export targets support the fast triangular solver, so we build the inverse by forward substitution then
    if True:
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    else:
        ut_system = -ut_system.tril(-1)  # ut_system is masked to only keep the strictly lower triangle
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()
            sub = ut_system[..., :i, :i].clone()
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut_system = ut_system + torch.eye(chunk_size, dtype=ut_system.dtype, device=ut_system.device)
        new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # Apply decay once rather than in each chunk
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Second phase: the sequential scan over chunks
    for i in range(num_chunks):
        # Compute attention output for the current chunk: add the read of the previous recurrent state
        # (inter_chunk_attn) with the within-chunk attention (intra_chunk_attn)
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # Update the recurrent state: S_t+1 = decayed old state (S_t * (I - βkk^T)) + update (βvk^T)
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Reshape the output to the orignal shape: flatten the chunk dimension, then drop padding
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes linear attention using the gated delta rule, by iterating over each token in the sequence dimension.
    Same args and return value as torch_chunk_gated_delta_rule, except for `chunk_size` because the sequence dim is not
    chunked."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (done once in fp32 for better accuracy)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)

    # Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
    # it is the starting point, otherwise start with a zeroed buffer.
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    # Loop over each token and update the recurrent state
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decay the recurrent state
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        # Update the recurrent state with the current token
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class HelixRecurrentStrand(nn.Module):
    """
    Strand R: a gated delta-rule recurrence with a matrix-valued state `S ∈ R^{d_k × d_v}` per head.

    `S_t = α_t · S_{t-1} · (I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ`

    The delta rule is what makes this more than a decaying sum: writing `v_t` *removes* whatever the state
    already associates with `k_t` before adding the new value, so re-binding a key does not pile up
    interference. Training uses the chunkwise-parallel (UT transform) form, so the whole sequence is one
    batch of matmuls; decoding uses the one-token recurrence and a state whose size does not depend on how
    much text came before.

    When `config.use_surprise_gating` is set, the write strength `β_t` is additionally driven by how novel
    `k_t` is with respect to a short causal pool of the keys just before it. Tokens that repeat what the
    state has just seen write weakly; genuinely new bindings write hard. The novelty signal is a
    depthwise convolution, so it stays parallel over the sequence.
    """

    def __init__(self, config: HelixConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_recurrent_heads
        self.head_k_dim = config.recurrent_head_dim
        self.head_v_dim = config.recurrent_value_head_dim
        self.chunk_size = config.recurrent_chunk_size
        self.conv_kernel_size = config.conv_kernel_size
        self.use_surprise_gating = config.use_surprise_gating
        self.surprise_kernel_size = config.surprise_kernel_size

        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkvz = nn.Linear(config.hidden_size, self.conv_dim + self.value_dim, bias=False)
        self.in_proj_ba = nn.Linear(config.hidden_size, 2 * self.num_heads, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_heads))
        self.surprise_gamma = nn.Parameter(torch.zeros(self.num_heads))
        self.surprise_weight = nn.Buffer(self.build_surprise_weight(), persistent=False)

        self.norm = HelixRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def build_surprise_weight(self, device: torch.device | None = None) -> torch.Tensor:
        """
        Fixed (non-learned) causal mean over the `surprise_kernel_size - 1` *previous* positions. The last
        tap is zero so the pool never sees the token it is judging.
        """
        weight = torch.zeros(self.key_dim, self.surprise_kernel_size, device=device)
        weight[:, :-1] = 1.0 / max(1, self.surprise_kernel_size - 1)
        return weight

    def _novelty(self, key: torch.Tensor, past_key_values: HelixCache | None, seq_len: int) -> torch.Tensor:
        """Per-head `1 - cos(k_t, mean(k_{t-W..t-1}))`, in `[0, 2]`, computed in parallel over the sequence."""
        key_stream = key.transpose(1, 2)  # (batch, key_dim, seq_len)
        if past_key_values is not None:
            key_stream = past_key_values.update_conv_state(
                key_stream, layer_idx=self.layer_idx, state_idx=1, conv_kernel_size=self.surprise_kernel_size
            )
        pooled = causal_conv1d_fn(key_stream, self.surprise_weight, None)[:, :, -seq_len:]
        pooled = pooled.transpose(1, 2).view(-1, seq_len, self.num_heads, self.head_k_dim)
        heads = key.view(-1, seq_len, self.num_heads, self.head_k_dim)
        return 1.0 - F.cosine_similarity(heads, pooled, dim=-1, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: HelixCache | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if padding_mask is not None:
            padding_mask = padding_mask[:, -seq_len:]
        hidden_states = apply_mask_to_padding_states(hidden_states, padding_mask)

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        mixed, z = torch.split(projected_states_qkvz, [self.conv_dim, self.value_dim], dim=-1)

        # Short depthwise causal convolution: the locality prior that lets the recurrence spend its state on
        # long-range bindings instead of re-deriving n-gram structure.
        conv_input = mixed.transpose(1, 2)
        if past_key_values is not None:
            conv_input = past_key_values.update_conv_state(
                conv_input, layer_idx=self.layer_idx, state_idx=0, conv_kernel_size=self.conv_kernel_size
            )
        mixed = causal_conv1d_fn(conv_input, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation="silu")
        mixed = mixed[:, :, -seq_len:].transpose(1, 2)

        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        beta_logits, a = torch.split(projected_states_ba, [self.num_heads, self.num_heads], dim=-1)
        if self.use_surprise_gating:
            beta_logits = beta_logits + self.surprise_gamma * self._novelty(key, past_key_values, seq_len)
        beta = beta_logits.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_v_dim)

        recurrent_state = None
        if past_key_values is not None:
            recurrent_state = past_key_values.layers[self.layer_idx].recurrent_states[0]

        if seq_len == 1 and recurrent_state is not None:
            core_out, recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_out, recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=self.chunk_size,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        if past_key_values is not None:
            past_key_values.update_recurrent_state(recurrent_state, layer_idx=self.layer_idx, state_idx=0)

        core_out = self.norm(core_out, z.view(batch_size, seq_len, self.num_heads, self.head_v_dim))
        return self.out_proj(core_out.reshape(batch_size, seq_len, self.value_dim))


# Finite stand-in for -inf used when ranking landmark nodes. Kept well inside fp32 range so that adding
# learned biases on top can never produce a NaN.
NEG_SCORE = -1e30


class HelixLandmarkPooler(nn.Module):
    """
    Turns a group of vectors into one landmark. Combines an unweighted mean (what is *typically* here) with
    a learned attention pool (what is *salient* here), because a mean alone erases the rare token that a
    later query will be looking for — exactly the failure mode that sinks fixed-state models on recall.
    """

    def __init__(self, input_dim: int, landmark_dim: int, eps: float):
        super().__init__()
        self.query = nn.Parameter(torch.empty(input_dim))
        self.proj = nn.Linear(2 * input_dim, landmark_dim, bias=False)
        self.norm = HelixRMSNorm(landmark_dim, eps=eps)

    def forward(self, states: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """`states` is `(..., group, dim)`; `valid` is a broadcastable `(..., group)` boolean mask."""
        scores = (states * self.query).sum(-1)
        if valid is not None:
            scores = scores.masked_fill(~valid, NEG_SCORE)
            counts = valid.sum(-1, keepdim=True).clamp(min=1)
            mean = (states * valid.unsqueeze(-1)).sum(-2) / counts
        else:
            mean = states.mean(-2)
        salient = (scores.softmax(-1).unsqueeze(-1) * states).sum(-2)
        return self.norm(self.proj(torch.cat([mean, salient], dim=-1)))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _duplicate_mask(index: torch.Tensor) -> torch.Tensor:
    """
    Mark every repeat of a value along the last axis except its first occurrence.

    The descent can propose the same node twice — a beam that collapsed onto one node, or a frontier node
    that is also a child of the beam. Without this, `topk` happily fills the beam with copies of one node
    and silently narrows the search.
    """
    order = index.argsort(dim=-1, stable=True)
    ordered = index.gather(-1, order)
    repeated = torch.zeros_like(ordered, dtype=torch.bool)
    repeated[..., 1:] = ordered[..., 1:] == ordered[..., :-1]
    return torch.zeros_like(repeated).scatter_(-1, order, repeated)


class HelixBraid(nn.Module):
    """
    The three-strand mixer.

    Strands L and I share one set of q/k/v projections and differ only in *which* keys they read:

    * **L** reads a contiguous staircase of the last `local_blocks + 1` memory blocks, with a per-head
      window so that different heads see geometrically different spans (the multi-scale prior).
    * **I** reads `index_topk` memory blocks chosen anywhere in the past by a beam descent over a
      landmark tree. Selection is per query *block* and is driven by the hidden state at the end of the
      *previous* block, which keeps it strictly causal while letting `block_size` queries share one gather.

    Their outputs are combined by a per-token, per-head softmax gate; strand R is added on top through its
    own output projection.
    """

    def __init__(self, config: HelixConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.has_index = config.layer_types[layer_idx] == "helix"

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.block_size = config.block_size
        self.local_blocks = config.local_blocks
        self.tile_blocks = config.attention_tile_blocks

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = HelixRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = HelixRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.local_window_sizes = nn.Buffer(torch.tensor(config.local_window_sizes), persistent=False)

        if self.has_index:
            self.branching = config.index_branching
            self.beam_width = config.index_beam_width
            self.index_topk = config.index_topk
            self.landmark_dim = config.landmark_dim
            self.num_distance_buckets = config.index_num_distance_buckets
            self.leaf_pooler = HelixLandmarkPooler(self.head_dim, self.landmark_dim, config.rms_norm_eps)
            # One pooler shared by every internal level: a memory of memories is summarized the same way at
            # every scale, which both saves parameters and biases the tree towards scale invariance.
            self.node_pooler = HelixLandmarkPooler(self.landmark_dim, self.landmark_dim, config.rms_norm_eps)
            self.route_q_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.landmark_dim, bias=False)
            self.num_level_biases = config.index_max_levels + 1
            self.level_bias = nn.Parameter(torch.zeros(self.num_key_value_heads, self.num_level_biases))
            # The descent scores landmarks, which carry no rotary phase of their own, so coarse distance
            # comes from a saturating bucket table. It keeps meaning at ranges where a rotary phase has
            # wrapped far past anything seen in training. (The attention that follows the descent reads the
            # rotated keys shared with strand L, so token order inside a retrieved block is preserved.)
            self.distance_bias = nn.Parameter(torch.zeros(self.num_key_value_heads, self.num_distance_buckets))
            self.strand_gate = nn.Linear(config.hidden_size, self.num_heads * 2, bias=False)

    def _distance_bucket(self, distance: torch.Tensor) -> torch.Tensor:
        """Logarithmic bucketing of a non-negative block distance, saturating at the last bucket."""
        distance = distance.clamp(min=0)
        bucket = torch.log2(distance.float() + 1.0).floor().long()
        return bucket.clamp(max=self.num_distance_buckets - 1)

    def _build_landmark_tree(self, leaves: torch.Tensor) -> list[torch.Tensor]:
        """
        Leaves first. Level `l + 1` holds the `n_leaves // branching**(l+1)` nodes that are *complete*; a
        partially filled node is never materialized, so a node summarizes the same span whether it is built
        during training over a whole sequence or incrementally while decoding.

        The tree always grows until the top level has fewer than `branching` nodes, which is what lets the
        descent seed its beam with every top-level node. `index_max_levels` only caps how many distinct
        learned per-level biases there are, so it can never silently put part of the memory out of reach.
        """
        levels = [leaves]
        num_leaves = leaves.shape[2]
        level = 1
        while num_leaves // self.branching**level >= 1:
            num_nodes = num_leaves // self.branching**level
            children = levels[-1][:, :, : num_nodes * self.branching]
            children = children.reshape(*children.shape[:2], num_nodes, self.branching, self.landmark_dim)
            levels.append(self.node_pooler(children))
            level += 1
        return levels

    def _score_nodes(
        self,
        levels: list[torch.Tensor],
        level: int,
        node_index: torch.Tensor,
        route_query: torch.Tensor,
        query_block: torch.Tensor,
        parent_path: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score `node_index` (`(batch, kv_heads, num_query_blocks, candidates)`) against `route_query`.

        Returns `(path, rank)`. `path` accumulates the scores of the eligible nodes along the descent, so
        the leaf bias that eventually reaches the attention logits carries gradient for *every* level of
        the tree, not just the leaves. `rank` is `path` with ineligible and duplicated candidates knocked
        out; it is what the beam is selected on.

        A node is eligible for query block `J` only if the *whole* span it summarizes ends at or before `J`.
        That is what keeps the index causal: a summary is never consulted by a query it partly describes.
        Eligibility is monotone downwards (an eligible node has only eligible descendants), so a path score
        is exactly the sum over the eligible suffix of the path.
        """
        landmarks = levels[level]
        num_nodes = landmarks.shape[2]
        span = self.branching**level
        eligible = (node_index < num_nodes) & ((node_index + 1) * span <= query_block[:, None])

        safe_index = node_index.clamp(0, max(num_nodes - 1, 0))
        batch, kv_heads, num_query_blocks, candidates = safe_index.shape
        gathered = landmarks.gather(
            2,
            safe_index.reshape(batch, kv_heads, num_query_blocks * candidates, 1).expand(
                -1, -1, -1, self.landmark_dim
            ),
        ).view(batch, kv_heads, num_query_blocks, candidates, self.landmark_dim)

        score = (route_query.unsqueeze(3) * gathered).sum(-1) * (self.landmark_dim**-0.5)
        distance = query_block[:, None] - (safe_index + 1) * span
        head_index = torch.arange(kv_heads, device=score.device).view(1, kv_heads, 1, 1)
        score = score + self.distance_bias[head_index, self._distance_bucket(distance)]
        # A tree deeper than `index_max_levels` shares the last bias rather than losing those levels.
        score = score + self.level_bias[:, min(level, self.num_level_biases - 1)].view(1, kv_heads, 1, 1)

        path = parent_path + score
        rank = path.masked_fill(~eligible | _duplicate_mask(node_index), NEG_SCORE)
        return torch.where(eligible, path, torch.zeros_like(path)), rank

    def _select_memory_blocks(
        self, levels: list[torch.Tensor], route_query: torch.Tensor, query_block: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Beam descent down the landmark tree. Returns `(block_index, path_score)`, both
        `(batch, kv_heads, num_query_blocks, index_topk)`.

        At every level the candidate set is the children of the current beam **plus** the level's
        *frontier* — the eligible nodes whose parent is not itself eligible. There are at most
        `branching - 1` of those per level, and together the frontiers across all levels tile `[0, J)`
        exactly, so no reachable block is ever cut off by a beam that happened to descend elsewhere.
        Cost per query block is `O(beam_width * branching * log_branching(N))`, which is where HELIX's
        `O(N log N)` comes from.
        """
        batch, kv_heads, num_query_blocks = route_query.shape[:3]
        device = route_query.device
        top_level = len(levels) - 1
        num_slots = self.beam_width * self.branching + self.branching - 1

        def broadcast(index: torch.Tensor) -> torch.Tensor:
            return index.reshape(1, 1, *index.shape[-2:]).expand(batch, kv_heads, -1, -1)

        candidates = broadcast(torch.arange(num_slots, device=device).expand(num_query_blocks, num_slots))
        parent_path = route_query.new_zeros(candidates.shape)
        beam, beam_path = None, None
        for level in range(top_level, -1, -1):
            if beam is not None:
                children = beam.unsqueeze(-1) * self.branching + torch.arange(self.branching, device=device)
                frontier_start = (query_block // self.branching ** (level + 1)) * self.branching
                frontier = frontier_start.view(num_query_blocks, 1) + torch.arange(self.branching - 1, device=device)
                candidates = torch.cat([children.flatten(-2), broadcast(frontier)], dim=-1)
                # A frontier node has no eligible ancestor, so it starts a fresh path.
                parent_path = torch.cat(
                    [
                        beam_path.unsqueeze(-1).expand(*beam.shape, self.branching).flatten(-2),
                        beam_path.new_zeros(*beam.shape[:-1], self.branching - 1),
                    ],
                    dim=-1,
                )
            path, rank = self._score_nodes(levels, level, candidates, route_query, query_block, parent_path)
            width = self.index_topk if level == 0 else self.beam_width
            top = rank.topk(min(width, rank.shape[-1]), dim=-1)
            beam, beam_path = candidates.gather(-1, top.indices), path.gather(-1, top.indices)
        return beam, top.values

    def _local_attention(
        self,
        query_blocks: torch.Tensor,
        key_blocks: torch.Tensor,
        value_blocks: torch.Tensor,
        token_valid: torch.Tensor,
        query_block: torch.Tensor,
        kv_block_offset: int,
    ) -> torch.Tensor:
        """Strand L: per-head multi-scale window attention over the last `local_blocks + 1` memory blocks."""
        batch, _, num_query_blocks, block_size, head_dim = query_blocks.shape
        num_blocks = key_blocks.shape[2]
        span = self.local_blocks + 1

        offsets = torch.arange(-self.local_blocks, 1, device=query_blocks.device)
        gather_index = query_block[:, None] - kv_block_offset + offsets[None, :]
        in_range = (gather_index >= 0) & (gather_index < num_blocks)
        gather_index = gather_index.clamp(0, max(num_blocks - 1, 0))

        keys = key_blocks[:, :, gather_index].flatten(3, 4)
        values = value_blocks[:, :, gather_index].flatten(3, 4)
        in_range = in_range.view(1, 1, num_query_blocks, span, 1).expand(-1, -1, -1, -1, block_size).flatten(3, 4)
        valid = token_valid[:, :, gather_index].flatten(3, 4) & in_range

        # Relative distance inside the gathered window is the same for every query block, so the per-head
        # window mask is built once and broadcast.
        within = torch.arange(block_size, device=query_blocks.device)
        positions = torch.arange(span * block_size, device=query_blocks.device)
        distance = self.local_blocks * block_size + within[:, None] - positions[None, :]
        window = self.local_window_sizes.to(query_blocks.device).view(-1, 1, 1)
        head_mask = (distance >= 0) & (distance < window)
        return self._blocked_attention(query_blocks, keys, values, valid, head_mask.unsqueeze(0), None)

    def _blocked_attention(
        self,
        query_blocks: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        head_mask: torch.Tensor | None,
        logit_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Attention of `(batch, heads, num_query_blocks, block_size, head_dim)` queries against per-query-block
        gathered keys/values of `(batch, kv_heads, num_query_blocks, num_keys, head_dim)`.

        The query-block axis is folded into the batch, so peak activation memory is
        `O(N * num_keys)` rather than `O(N^2)`.
        """
        batch, num_heads, num_query_blocks, block_size, head_dim = query_blocks.shape
        num_keys = keys.shape[3]
        folded = batch * num_query_blocks
        dtype = query_blocks.dtype
        min_value = torch.finfo(dtype).min

        queries = query_blocks.permute(0, 2, 1, 3, 4).reshape(folded, num_heads, block_size, head_dim)
        keys = repeat_kv(
            keys.permute(0, 2, 1, 3, 4).reshape(folded, -1, num_keys, head_dim), self.num_key_value_groups
        )
        values = repeat_kv(
            values.permute(0, 2, 1, 3, 4).reshape(folded, -1, num_keys, head_dim), self.num_key_value_groups
        )

        # `valid` is per key/value group; it is constant along the query axis of the folded batch.
        mask = valid.permute(0, 2, 1, 3).reshape(folded, -1, 1, num_keys)
        mask = mask.repeat_interleave(self.num_key_value_groups, dim=1)
        if head_mask is not None:
            mask = mask & head_mask
        # Rows with nothing to attend to would make softmax produce NaNs, so they are left uniform here and
        # zeroed by the caller instead. Strand L carries no additive bias, so it hands SDPA the boolean mask
        # directly rather than materializing a float one -- that tensor is `O(N * num_keys)` and is the
        # largest single allocation in the block.
        if logit_bias is None:
            attn_mask = mask
        else:
            bias = logit_bias.permute(0, 2, 1, 3).reshape(folded, -1, 1, num_keys)
            bias = bias.repeat_interleave(self.num_key_value_groups, dim=1).to(dtype)
            attn_mask = torch.where(mask, bias, torch.full_like(bias, min_value).expand_as(mask))
        attn = F.scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask, scale=self.scaling)
        return attn.view(batch, num_query_blocks, num_heads, block_size, head_dim).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: HelixCache | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        block_size = self.block_size
        device = hidden_states.device
        # The memory-block grid is anchored to absolute positions, so the braid needs to know how many
        # tokens came before this chunk. That is exactly what the cache has already counted.
        past_len = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        total_len = past_len + seq_len

        query = self.q_norm(self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        kv_len = key.shape[2]
        kv_start = total_len - kv_len

        # --- align queries and keys to the memory-block grid -------------------------------------------
        query_lo = (past_len // block_size) * block_size
        left_pad = past_len - query_lo
        right_pad = -(left_pad + seq_len) % block_size
        num_query_blocks = (left_pad + seq_len + right_pad) // block_size
        query_blocks = F.pad(query, (0, 0, left_pad, right_pad)).view(
            batch_size, -1, num_query_blocks, block_size, self.head_dim
        )
        query_block = torch.arange(num_query_blocks, device=device) + query_lo // block_size

        kv_lo = (kv_start // block_size) * block_size
        kv_left = kv_start - kv_lo
        kv_right = -(kv_left + kv_len) % block_size
        num_blocks = (kv_left + kv_len + kv_right) // block_size
        kv_block_offset = kv_lo // block_size
        key_blocks = F.pad(key, (0, 0, kv_left, kv_right)).view(batch_size, -1, num_blocks, block_size, self.head_dim)
        value_blocks = F.pad(value, (0, 0, kv_left, kv_right)).view(
            batch_size, -1, num_blocks, block_size, self.head_dim
        )

        token_valid = torch.zeros(batch_size, num_blocks * block_size, dtype=torch.bool, device=device)
        token_valid[:, kv_left : kv_left + kv_len] = True
        if padding_mask is not None:
            token_valid[:, kv_left : kv_left + kv_len] &= padding_mask[:, -kv_len:].bool()
        token_valid = token_valid.view(batch_size, 1, num_blocks, block_size).expand(-1, key_blocks.shape[1], -1, -1)

        # Query blocks are processed a tile at a time. The gathered keys, values and masks are the largest
        # transient tensors in the block and they are proportional to the number of query blocks in flight,
        # so tiling turns peak activation memory into a constant while total work stays linear in `N`. It
        # changes nothing numerically -- tiles never interact.
        index_state = self._prepare_index(hidden_states, key_blocks, token_valid, past_key_values, past_len)
        tile = self.tile_blocks or num_query_blocks
        local_tiles, index_tiles, live_tiles = [], [], []
        for start in range(0, num_query_blocks, tile):
            stop = min(start + tile, num_query_blocks)
            tile_queries, tile_blocks_ids = query_blocks[:, :, start:stop], query_block[start:stop]
            local_tiles.append(
                self._local_attention(
                    tile_queries, key_blocks, value_blocks, token_valid, tile_blocks_ids, kv_block_offset
                )
            )
            if not self.has_index:
                continue
            if index_state is None:
                # Not one memory block has closed yet, so nothing is eligible and strand L covers everything.
                index_tiles.append(torch.zeros_like(tile_queries))
                live_tiles.append(tile_queries.new_zeros((*tile_queries.shape[:-1], 1), dtype=torch.bool))
            else:
                levels, route_query = index_state
                attended, live = self._index_attention(
                    levels,
                    route_query[:, :, start:stop],
                    tile_queries,
                    key_blocks,
                    value_blocks,
                    token_valid,
                    tile_blocks_ids,
                )
                index_tiles.append(attended)
                live_tiles.append(live)

        local_out = torch.cat(local_tiles, dim=2) if len(local_tiles) > 1 else local_tiles[0]

        def unpad(blocked: torch.Tensor) -> torch.Tensor:
            flat = blocked.reshape(batch_size, blocked.shape[1], num_query_blocks * block_size, blocked.shape[-1])
            return flat[:, :, left_pad : left_pad + seq_len].transpose(1, 2)

        attn_out = unpad(local_out)
        if self.has_index:
            gate = self.strand_gate(hidden_states).view(batch_size, seq_len, self.num_heads, 2).softmax(-1)
            index_out = unpad(torch.cat(index_tiles, dim=2)) * unpad(torch.cat(live_tiles, dim=2)).to(attn_out.dtype)
            attn_out = gate[..., :1] * attn_out + gate[..., 1:] * index_out

        return self.o_proj(attn_out.reshape(batch_size, seq_len, -1))

    def _prepare_index(
        self,
        hidden_states: torch.Tensor,
        key_blocks: torch.Tensor,
        token_valid: torch.Tensor,
        past_key_values: HelixCache | None,
        past_len: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor] | None:
        """
        Bring the episodic memory up to date and project the routing queries. Shared by every tile.

        Returns `None` when no memory block has closed yet. The landmark tree is only rebuilt when a block
        closes -- once every `block_size` tokens -- so its cost amortizes to `O(num_blocks / block_size)`
        per generated token.
        """
        if not self.has_index:
            return None
        batch_size, seq_len, hidden_size = hidden_states.shape
        block_size = self.block_size
        kv_heads = key_blocks.shape[1]
        total_len = past_len + seq_len
        num_leaves = total_len // block_size
        layer_cache = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
        if num_leaves == 0:
            return None

        cached_leaves = (
            0 if layer_cache is None or layer_cache.leaf_landmarks is None else layer_cache.leaf_landmarks.shape[2]
        )
        if layer_cache is not None and num_leaves == cached_leaves and layer_cache.landmark_levels:
            levels = layer_cache.landmark_levels
        else:
            new_leaves = self.leaf_pooler(
                key_blocks[:, :, cached_leaves:num_leaves], token_valid[:, :, cached_leaves:num_leaves]
            )
            leaves = new_leaves if cached_leaves == 0 else torch.cat([layer_cache.leaf_landmarks, new_leaves], dim=2)
            levels = self._build_landmark_tree(leaves)
            if layer_cache is not None:
                layer_cache.leaf_landmarks = leaves
                layer_cache.landmark_levels = levels

        # Routing query for block J is the hidden state at the end of block J - 1: strictly in the past for
        # every token of block J, so one gather serves the whole block without leaking anything.
        query_lo = (past_len // block_size) * block_size
        num_query_blocks = (
            past_len - query_lo + seq_len + -(past_len - query_lo + seq_len) % block_size
        ) // block_size
        query_block = torch.arange(num_query_blocks, device=hidden_states.device) + query_lo // block_size
        route_source = hidden_states.new_zeros(batch_size, num_query_blocks, hidden_size)
        in_chunk = query_block * block_size - 1 - past_len
        available = in_chunk >= 0
        if available.any():
            route_source[:, available] = hidden_states[:, in_chunk[available]]
        if (~available).any() and layer_cache is not None and layer_cache.route_hidden is not None:
            route_source[:, ~available] = layer_cache.route_hidden.unsqueeze(1)

        if layer_cache is not None:
            boundary = num_leaves * block_size - 1
            if boundary >= past_len:
                layer_cache.route_hidden = hidden_states[:, boundary - past_len]

        route_query = (
            self.route_q_proj(route_source)
            .view(batch_size, num_query_blocks, kv_heads, self.landmark_dim)
            .transpose(1, 2)
        )
        return levels, route_query

    def _index_attention(
        self,
        levels: list[torch.Tensor],
        route_query: torch.Tensor,
        query_blocks: torch.Tensor,
        key_blocks: torch.Tensor,
        value_blocks: torch.Tensor,
        token_valid: torch.Tensor,
        query_block: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Strand I: landmark descent, then exact attention over the selected memory blocks."""
        batch_size, kv_heads = key_blocks.shape[0], key_blocks.shape[1]
        block_size = self.block_size
        num_query_blocks = query_block.shape[0]

        selected, scores = self._select_memory_blocks(levels, route_query, query_block)

        # `topk` still returns candidates even when every one of them was masked out, so clamp before
        # gathering; `usable` below is what actually decides whether a selection counts.
        usable = scores > NEG_SCORE / 2
        flat = selected.clamp(0, key_blocks.shape[2] - 1).reshape(batch_size, kv_heads, -1)
        selected_keys = key_blocks.gather(2, flat[..., None, None].expand(-1, -1, -1, block_size, self.head_dim)).view(
            batch_size, kv_heads, num_query_blocks, -1, self.head_dim
        )
        selected_values = value_blocks.gather(
            2, flat[..., None, None].expand(-1, -1, -1, block_size, self.head_dim)
        ).view(batch_size, kv_heads, num_query_blocks, -1, self.head_dim)
        selected_valid = token_valid.gather(2, flat[..., None].expand(-1, -1, -1, block_size)).view(
            batch_size, kv_heads, num_query_blocks, -1
        )

        selected_valid = selected_valid & usable.repeat_interleave(block_size, dim=-1)
        # Feeding the routing score back in as an additive logit is what makes the discrete top-k
        # differentiable: gradients reach the landmark poolers through the blocks that were chosen.
        logit_bias = F.logsigmoid(scores.float()).to(query_blocks.dtype).repeat_interleave(block_size, dim=-1)

        index_out = self._blocked_attention(
            query_blocks, selected_keys, selected_values, selected_valid, None, logit_bias
        )
        live = selected_valid.any(-1).view(batch_size, kv_heads, num_query_blocks, 1, 1)
        live = live.repeat_interleave(self.num_key_value_groups, dim=1).expand(-1, -1, -1, block_size, 1)
        return index_out, live


class HelixDecoderLayer(nn.Module):
    def __init__(self, config: HelixConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = HelixBraid(config, layer_idx)
        self.recurrent = HelixRecurrentStrand(config, layer_idx)
        self.post_attention_layernorm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HelixMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: HelixCache | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # The two halves of the braid read the same normalized input and are summed back into the residual
        # stream, so the block stays a single parallel branch rather than two stacked sub-layers.
        mixed = self.mixer(
            hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            padding_mask=padding_mask,
            **kwargs,
        )
        mixed = mixed + self.recurrent(
            hidden_states, past_key_values=past_key_values, padding_mask=padding_mask, **kwargs
        )
        hidden_states = residual + mixed

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@dataclass
class HelixOutput:
    """What :meth:`HelixForCausalLM.forward` returns."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: HelixCache | None = None
    hidden_states: torch.Tensor | None = None


@torch.no_grad()
def init_weights(module: nn.Module, config: HelixConfig) -> None:
    """
    Initialize one module. Applied recursively by :class:`HelixModel`.

    The non-obvious ones: ``A_log`` spans several orders of magnitude so different heads start with very
    different memory half-lives, ``dt_bias`` starts at one so the recurrence writes from the first step,
    and ``surprise_gamma`` starts at zero so surprise gating begins as a no-op and has to earn its effect.
    """
    if isinstance(module, nn.Linear):
        module.weight.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight[module.padding_idx].zero_()
    elif isinstance(module, (HelixRMSNorm, HelixRMSNormGated)):
        module.weight.fill_(1.0)
    elif isinstance(module, HelixRecurrentStrand):
        module.A_log.copy_(torch.empty(module.num_heads, device=module.A_log.device).uniform_(0.01, 16).log_())
        module.dt_bias.fill_(1.0)
        module.surprise_gamma.zero_()
        module.surprise_weight.copy_(module.build_surprise_weight(module.surprise_weight.device))
    elif isinstance(module, HelixLandmarkPooler):
        module.query.normal_(mean=0.0, std=config.initializer_range)
    elif isinstance(module, HelixBraid):
        module.local_window_sizes.copy_(
            torch.tensor(config.local_window_sizes, device=module.local_window_sizes.device)
        )
        if module.has_index:
            module.level_bias.zero_()
            module.distance_bias.zero_()


class HelixModel(nn.Module):
    """The stack of HELIX blocks, without a language-modelling head."""

    def __init__(self, config: HelixConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(HelixDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HelixRotaryEmbedding(config)
        self.apply(lambda module: init_weights(module, config))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HelixCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, HelixCache | None]:
        """
        Args:
            input_ids: ``(batch, sequence)`` token ids. Exactly one of this or ``inputs_embeds``.
            attention_mask: ``(batch, sequence)`` padding mask, 1 for real tokens. HELIX builds its own
                block-structured masks, so a prepared 4D mask is rejected -- causality comes from the block
                geometry, not from the mask.
            past_key_values: cache to read and extend. Created for you when ``use_cache`` is set.
            use_cache: return a cache for incremental decoding.

        Returns:
            ``(hidden_states, cache_or_None)``.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                f"HelixModel expects a 2D padding mask (batch, sequence), got shape {tuple(attention_mask.shape)}. "
                "Causality is enforced by the block geometry, not by the mask."
            )
        if use_cache and past_key_values is None:
            past_key_values = HelixCache(self.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(seen, seen + inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                padding_mask=attention_mask,
            )
        return self.norm(hidden_states), past_key_values


class HelixForCausalLM(nn.Module):
    """HELIX with a language-modelling head. See :meth:`generate` for sampling."""

    def __init__(self, config: HelixConfig) -> None:
        super().__init__()
        self.config = config
        self.model = HelixModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        init_weights(self.lm_head, config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HelixCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool = False,
    ) -> HelixOutput:
        """Returns logits, the loss when ``labels`` are given, and the cache when ``use_cache`` is set."""
        hidden_states, cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100
            )
        return HelixOutput(logits=logits, loss=loss, past_key_values=cache, hidden_states=hidden_states)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """
        Greedy (``temperature=0``) or sampled continuation, decoded one token at a time from the cache.

        Args:
            input_ids: ``(batch, prompt)`` prompt tokens.
            max_new_tokens: how many tokens to append.
            temperature: 0 for greedy; higher is more random.
            top_k: restrict sampling to the k most likely tokens.
            eos_token_id: stop early once every sequence has produced this token.
        """
        self.eval()
        cache = HelixCache(self.config)
        output = input_ids
        step_input, step_mask = input_ids, attention_mask
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id

        for _ in range(max_new_tokens):
            logits = self(step_input, attention_mask=step_mask, past_key_values=cache, use_cache=True).logits[:, -1]
            if temperature <= 0:
                next_token = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_token = torch.multinomial(logits.softmax(-1), num_samples=1).squeeze(-1)
            if eos_token_id is not None:
                next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
                finished |= next_token == eos_token_id
            output = torch.cat([output, next_token[:, None]], dim=1)
            step_input = next_token[:, None]
            if step_mask is not None:
                step_mask = torch.cat([step_mask, torch.ones_like(next_token[:, None])], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return output

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write ``config.json`` and ``model.pt`` into ``directory``."""
        directory = Path(directory)
        self.config.save_pretrained(directory)
        torch.save(self.state_dict(), directory / "model.pt")
        return directory

    @classmethod
    def from_pretrained(cls, directory: str | Path, **overrides) -> HelixForCausalLM:
        """Rebuild a model saved by :meth:`save_pretrained`."""
        directory = Path(directory)
        config = HelixConfig.from_dict({**json.loads((directory / "config.json").read_text()), **overrides})
        model = cls(config)
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        return model

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Parameter count, for reporting."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
