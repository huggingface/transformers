# Copyright 2026 The HuggingFace Inc. team.
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
"""Alternative functions used in linear attention models for when the sequences are packed together."""
import torch

from ..activations import ACT2FN


def packed_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    max_seqlen_q: int,
    cu_seq_lens_q: torch.Tensor,
    read_index: torch.Tensor,
    has_previous_state: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs,
) -> torch.Tensor:
    if max_seqlen_q == 1:
        print("BORDEL MAIS DECODING IS NOT IN YET!!!")
    return conv1d_and_update_varlen(
        hidden_states, conv_state, weight, bias, cu_seq_lens_q, read_index, has_previous_state, activation
    )


def conv1d_and_update_varlen(
    hidden_states: torch.Tensor,                     # vLLM equivalent is "x"
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cu_seq_lens_q: torch.Tensor,                     # vLLM equivalent is "query_start_loc"
    read_index: torch.Tensor,                        # vLLM equivalent is "cache_indices"
    has_previous_state: torch.Tensor | None = None,  # vLLM equivalent is "has_initial_state"
    activation: str | None = None,
) -> torch.Tensor:
    """Depthwise causal conv1d over packed sequences, with per-sequence conv states that are updated in place.

    Args:
        hidden_states: Packed inputs of shape [conv_dim, num_tokens]. Tokens past `cu_seq_lens_q[-1]` are padding.
        weight: Conv weight of shape [conv_dim, conv_kernel_size].
        bias: Optional conv bias of shape [conv_dim].
        conv_state: Cache of shape [cache_size, state_len, conv_dim], updated in place. Since the conv only sees
            conv_kernel_size at a time, state_len = conv_kernel_size - 1. Like vLLM, it is expected to be the transpose
            of a [cache_size, state_len, conv_dim] tensor (channels-last), which avoids copies.
        cu_seq_lens_q: Cumulative sequence lengths, shape [batch_size + 1].
        read_index: Physical index of each cached state, shape [batch_size].
        has_previous_state: Optional bool tensor of shape [batch_size] indicating if the state in cache is initialized.
            If None, it is assumed that all states are initialized.
        activation: Name of the activation applied to the output or `None` for no activation.

    Returns:
        The output of shape [conv_dim, num_tokens], in the dtype of `hidden_states`.
    """
    conv_dim, num_tokens = hidden_states.shape
    conv_kernel_size = weight.shape[-1]
    state_len = conv_kernel_size - 1
    batch_size = cu_seq_lens_q.shape[0] - 1
    device = hidden_states.device
    cu_seq_lens_q = cu_seq_lens_q.long()
    read_index = read_index.long()

    # For each token, find its sequence index and its position in that sequence. Padding tokens are considered part of
    # the last sequence.
    token_idx = torch.arange(num_tokens, device=device)
    seq_idx = torch.searchsorted(cu_seq_lens_q[1:], token_idx, right=True).clamp_(max=batch_size - 1)
    positions = token_idx - cu_seq_lens_q[seq_idx]

    # Gather old states
    old_states = conv_state[read_index]  # [batch_size, state_len, conv_dim]
    # Zero out un-initialized states (TODO: do once at start of fwd)
    if has_previous_state is not None:
        old_states = old_states * has_previous_state[:, None, None].to(old_states.dtype)
    # Flatten the first two dims to get a token-major view
    num_state_tokens = batch_size * state_len
    old_states = old_states.reshape(num_state_tokens, conv_dim)
    # Attach the old_states to the hidden_states so we can gather tokens from one tensor
    all_states = torch.cat([old_states, hidden_states.transpose(0, 1).to(old_states.dtype)])

    # Perform the convolution in fp32 into a zero-init buffer
    out = torch.zeros((num_tokens, conv_dim), dtype=torch.float32, device=device)
    for i in range(conv_kernel_size):
        current_idx = conv_kernel_size - 1 - i
        relative_pos = positions - current_idx
        rows = torch.where(
            relative_pos >= 0,
            num_state_tokens + relative_pos,
            (seq_idx + 1) * state_len + relative_pos,
        )
        out += all_states[rows].float() * weight[:, i].float()

    # Apply bias and activation if needed
    out = out + bias.float() if bias is not None else out
    out = ACT2FN[activation](out) if activation is not None else out

    # New state = last state_len inputs of (old state ++ new tokens), gathered before writing back
    # TODO: can this be optimized / simplified?
    seqlens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]  # [batch_size]
    rel_pos = seqlens[:, None] - state_len + torch.arange(state_len, device=device)[None]  # [batch_size, state_len]
    new_token_rows = num_state_tokens + cu_seq_lens_q[:-1, None] + rel_pos
    old_state_rows = torch.arange(1, batch_size + 1, device=device)[:, None] * state_len + rel_pos
    rows = torch.where(rel_pos >= 0, new_token_rows, old_state_rows)
    new_states = all_states[rows]  # [batch_size, state_len, conv_dim]
    conv_state.index_copy_(0, read_index, new_states.to(conv_state.dtype))

    return out.t().to(hidden_states.dtype)


# def causal_conv1d_update(
#     hidden_states: torch.Tensor,             # vLLM equivalent is "x"
#     conv_state: torch.Tensor,
#     weight: torch.Tensor,
#     bias: torch.Tensor | None = None,
#     activation: str | None = None,
#     read_index: torch.Tensor | None = None,  # vLLM equivalent is "conv_state_indices"
# ) -> torch.Tensor:
#     """Depthwise causal conv1d for decoding, where every sequence has the same number of new tokens (usually one).

#     Args:
#         hidden_states: New inputs of shape [batch_size, conv_dim] or [batch_size, conv_dim, seqlen].
#         conv_state: Cache of shape [cache_size, conv_dim, state_len], updated in place. Since the conv only sees
#             conv_kernel_size at a time, state_len = conv_kernel_size - 1. Like vLLM, it is expected to be the transpose
#             of a [cache_size, state_len, conv_dim] tensor (channels-last), which avoids copies.
#         weight: Conv weight of shape [conv_dim, conv_kernel_size].
#         bias: Optional conv bias of shape [conv_dim].
#         activation: Name of the activation applied to the output (e.g. "silu"), or `None` for no activation.
#         read_index: Physical index of each cached state, shape [batch_size]. Defaults to the sequence index.

#     Returns:
#         The output, with the same shape and dtype as `hidden_states`.
#     """
#     unsqueeze = hidden_states.dim() == 2
#     inputs = hidden_states.unsqueeze(-1) if unsqueeze else hidden_states
#     batch_size, _, seqlen = inputs.shape
#     conv_kernel_size = weight.shape[-1]
#     state_len = conv_state.shape[-1]

#     # Window of the last conv_kernel_size - 1 + seqlen inputs of each sequence
#     if read_index is not None:
#         read_index = read_index.long()  # index_copy_ needs int64 indices
#     old_states = conv_state[read_index] if read_index is not None else conv_state[:batch_size]
#     window = torch.cat([old_states, inputs.to(old_states.dtype)], dim=-1)  # [batch_size, conv_dim, state_len + seqlen]

#     # Convolution: each output token sees the conv_kernel_size inputs ending at it. Accumulate in fp32 like the kernels
#     taps = window[..., state_len - (conv_kernel_size - 1) :].unfold(-1, conv_kernel_size, 1)
#     out = torch.einsum("bdlw,dw->bdl", taps.float(), weight.float())
#     if bias is not None:
#         out += bias.float()[:, None]
#     if activation is not None:
#         out = ACT2FN[activation](out)

#     # New state = last state_len inputs of (old state ++ new tokens)
#     new_states = window[..., seqlen:].to(conv_state.dtype)
#     if read_index is not None:
#         conv_state.index_copy_(0, read_index, new_states)
#     else:
#         conv_state[:batch_size].copy_(new_states)

#     out = out.to(hidden_states.dtype)
#     return out.squeeze(-1) if unsqueeze else out
