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

import torch

from ..activations import ACT2FN


def causal_conv1d_varlen(
    hidden_states: torch.Tensor,
    conv_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution over a batch of sequences packed in one row, each continuing from its own state.

    This is the convolution of the linear attention layers under continuous batching: the batch is one flat row that
    mixes sequences of any length (prefill chunks and single decode tokens), and every sequence starts from the last
    inputs its previous forward pass left in the cache. Inside a sequence, the window of a token holds the previous
    inputs of that sequence; before its start, it holds the conv state. The window of the last token of each sequence
    is its next conv state.

    Args:
        hidden_states (`torch.Tensor`): The packed inputs, of shape `[1, dim, total_tokens]`.
        conv_states (`torch.Tensor`): The conv state of every sequence, of shape `[num_sequences, dim, width - 1]`.
        cu_seqlens (`torch.Tensor`): The cumulative sequence lengths, of shape `[num_sequences + 1]`.
        weight (`torch.Tensor`): The convolution weight, of shape `[dim, width]`.
        bias (`torch.Tensor`, *optional*): The convolution bias, of shape `[dim]`.
        activation (`str`, *optional*): The activation applied to the output.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The output, of shape `[1, dim, total_tokens]`, and the next conv state
        of every sequence, of shape `[num_sequences, dim, width - 1]`.
    """
    inputs = hidden_states[0].transpose(0, 1)  # [total_tokens, dim]
    states = conv_states.transpose(1, 2)  # [num_sequences, width - 1, dim]
    total_tokens, state_len = inputs.shape[0], states.shape[1]
    positions = torch.arange(total_tokens, device=inputs.device, dtype=cu_seqlens.dtype)
    # The sequence of each token and the position at which it starts. The row may be padded past the last sequence:
    # those tokens are counted in the last sequence, whose state and output they cannot reach
    sequence = torch.searchsorted(cu_seqlens[1:], positions, right=True).clamp(max=states.shape[0] - 1)
    start = cu_seqlens[sequence]

    def window(sequence: torch.Tensor, start: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """The input at `position` for the given sequences: an input of the row from the start of the sequence on,
        an entry of the conv state before. Positions are relative to the start of the state, which ends at the start
        of the sequence."""
        from_inputs = position >= start
        state_position = (position - start + state_len).clamp(0, state_len - 1)
        return torch.where(from_inputs.unsqueeze(-1), inputs[position.clamp(min=0)], states[sequence, state_position])

    # Conv1d is a sum over the window, where the last weight applies to the token itself
    output = inputs * weight[:, -1]
    for offset in range(1, state_len + 1):
        output = output + window(sequence, start, positions - offset) * weight[:, state_len - offset]
    if bias is not None:
        output = output + bias
    if activation is not None:
        output = ACT2FN[activation](output)

    # The next state of a sequence is the window of its last token, which is the state itself for an empty sequence
    last = cu_seqlens[1:] - 1  # [num_sequences]
    offsets = torch.arange(state_len, device=inputs.device)
    sequences = torch.arange(states.shape[0], device=inputs.device).unsqueeze(-1)
    next_states = window(sequences, cu_seqlens[:-1].unsqueeze(-1), last.unsqueeze(-1) - (state_len - 1 - offsets))

    return output.transpose(0, 1).unsqueeze(0).to(hidden_states.dtype), next_states.transpose(1, 2)
