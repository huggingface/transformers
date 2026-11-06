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

from collections.abc import Callable

import torch

from ..activations import ACT2FN


def _previous_inputs_varlen(
    inputs: torch.Tensor, states: torch.Tensor, cu_seqlens: torch.Tensor
) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor]:
    """Indexing into a batch of sequences packed in one row, each continuing from the last inputs of its previous
    forward pass, which its state holds.

    Args:
        inputs (`torch.Tensor`): The packed inputs, of shape `[total_tokens, *dims]`.
        states (`torch.Tensor`): The state of every sequence, its last inputs, of shape `[num_sequences, state_len, *dims]`.
        cu_seqlens (`torch.Tensor`): The cumulative sequence lengths, of shape `[num_sequences + 1]`.

    Returns:
        The function giving, for given sequences, the input at a given row position: an input of the row from the start
        of the sequence on, an entry of the state before; then, for every token of the row, its sequence and the
        position at which that sequence starts.
    """
    total_tokens, state_len = inputs.shape[0], states.shape[1]
    positions = torch.arange(total_tokens, device=inputs.device, dtype=cu_seqlens.dtype)
    # The row may be padded past the last sequence: those tokens are counted in the last sequence, whose state and
    # output they cannot reach
    sequence = torch.searchsorted(cu_seqlens[1:], positions, right=True).clamp(max=states.shape[0] - 1)
    start = cu_seqlens[sequence]

    def input_at(sequence: torch.Tensor, start: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        from_inputs = (position >= start).view(*position.shape, *([1] * (inputs.dim() - 1)))
        state_position = (position - start + state_len).clamp(0, state_len - 1)
        return torch.where(from_inputs, inputs[position.clamp(min=0)], states[sequence, state_position])

    return input_at, sequence, start


def _next_states_varlen(input_at: Callable, cu_seqlens: torch.Tensor, state_len: int) -> torch.Tensor:
    """The next state of every sequence, its last `state_len` inputs, which is its state itself when it is empty."""
    offsets = torch.arange(state_len, device=cu_seqlens.device)
    sequences = torch.arange(cu_seqlens.shape[0] - 1, device=cu_seqlens.device).unsqueeze(-1)
    last = (cu_seqlens[1:] - 1).unsqueeze(-1)
    return input_at(sequences, cu_seqlens[:-1].unsqueeze(-1), last - (state_len - 1 - offsets))


def causal_conv1d_varlen(
    hidden_states: torch.Tensor,
    conv_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    dilation: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution over a batch of sequences packed in one row, each continuing from its own state.

    This is the convolution of the linear attention layers under continuous batching: the batch is one flat row that
    mixes sequences of any length (prefill chunks and single decode tokens), and every sequence starts from the last
    inputs its previous forward pass left in the cache. Inside a sequence, the window of a token holds the previous
    inputs of that sequence; before its start, it holds the conv state. The window of the last token of each sequence
    is its next conv state.

    Args:
        hidden_states (`torch.Tensor`): The packed inputs, of shape `[1, dim, total_tokens]`.
        conv_states (`torch.Tensor`): The conv state of every sequence, of shape
            `[num_sequences, dim, (width - 1) * dilation]`.
        cu_seqlens (`torch.Tensor`): The cumulative sequence lengths, of shape `[num_sequences + 1]`.
        weight (`torch.Tensor`): The convolution weight, of shape `[dim, width]`.
        bias (`torch.Tensor`, *optional*): The convolution bias, of shape `[dim]`.
        activation (`str`, *optional*): The activation applied to the output.
        dilation (`int`, *optional*, defaults to `1`): The spacing between the inputs the convolution weighs.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The output, of shape `[1, dim, total_tokens]`, and the next conv state
        of every sequence, of the shape of `conv_states`.
    """
    inputs = hidden_states[0].transpose(0, 1)  # [total_tokens, dim]
    states = conv_states.transpose(1, 2)  # [num_sequences, state_len, dim]
    input_at, sequence, start = _previous_inputs_varlen(inputs, states, cu_seqlens)
    positions = torch.arange(inputs.shape[0], device=inputs.device, dtype=cu_seqlens.dtype)

    # Conv1d is a sum over the window, where the last weight applies to the token itself
    width = weight.shape[-1]
    output = inputs * weight[:, -1]
    for tap in range(1, width):
        output = output + input_at(sequence, start, positions - tap * dilation) * weight[:, width - 1 - tap]
    if bias is not None:
        output = output + bias
    if activation is not None:
        output = ACT2FN[activation](output)

    next_states = _next_states_varlen(input_at, cu_seqlens, states.shape[1])
    return output.transpose(0, 1).unsqueeze(0).to(hidden_states.dtype), next_states.transpose(1, 2)


def sliding_windows_varlen(
    inputs: torch.Tensor, states: torch.Tensor, cu_seqlens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The window of every token of a batch of sequences packed in one row: its `state_len` previous inputs in its
    sequence, taken from the sequence's state before its start, then itself.

    Args:
        inputs (`torch.Tensor`): The packed inputs, of shape `[total_tokens, *dims]`.
        states (`torch.Tensor`): The last inputs of every sequence, of shape `[num_sequences, state_len, *dims]`.
        cu_seqlens (`torch.Tensor`): The cumulative sequence lengths, of shape `[num_sequences + 1]`.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The windows, of shape `[total_tokens, state_len + 1, *dims]`, and the next
        state of every sequence, of the shape of `states`.
    """
    input_at, sequence, start = _previous_inputs_varlen(inputs, states, cu_seqlens)
    state_len = states.shape[1]
    positions = torch.arange(inputs.shape[0], device=inputs.device, dtype=cu_seqlens.dtype)
    offsets = torch.arange(state_len, -1, -1, device=inputs.device, dtype=cu_seqlens.dtype)
    windows = input_at(sequence.unsqueeze(-1), start.unsqueeze(-1), positions.unsqueeze(-1) - offsets)
    return windows, _next_states_varlen(input_at, cu_seqlens, state_len)
