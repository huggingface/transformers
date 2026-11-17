# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Building blocks shared by linear attention layers."""

import torch
import torch.nn.functional as F
from torch import nn

from .activations import ACT2FN
from .cache_utils import Cache
from .integrations import use_kernel_func_from_hub_with_fallback


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def torch_causal_conv1d_fn(
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


@use_kernel_func_from_hub_with_fallback("causal_conv1d_update", "causal_conv1d")
def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
):
    return torch_causal_conv1d_update(hidden_states, conv_state, weight, bias, activation)


@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    return torch_causal_conv1d_fn(hidden_states, weight, bias, activation, **kwargs)


# The `causal_conv1d` kernels only support these kernel widths
CAUSAL_CONV1D_KERNEL_WIDTHS = (2, 3, 4)


def causal_conv1d_forward(
    conv1d: nn.Conv1d,
    hidden_states: torch.Tensor,
    cache: Cache | None,
    layer_idx: int,
    activation: str | None = None,
    state_idx: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Causal depthwise convolution of a linear attention layer, reading and updating its conv state in `cache`.

    Args:
        conv1d (`nn.Conv1d`):
            The depthwise convolution, whose weights and kernel size are used.
        hidden_states (`torch.Tensor` of shape `(batch_size, channels, seq_len)`):
            The input to the convolution.
        cache (`Cache`, *optional*):
            The cache holding the conv state, updated in place.
        layer_idx (`int`):
            The index of the layer in `cache`.
        activation (`str`, *optional*):
            The activation applied to the output of the convolution.
        state_idx (`int`, *optional*, defaults to 0):
            Which of the layer's conv states to use.
        kwargs:
            Passed to the convolution kernel, e.g. `seq_idx` for packed sequences.

    Returns:
        `torch.Tensor` of shape `(batch_size, channels, seq_len)`: The output of the convolution.
    """
    seq_len = hidden_states.shape[-1]
    conv_state = cache.get_conv_state(layer_idx, state_idx=state_idx) if cache is not None else None
    weight = conv1d.weight.squeeze(1)
    if conv1d.kernel_size[0] in CAUSAL_CONV1D_KERNEL_WIDTHS:
        update_fn, conv_fn = causal_conv1d_update, causal_conv1d_fn
    else:
        update_fn, conv_fn = torch_causal_conv1d_update, torch_causal_conv1d_fn

    # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
    if conv_state is not None and seq_len == 1 and not cache.is_recording_past(layer_idx):
        return update_fn(hidden_states, conv_state, weight, conv1d.bias, activation=activation)

    if cache is not None:
        hidden_states = cache.update_conv_state(
            hidden_states, layer_idx, state_idx=state_idx, conv_kernel_size=conv1d.kernel_size[0]
        )
    hidden_states = conv_fn(hidden_states, weight, conv1d.bias, activation=activation, **kwargs)
    # Drop the previous states prepended by the cache
    if cache is not None:
        hidden_states = hidden_states[:, :, -seq_len:]
    return hidden_states
