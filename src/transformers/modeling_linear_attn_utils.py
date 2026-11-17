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
from .integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub_with_fallback


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


@use_kernel_forward_from_hub("causal_conv1d_forward")
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
    Layers list it in `use_kernelized_func`, so that `kernelize` can replace it, e.g. with a runtime that manages
    the conv state itself.

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


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


@use_kernel_func_from_hub_with_fallback("selective_state_update", "mamba_ssm")
def ssd_selective_state_update(
    state: torch.Tensor,
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    z: torch.Tensor | None = None,
    **kwargs,
):
    batch_size, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[1]
    state_size = B.shape[-1]

    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = dt[..., None]

    # Discretize A
    dA = torch.exp(dt.float() * A.float()).to(device=state.device)

    # Discretize B
    B = B.reshape(batch_size, num_groups, 1, state_size)
    B = B.expand(batch_size, num_groups, num_heads // num_groups, state_size).contiguous()
    B = B.reshape(batch_size, num_heads, 1, state_size)
    dB = dt * B

    # Discretize x into dB
    dBx = (dB * hidden_states[..., None]).to(device=state.device)

    # State calculation
    ssm_states = state * dA + dBx
    state.copy_(ssm_states.to(state.dtype))

    # Subsequent output
    C = C.reshape(batch_size, num_groups, 1, state_size)
    C = C.expand(batch_size, num_groups, num_heads // num_groups, state_size).contiguous()
    C = C.reshape(batch_size, num_heads, state_size)

    # Reshape ssm_states to merge the first two dimensions
    ssm_states = ssm_states.to(device=C.device, dtype=C.dtype)
    ssm_states_reshaped = ssm_states.view(batch_size * num_heads, head_dim, state_size)
    C_reshaped = C.view(batch_size * num_heads, state_size, 1)
    out = torch.bmm(ssm_states_reshaped, C_reshaped)
    out = out.view(batch_size, num_heads, head_dim)

    # D skip connection
    if D is not None:
        out = (out + hidden_states * D).to(out.dtype)

    if z is not None:
        out = out * F.silu(z)

    return out.to(hidden_states.dtype)


@use_kernel_func_from_hub_with_fallback("mamba_chunk_scan_combined", "mamba_ssm")
def ssd_chunk_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    **kwargs,
):
    batch_size, sequence_length, num_heads, head_dim = hidden_states.shape
    num_groups = B.shape[2]

    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])

    hidden_states = hidden_states.float()
    B = B.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)
    C = C.float().repeat_interleave(num_heads // num_groups, dim=2, output_size=num_heads)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    D_residual = None
    if D is not None:
        D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None].float()
    A = A.to(hidden_states.dtype) * dt.float()

    # Rearrange into blocks/chunks. This fixes the layout the einsums below are written against:
    # b = batch, c = chunk index, l and s = positions within a chunk, h = head, p = head_dim,
    # n = ssm state. So hidden_states is (b, c, l, h, p) and B and C are (b, c, l, h, n).
    hidden_states, A, B, C = [reshape_into_chunks(tensor, pad_size, chunk_size) for tensor in (hidden_states, A, B, C)]

    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    # This is the analog of a causal mask
    L = torch.exp(segment_sum(A))

    # Contraction of C and B to get G (attention-weights like): sum over the state n, leaving a
    # position-by-position score per head. (b,c,l,h,n) x (b,c,s,h,n) -> (b,c,l,s,h).
    G = torch.einsum("bclhn,bcshn->bclsh", C, B)

    # Compute M, equivalent to applying attention mask to weights
    M = G * L.permute(0, 2, 3, 4, 1)

    # Compute Y_diag (apply to values): sum over the source position s. (b,c,l,s,h) x (b,c,s,h,p)
    # -> (b,c,l,h,p).
    Y_diag = torch.einsum("bclsh,bcshp->bclhp", M, hidden_states)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    # Sum over the positions l in the chunk: (b,c,l,h,n) x (b,c,l,h,p) -> (b,c,h,p,n).
    states = torch.einsum("bclhn,bclhp->bchpn", B_decay, hidden_states)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    previous_states = (
        initial_states[:, None].to(dtype=states.dtype, device=states.device)
        if initial_states is not None
        else torch.zeros_like(states[:, :1])
    )
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    # Sum over the source chunk z to get each chunk's start state: (b,z,c,h) x (b,z,h,p,n) -> (b,c,h,p,n).
    new_states = torch.einsum("bzch,bzhpn->bchpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    # Sum over the state n again, now against each chunk's start state: (b,c,l,h,n) x (b,c,h,p,n)
    # -> (b,c,l,h,p).
    C_times_states = torch.einsum("bclhn,bchpn->bclhp", C, states)
    Y_off = C_times_states * state_decay_out.permute(0, 2, 3, 1)[..., None]

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks).
    # `contiguous` because einsum may return a permuted view, and callers such as FalconH1Mixer
    # call `.view()` on this return value, which a permuted view refuses.
    output = (Y_diag + Y_off).contiguous()
    output = output.reshape(batch_size, -1, num_heads, head_dim)

    if D_residual is not None:
        output = output + D_residual

    # Cutting off padded chunks
    if pad_size > 0:
        output = output[:, :sequence_length]

    if return_final_states:
        return output, final_state

    return output


@use_kernel_forward_from_hub("ssd_forward")
def ssd_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    recurrent_state: torch.Tensor | None,
    cache: Cache | None,
    **kwargs,
) -> torch.Tensor:
    """
    Selective scan of a Mamba2 layer (state space duality, SSD), reading and updating its recurrent state in `cache`.
    Layers list it in `use_kernelized_func`, so that `kernelize` can replace it, e.g. with a runtime that manages the
    recurrent state itself.

    Args:
        module (`nn.Module`):
            The Mamba2 mixer, from which `A_log`, `D`, `dt_bias`, `time_step_limit`, `chunk_size`, `layer_idx` and the
            layer geometry are read.
        hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, num_heads * head_dim)`):
            The input to the scan.
        B (`torch.Tensor` of shape `(batch_size, seq_len, n_groups * ssm_state_size)`):
            The input projection of the state space model.
        C (`torch.Tensor` of shape `(batch_size, seq_len, n_groups * ssm_state_size)`):
            The output projection of the state space model.
        dt (`torch.Tensor` of shape `(batch_size, seq_len, num_heads)`):
            The time step, before `dt_bias` and softplus.
        recurrent_state (`torch.Tensor`, *optional*):
            The layer's recurrent state in `cache` from before this forward pass, or `None` if it has none.
        cache (`Cache`, *optional*):
            The cache holding the recurrent state, updated in place.
        kwargs:
            Passed to the scan kernels, e.g. `seq_idx` for packed sequences.

    Returns:
        `torch.Tensor` of shape `(batch_size, seq_len, num_heads * head_dim)`: The output of the scan, including the
        `D` skip connection.
    """
    batch_size, seq_len, _ = hidden_states.shape
    A = -torch.exp(module.A_log.float())

    if recurrent_state is not None and seq_len == 1:
        hidden_states = hidden_states.view(batch_size, module.num_heads, module.head_dim)
        dt = dt.transpose(1, 2).expand(-1, -1, module.head_dim)
        A = A[:, None, None].expand(-1, module.head_dim, module.ssm_state_size)
        B = B.view(batch_size, module.n_groups, module.ssm_state_size)
        C = C.view(batch_size, module.n_groups, module.ssm_state_size)
        D = module.D[:, None].expand(-1, module.head_dim)
        dt_bias = module.dt_bias[:, None].expand(-1, module.head_dim)
        scan_output = ssd_selective_state_update(
            recurrent_state, hidden_states, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True, **kwargs
        )
        return scan_output.view(batch_size, 1, -1)

    output_final_state = cache is not None
    scan_result = ssd_chunk_scan(
        hidden_states.view(batch_size, seq_len, module.num_heads, module.head_dim),
        dt,
        A,
        B.view(batch_size, seq_len, module.n_groups, module.ssm_state_size),
        C.view(batch_size, seq_len, module.n_groups, module.ssm_state_size),
        chunk_size=module.chunk_size,
        D=module.D,
        z=None,
        return_final_states=output_final_state,
        dt_bias=module.dt_bias,
        dt_softplus=True,
        initial_states=recurrent_state,
        dt_limit=module.time_step_limit,
        **kwargs,
    )
    if output_final_state:
        scan_output, final_state = scan_result
        cache.update_recurrent_state(final_state, layer_idx=module.layer_idx)
    else:
        scan_output = scan_result
    return scan_output.reshape(batch_size, seq_len, -1)
