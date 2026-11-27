# [scratch] Triton version of `linear_attention_paged.causal_conv1d_varlen`: the depthwise causal convolution of a row of
# packed sequences, each continuing from its conv state. Compiled from torch, the gathers of the window cost ~0.74 ms a
# layer of an 8,192-token step (Qwen3-Next-80B-A3B on a B300, 8,192 channels), for ~0.27 ms of reads and writes.
import torch
import triton
import triton.language as tl


@triton.jit
def _conv_varlen_kernel(
    X, States, CuSeqlens, Weight, Bias, Out, total_tokens, num_seqs, dim,
    stride_x_t, stride_x_d, stride_s_n, stride_s_d, stride_s_w, stride_o_t, stride_o_d,
    WIDTH: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    LOG_SEQS: tl.constexpr,
):  # fmt: skip
    """One program per (BLOCK_T tokens, BLOCK_D channels). Each token finds its sequence by a binary search of the
    cumulative lengths (padding past the last sequence counts in the last one), then sums the window: the inputs from its
    sequence's start on, the conv state before."""
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = tokens < total_tokens
    channel_mask = channels < dim
    # sequence = the last s with cu_seqlens[s] <= token, in [0, num_seqs - 1]
    lo = tl.zeros([BLOCK_T], dtype=tl.int32)
    hi = tl.full([BLOCK_T], num_seqs - 1, dtype=tl.int32)
    for _ in range(LOG_SEQS):
        mid = (lo + hi + 1) // 2
        start_mid = tl.load(CuSeqlens + mid)
        go_up = start_mid <= tokens
        lo = tl.where(go_up, mid, lo)
        hi = tl.where(go_up, hi, mid - 1)
    sequence = lo
    start = tl.load(CuSeqlens + sequence)
    mask = token_mask[:, None] & channel_mask[None, :]
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(WIDTH):
        position = tokens - (WIDTH - 1) + k  # [BLOCK_T]
        from_inputs = position >= start
        x = tl.load(
            X + tl.maximum(position, 0)[:, None] * stride_x_t + channels[None, :] * stride_x_d,
            mask=mask & from_inputs[:, None], other=0.0,
        )
        state_index = tl.minimum(tl.maximum(position - start + (WIDTH - 1), 0), WIDTH - 2)
        s = tl.load(
            States + sequence[:, None] * stride_s_n + channels[None, :] * stride_s_d + state_index[:, None] * stride_s_w,
            mask=mask & (~from_inputs)[:, None], other=0.0,
        )
        w = tl.load(Weight + channels * WIDTH + k, mask=channel_mask, other=0.0).to(tl.float32)
        acc += (x.to(tl.float32) + s.to(tl.float32)) * w[None, :]
    if HAS_BIAS:
        acc += tl.load(Bias + channels, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    if SILU:
        acc = acc * tl.sigmoid(acc)
    tl.store(Out + tokens[:, None] * stride_o_t + channels[None, :] * stride_o_d, acc.to(Out.dtype.element_ty), mask=mask)


@triton.jit
def _conv_varlen_states_kernel(
    X, States, CuSeqlens, NewStates, dim,
    stride_x_t, stride_x_d, stride_s_n, stride_s_d, stride_s_w, stride_n_n, stride_n_d, stride_n_w,
    WIDTH: tl.constexpr, BLOCK_D: tl.constexpr,
):  # fmt: skip
    """One program per (sequence, BLOCK_D channels): the next state is the window of the sequence's last token, so the
    state itself for an empty sequence."""
    sequence = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    channel_mask = channels < dim
    start = tl.load(CuSeqlens + sequence)
    end = tl.load(CuSeqlens + sequence + 1)
    for j in tl.static_range(WIDTH - 1):
        position = end - (WIDTH - 1) + j
        if position >= start:
            value = tl.load(X + position * stride_x_t + channels * stride_x_d, mask=channel_mask, other=0.0)
        else:
            index = position - start + (WIDTH - 1)
            value = tl.load(
                States + sequence * stride_s_n + channels * stride_s_d + index * stride_s_w, mask=channel_mask, other=0.0
            )
        tl.store(NewStates + sequence * stride_n_n + channels * stride_n_d + j * stride_n_w, value, mask=channel_mask)


@torch.library.custom_op("transformers::conv_varlen", mutates_args=())
def conv_varlen(
    x: torch.Tensor, states: torch.Tensor, cu_seqlens: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
    silu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """`x` [tokens, dim] (any strides), `states` [sequences, dim, width - 1], `cu_seqlens` [sequences + 1], `weight`
    [dim, width] -> (output [tokens, dim], next states [sequences, dim, width - 1])."""
    total_tokens, dim = x.shape
    num_seqs, _, state_len = states.shape
    width = weight.shape[1]
    out = torch.empty(total_tokens, dim, dtype=x.dtype, device=x.device)
    new_states = torch.empty_like(states)
    block_t, block_d = 32, 128
    grid = (triton.cdiv(total_tokens, block_t), triton.cdiv(dim, block_d))
    _conv_varlen_kernel[grid](
        x, states, cu_seqlens, weight, bias if bias is not None else weight, out, total_tokens, num_seqs, dim,
        x.stride(0), x.stride(1), states.stride(0), states.stride(1), states.stride(2), out.stride(0), out.stride(1),
        WIDTH=width, HAS_BIAS=bias is not None, SILU=silu, BLOCK_T=block_t, BLOCK_D=block_d,
        LOG_SEQS=max(1, (num_seqs - 1).bit_length()),
    )  # fmt: skip
    _conv_varlen_states_kernel[(num_seqs, triton.cdiv(dim, block_d))](
        x, states, cu_seqlens, new_states, dim,
        x.stride(0), x.stride(1), states.stride(0), states.stride(1), states.stride(2),
        new_states.stride(0), new_states.stride(1), new_states.stride(2), WIDTH=width, BLOCK_D=block_d,
    )  # fmt: skip
    return out, new_states


@conv_varlen.register_fake
def _(x, states, cu_seqlens, weight, bias, silu):
    return x.new_empty(x.shape), torch.empty_like(states)
