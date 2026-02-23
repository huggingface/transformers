# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from functools import wraps

from ..utils import logging
from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available, is_torchdynamo_compiling


if is_torch_available():
    import torch
    import triton
    import triton.language as tl
    from torch.library import wrap_triton


logger = logging.get_logger(__name__)

# Examples of experts class with its eager mm implementation
# class Experts(torch.nn.Module):
#     """Collection of expert weights stored as 3D tensors."""

#     def __init__(self, config):
#         super().__init__()
#         self.num_experts = config.n_routed_experts
#         self.hidden_dim = config.hidden_size
#         self.intermediate_dim = config.moe_intermediate_size
#         self.gate_up_proj = torch.nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
#         self.down_proj = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         top_k_index: torch.Tensor,
#         top_k_weights: torch.Tensor,
#     ) -> torch.Tensor:
#         final_hidden_states = torch.zeros_like(hidden_states)
#         with torch.no_grad():
#             expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
#             expert_mask = expert_mask.permute(2, 1, 0)
#             expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

#         for expert_idx in expert_hit:
#             expert_idx = expert_idx[0]
#             if expert_idx == self.num_experts:
#                 continue
#             top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
#             current_state = hidden_states[token_idx]
#             gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
#             current_hidden_states = self.act_fn(gate) * up
#             current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
#             current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
#             final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

#         return final_hidden_states


def _batched_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Batched linear layer supporting optional bias and transposed weights.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (batch_size, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (batch_size, output_dim, input_dim) if transposed is `False`,
            else of shape (batch_size, input_dim, output_dim).
        bias (`torch.Tensor`, *optional*):
            Bias tensor of shape (batch_size, output_dim). Default is `None`.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the weight tensor is transposed.
    Returns:
        `torch.Tensor`: Output tensor of shape (batch_size, output_dim).
    """
    if is_transposed:
        # (batch_size, 1, input_dim) @ (batch_size, input_dim, output_dim) -> (batch_size, 1, output_dim) -> (batch_size, output_dim)
        out = torch.bmm(input.unsqueeze(1), weight).squeeze(1)
    else:
        # (batch_size, output_dim, input_dim) @ (batch_size, input_dim, 1) -> (batch_size, output_dim, 1) -> (batch_size, output_dim)
        out = torch.bmm(weight, input.unsqueeze(-1)).squeeze(-1)

    if bias is not None:
        out = out + bias

    return out


def batched_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # Reshape for easier indexing
    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Handle invalid expert IDs from Expert Parallelism (EP)
    # When EP is enabled, tokens assigned to experts on other devices are marked with sentinel value >= num_experts
    valid_mask = expert_ids < self.num_experts
    expert_ids_clamped = expert_ids.clamp(0, self.num_experts - 1)

    # Get current hidden states for selected samples
    selected_hidden_states = hidden_states[token_idx]

    # Select expert weights and biases for selected samples (using clamped IDs for safe indexing)
    selected_gate_up = self.gate_up_proj[expert_ids_clamped]
    selected_down = self.down_proj[expert_ids_clamped]
    selected_gate_up_bias = self.gate_up_proj_bias[expert_ids_clamped] if self.has_bias else None
    selected_down_bias = self.down_proj_bias[expert_ids_clamped] if self.has_bias else None

    # --- Up projection per expert (batched) ---
    gate_up_out = _batched_linear(
        selected_hidden_states, selected_gate_up, selected_gate_up_bias, is_transposed=self.is_transposed
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (batched) ---
    out_per_sample = _batched_linear(
        gated_out, selected_down, selected_down_bias, is_transposed=self.is_transposed
    )  # (S, hidden_dim)

    # Apply routing weights and zero out invalid expert contributions
    if sample_weights.shape != expert_ids_clamped.shape:
        sample_weights = sample_weights.gather(0, expert_ids_clamped)
    out_per_sample = out_per_sample * sample_weights.unsqueeze(-1)  # (S, hidden_dim)
    out_per_sample = out_per_sample * valid_mask.unsqueeze(-1).to(out_per_sample.dtype)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


# torch.compiler.disable does not work with fullgraph=True, so we implement a custom operator to opaque this function.
# This is not "free compilation compatibility" because now inductor won't be able to optimize matmuls inside the loop,
# but since the matmuls here have dynamic shapes, inductor wouldn't have been able to optimize them anyway.
def _grouped_mm_fallback(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """
    Fallback grouped matrix multiplication used when `torch.nn.functional.grouped_mm` and `torch._grouped_mm`
    are unavailable or incompatible with `torch.compile` (e.g. non-bfloat16 weights).

    Args:
        input (`torch.Tensor`): Input of shape (S, input_dim), sorted by expert id.
        weight (`torch.Tensor`): Expert weights of shape (num_experts, input_dim, output_dim).
        offs (`torch.Tensor`): Cumulative token counts per expert of shape (num_experts,).
    Returns:
        `torch.Tensor`: Output of shape (S, output_dim).
    """
    output = torch.zeros(input.size(0), weight.size(2), device=input.device, dtype=input.dtype)  # (S, output_dim)

    start = 0
    # single cpu<->gpu sync point here,
    # avoids multiple syncs inside the loop
    for i, end in enumerate(offs.tolist()):
        if start == end:
            continue
        torch.mm(input[start:end], weight[i], out=output[start:end])
        start = end

    return output


def _grouped_mm_fallback_fake(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Shape/dtype inference stub for `_grouped_mm_fallback` required by `torch.compile`."""
    assert input.dim() == 2, f"input must be 2D (S, input_dim), got shape {tuple(input.shape)}"
    assert weight.dim() == 3, (
        f"weight must be 3D (num_experts, input_dim, output_dim), got shape {tuple(weight.shape)}"
    )
    assert offs.dim() == 1, f"offs must be 1D (num_experts,), got shape {tuple(offs.shape)}"
    assert offs.size(0) == weight.size(0), f"offs length {offs.size(0)} must match number of experts {weight.size(0)}"
    assert input.size(1) == weight.size(1), (
        f"input_dim mismatch: input has {input.size(1)}, weight has {weight.size(1)}"
    )
    assert offs.dtype in (torch.int32, torch.int64), f"offs must be an integer tensor, got {offs.dtype}"
    return torch.empty(input.size(0), weight.size(2), device=input.device, dtype=input.dtype)


def _grouped_mm_fallback_setup_context(ctx, inputs, output):
    """Saves input and weight for backward; offs is stored directly as it is a non-differentiable integer tensor."""
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.offs = inputs[2]


def _grouped_mm_fallback_backward(ctx, grad_output):
    """Backward pass for `_grouped_mm_fallback`. Computes grad_input and grad_weight per expert group; offs has no gradient."""
    input, weight = ctx.saved_tensors
    grad_input = torch.zeros_like(input)
    grad_weight = torch.zeros_like(weight)

    start = 0
    # single cpu<->gpu sync point here,
    # avoids multiple syncs inside the loop
    for i, end in enumerate(ctx.offs.tolist()):
        if start == end:
            continue
        torch.mm(grad_output[start:end], weight[i].T, out=grad_input[start:end])
        torch.mm(input[start:end].T, grad_output[start:end], out=grad_weight[i])
        start = end

    return grad_input, grad_weight, None


if is_torch_available():
    torch.library.custom_op("transformers::grouped_mm_fallback", _grouped_mm_fallback, mutates_args=())
    torch.library.register_fake("transformers::grouped_mm_fallback", _grouped_mm_fallback_fake)
    torch.library.register_autograd(
        "transformers::grouped_mm_fallback",
        _grouped_mm_fallback_backward,
        setup_context=_grouped_mm_fallback_setup_context,
    )


def _can_use_grouped_mm(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> bool:
    """
    Check if torch.nn.functional.grouped_mm or torch._grouped_mm can be used based on availability and compatibility with torch.compile.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, input_dim, output_dim).
        offs (`torch.Tensor`):
            Offsets tensor indicating the boundaries of each group in the input tensor.
    Returns:
        `bool`: True if grouped_mm can be used, False otherwise.
    """
    if is_torchdynamo_compiling() and weight.dtype != torch.bfloat16:
        # torch.grouped_mm is not supported in torch.compile with dtypes other than bfloat16
        return False

    return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")


def _grouped_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """Grouped matrix multiplication dispatcher that uses torch.nn.functional.grouped_mm if available, else falls back to torch._grouped_mm.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, input_dim, output_dim).
        offs (`torch.Tensor`):
            Offsets tensor indicating the boundaries of each group in the input tensor.
    Returns:
        `torch.Tensor`: Output tensor of shape (S, output_dim).
    """

    if _can_use_grouped_mm(input, weight, offs):
        # torch.nn.functional.grouped_mm and torch._grouped_mm are not autocast-enabled,
        # when autocast is enabled we can end up with intermediate tensors in fp32 (e.g. LayerNorm output) and weight tensors in bf16
        # In that case we need to cast the input to the weight dtype to avoid dtype mismatch errors.
        # See: https://github.com/pytorch/pytorch/issues/174763
        if hasattr(torch.nn.functional, "grouped_mm"):
            return torch.nn.functional.grouped_mm(input.to(weight.dtype), weight, offs=offs)
        elif hasattr(torch, "_grouped_mm"):
            return torch._grouped_mm(input.to(weight.dtype), weight, offs=offs)

    return torch.ops.transformers.grouped_mm_fallback(input, weight, offs=offs)


def _grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Grouped linear layer supporting optional bias and transposed weights.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, input_dim, output_dim) if `is_transposed`,
            else of shape (num_experts, output_dim, input_dim).
        offs (`torch.Tensor`):
            Offsets tensor indicating the boundaries of each group in the input tensor.
        bias (`torch.Tensor`, *optional*):
            Bias tensor of shape (num_experts, output_dim). Default is `None`.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the weight tensor is transposed.
    Returns:
        `torch.Tensor`: Output tensor of shape (S, output_dim).
    """
    if is_transposed:
        # (S, input_dim) @ grouped (num_experts, input_dim, output_dim) -> (S, output_dim)
        out = _grouped_mm(input, weight, offs=offs)
    else:
        # (S, input_dim) @ grouped (num_experts, output_dim, input_dim).T -> (S, output_dim)
        out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)

    if bias is not None:
        # We should be able to pass bias to the grouped_mm call, but it's not yet supported.
        out = out + bias

    return out


def grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # Reshape for easier indexing
    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Get current hidden states for selected samples
    selected_hidden_states = hidden_states[token_idx]

    # Sort by expert for grouped processing.
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    # Select expert weights and biases for selected samples
    # NOTE: We keep all experts here and rely on offsets to target the active ones.
    # I have already implemented a version that only passes the active experts, but
    # to do so I had to use torch.unique which breaks the graph capture (data-dependent).
    # Also there were no speedup gains from it in my experiments, even in eager mode.
    selected_gate_up = self.gate_up_proj
    selected_down = self.down_proj
    selected_gate_up_bias = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
    selected_down_bias = self.down_proj_bias[expert_ids_g] if self.has_bias else None

    # Compute offsets for grouped_mm
    # using histc instead of bincount to avoid cuda graph issues
    # With deterministic algorithms, CPU only supports float input, CUDA only supports int input.
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # --- Up projection per expert (grouped) ---
    gate_up_out = _grouped_linear(
        selected_hidden_states_g,
        selected_gate_up,
        offs=offsets,
        bias=selected_gate_up_bias,
        is_transposed=self.is_transposed,
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (grouped) ---
    out_per_sample_g = _grouped_linear(
        gated_out,
        selected_down,
        offs=offsets,
        bias=selected_down_bias,
        is_transposed=self.is_transposed,
    )  # (S, hidden_dim)

    # Apply routing weights
    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)

    # Restore original order
    out_per_sample = out_per_sample_g[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


class ExpertsInterface(GeneralInterface):
    """Interface for registering custom experts implementations."""

    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
    }

    def get_interface(self, experts_implementation: str, default: Callable) -> Callable:
        """Return the requested `experts_implementation`. Also strictly check its validity, and raise if invalid."""
        if experts_implementation is None:
            logger.warning_once(
                "You tried to access the `ExpertsInterface` with a `config._experts_implementation` set to `None`. This "
                "is expected if you use an Expert Module as a standalone Module. If this is not the case, something went "
                "wrong with the dispatch of `config._experts_implementation`"
            )
        elif experts_implementation != "eager" and experts_implementation not in self:
            raise KeyError(
                f"`{experts_implementation}` is not a valid experts implementation registered in the `ExpertsInterface`"
            )
        return super().get(experts_implementation, default)


ALL_EXPERTS_FUNCTIONS = ExpertsInterface()


def _default_apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
    """
    Default gating mechanism: splits the gate_up_out into gate and up parts,
    applies the activation function to the gate part, and multiplies it with the up part.
    Args:
        gate_up_out (`torch.Tensor`):
            The output tensor from the gate and up projection of shape (S, 2 * intermediate_dim).
    Returns:
        `torch.Tensor`: The gated output tensor of shape (S, intermediate_dim).
    """
    gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
    return self.act_fn(gate) * up  # (S, intermediate_dim)


def use_experts_implementation(
    experts_class: type[torch.nn.Module] | None = None, *, is_transposed: bool = False, has_bias: bool = False
) -> type[torch.nn.Module]:
    """Decorator to modify experts class to support different experts implementations.

    Args:
        experts_class (`type[torch.nn.Module]`, *optional*):
            The experts class to modify. If not provided, returns a decorator that can be applied to the class.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the expert weights are stored in transposed format.
        has_bias (`bool`, *optional*, defaults to `False`):
            Whether the expert layers include bias terms.

    Returns:
        `type[torch.nn.Module]`: The modified experts class.
    """

    def wrapper(experts_class: type[torch.nn.Module]) -> type[torch.nn.Module]:
        original_init = experts_class.__init__
        original_forward = experts_class.forward

        @wraps(original_init)
        def __init__(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            self.config = config
            self.has_bias = has_bias
            self.is_transposed = is_transposed

        @wraps(original_forward)
        def forward(self, *args, **kwargs):
            experts_forward = ALL_EXPERTS_FUNCTIONS.get_interface(
                self.config._experts_implementation, original_forward
            )
            return experts_forward(self, *args, **kwargs)

        if not hasattr(experts_class, "_apply_gate"):
            experts_class._apply_gate = _default_apply_gate
        experts_class.__init__ = __init__
        experts_class.forward = forward
        return experts_class

    if experts_class is not None:
        return wrapper(experts_class)

    return wrapper


@triton.jit
def _w8a8_block_fp8_matmul_batched_fused(
    A,  # (S, K)  raw BF16/FP16 activations — fused: quantized inline
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N)  output
    Bs,  # (E, N // group_n, K // group_k) weight scales
    ExpertIds,  # (S,) — which expert each batch element routes to
    # Shape
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Per-row strides
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    stride_Bs_k,
    stride_Bs_n,
    # Batch / expert strides
    stride_Ab,  # stride between rows in A (one token per program)
    stride_Eb,  # stride between experts in B
    stride_Cb,
    stride_Esb,  # stride between experts in Bs
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)

    # Advance base pointers to this token's activation row and its expert's
    # weight / scale slice.  No pre-gather of weights needed.
    expert_id = tl.load(ExpertIds + batch_id)
    A = A + batch_id * stride_Ab
    B = B + expert_id * stride_Eb
    C = C + batch_id * stride_Cb
    Bs = Bs + expert_id * stride_Esb

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # M=1: broadcast the single activation row to BLOCK_SIZE_M identical rows
    # so tl.dot gets the required (BLOCK_SIZE_M, BLOCK_SIZE_K) shape.
    # BLOCK_SIZE_M=16 (set by wrapper) — smallest legal FP8 WGMMA tile, matching
    # adaptive eager behaviour for M=1 and minimising register pressure.
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ---- fused act_quant (replaces: a = tl.load(a_ptrs); a_s = tl.load(As_ptrs)) ----
        a_raw = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw)) / 448.0  # per-block scale (scalar for M=1)
        a = (a_raw / tl.maximum(a_s, 1e-12)).to(tl.float8e4nv)
        # ---- same as baseline from here ----
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    # Only write row 0 (M=1); the broadcast rows are discarded.
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < 1) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_block_fp8_matmul_batched_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    """Batched FP8 block-wise matmul with fused activation quantization.

    Mirrors ``_batched_linear`` for FP8 weights: A is the raw (BF16/FP16)
    activation matrix, B / Bs are the stacked expert weights / scales.
    The kernel looks up ``expert_ids[batch_id]`` to address the correct expert
    slice of B directly — no (S, N, K) weight gather is needed.
    Activation quantization (``act_quant``) is fused into the matmul loop.
    """
    assert A.ndim == 2, "A must be (S, K)"
    assert A.is_contiguous()
    S, K = A.shape
    E, N, _ = B.shape

    assert B.is_contiguous()
    assert B.shape[-1] == K, "K dimension mismatch"
    assert Bs.shape[0] == E
    assert expert_ids.shape[0] == S

    C = A.new_empty(S, N)

    # Adaptive BLOCK_SIZE_M: match the tile to the average tokens per expert —
    # same heuristic as the grouped kernel so all three paths stay in sync.
    # Pure integer arithmetic, no GPU sync, CUDA-graph safe.
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)

    grid = (triton.cdiv(N, block_n), S)
    wrap_triton(_w8a8_block_fp8_matmul_batched_fused)[grid](
        A,
        B,
        C,
        Bs,
        expert_ids,
        N,
        K,
        block_n,
        block_k,
        A.stride(1),  # stride_ak
        B.stride(2),  # stride_bk
        B.stride(1),  # stride_bn
        C.stride(1),  # stride_cn
        Bs.stride(2),  # stride_Bs_k
        Bs.stride(1),  # stride_Bs_n
        A.stride(0),  # stride_Ab
        B.stride(0),  # stride_Eb
        C.stride(0),  # stride_Cb
        Bs.stride(0),  # stride_Esb
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )

    return C


@triton.jit
def _w8a8_block_fp8_grouped_mm_fused(
    A,  # (S, K)  raw BF16/FP16 activations, sorted by expert id
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N)  output
    Bs,  # (E, N // group_n, K // group_k) weight scales
    Offsets,  # (E,) int32 — cumulative row-end per expert
    TileOffsets,  # (E,) int32 — cumulative tile-end per expert
    # Shape
    S,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Strides
    stride_am,
    stride_ak,
    stride_Eb,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_Esb,
    stride_Bsk,
    stride_Bsn,
    # Meta-parameters
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Grouped FP8 block-wise matmul with fused activation quantization.

    Mirrors ``_grouped_linear`` / ``_grouped_mm`` for FP8 weights.
    Activation quantization (``act_quant``) is fused into the matmul loop,
    eliminating the separate HBM round-trip for quantized activations and their
    scales.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Exit early for programs beyond the actual tile count.
    total_tiles = tl.load(TileOffsets + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    # Binary search in TileOffsets to find the owning expert.
    # Finds the smallest e such that TileOffsets[e] > pid_m (upper_bound semantics),
    # which is the expert whose tile range contains pid_m.
    # O(log2(NUM_EXPERTS)) loads instead of the O(NUM_EXPERTS) linear scan.
    # NUM_EXPERTS.bit_length() is ceil(log2(E))+1 for powers-of-two, giving one
    # harmless extra iteration when lo==hi; it's a compile-time constant so the
    # loop is fully unrolled by the compiler.
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(NUM_EXPERTS.bit_length()):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets + mid)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    expert_id = lo

    prev_eid = tl.maximum(expert_id - 1, 0)

    expert_start = tl.where(expert_id == 0, 0, tl.load(Offsets + prev_eid))
    expert_end = tl.load(Offsets + expert_id)
    M_expert = expert_end - expert_start

    expert_tile_start = tl.where(expert_id == 0, 0, tl.load(TileOffsets + prev_eid))
    local_tile = pid_m - expert_tile_start
    m_off = local_tile * BLOCK_SIZE_M

    offs_am = m_off + tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs_am < M_expert
    offs_global_m = expert_start + offs_am

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_bn_safe = offs_bn % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_am_safe = offs_global_m % S

    a_ptrs = A + offs_am_safe[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + expert_id * stride_Eb + offs_k[:, None] * stride_bk + offs_bn_safe[None, :] * stride_bn
    offs_bsn_safe = offs_bn_safe // group_n
    Bs_ptrs = Bs + expert_id * stride_Esb + offs_bsn_safe * stride_Bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ---- fused act_quant (replaces: a = tl.load(a_ptrs); a_s = tl.load(As_ptrs)) ----
        a_raw = tl.load(a_ptrs, mask=row_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0).to(
            tl.float32
        )
        a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0  # per-row scale  (BLOCK_SIZE_M,)
        # clamp denominator so masked all-zero rows don't produce NaN
        # (their a_s multiplier is 0 anyway, so the output row is correct)
        a = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)
        # ---- same as baseline from here ----
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bsk)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    c_mask = row_mask[:, None] & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_block_fp8_matmul_grouped_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    """Grouped FP8 block-wise matmul with fused activation quantization.

    Mirrors ``_grouped_linear`` / ``_grouped_mm`` for FP8 weights: A is the
    raw (BF16/FP16) activation matrix sorted by expert, B / Bs are the stacked
    expert weights / scales.  Activation quantization (``act_quant``) is fused
    into the matmul loop.  ``tokens_per_expert`` is needed (in addition to
    ``offsets``) to build the per-expert tile schedule inside the kernel.
    """
    S, K = A.shape
    E, N, _ = B.shape

    assert A.is_contiguous() and B.is_contiguous()
    assert Bs.is_contiguous()
    assert offsets.is_contiguous()

    C = A.new_empty(S, N)

    # Adaptive BLOCK_SIZE_M: match tile to average tokens per expert.
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    tiles_per_expert = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    # Upper bound on M-tiles: sum_e ceil(M_e / BLOCK_M) <= ceil(S / BLOCK_M) + E.
    # Using a static upper bound keeps the grid size data-independent, which is
    # required for cuda-graph compatibility.  Programs beyond the real tile count
    # exit immediately via the early-return guard inside the kernel.
    max_M_tiles = triton.cdiv(S, BLOCK_SIZE_M) + E

    grid = (max_M_tiles * triton.cdiv(N, block_n),)
    wrap_triton(_w8a8_block_fp8_grouped_mm_fused)[grid](
        A,
        B,
        C,
        Bs,
        offsets,
        tile_offsets,
        S,
        N,
        K,
        block_n,
        block_k,
        A.stride(0),  # stride_am
        A.stride(1),  # stride_ak
        B.stride(0),  # stride_Eb
        B.stride(2),  # stride_bk
        B.stride(1),  # stride_bn
        C.stride(0),  # stride_cm
        C.stride(1),  # stride_cn
        Bs.stride(0),  # stride_Esb
        Bs.stride(2),  # stride_Bsk
        Bs.stride(1),  # stride_Bsn
        NUM_EXPERTS=E,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )

    return C


if is_torch_available():
    torch.library.triton_op(
        "transformers::w8a8_block_fp8_matmul_batched_fused",
        w8a8_block_fp8_matmul_batched_fused,
        mutates_args=(),
    )
    torch.library.triton_op(
        "transformers::w8a8_block_fp8_matmul_grouped_fused",
        w8a8_block_fp8_matmul_grouped_fused,
        mutates_args=(),
    )


def fp8_batched_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # Reshape for easier indexing
    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Get current hidden states for selected samples
    selected_hidden_states = hidden_states[token_idx]

    # --- Up projection per expert (FP8 batched) ---
    gate_up_out = torch.ops.transformers.w8a8_block_fp8_matmul_batched_fused(
        selected_hidden_states,
        self.gate_up_proj,
        self.gate_up_proj_scale_inv,
        expert_ids,
        self.block_size[0],
        self.block_size[1],
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 batched) ---
    out_per_sample = torch.ops.transformers.w8a8_block_fp8_matmul_batched_fused(
        gated_out,
        self.down_proj,
        self.down_proj_scale_inv,
        expert_ids,
        self.block_size[0],
        self.block_size[1],
    )  # (S, hidden_dim)

    # Apply routing weights
    out_per_sample = out_per_sample * sample_weights.to(out_per_sample.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


def fp8_grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    selected_hidden_states = hidden_states[token_idx]

    # Sort by expert for grouped processing.
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    # Compute offsets for grouped processing.
    # histc instead of bincount avoids cuda-graph issues;
    # CPU requires float input, CUDA requires int input (deterministic mode).
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # --- Up projection per expert (FP8 grouped) ---
    gate_up_out = torch.ops.transformers.w8a8_block_fp8_matmul_grouped_fused(
        selected_hidden_states_g,
        self.gate_up_proj,
        self.gate_up_proj_scale_inv,
        offsets,
        tokens_per_expert,
        self.block_size[0],
        self.block_size[1],
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 grouped) ---
    out_per_sample_g = torch.ops.transformers.w8a8_block_fp8_matmul_grouped_fused(
        gated_out,
        self.down_proj,
        self.down_proj_scale_inv,
        offsets,
        tokens_per_expert,
        self.block_size[0],
        self.block_size[1],
    )  # (S, hidden_dim)

    # Apply routing weights
    out_per_sample_g = out_per_sample_g * sample_weights_g.to(out_per_sample_g.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # Restore original order
    out_per_sample = out_per_sample_g[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)
