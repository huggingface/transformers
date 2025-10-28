# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional
import re
from ..utils import is_accelerate_available, is_torch_accelerator_available, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn
    import triton
    import triton.language as tl
    from torch.nn import functional as F

if is_accelerate_available():
    from accelerate import init_empty_weights


logger = logging.get_logger(__name__)


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization.
    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should
        be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.
    Returns:
        torch.Tensor: The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    BLOCK_SIZE_M = 128
    if M < BLOCK_SIZE_M:
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_M = max(BLOCK_SIZE_M, 16)
    BLOCK_SIZE_K = block_k
    assert block_k % BLOCK_SIZE_K == 0
    BLOCK_SIZE_N = block_n

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
    )

    return C


# Python version of the above triton function, it's much slower than the triton version, for testing
@torch.compile
def w8a8_block_fp8_matmul_compile(
    input_q: torch.Tensor,  # [batch, seq_len, hidden_dim]
    weight_q: torch.Tensor,  # [out_features, hidden_dim]
    input_scale: torch.Tensor,  # [batch * seq_len, num_input_groups]
    weight_scale: torch.Tensor,  # [num_weight_blocks_m, num_weight_blocks_n]
    block_size: Optional[tuple[int, int]] = None,  # (M=128, N=128) for weights for example
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Performs blocked matrix multiplication with FP8 quantized matrices.

    Args:
        input_q: Quantized input tensor with 1x128 block quantization
        weight_q: Quantized weight tensor with 128x128 block quantization
        input_scale: Scaling factors for input blocks
        weight_scale: Scaling factors for weight blocks
        block_size: Tuple of (M, N) for weight block dimensions
        output_dtype: Desired output dtype
    """
    batch_size, seq_len, hidden_dim = input_q.shape if input_q.ndim == 3 else (1, input_q.shape[0], input_q.shape[1])
    out_features = weight_q.shape[0]

    # Reshape input for batched matmul
    input_reshaped = input_q.view(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
    input_scale_reshaped = input_scale.view(input_scale.shape[0], -1)  # [batch*seq_len, 1]
    # Calculate number of blocks
    num_weight_blocks_m = out_features // block_size[0]
    num_weight_blocks_n = hidden_dim // block_size[1]

    output = torch.zeros((batch_size * seq_len, out_features), dtype=torch.float32, device=input_q.device)

    for i in range(num_weight_blocks_m):
        m_start = i * block_size[0]
        m_end = m_start + block_size[0]

        for j in range(num_weight_blocks_n):
            n_start = j * block_size[1]
            n_end = n_start + block_size[1]

            # Extract current blocks
            input_block = input_reshaped[:, n_start:n_end]
            weight_block = weight_q[m_start:m_end, n_start:n_end]

            # Get corresponding scales
            curr_input_scale = input_scale_reshaped[:, j : j + 1]  # [batch*seq_len, 1]
            curr_weight_scale = weight_scale[i, j]  # scalar

            block_result = (
                torch._scaled_mm(
                    input_block,
                    weight_block.t(),
                    scale_a=torch.tensor(1, dtype=torch.float32, device=input_q.device),
                    scale_b=curr_weight_scale,
                    out_dtype=output_dtype,
                )
                * curr_input_scale
            )

            output[:, m_start:m_end] += block_result

    output = output.view(batch_size, seq_len, out_features)

    return output.to(output_dtype)


class FP8Linear(nn.Linear):
    dtype = torch.float8_e4m3fn

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        block_size: Optional[tuple[int, int]] = None,
        device=None,
        activation_scheme="dynamic",
    ):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=FP8Linear.dtype, device=device))

        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size[0] - 1) // block_size[0]
            scale_in_features = (in_features + block_size[1] - 1) // block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("weight_scale_inv", None)

        self.block_size = block_size

        self.activation_scheme = activation_scheme

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.element_size() > 1:
            return F.linear(input, self.weight, self.bias)
        else:
            # Context manager used to switch among the available accelerators
            device_type = torch.accelerator.current_accelerator().type if is_torch_accelerator_available() else "cuda"
            torch_accelerator_module = getattr(torch, device_type, torch.cuda)
            with torch_accelerator_module.device(input.device):
                qinput, scale = act_quant(input, self.block_size[1])
                output = w8a8_block_fp8_matmul_triton(
                    qinput,
                    self.weight,
                    scale,
                    self.weight_scale_inv,
                    self.block_size,
                    output_dtype=input.dtype,
                )
            # Blocks the CPU until all accelerator operations on the specified device are complete. It is used to ensure that the results of the
            # preceding operations are ready before proceeding
            torch_accelerator_module.synchronize()
            if self.bias is not None:
                output = output + self.bias
            return output.to(dtype=input.dtype)

def _ceil_div(a, b):
    return (a + b - 1) // b


class FP8Expert(nn.Parameter):
    dtype = torch.float8_e4m3fn

    def __init__(self, config, block_size, device):
        super().__init__()

        from ...activations import ACT2FN
        self.block_size = block_size
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size

        # Shapes mirror Linear(out_features, in_features)
        # gate_up: (2*intermediate, hidden) ; down: (hidden, intermediate)
        Wg_out, Wg_in = 2 * self.intermediate_dim, self.hidden_dim
        Wd_out, Wd_in = self.hidden_dim, self.intermediate_dim

        # FP8 weight tensors (packed per-expert)
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, Wg_out, Wg_in, dtype=FP8Expert.dtype, device=device)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, Wd_out, Wd_in, dtype=FP8Expert.dtype, device=device)
        )

        # Create inverse scale tiles only when using 1-byte types (fp8)
        if self.gate_up_proj.element_size() == 1:
            bo, bi = self.block_size

            # gate_up tiles: ceil(Wg_out/bo) x ceil(Wg_in/bi)
            gu_scale_o = _ceil_div(Wg_out, bo)
            gu_scale_i = _ceil_div(Wg_in, bi)
            self.gate_up_proj_scales_inv = nn.Parameter(
                torch.empty(self.num_experts, gu_scale_o, gu_scale_i, dtype=torch.float32, device=device)
            )

            # down tiles: ceil(Wd_out/bo) x ceil(Wd_in/bi)
            dp_scale_o = _ceil_div(Wd_out, bo)
            dp_scale_i = _ceil_div(Wd_in, bi)
            self.down_proj_scales_inv = nn.Parameter(
                torch.empty(self.num_experts, dp_scale_o, dp_scale_i, dtype=torch.float32, device=device)
            )
        else:
            # Match FP8Linear behavior when not using 1-byte weights
            self.register_parameter("gate_up_proj_scale_inv", None)
            self.register_parameter("down_proj_scale_inv", None)

        # (Optional) bias per projection — many MoEs omit bias; keep None to match your FP8Linear default
        self.register_parameter("gate_up_bias", None)
        self.register_parameter("down_bias", None)

        # Activation used in the MLP (same as your config / ACT2FN)
        # Keep a handle here; actual usage happens in forward of your MoE block
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()

        for expert_idx in expert_hit.tolist():
            expert_selection = expert_mask[expert_idx].squeeze(0)
            top_indices, token_positions = torch.where(expert_selection)
            if token_positions.numel() == 0:
                continue

            current_state = hidden_states.index_select(0, token_positions)
            gate, up = self.linear(current_state, self.gate_up_proj[expert_idx], self.gate_up_proj_scales[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.linear(current_hidden_states, self.down_proj[expert_idx], self.down_proj_scales[expert_idx])

            routing_weights = top_k_weights[token_positions, top_indices].unsqueeze(-1)
            current_hidden_states = current_hidden_states * routing_weights.to(current_hidden_states.dtype)
            final_hidden_states.index_add_(0, token_positions, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def linear(self, input: torch.Tensor, weight: torch.Tensor, weight_scale_inv: torch.Tensor) -> torch.Tensor:
        if weight.element_size() > 1:
            return F.linear(input, weight, self.bias)
        else:
            # Context manager used to switch among the available accelerators
            device_type = torch.accelerator.current_accelerator().type if is_torch_accelerator_available() else "cuda"
            torch_accelerator_module = getattr(torch, device_type, torch.cuda)
            with torch_accelerator_module.device(input.device):
                qinput, scale = act_quant(input, self.block_size[1])
                output = w8a8_block_fp8_matmul_triton(
                    qinput,
                    weight,
                    scale,
                    weight_scale_inv,
                    self.block_size,
                    output_dtype=input.dtype,
                )
            # Blocks the CPU until all accelerator operations on the specified device are complete. It is used to ensure that the results of the
            # preceding operations are ready before proceeding
            torch_accelerator_module.synchronize()
            if self.bias is not None:
                output = output + self.bias
            return output.to(dtype=input.dtype)

# TODO: we do need this.... but not recursive...
def _replace_with_fp8_linear(
    model,
    tp_plan=None,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """Replace Linear layers with FP8Linear."""
    if current_key_name is None:
        current_key_name = []

    iterator = list(model.named_parameters()).copy()
    for name, empty_tensor in iterator:
        current_key_name.append(name)
        name = name.rsplit(".", 1)[0] if '.' in name else name
        module = model.get_submodule(name)

        if isinstance(module, nn.Linear) and name not in (modules_to_not_convert or []) or "gate_up_proj" in name or "down_proj" in name :
            current_key_name_str = re.sub(r"\d+","*" ,".".join(current_key_name))
            if not any(key in current_key_name_str for key in (modules_to_not_convert or [])):
                with init_empty_weights():
                    if "gate_up_proj" in name or "down_proj" in name and "experts" in name: # Experts!
                        in_features = module.size(-2)
                        out_features = module.size(-1)
                        model.set_submodule(name, FP8Expert(
                            config=model.config,
                            block_size = quantization_config.weight_block_size,
                            device=module.weight.device,
                        ))

                    else:
                        in_features=module.in_features
                        out_features=module.out_features
                        model.set_submodule(name, FP8Linear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=module.bias is not None,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            activation_scheme=quantization_config.activation_scheme,
                            block_size=quantization_config.weight_block_size,
                        ))
                    has_been_replaced = True
        # when changing a layer the TP PLAN for that layer should be updated. TODO
        current_key_name.pop(-1)

    return model, has_been_replaced


def replace_with_fp8_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
):
    """Helper function to replace model layers with FP8 versions."""
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_fp8_linear(
        model,
        tp_plan=model._tp_plan,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )

    return model
