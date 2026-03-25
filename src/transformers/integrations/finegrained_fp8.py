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
import torch
import torch.nn as nn
import triton
from torch.nn import functional as F

from ..activations import ACT2FN
from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_kernels_available, is_torch_available, logging
from .hub_kernels import get_kernel
from .moe import ExpertsInterface, use_experts_implementation


logger = logging.get_logger(__name__)


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max

# Global for the Triton quantization kernel (lazily compiled)
_triton_kernel = None
_triton_kernel_available = None


# Global for the CUTLASS quantization kernel (lazily loaded)
_cutlass_kernel = None
_cutlass_kernel_available = None


def _get_triton_kernel():
    """Lazily compile the Triton quantization kernel."""
    global _triton_kernel, _triton_kernel_available

    if _triton_kernel_available is None:
        # this kernel is universal so should be usable independently of torch version
        _triton_kernel = get_kernel("kernels-community/finegrained-fp8")
        _triton_kernel_available = True

    return _triton_kernel


def _get_cutlass_kernel():
    """Lazily load the CUTLASS quantization kernel from HuggingFace Hub."""
    global _cutlass_kernel, _cutlass_kernel_available

    if _cutlass_kernel_available is None:
        try:
            # this kernel's build was not updated since torch 2.8
            _cutlass_kernel = get_kernel("RedHatAI/quantization")
            _cutlass_kernel_available = True
        except Exception as e:
            logger.warning_once(f"Failed to load CUTLASS quantization kernel: {e}. Falling back to Triton.")
            _cutlass_kernel_available = False

    return _cutlass_kernel


def _supports_cutlass(
    A: torch.Tensor,
    B: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype,
) -> bool:
    """
    Check if CUTLASS blockwise FP8 matmul is supported for the given inputs, output dtype, and block size.

    CUTLASS blockwise kernels require:
    - SM90+ (Hopper or newer)
    - Block size [128, 128] for weights
    - Block size [1, 128] for activations (handled implicitly)
    - Output dtype bfloat16 or float16
    - K and N divisible by 16
    """

    if torch.compiler.is_compiling():
        # the checks after this, using importlib fail during torch.compile :/
        return False

    if not is_torch_available() or not torch.cuda.is_available() or not is_kernels_available():
        return False

    # CUTLASS only supports bfloat16/float16 output
    if output_dtype not in (torch.bfloat16, torch.float16):
        return False

    # Check block size compatibility - CUTLASS only supports [128, 128]
    if block_size is None:
        return False
    if len(block_size) != 2 or block_size[0] != 128 or block_size[1] != 128:
        return False

    # CUTLASS requires K and N divisible by 16
    K, N = A.shape[-1], B.shape[0]
    if K % 16 != 0 or N % 16 != 0:
        return False

    # Check GPU capability (SM90+)
    capability = torch.cuda.get_device_capability()
    cuda_capability = capability[0] * 10 + capability[1]

    # Try to load the kernel and check if blockwise FP8 is supported
    kernel = _get_cutlass_kernel()
    if kernel is None:
        return False

    try:
        return kernel.cutlass_scaled_mm_supports_block_fp8(cuda_capability)
    except Exception:
        return False


def w8a8_block_fp8_matmul_cutlass(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Call the CUTLASS blockwise FP8 matmul kernel.

    Handles all layout conversions required by CUTLASS:
      - A:  [M, K]           row-major    float8_e4m3fn
      - B:  [K, N]           column-major float8_e4m3fn
      - As: [M,  K//128]     M-major      (stride(0)==1)
      - Bs: [K//128, N//128] K-major      (stride(0)==1)
    """
    kernel = _get_cutlass_kernel()

    original_shape = A.shape
    M = A.numel() // A.shape[-1]
    K = A.shape[-1]
    N = B.shape[0]

    A_2d = A.view(M, K).contiguous()
    # B is [N, K] row-major; CUTLASS needs [K, N] column-major (stride(0)==1).
    # .contiguous().t() gives [K, N] with stride=(1, K) — do NOT call .contiguous() after!
    B_col_major = B.contiguous().t()

    # As: reshape to [M, K//128], then force M-major layout via t().contiguous().t()
    As_2d = As.view(M, -1).contiguous()
    As_2d = As_2d.t().contiguous().t()  # [M, K//128] with stride(0)==1

    # Bs: our layout is [N//128, K//128]; CUTLASS needs [K//128, N//128] K-major (stride(0)==1)
    Bs_km = Bs.contiguous().t()  # [K//128, N//128]
    Bs_km = Bs_km.t().contiguous().t()  # force K-major (stride(0)==1)

    C = kernel.cutlass_scaled_mm(A_2d, B_col_major, As_2d, Bs_km, output_dtype, None)
    return C.view(original_shape[:-1] + (N,))


def w8a8_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dispatch to CUTLASS or Triton for block-wise FP8 matmul.

    Uses CUTLASS when:
    - Block size is [128, 128] (the only size CUTLASS supports)
    - Running on SM90+ (Hopper or newer)
    - The CUTLASS kernel is available
    - Output dtype is bfloat16 or float16 (CUTLASS requirement)
    - Tensor dimensions are compatible (divisible by 16)

    Otherwise falls back to Triton.
    """

    if _supports_cutlass(A, B, block_size, output_dtype):
        return w8a8_block_fp8_matmul_cutlass(A, B, As, Bs, output_dtype)

    # Ensure correct CUDA device context for Triton JIT on multi-GPU
    torch.cuda.set_device(A.device)

    # TODO(kernels-community/finegrained-fp8): remove once the hub tensor-scale kernel
    # handles non-power-of-2 dimensions internally (e.g. N=320 for MLA kv_a_proj).
    # The kernel uses tl.arange(0, N) which requires N to be a power of 2.
    if block_size is None:
        N, K = B.shape
        n_needs_pad = (N % 128 != 0) and (N & (N - 1)) != 0
        k_needs_pad = (K % 128 != 0) and (K & (K - 1)) != 0
        if n_needs_pad or k_needs_pad:
            orig_N = N
            if n_needs_pad:
                pad_n = ((N + 127) // 128 * 128) - N
                B = F.pad(B, [0, 0, 0, pad_n])
            if k_needs_pad:
                pad_k = ((K + 127) // 128 * 128) - K
                B = F.pad(B, [0, pad_k])
                A = F.pad(A, [0, pad_k])
            kernel = _get_triton_kernel()
            result = kernel.w8a8_fp8_matmul(A, B, As, Bs, None, output_dtype)
            return result[..., :orig_N]

    kernel = _get_triton_kernel()
    return kernel.w8a8_fp8_matmul(A, B, As, Bs, block_size, output_dtype)


class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        has_bias: bool = False,
        dtype=_FP8_DTYPE,
    ):
        super().__init__(in_features, out_features)

        self.has_bias = has_bias
        self.block_size = block_size
        self.activation_scheme = activation_scheme
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

        if self.block_size is None:
            # If block size is None, it means that we are doing per-tensor quantization
            self.weight_scale_inv = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            scale_out_features = (out_features + self.block_size[0] - 1) // self.block_size[0]
            scale_in_features = (in_features + self.block_size[1] - 1) // self.block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )

        if self.activation_scheme == "static":
            self.activation_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_parameter("activation_scale", None)

        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.element_size() > 1:
            return F.linear(input, self.weight, self.bias)

        if isinstance(self.weight, torch.distributed.tensor.DTensor):
            weight = self.weight._local_tensor.contiguous()
            scale_inv = self.weight_scale_inv._local_tensor.contiguous()
        else:
            # why wouldn't it be contiguous?
            weight = self.weight.contiguous()
            scale_inv = self.weight_scale_inv.contiguous()

        if self.activation_scheme == "dynamic":
            kernel = _get_triton_kernel()
            qinput, scale = kernel.fp8_act_quant(
                input, self.block_size[1] if self.block_size is not None else input.shape[-1]
            )
        elif self.activation_scheme == "static":
            scale = self.activation_scale.to(torch.float32)
            qinput = (input / scale).clamp(min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        else:
            raise NotImplementedError(f"Unsupported activation scheme: {self.activation_scheme}")

        output = w8a8_fp8_matmul(
            qinput,
            weight,
            scale,
            scale_inv,
            self.block_size,
            output_dtype=input.dtype,
        )

        if self.bias is not None:
            output = output + self.bias

        return output.to(dtype=input.dtype)


def fp8_batched_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    kernel = _get_triton_kernel()
    device = hidden_states.device
    torch.cuda.set_device(device)
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
    proj_out = kernel.w8a8_fp8_matmul_batched(
        selected_hidden_states,
        self.gate_up_proj if self.has_gate else self.up_proj,
        self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv,
        block_size=self.block_size,
        expert_ids=expert_ids,
    )  # (S, 2 * intermediate_dim) or (S, intermediate_dim) depending on gating

    # Apply gating or activation
    if self.has_gate:
        # for gated experts we apply the custom/default gating mechanism
        proj_out = self._apply_gate(proj_out)  # (S, intermediate_dim)
    else:
        # for non-gated experts we just apply the activation function
        proj_out = self.act_fn(proj_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 batched) ---
    proj_out = kernel.w8a8_fp8_matmul_batched(
        proj_out,
        self.down_proj,
        self.down_proj_scale_inv,
        block_size=self.block_size,
        expert_ids=expert_ids,
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights.to(proj_out.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


def fp8_grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    kernel = _get_triton_kernel()
    torch.cuda.set_device(hidden_states.device)
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    selected_hidden_states = hidden_states[token_idx]

    # Sort by expert for grouped processing
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
    proj_out = kernel.w8a8_fp8_matmul_grouped(
        selected_hidden_states_g,
        self.gate_up_proj if self.has_gate else self.up_proj,
        self.gate_up_proj_scale_inv if self.has_gate else self.up_proj_scale_inv,
        tokens_per_expert=tokens_per_expert,
        block_size=self.block_size,
        offsets=offsets,
    )  # (S, 2 * intermediate_dim)

    # Apply gating or activation
    if self.has_gate:
        # for gated experts we apply the custom/default gating mechanism
        proj_out = self._apply_gate(proj_out)  # (S, intermediate_dim)
    else:
        # for non-gated experts we just apply the activation function
        proj_out = self.act_fn(proj_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 grouped) ---
    proj_out = kernel.w8a8_fp8_matmul_grouped(
        proj_out,
        self.down_proj,
        self.down_proj_scale_inv,
        tokens_per_expert=tokens_per_expert,
        block_size=self.block_size,
        offsets=offsets,
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights_g.to(proj_out.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # Restore original order
    weighted_out = weighted_out[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


class FP8Experts(nn.Module):
    def __init__(
        self,
        config,
        block_size: tuple[int, int] | None = None,
        activation_scheme: str = "dynamic",
        has_bias: bool = False,
        has_gate: bool = True,
        dtype=_FP8_DTYPE,
    ):
        super().__init__()

        assert has_bias is False, (
            "FP8Experts does not support bias for now, please open an issue if you want this feature"
        )

        self.config = config
        self.has_bias = has_bias
        self.has_gate = has_gate
        self.block_size = block_size
        self.hidden_dim = config.hidden_size
        self.activation_scheme = activation_scheme
        self.num_experts = config.num_local_experts if hasattr(config, "num_local_experts") else config.num_experts
        self.intermediate_dim = (
            config.moe_intermediate_size if hasattr(config, "moe_intermediate_size") else config.intermediate_size
        )

        if self.has_gate:
            gu_proj_out, gu_proj_in = 2 * self.intermediate_dim, self.hidden_dim
            self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, gu_proj_out, gu_proj_in, dtype=dtype))
            gu_scale_out = triton.cdiv(gu_proj_out, self.block_size[0]) if self.block_size is not None else 1
            gu_scale_in = triton.cdiv(gu_proj_in, self.block_size[1]) if self.block_size is not None else 1
            self.gate_up_proj_scale_inv = nn.Parameter(
                torch.empty(self.num_experts, gu_scale_out, gu_scale_in, dtype=torch.float32)
            )
            self.register_parameter("gate_up_proj_bias", None)
        else:
            u_proj_out, u_proj_in = self.intermediate_dim, self.hidden_dim
            self.up_proj = nn.Parameter(torch.empty(self.num_experts, u_proj_out, u_proj_in, dtype=dtype))
            u_scale_out = triton.cdiv(u_proj_out, self.block_size[0]) if self.block_size is not None else 1
            u_scale_in = triton.cdiv(u_proj_in, self.block_size[1]) if self.block_size is not None else 1
            self.up_proj_scale_inv = nn.Parameter(
                torch.empty(self.num_experts, u_scale_out, u_scale_in, dtype=torch.float32)
            )
            self.register_parameter("up_proj_bias", None)

        d_proj_out, d_proj_in = self.hidden_dim, self.intermediate_dim
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, d_proj_out, d_proj_in, dtype=dtype))
        d_scale_out = triton.cdiv(d_proj_out, self.block_size[0]) if self.block_size is not None else 1
        d_scale_in = triton.cdiv(d_proj_in, self.block_size[1]) if self.block_size is not None else 1
        self.down_proj_scale_inv = nn.Parameter(
            torch.empty(self.num_experts, d_scale_out, d_scale_in, dtype=torch.float32)
        )
        self.register_parameter("down_proj_bias", None)

        if self.activation_scheme == "static":
            self.gate_up_proj_activation_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))
            self.down_proj_activation_scale = nn.Parameter(torch.ones(self.num_experts, dtype=torch.float32))

        self.act_fn = ACT2FN[config.hidden_act]

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # index_add_ will accumulate using the dtype of the tensor we write into
        # so we use float32 for the accumulation to avoid numerical issues in bf16/fp16
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx in expert_hit:
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up_act_scale = (
                self.gate_up_proj_activation_scale[expert_idx] if self.activation_scheme == "static" else None
            )
            proj_out = self.linear(
                current_state,
                self.gate_up_proj[expert_idx] if self.has_gate else self.up_proj[expert_idx],
                self.gate_up_proj_scale_inv[expert_idx] if self.has_gate else self.up_proj_scale_inv[expert_idx],
                activation_scale=gate_up_act_scale,
            )
            proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)
            down_act_scale = (
                self.down_proj_activation_scale[expert_idx] if self.activation_scheme == "static" else None
            )
            proj_out = self.linear(
                proj_out,
                self.down_proj[expert_idx],
                self.down_proj_scale_inv[expert_idx],
                activation_scale=down_act_scale,
            )
            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            weighted_out = proj_out * routing_weights.to(proj_out.dtype)
            final_hidden_states.index_add_(0, token_idx, weighted_out.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    def linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        activation_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weight.element_size() > 1:
            return F.linear(input, weight, None)

        if self.activation_scheme == "static" and activation_scale is not None:
            scale = activation_scale.to(torch.float32)
            qinput = (input / scale).clamp(min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        else:
            kernel = _get_triton_kernel()
            qinput, scale = kernel.fp8_act_quant(
                input, self.block_size[1] if self.block_size is not None else input.shape[-1]
            )
        output = w8a8_fp8_matmul(
            qinput,
            weight,
            scale,
            weight_scale_inv,
            self.block_size,
            output_dtype=input.dtype,
        )
        return output.to(dtype=input.dtype)


class FP8ExpertsInterface(ExpertsInterface):
    """Interface for registering custom FP8 experts forward functions."""

    _global_mapping = {
        "batched_mm": fp8_batched_mm_experts_forward,
        "grouped_mm": fp8_grouped_mm_experts_forward,
    }


ALL_FP8_EXPERTS_FUNCTIONS = FP8ExpertsInterface()


def replace_with_fp8_linear(
    model, modules_to_not_convert: list[str] | None = None, quantization_config=None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `FP8Linear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `None`):
            Names of the modules to not convert. In practice we keep the `lm_head` in full precision for numerical stability reasons.
        quantization_config (`FbgemmFp8Config`):
            The quantization config object that contains the quantization parameters.
        pre_quantized (`book`, defaults to `False`):
            Whether the model is pre-quantized or not
    """

    if quantization_config.dequantize:
        return model

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        # we need this to correctly materialize the weights during quantization
        module_kwargs = {} if pre_quantized else {"dtype": None}
        new_module = None
        with torch.device("meta"):
            if module_name.endswith(".experts"):
                has_gate = getattr(module, "has_gate", True)
                has_bias = getattr(module, "has_bias", False)
                config = getattr(module, "config", model.config.get_text_config())
                new_class = use_experts_implementation(
                    experts_class=FP8Experts,
                    experts_interface=ALL_FP8_EXPERTS_FUNCTIONS,
                    has_bias=has_bias,
                    has_gate=has_gate,
                )
                new_module = new_class(
                    config=config,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    has_bias=has_bias,
                    has_gate=has_gate,
                    **module_kwargs,
                )
            elif isinstance(module, nn.Linear):
                new_module = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    block_size=quantization_config.weight_block_size,
                    activation_scheme=quantization_config.activation_scheme,
                    has_bias=module.bias is not None,
                    **module_kwargs,
                )
            if new_module is not None:
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using fp8 but no linear modules were found in your model."
            " Please double check your model architecture."
        )
    return model


class Fp8Quantize(ConversionOps):
    """
    A quantization operation that creates two tensors, weight and scale out of a weight.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        # Unpack single key/value (value may be wrapped in a list)
        target_keys, value = tuple(input_dict.items())[0]
        value = value[0]

        # Resolve block size (support dict-like or attr-like quant_config)
        block_size = None
        if self.hf_quantizer.quantization_config is not None:
            if isinstance(self.hf_quantizer.quantization_config, dict):
                block_size = self.hf_quantizer.quantization_config.get("weight_block_size")
            else:
                block_size = getattr(self.hf_quantizer.quantization_config, "weight_block_size", None)
        if block_size is None:
            block_size = (value.shape[-2], value.shape[-1])

        block_m, block_n = block_size
        rows, cols = value.shape[-2], value.shape[-1]

        # Enforce exact tiling like your original
        if rows % block_m != 0 or cols % block_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n}). for {target_keys}"
            )

        # Leading dims can be empty (2D) or include num_experts/... (3D+)
        leading_shape = value.shape[:-2]
        rows_tiles = rows // block_m
        cols_tiles = cols // block_n

        original_shape = value.shape
        value_fp32 = value.to(torch.float32)

        # Reshape to (..., rows_tiles, block_m, cols_tiles, block_n)
        reshaped = value_fp32.reshape(*leading_shape, rows_tiles, block_m, cols_tiles, block_n)

        # Per-tile max-abs over the block dims
        # dims: block_m is at -3, block_n is at -1 after the reshape
        max_abs = reshaped.abs().amax(dim=(-3, -1))
        safe_max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))

        # Tile scale (we store inverse scale like your Linear: weight_scale_inv)
        scales = _FP8_MAX / safe_max_abs
        scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))  # keep zeros stable

        # Broadcast scales back over the block dims and quantize
        # max_abs/scales shape: (..., rows_tiles, cols_tiles)
        scales_broadcast = scales.unsqueeze(-1).unsqueeze(-3)  # -> (..., rows_tiles, 1, cols_tiles, 1)
        scaled = reshaped * scales_broadcast

        quantized = torch.clamp(scaled, min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)

        quantized = quantized.reshape(original_shape)

        inv_scales = (1.0 / scales).to(torch.float32)  # shape: (*leading, rows_tiles, cols_tiles)
        if target_keys.endswith("weight"):
            scale_key = target_keys.rsplit(".", 1)[0] + ".weight_scale_inv"
        else:
            scale_key = target_keys + "_scale_inv"

        # Return both quantized weights and per-tile inverse scales (keeps leading dims, e.g., num_experts)
        return {
            target_keys: quantized,
            scale_key: inv_scales,
        }


class Fp8Dequantize(ConversionOps):
    """Inverse operation of :class:`Fp8Quantize`. Takes a pair (weight, scale) and reconstructs the fp32 tensor."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if len(input_dict) < 2:
            # case where we only got weights, need to check for "weight$"
            return {full_layer_name: input_dict["weight$"]}

        quantized = input_dict["weight$"][0]
        scales = input_dict["weight_scale_inv"][0]

        rows, cols = quantized.shape[-2:]
        block_size = self.hf_quantizer.quantization_config.weight_block_size
        if block_size is None:
            block_size = (quantized.shape[-2], quantized.shape[-1])

        block_m, block_n = block_size

        if rows % block_m != 0 or cols % block_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_m}, {block_n})."
            )
        quantized = quantized.to(scales.dtype)
        reshaped = quantized.reshape(-1, rows // block_m, block_m, cols // block_n, block_n)
        expanded_scales = scales.reshape(-1, rows // block_m, cols // block_n)
        expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(2)
        dequantized = reshaped * expanded_scales

        return {
            full_layer_name: dequantized.reshape(quantized.shape),
        }
