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

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_kernels_available, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn
    import triton
    import triton.language as tl
    from torch.library import custom_op, triton_op, wrap_triton
    from torch.nn import functional as F

    from ..activations import ACT2FN


logger = logging.get_logger(__name__)

# Global for the CUTLASS quantization kernel (lazily loaded)
_cutlass_kernel = None


def _get_cutlass_kernel():
    """Lazily load the CUTLASS quantization kernel from HuggingFace Hub."""
    global _cutlass_kernel
    if _cutlass_kernel is None:
        try:
            from .hub_kernels import get_kernel

            # this kernel's build was not updated since torch 2.8
            _cutlass_kernel = get_kernel("RedHatAI/quantization")
        except Exception as e:
            logger.warning_once(f"Failed to load CUTLASS quantization kernel: {e}. Falling back to Triton.")
            _cutlass_kernel = False  # Mark as unavailable to avoid future attempts
    return _cutlass_kernel if _cutlass_kernel else None


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


@custom_op("transformers::w8a8_block_fp8_matmul_cutlass", mutates_args=())
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
      - B:  [K, N]           column-major float8_e4m3fn  (our [N, K] transposed)
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


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def fp8_act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0  # _FP8_MAX
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


@triton_op("transformers::fp8_act_quant", mutates_args=())
def fp8_act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=_FP8_DTYPE)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    grid = (triton.cdiv(x.numel(), block_size),)
    wrap_triton(fp8_act_quant_kernel)[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.jit
def w8a8_block_fp8_matmul_kernel(
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


@triton_op("transformers::w8a8_block_fp8_matmul_triton", mutates_args=())
def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int] | None,
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
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    # if we have per-tensor quantization, we use 128x128 block size for tiled matmul multiplication
    if block_n == B.shape[-2] and block_k == B.shape[-1]:
        block_n = 128
        block_k = 128

    assert A.shape[-1] == B.shape[-1]
    assert A.is_contiguous()

    assert B.ndim == 2
    assert B.is_contiguous()

    N, K = B.shape
    M = A.numel() // A.shape[-1]

    # For per-tensor scales (scalar), expand to block-scale shape with strides (0, 0).
    # This is a zero-copy view; all loads inside the kernel hit the same cached value.
    if As.numel() == 1:
        As = As.reshape(1, 1).expand(M, triton.cdiv(K, block_k))
    else:
        assert A.shape[:-1] == As.shape[:-1]
        assert triton.cdiv(K, block_k) == As.shape[-1]
    if Bs.numel() == 1:
        Bs = Bs.reshape(1, 1).expand(triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    else:
        assert Bs.ndim == 2
        assert triton.cdiv(N, block_n) == Bs.shape[0], f"{N}, {block_n}, {Bs.shape}"
        assert triton.cdiv(K, block_k) == Bs.shape[1], f"{K}, {block_k}, {Bs.shape}"

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    # Adaptive BLOCK_SIZE_M: smallest power-of-2 >= M, floored at 16, capped at 128.
    # Matches the WGMMA tile to the actual row count — smaller tiles use less
    # register pressure and a better-matched FP8 WGMMA instruction, improving
    # both accuracy and performance for small M (decode).
    BLOCK_SIZE_M = min(max(triton.next_power_of_2(M), 16), 128)
    BLOCK_SIZE_K = block_k
    BLOCK_SIZE_N = block_n

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    wrap_triton(w8a8_block_fp8_matmul_kernel)[grid](
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


def w8a8_block_fp8_matmul(
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
        try:
            return torch.ops.transformers.w8a8_block_fp8_matmul_cutlass(A, B, As, Bs, output_dtype)
        except Exception as e:
            global _cutlass_kernel
            logger.warning_once(f"CUTLASS kernel failed: {e}. Falling back to Triton.")
            _cutlass_kernel = False  # mark unavailable to avoid future attempts

    # Fall back to Triton
    return w8a8_block_fp8_matmul_triton(A, B, As, Bs, block_size, output_dtype)


class FP8Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=torch.float8_e4m3fn,
        block_size: tuple[int, int] | None = None,
        activation_scheme="dynamic",
    ):
        super().__init__(in_features, out_features)

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

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.element_size() > 1:
            # QUESTION: not sure why we would want the fp8 linear to support non-fp8 weights
            return F.linear(input, self.weight, self.bias)

        if isinstance(self.weight, torch.distributed.tensor.DTensor):
            weight = self.weight._local_tensor.contiguous()
            scale_inv = self.weight_scale_inv._local_tensor.contiguous()
        else:
            weight = self.weight.contiguous()
            scale_inv = self.weight_scale_inv.contiguous()

        if self.activation_scheme == "dynamic":
            qinput, scale = torch.ops.transformers.fp8_act_quant(input, self.block_size[1])
        elif self.activation_scheme == "static":
            scale = self.activation_scale.to(torch.float32)
            qinput = (input / scale).clamp(min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
        else:
            raise NotImplementedError("Not supported")

        output = w8a8_block_fp8_matmul(
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


@triton.jit
def w8a8_block_fp8_matmul_batched_kernel(
    A,  # (S, K)  raw BF16/FP16 activations
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
    # weight / scale slice. No pre-gather of weights needed (like in non-fp8 impls)
    expert_id = tl.load(ExpertIds + batch_id)
    A = A + batch_id * stride_Ab
    B = B + expert_id * stride_Eb
    C = C + batch_id * stride_Cb
    Bs = Bs + expert_id * stride_Esb

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # M=1: broadcast the single activation row to BLOCK_SIZE_M identical rows
    # so tl.dot gets the required (BLOCK_SIZE_M, BLOCK_SIZE_K) shape.
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


@triton.jit
def w8a8_block_fp8_grouped_mm_kernel(
    A,  # (S, K)  raw BF16/FP16 activations, sorted/grouped by expert id
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


@triton_op("transformers::w8a8_block_fp8_matmul_batched", mutates_args=())
def w8a8_block_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
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

    assert B.ndim == 3, "B must be (E, N, K)"
    assert B.is_contiguous()

    assert A.shape[1] == B.shape[2], "K dimension mismatch between A and B"
    assert expert_ids.is_contiguous()
    assert Bs.is_contiguous()

    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    S, K = A.shape
    E, N, _ = B.shape
    C = A.new_empty(S, N)

    # Adaptive BLOCK_SIZE_M: match the tile to the average tokens per expert
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)

    grid = (triton.cdiv(N, block_n), S)
    wrap_triton(w8a8_block_fp8_matmul_batched_kernel)[grid](
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


@triton_op("transformers::w8a8_block_fp8_matmul_grouped", mutates_args=())
def w8a8_block_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Grouped FP8 block-wise matmul with fused activation quantization.

    Mirrors ``_grouped_linear`` / ``_grouped_mm`` for FP8 weights: A is the
    raw (BF16/FP16) activation matrix sorted by expert, B / Bs are the stacked
    expert weights / scales.  Activation quantization (``act_quant``) is fused
    into the matmul loop.  ``tokens_per_expert`` is needed (in addition to
    ``offsets``) to build the per-expert tile schedule inside the kernel.
    """

    assert A.ndim == 2, "A must be (S, K)"
    assert A.is_contiguous()

    assert B.ndim == 3, "B must be (E, N, K)"
    assert B.is_contiguous()

    assert tokens_per_expert.is_contiguous()
    assert offsets.is_contiguous()
    assert Bs.is_contiguous()

    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    S, K = A.shape
    E, N, _ = B.shape
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
    wrap_triton(w8a8_block_fp8_grouped_mm_kernel)[grid](
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
    gate_up_out = torch.ops.transformers.w8a8_block_fp8_matmul_batched(
        selected_hidden_states,
        self.gate_up_proj,
        self.gate_up_proj_scale_inv,
        expert_ids,
        self.block_size,
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 batched) ---
    out_per_sample = torch.ops.transformers.w8a8_block_fp8_matmul_batched(
        gated_out,
        self.down_proj,
        self.down_proj_scale_inv,
        expert_ids,
        self.block_size,
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
    gate_up_out = torch.ops.transformers.w8a8_block_fp8_matmul_grouped(
        selected_hidden_states_g,
        self.gate_up_proj,
        self.gate_up_proj_scale_inv,
        offsets,
        tokens_per_expert,
        self.block_size,
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 grouped) ---
    out_per_sample_g = torch.ops.transformers.w8a8_block_fp8_matmul_grouped(
        gated_out,
        self.down_proj,
        self.down_proj_scale_inv,
        offsets,
        tokens_per_expert,
        self.block_size,
    )  # (S, hidden_dim)

    # Apply routing weights
    out_per_sample_g = out_per_sample_g * sample_weights_g.to(out_per_sample_g.dtype).unsqueeze(-1)  # (S, hidden_dim)

    # Restore original order
    out_per_sample = out_per_sample_g[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


class FP8Expert(nn.Module):
    def __init__(self, config, block_size, dtype=torch.float8_e4m3fn):
        super().__init__()

        self.config = config
        self.block_size = block_size
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts if hasattr(config, "num_local_experts") else config.num_experts
        self.intermediate_dim = (
            config.moe_intermediate_size if hasattr(config, "moe_intermediate_size") else config.intermediate_size
        )

        Wg_out, Wg_in = 2 * self.intermediate_dim, self.hidden_dim
        Wd_out, Wd_in = self.hidden_dim, self.intermediate_dim

        self.gate_up_proj = nn.Parameter(torch.zeros(self.num_experts, Wg_out, Wg_in, dtype=dtype))
        self.down_proj = nn.Parameter(torch.zeros(self.num_experts, Wd_out, Wd_in, dtype=dtype))

        bo, bi = self.block_size

        # gate_up tiles: ceil(Wg_out/bo) x ceil(Wg_in/bi)
        gu_scale_o = triton.cdiv(Wg_out, bo)
        gu_scale_i = triton.cdiv(Wg_in, bi)
        self.gate_up_proj_scale_inv = nn.Parameter(
            torch.zeros(self.num_experts, gu_scale_o, gu_scale_i, dtype=torch.float32)
        )

        # down tiles: ceil(Wd_out/bo) x ceil(Wd_in/bi)
        dp_scale_o = triton.cdiv(Wd_out, bo)
        dp_scale_i = triton.cdiv(Wd_in, bi)
        self.down_proj_scale_inv = nn.Parameter(
            torch.zeros(self.num_experts, dp_scale_o, dp_scale_i, dtype=torch.float32)
        )

        # (Optional) bias per projection — many MoEs omit bias; keep None to match your FP8Linear default
        self.register_parameter("gate_up_bias", None)
        self.register_parameter("down_bias", None)

        # Activation used in the MLP (same as your config / ACT2FN)
        # Keep a handle here; actual usage happens in forward of your MoE block
        self.act_fn = ACT2FN[config.hidden_act]

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    # We follow the mixtral "eager" moe implementation at
    # https://github.com/huggingface/transformers/blob/457048fbfdba9a7dee8bd03328c62f49e57b95f9/src/transformers/models/mixtral/modular_mixtral.py#L148
    # The core changes in this FP8 version should only relate to how we call the linear projections
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.config._experts_implementation == "grouped_mm":
            return fp8_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
        elif self.config._experts_implementation == "batched_mm":
            return fp8_batched_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
        elif self.config._experts_implementation != "eager":
            raise ValueError(f"Unsupported experts implementation: {self.config._experts_implementation}")

        # index_add_ will accumulate using the dtype of the tensor we write into
        # so we use float32 for the accumulation to avoid numerical issues in bf16/fp16
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.gate_up_proj.size(0):
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = self.linear(
                current_state, self.gate_up_proj[expert_idx], self.gate_up_proj_scale_inv[expert_idx]
            )
            current_hidden_states = self._apply_gate(gate_up)
            current_hidden_states = self.linear(
                current_hidden_states, self.down_proj[expert_idx], self.down_proj_scale_inv[expert_idx]
            )

            routing_weights = top_k_weights[token_idx, top_k_pos, None]
            current_hidden_states = current_hidden_states * routing_weights.to(current_hidden_states.dtype)
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)

    def linear(self, input: torch.Tensor, weight: torch.Tensor, weight_scale_inv: torch.Tensor) -> torch.Tensor:
        if weight.element_size() > 1:
            # QUESTION: not sure why we would want the fp8 experts to support a fallback to fp16/bf16/fp32
            return F.linear(input, weight, None)

        qinput, scale = torch.ops.transformers.fp8_act_quant(input, self.block_size[1])
        output = w8a8_block_fp8_matmul(
            qinput,
            weight,
            scale,
            weight_scale_inv,
            self.block_size,
            output_dtype=input.dtype,
        )
        return output.to(dtype=input.dtype)


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
                new_module = FP8Expert(
                    config=model.config.get_text_config(),
                    block_size=quantization_config.weight_block_size,
                    **module_kwargs,
                )
            elif isinstance(module, nn.Linear):
                new_module = FP8Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    activation_scheme=quantization_config.activation_scheme,
                    block_size=quantization_config.weight_block_size,
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
