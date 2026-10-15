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

"""DeepEP integration: the expert-parallel token exchange of `kernels-community/deep-ep`.

Provides:
- `is_deepep_loadable` / `load_deepep_kernel`: whether DeepEP runs here, and the kernel module.
- `deepep_dispatch` / `deepep_combine`: DeepEP's dispatch and combine in its expanded layout, differentiable, as custom
  ops so they run eagerly, under CUDA graph capture and under `torch.compile`.

Requirements: CUDA, Hopper (SM90+), the NCCL PyTorch loads ≥ 2.32.3, and a CUDA toolkit ≥ 13.1 (`nvcc`) at runtime,
since DeepEP JIT-compiles its kernels.
"""

import ctypes
import functools
import os

import torch
import torch.distributed as dist

from ..utils import logging
from ..utils.import_utils import is_kernels_available, maybe_import_error
from .deepgemm import _get_cuda_home, _get_nvcc_version
from .hub_kernels import _MISSING_KERNELS_MESSAGE, lazy_load_kernel


logger = logging.get_logger(__name__)

MIN_NCCL_VERSION = (2, 32, 3)
MIN_NVCC_VERSION = (13, 1)


# ── Kernel loading ─────────────────────────────────────────────────────────────


def _get_loaded_nccl_libraries() -> list[str]:
    """Paths of the NCCL libraries mapped into this process: the ones PyTorch loaded, which DeepEP's NCCL symbols
    resolve to."""
    libraries = []
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            for line in f:
                path = line.split()[-1]
                if "libnccl" in os.path.basename(path) and ".so" in path and path not in libraries:
                    libraries.append(path)
    except OSError:
        pass
    return libraries


def _get_nccl_runtime_version(library: str) -> tuple[int, int, int] | None:
    """Version of the NCCL library at `library`, from its `ncclGetVersion`. `torch.cuda.nccl.version()` is the
    version PyTorch was built against, not necessarily the one it loaded."""
    try:
        code = ctypes.c_int()
        if ctypes.CDLL(library).ncclGetVersion(ctypes.byref(code)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return code.value // 10000, (code.value % 10000) // 100, code.value % 100


@functools.lru_cache
def is_deepep_loadable(raise_error: bool = False) -> bool:
    """Whether DeepEP runs in this environment: `kernels` installed, a CUDA GPU of compute capability 9.0 or newer,
    a CUDA toolkit ≥ 13.1 for DeepEP's JIT, one NCCL ≥ 2.32.3 loaded by PyTorch (DeepEP uses PyTorch's NCCL), and
    `kernels-community/deep-ep` loading from the Hub. Decided once per process. With `raise_error=True` (used by
    `load_deepep_kernel`) it raises the `ImportError` explaining what is missing instead of returning `False`.
    """
    if not is_kernels_available():
        return maybe_import_error(f"DeepEP kernel unavailable: {_MISSING_KERNELS_MESSAGE}", raise_error=raise_error)
    if not torch.cuda.is_available():
        return maybe_import_error("DeepEP requires CUDA, but CUDA is not available.", raise_error=raise_error)
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        return maybe_import_error(
            f"DeepEP requires a GPU of compute capability 9.0 (Hopper) or newer; the current device is SM{major}{minor}.",
            raise_error=raise_error,
        )

    # DeepEP JIT-compiles its kernels with nvcc, found the same way DeepGEMM's JIT finds it.
    required_nvcc = ".".join(map(str, MIN_NVCC_VERSION))
    cuda_home = _get_cuda_home()
    if cuda_home is None or not os.path.isfile(os.path.join(cuda_home, "bin", "nvcc")):
        return maybe_import_error(
            f"DeepEP compiles its kernels at runtime and needs a CUDA toolkit ≥ {required_nvcc} with `nvcc`, but none "
            f"was found{f' in `{cuda_home}/bin`' if cuda_home else ''}. Set `CUDA_HOME` to one, e.g. after "
            f'`pip install "nvidia-cuda-nvcc>={required_nvcc}"`, its `nvidia/cu13` directory.',
            raise_error=raise_error,
        )
    nvcc_version = _get_nvcc_version()
    if nvcc_version is None or nvcc_version < MIN_NVCC_VERSION:
        found = "an unknown version" if nvcc_version is None else ".".join(map(str, nvcc_version))
        return maybe_import_error(
            f"DeepEP needs a CUDA toolkit ≥ {required_nvcc} to compile its kernels, but `{cuda_home}` holds "
            f"{found}. Point `CUDA_HOME` at a newer toolkit.",
            raise_error=raise_error,
        )

    # DeepEP links no NCCL of its own: it runs on the one PyTorch loaded, which needs the device API of 2.32.3.
    required_nccl = ".".join(map(str, MIN_NCCL_VERSION))
    libraries = _get_loaded_nccl_libraries()
    if len(libraries) != 1:
        return maybe_import_error(
            f"DeepEP runs on the NCCL library PyTorch loads, but {len(libraries)} are loaded"
            f"{': ' + ', '.join(libraries) if libraries else ''}. Use a PyTorch build that loads one NCCL "
            f"≥ {required_nccl}.",
            raise_error=raise_error,
        )
    nccl_version = _get_nccl_runtime_version(libraries[0])
    if nccl_version is None or nccl_version < MIN_NCCL_VERSION:
        found = "an unknown version" if nccl_version is None else ".".join(map(str, nccl_version))
        return maybe_import_error(
            f"DeepEP requires NCCL ≥ {required_nccl}, but PyTorch loaded {found} ({libraries[0]}). Upgrade the NCCL "
            f'PyTorch loads, e.g. `pip install "nvidia-nccl-cu13>={required_nccl}"` for a CUDA 13 build of PyTorch.',
            raise_error=raise_error,
        )

    try:
        kernel = lazy_load_kernel("deep-ep")
    except Exception as error:  # a Hub or import failure: report it like the other unmet requirements
        return maybe_import_error(f"Failed to load `kernels-community/deep-ep`: {error}", raise_error=raise_error)
    if kernel is None or not hasattr(kernel, "EPBuffer"):
        return maybe_import_error(
            "Failed to load `kernels-community/deep-ep`: check that a build matches the current torch and CUDA.",
            raise_error=raise_error,
        )
    return True


def load_deepep_kernel():
    """The `kernels-community/deep-ep` module, raising an `ImportError` that says what is missing when DeepEP cannot
    run here."""
    is_deepep_loadable(raise_error=True)
    return lazy_load_kernel("deep-ep")


# ── Buffers ────────────────────────────────────────────────────────────────────

# EP group name -> (DeepEP buffer, (tokens per rank, hidden size, top-k) it holds). Every layer of an EP group shares
# the group's buffer, since the layers run one after another.
_BUFFERS: dict[str, tuple] = {}


def _get_buffer(group_name: str, capacity: int, hidden_dim: int, num_top_k: int):
    """The DeepEP buffer of the EP group `group_name`, rebuilt larger (a collective) when it does not hold `capacity`
    tokens per rank of `hidden_dim` and `num_top_k`. Never built under CUDA graph capture."""
    buffer, size = _BUFFERS.get(group_name, (None, (0, 0, 0)))
    if buffer is not None and capacity <= size[0] and hidden_dim <= size[1] and num_top_k <= size[2]:
        return buffer
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"The DeepEP buffer of this expert-parallel group does not hold {capacity} tokens per rank yet, and it "
            "cannot be allocated during CUDA graph capture: run the forward eagerly at this batch size before "
            "capturing it (the usual warm-up), so the buffer is allocated outside the capture."
        )
    size = (max(capacity, size[0]), max(hidden_dim, size[1]), max(num_top_k, size[2]))
    if buffer is not None:
        # every rank rebuilds here, since they agreed on the capacity; let the old buffer's kernels finish first
        torch.cuda.synchronize()
        buffer.destroy()
    group = dist.distributed_c10d._resolve_process_group(group_name)
    # DeepEP builds on the group's NCCL communicator, which NCCL creates lazily on the group's first collective
    dist.all_reduce(torch.zeros(1, device=torch.cuda.current_device()), group=group)
    buffer = load_deepep_kernel().EPBuffer(
        group, num_max_tokens_per_rank=size[0], hidden=size[1], num_topk=size[2], explicitly_destroy=True
    )
    _BUFFERS[group_name] = (buffer, size)
    return buffer


def destroy_deepep_buffers():
    """Free the DeepEP buffers of every EP group. A collective: every rank calls it, before the process groups are
    destroyed."""
    for buffer, _ in _BUFFERS.values():
        buffer.destroy()
    _BUFFERS.clear()


# ── Dispatch and combine ───────────────────────────────────────────────────────
# The DeepEP handle is a Python object, which custom ops cannot pass along: the ops return its tensors and rebuild it
# from them. `token_metadata_at_forward` and `channel_linked_list` only exist across nodes (hybrid mode); within a node
# they travel as empty tensors.


def _handle(buffer, top_k_index, num_experts, capacity, num_rows, handle_tensors):
    psum_per_rank, psum_per_expert, per_expert, src_metadata, slot_idx, token_metadata, linked_list = handle_tensors
    return load_deepep_kernel().EPHandle(
        True,  # the expanded layout
        num_experts,
        1,  # expert alignment
        capacity,
        buffer.get_theoretical_num_sms(num_experts, top_k_index.size(-1)),
        top_k_index,
        src_metadata.size(0),
        num_rows,
        [],
        psum_per_rank,
        psum_per_expert,
        per_expert,
        src_metadata,
        slot_idx,
        token_metadata if token_metadata.numel() else None,
        linked_list if linked_list.numel() else None,
    )


@torch.library.custom_op("transformers::deepep_dispatch", mutates_args=())
def _deepep_dispatch(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    group_name: str,
    num_experts: int,
    capacity: int,
    cpu_sync: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    buffer = _get_buffer(group_name, capacity, hidden_states.size(-1), top_k_index.size(-1))
    recv_hidden_states, _, recv_weights, handle, _ = buffer.dispatch(
        hidden_states,
        topk_idx=top_k_index,
        topk_weights=top_k_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=capacity,
        do_cpu_sync=cpu_sync,
        do_expand=True,
    )
    empty = top_k_index.new_empty(0, dtype=torch.int32)
    return (
        recv_hidden_states,
        recv_weights,
        handle.psum_num_recv_tokens_per_scaleup_rank,
        handle.psum_num_recv_tokens_per_expert.clone(),
        handle.num_unaligned_recv_tokens_per_expert,
        handle.recv_src_metadata,
        handle.dst_buffer_slot_idx,
        empty if handle.token_metadata_at_forward is None else handle.token_metadata_at_forward,
        empty.clone() if handle.channel_linked_list is None else handle.channel_linked_list,
    )


@_deepep_dispatch.register_fake
def _(hidden_states, top_k_index, top_k_weights, group_name, num_experts, capacity, cpu_sync):
    if cpu_sync:
        raise NotImplementedError("DeepEP's dispatch with a host sync has data-dependent shapes: trace it sync-free.")
    # sync-free, within a node: every receive tensor sized for the worst case
    num_ranks = dist.get_world_size(dist.distributed_c10d._resolve_process_group(group_name))
    num_top_k = top_k_index.size(-1)
    num_local_experts = num_experts // num_ranks
    num_rows = num_ranks * capacity * min(num_top_k, num_local_experts)
    int_tensor = top_k_index.new_empty
    return (
        hidden_states.new_empty(num_rows, hidden_states.size(-1)),
        top_k_weights.new_empty(num_rows),
        int_tensor(num_ranks, dtype=torch.int32),
        int_tensor(num_local_experts, dtype=torch.int32),
        int_tensor(num_local_experts, dtype=torch.int32),
        int_tensor(num_ranks * capacity, num_top_k + 2, dtype=torch.int32),
        int_tensor(top_k_index.size(0), num_top_k, dtype=torch.int32),
        int_tensor(0, dtype=torch.int32),
        int_tensor(0, dtype=torch.int32),
    )


@torch.library.custom_op("transformers::deepep_combine", mutates_args=())
def _deepep_combine(
    expert_output: torch.Tensor,
    top_k_index: torch.Tensor,
    handle_tensors: list[torch.Tensor],
    slot_weights: torch.Tensor | None,
    group_name: str,
    num_experts: int,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    buffer = _get_buffer(group_name, capacity, expert_output.size(-1), top_k_index.size(-1))
    handle = _handle(buffer, top_k_index, num_experts, capacity, expert_output.size(0), handle_tensors)
    combined, combined_weights, _ = buffer.combine(expert_output, handle, topk_weights=slot_weights)
    return combined, top_k_index.new_empty(0, dtype=torch.float32) if combined_weights is None else combined_weights


@_deepep_combine.register_fake
def _(expert_output, top_k_index, handle_tensors, slot_weights, group_name, num_experts, capacity):
    num_tokens = top_k_index.size(0)
    weights_shape = (0,) if slot_weights is None else (num_tokens, top_k_index.size(-1))
    return (
        expert_output.new_empty(num_tokens, expert_output.size(-1)),
        expert_output.new_empty(weights_shape, dtype=torch.float32),
    )


@torch.library.custom_op("transformers::deepep_dispatch_cached", mutates_args=())
def _deepep_dispatch_cached(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    handle_tensors: list[torch.Tensor],
    num_rows: int,
    group_name: str,
    num_experts: int,
    capacity: int,
) -> torch.Tensor:
    """DeepEP's dispatch along the layout of an earlier one: the backward of the combine."""
    buffer = _get_buffer(group_name, capacity, hidden_states.size(-1), top_k_index.size(-1))
    handle = _handle(buffer, top_k_index, num_experts, capacity, num_rows, handle_tensors)
    recv_hidden_states, _, _, _, _ = buffer.dispatch(
        hidden_states, handle=handle, num_experts=num_experts, do_expand=True
    )
    return recv_hidden_states


@_deepep_dispatch_cached.register_fake
def _(hidden_states, top_k_index, handle_tensors, num_rows, group_name, num_experts, capacity):
    return hidden_states.new_empty(num_rows, hidden_states.size(-1))


def _mask_padding(rows: torch.Tensor, psum_per_expert: torch.Tensor) -> torch.Tensor:
    """Zero the rows past the received total, which the sync-free dispatch leaves uninitialized."""
    received = torch.arange(rows.size(0), device=rows.device) < psum_per_expert[-1]
    return rows.masked_fill(~received.view(-1, *([1] * (rows.dim() - 1))), 0)


def _dispatch_setup_context(ctx, inputs, output):
    _, top_k_index, _, group_name, num_experts, capacity, _ = inputs
    ctx.save_for_backward(top_k_index, *output[2:])
    ctx.args = (group_name, num_experts, capacity)


def _dispatch_backward(ctx, grad_hidden_states, grad_weights, *_):
    # the combine sends each slot's gradient back to its token and sums them; the slot weights' gradients come
    # back as `[tokens, top-k]`
    top_k_index, *handle_tensors = ctx.saved_tensors
    grad_hidden_states, grad_top_k_weights = torch.ops.transformers.deepep_combine(
        grad_hidden_states.contiguous(),
        top_k_index,
        handle_tensors,
        None if grad_weights is None else grad_weights.float().contiguous(),
        *ctx.args,
    )
    return grad_hidden_states, None, None if grad_weights is None else grad_top_k_weights, None, None, None, None


def _combine_setup_context(ctx, inputs, output):
    expert_output, top_k_index, handle_tensors, _, group_name, num_experts, capacity = inputs
    ctx.save_for_backward(top_k_index, *handle_tensors)
    ctx.args = (expert_output.size(0), group_name, num_experts, capacity)


def _combine_backward(ctx, grad_combined, _):
    # dispatching the gradient along the forward's layout hands each slot its token's gradient
    top_k_index, *handle_tensors = ctx.saved_tensors
    grad_expert_output = torch.ops.transformers.deepep_dispatch_cached(
        grad_combined.contiguous(), top_k_index, handle_tensors, *ctx.args
    )
    no_grads = [None] * len(handle_tensors)
    return _mask_padding(grad_expert_output, handle_tensors[1]), None, no_grads, None, None, None, None


_deepep_dispatch.register_autograd(_dispatch_backward, setup_context=_dispatch_setup_context)
_deepep_combine.register_autograd(_combine_backward, setup_context=_combine_setup_context)


def deepep_dispatch(hidden_states, top_k_index, top_k_weights, group, num_experts, capacity, cpu_sync):
    """
    DeepEP's dispatch in its expanded layout: one received row per (token, local expert) slot, grouped by local
    expert, with that slot's routing weight. `capacity` bounds the tokens of every rank of `group` and must be the same
    on all of them.

    With `cpu_sync`, DeepEP reads the received counts back to the host and returns exactly the received rows. Without
    it, nothing waits on the host (CUDA graph capture, `torch.compile`): the rows are sized for the worst case, every
    rank sending `capacity` tokens with all of their top-k experts on this rank, so they can never overflow, and the
    rows past the received total are zero and routed to the local expert id `num_local_experts`, which the experts
    skip.

    Returns the rows, their local expert ids, their routing weights and the handle `deepep_combine` takes. Tokens travel
    in bfloat16.
    """
    if hidden_states.size(-1) % 256:
        # DeepEP's combine moves rows in chunks of 32 lanes x 8 bfloat16 values
        raise ValueError(f"DeepEP needs a hidden size that is a multiple of 256, got {hidden_states.size(-1)}.")
    top_k_index = top_k_index.to(torch.int64).contiguous()
    outputs = torch.ops.transformers.deepep_dispatch(
        hidden_states.to(torch.bfloat16).contiguous(),
        top_k_index,
        top_k_weights.float().contiguous(),
        group.group_name,
        num_experts,
        capacity,
        cpu_sync,
    )
    recv_hidden_states, recv_weights, *handle_tensors = outputs
    # inclusive prefix sum of the rows each local expert received: row -> local expert, `num_local_experts` past them
    psum_per_expert = handle_tensors[1]
    rows = torch.arange(recv_hidden_states.size(0), device=recv_hidden_states.device, dtype=psum_per_expert.dtype)
    expert_ids = torch.searchsorted(psum_per_expert, rows, right=True)
    if not cpu_sync:
        recv_hidden_states = _mask_padding(recv_hidden_states, psum_per_expert)
        recv_weights = _mask_padding(recv_weights, psum_per_expert)
    handle = (top_k_index, handle_tensors, group.group_name, num_experts, capacity)
    return recv_hidden_states, expert_ids, recv_weights, handle


def deepep_combine(expert_output, handle):
    """DeepEP's combine: sums the rows of each token, already weighted by the experts, back on the token's rank."""
    top_k_index, handle_tensors, group_name, num_experts, capacity = handle
    combined, _ = torch.ops.transformers.deepep_combine(
        expert_output.to(torch.bfloat16).contiguous(),
        top_k_index,
        handle_tensors,
        None,
        group_name,
        num_experts,
        capacity,
    )
    return combined
