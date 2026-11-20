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
import atexit
import contextlib
import os
import re

from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available, is_torch_distributed_available


if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed as dist
    from torch.distributed._functional_collectives import wait_tensor
    from torch.distributed.tensor import DTensor


class ParallelInterface(GeneralInterface):
    """Registry of the named parallel styles: `tensor_parallel.py` registers the TP styles, `expert_parallel.py` the EP ones."""

    _global_mapping = {}


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()


def replace_layer_number_by_wildcard(name: str) -> str:
    """
    Replace the numbers in the `name` by wildcards, only if they are in-between dots (`.`) or if they are between
    a dot (`.`) and the end of the string.
    This matches how modules are named/numbered when using a nn.ModuleList or nn.Sequential, but will NOT match
    numbers in a parameter name itself, e.g. if the param is named `"w1"` or `"w2"`.
    """
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def _get_parameter_plan(parameter_name: str, plan: dict[str, str], is_weight=True) -> str | None:
    """
    Get the parallel style for a parameter or module from a TP or EP plan.

    The plan is a dictionary that maps parameter or module names to parallel styles.
    The parameter name can be a generic name with wildcards (e.g. "*.weight") or a specific name (e.g. "layer_1.weight").

    The `is_weight` is important because for weights, we want to support `.weights` and `.bias` cases seamlessly! but
    not parent classes for `post_init` calls
    """
    generic_param_name = replace_layer_number_by_wildcard(parameter_name)
    if generic_param_name in plan:
        return plan[generic_param_name]
    elif is_weight and "." in generic_param_name and (module_name := generic_param_name.rsplit(".", 1)[0]) in plan:
        return plan[module_name]
    return None


@contextlib.contextmanager
def _use_local_dtensor_params(module):
    # Kernels as DeepGEMM require local tensors rather than DTensors.
    # We temporarily convert the DTensors to local tensors for the duration of forward() and swap them back after.
    originals = {name: param for name, param in module.named_parameters(recurse=False) if isinstance(param, DTensor)}
    local_params = {name: param.to_local() for name, param in originals.items()}
    module._parameters.update(local_params)
    try:
        yield
    finally:
        for name, original in originals.items():
            current = module._parameters[name]

            if current is local_params[name]:
                module._parameters[name] = original

            # Some kernels such as Megamoe performs changing on FP4 weights and scale factors during its forward pass.
            # That implies creating a new Parameter so we should not restore the original DTensor.
            elif current is not None and not isinstance(current, DTensor):
                replacement = DTensor.from_local(
                    current,
                    original.device_mesh,
                    original.placements,
                    run_check=False,
                )
                module._parameters[name] = torch.nn.Parameter(
                    replacement,
                    requires_grad=current.requires_grad,
                )


if is_torch_distributed_available():
    # [gap2 prototype, bench only] Inference all-reduce of the TP styles through flashinfer's MNNVL kernels, the ones
    # vLLM runs (`flashinfer.comm.allreduce_fusion`, backend "mnnvl": a two-shot reduce over NVLink multicast memory,
    # 7 us at TP8 where NCCL's ring takes 24). `HF_TP_ALL_REDUCE=flashinfer` swaps the reduce of `rowwise`,
    # `all_reduce` and `moe_tp_experts`; `HF_TP_ALL_REDUCE=flashinfer_fused` also lets a model fuse the reduce with
    # the residual add and the RMSNorm that follow it (`tp_all_reduce_add_rms_norm`). One workspace per process,
    # created by the first eligible call outside a graph capture and never reallocated: CUDA graphs record its
    # pointers. A call that does not fit it (rows, dtype, layout) takes NCCL, on every rank alike.
    _TP_ALL_REDUCE = os.environ.get("HF_TP_ALL_REDUCE", "nccl")
    _TP_GROUP = None  # the TP process group, recorded by `apply_tensor_parallelism`
    # Programmatic dependent launch, as vLLM launches these kernels. "0" is the configuration that was clean before
    # flashinfer-ai/flashinfer#7782 fixed the fused kernel's cluster barrier (shipped in flashinfer 0.7.3).
    _FLASHINFER_PDL = os.environ.get("HF_TP_FLASHINFER_PDL", "1") == "1"
    _FLASHINFER = {"workspace": None, "calls": {"flashinfer": 0, "flashinfer_fused": 0, "nccl": 0}}

    def _flashinfer_workspace(x, process_group):
        """The process-wide flashinfer workspace when `x` can be reduced through it, else `None`."""
        hidden = x.shape[-1]
        if (
            x.dim() < 2
            or not x.is_cuda
            or not x.is_contiguous()
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or (hidden * x.element_size()) % 16 != 0  # the kernels load a row as 16-byte vectors
            or dist.get_world_size(process_group) not in (2, 4, 8, 16, 32, 64)  # the kernels exist for these only
        ):
            return None
        workspace = _FLASHINFER["workspace"]
        if workspace is not None and process_group.group_name != _FLASHINFER["group_name"]:
            return None  # one workspace, for the first group that reduces through it (the TP group)
        if workspace is None:
            if torch.cuda.is_current_stream_capturing():
                return None  # creating the workspace is a collective: the warm-up forward before a capture does it
            import flashinfer.comm as flashinfer_comm
            from flashinfer.comm.mnnvl import TorchDistBackend

            # the memory handles are exchanged over a CPU group, as vLLM does
            cpu_group = dist.new_group(ranks=dist.get_process_group_ranks(process_group), backend="gloo")
            workspace = flashinfer_comm.create_allreduce_fusion_workspace(
                backend="mnnvl",
                world_size=dist.get_world_size(process_group),
                rank=dist.get_rank(process_group),
                max_token_num=int(os.environ.get("HF_TP_FLASHINFER_MAX_TOKENS", "16384")),
                hidden_dim=hidden,
                dtype=x.dtype,
                comm_backend=TorchDistBackend(group=cpu_group),
                group=cpu_group,
            )
            if not workspace.mc_ptr:
                raise RuntimeError("flashinfer's MNNVL workspace has no multicast pointer on this topology")
            atexit.register(workspace.destroy)  # unmap the shared memory while the CUDA context is alive, as vLLM does
            if dist.get_rank(process_group) == 0:
                # which path the forwards took (Python calls: eager steps and graph captures, not replays)
                atexit.register(lambda: print(f"TP_ALL_REDUCE {_TP_ALL_REDUCE} calls {_FLASHINFER['calls']}", flush=True))
            _FLASHINFER.update(
                workspace=workspace, comm=flashinfer_comm, cpu_group=cpu_group, group_name=process_group.group_name
            )
        rows = x.numel() // hidden
        if not workspace.is_buffer_size_sufficient(
            tp_size=dist.get_world_size(process_group), num_tokens=rows, hidden_dim=hidden, dtype=x.dtype
        ):
            return None
        return workspace

    def _tp_inference_all_reduce(x, process_group):
        """All-reduce (sum) of this rank's partial tensor on the inference path. Returns the reduced tensor: `x`
        itself, reduced in place, with NCCL; a new tensor with flashinfer."""
        if _TP_ALL_REDUCE != "nccl" and (workspace := _flashinfer_workspace(x, process_group)) is not None:
            comm = _FLASHINFER["comm"]
            x2d = x.view(-1, x.shape[-1])
            # the arguments of vLLM's `FlashInferAllReduce.all_reduce`
            out = comm.allreduce_fusion(
                input=x2d,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kAllReduce,
                launch_with_pdl=_FLASHINFER_PDL,
                trigger_completion_at_end=x2d.shape[0] > 16,
            )
            _FLASHINFER["calls"]["flashinfer"] += 1
            return out.view(x.shape)
        dist.all_reduce(x, group=process_group)
        _FLASHINFER["calls"]["nccl"] += 1
        return x

    def tp_all_reduce_add_rms_norm(partial, residual, norm):
        """[gap2 prototype] `residual + all_reduce(partial)` and its RMSNorm, for a block whose TP plan leaves its
        output partial. Returns `(normed, new_residual)`. One fused flashinfer call when the tensors fit the
        workspace (the arguments of vLLM's `call_trtllm_fused_allreduce_norm`, with fresh outputs); otherwise the
        reduce, the add and `norm` one after the other. Without tensor parallelism `partial` is the block's full
        output and nothing is reduced."""
        process_group = _TP_GROUP
        if process_group is None or dist.get_world_size(process_group) == 1:
            hidden_states = residual + partial
            return norm(hidden_states), hidden_states
        fused = _TP_ALL_REDUCE == "flashinfer_fused" and not torch.is_grad_enabled() and residual.is_contiguous()
        if fused and (workspace := _flashinfer_workspace(partial, process_group)) is not None:
            comm = _FLASHINFER["comm"]
            hidden = partial.shape[-1]
            x2d = partial.view(-1, hidden)
            normed = torch.empty_like(x2d)
            new_residual = torch.empty_like(x2d)
            comm.allreduce_fusion(
                input=x2d,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=_FLASHINFER_PDL,
                output=None,
                residual_out=new_residual,
                norm_out=normed,
                quant_out=None,
                scale_out=None,
                residual_in=residual.view(-1, hidden),
                rms_gamma=norm.weight,
                rms_eps=norm.variance_epsilon,
                scale_factor=None,
                layout_code=comm.QuantizationSFLayout.SWIZZLED_128x4,
                use_oneshot=None,
                fp32_acc=True,
                weight_bias=0.0,
                trigger_completion_at_end=x2d.shape[0] > 16,
            )
            _FLASHINFER["calls"]["flashinfer_fused"] += 1
            return normed.view(partial.shape), new_residual.view(partial.shape)
        if torch.is_grad_enabled() and partial.requires_grad:
            partial = _AllReduceForward.apply(partial, process_group)
        else:
            partial = _tp_inference_all_reduce(partial, process_group)
        hidden_states = residual + partial
        return norm(hidden_states), hidden_states

    class _AllReduceForward(torch.autograd.Function):
        """Allreduce-sum forward, identity backward.

        `apply` reduces directly when no gradient can flow, like `_AllReduceBackward.apply`: that is the inference
        path, where the reduce can go through `_tp_inference_all_reduce`."""

        @classmethod
        def apply(cls, x, process_group):
            if torch.is_grad_enabled() and x.requires_grad:
                return super().apply(x, process_group)
            if dist.get_world_size(process_group) > 1:
                x = _tp_inference_all_reduce(x, process_group)
            return x

        @staticmethod
        def forward(ctx, x, process_group):
            if dist.get_world_size(process_group) > 1:
                dist.all_reduce(x, group=process_group)
            return x

        @staticmethod
        def backward(ctx, grad):
            return grad, None

    class _AllReduceBackward(torch.autograd.Function):
        """Identity forward, allreduce-sum backward.

        Used for MoE routing weights: the forward value is replicated (same on all
        ranks), but the backward gradient is partial (each rank has 1/tp_size from
        its expert shard). We need to sum the partial gradients without dividing by
        world_size, which is what DTensor's Replicate backward does incorrectly.

        `apply` returns `x` as is when no gradient can flow: at inference the per-call overhead of an
        `autograd.Function` compounds across layers, e.g. in eager decode.
        """

        @classmethod
        def apply(cls, x, process_group):
            if torch.is_grad_enabled() and x.requires_grad:
                return super().apply(x, process_group)
            return x

        @staticmethod
        def forward(ctx, x, process_group):
            ctx.process_group = process_group
            return x

        @staticmethod
        def backward(ctx, grad):
            grad = grad.contiguous()
            dist.all_reduce(grad, group=ctx.process_group)
            return grad, None

    class _WaitTensor(torch.autograd.Function):
        """Wait on a functional collective's result, identity backward (`wait_tensor` has no autograd kernel before
        torch 2.11)."""

        @staticmethod
        def forward(ctx, x):
            return wait_tensor(x)

        @staticmethod
        def backward(ctx, grad):
            return grad
