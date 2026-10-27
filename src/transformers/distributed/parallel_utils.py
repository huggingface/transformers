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
import contextlib
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


def _same_ranks(mesh_a, mesh_b) -> bool:
    """Whether two 1D device meshes group the same ranks, e.g. an EP group whose ranks all hold one batch (its TP group)."""
    return dist.get_process_group_ranks(mesh_a.get_group()) == dist.get_process_group_ranks(mesh_b.get_group())


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

    class _AllReduceForward(torch.autograd.Function):
        """Allreduce-sum forward, identity backward."""

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
        """

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
