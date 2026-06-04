# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import contextlib
import functools
import re

from ..utils import logging
from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal


if is_torch_available():
    import torch

if is_torch_available() and is_torch_greater_or_equal("2.5"):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.placement_types import _StridedShard

    from .sharding_utils import _find_strided_shard_placement_from_fused_params

    # Cache this result has it's a C FFI call which can be pretty time-consuming
    _torch_distributed_available = torch.distributed.is_available()


logger = logging.get_logger(__name__)


def replace_layer_number_by_wildcard(name: str) -> str:
    """
    Replace the numbers in the `name` by wildcards, only if they are in-between dots (`.`) or if they are between
    a dot (`.`) and the end of the string.
    This matches how modules are named/numbered when using a nn.ModuleList or nn.Sequential, but will NOT match
    numbers in a parameter name itself, e.g. if the param is named `"w1"` or `"w2"`.
    """
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def _get_parameter_tp_plan(parameter_name: str, tp_plan: dict[str, str], is_weight=True) -> str | None:
    """
    Get the TP style for a parameter from the TP plan.

    The TP plan is a dictionary that maps parameter names to TP styles.
    The parameter name can be a generic name with wildcards (e.g. "*.weight") or a specific name (e.g. "layer_1.weight").

    The `is_weight` is important because for weights, we want to support `.weights` and `.bias` cases seamlessly! but
    not parent classes for `post_init` calls
    """
    # Try exact (indexed) keys before wildcard ones, so a per-layer override wins over a
    # generic rule.
    # e.g. for param "layers.0.mlp.experts.gate_up_proj", an indexed plan key
    # "layers.0.mlp.experts.gate_up_proj" is matched here, before collapsing to the generic
    # "layers.*.mlp.experts.gate_up_proj".
    if parameter_name in tp_plan:
        return tp_plan[parameter_name]
    elif is_weight and "." in parameter_name and (module_name := parameter_name.rsplit(".", 1)[0]) in tp_plan:
        return tp_plan[module_name]
    generic_param_name = replace_layer_number_by_wildcard(parameter_name)
    if generic_param_name in tp_plan:
        return tp_plan[generic_param_name]
    elif is_weight and "." in generic_param_name and (module_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
        return tp_plan[module_name]
    return None


class TensorParallelLayer:
    def shard_param(self, module, param, mesh):
        """Wrap ONE parameter as a DTensor placeholder. Default: no-op."""
        pass

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        return args, kwargs

    def context_around_forward(self, module):
        return contextlib.nullcontext()

    def transform_output_post_forward(self, module, output, mesh):
        return output

    def install_forward(self, module, mesh):
        """Install pre / around / post transforms by replacing module.forward."""
        original_forward = module.forward

        def tp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(module, args, kwargs, mesh)
            with self.context_around_forward(module):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, mesh)

        module.forward = tp_forward
        return module


def _verify_plan_weight_sharding(
    expected_keys: list[str], plan: dict[str, str] | None, plan_label: str, *, check_unsharded: bool
) -> None:
    """
    Log warnings when weight-sharding rules in ``plan`` do not match any parameter, and optionally
    when parameters have no matching rule. Forward-comm-only styles are excluded.
    """
    if plan is None:
        return

    # Keep only entries whose style overrides shard_param (actual weight sharding).
    weight_plan = {
        k: v
        for k, v in plan.items()
        if v in ALL_PARALLEL_STYLES and type(ALL_PARALLEL_STYLES[v]).shard_param is not TensorParallelLayer.shard_param
    }
    if not weight_plan:
        return

    generic_keys = {replace_layer_number_by_wildcard(key) for key in expected_keys}
    unsharded_layers = set(generic_keys) if check_unsharded else set()
    unused_rules = weight_plan.copy()

    for key in generic_keys:
        if key in weight_plan:
            unused_rules.pop(key, None)
            unsharded_layers.discard(key)
            continue

        param_name = key.rsplit(".", 1)[0] if "." in key else key
        generic_param_name = re.sub(r"\d+", "*", param_name)

        if generic_param_name in weight_plan:
            unused_rules.pop(generic_param_name, None)
            unsharded_layers.discard(key)
        elif "." in generic_param_name and (parent_param_name := generic_param_name.rsplit(".", 1)[0]) in weight_plan:
            unused_rules.pop(parent_param_name, None)
            unsharded_layers.discard(key)

    if len(unused_rules) > 0:
        logger.warning(f"The following {plan_label} rules were not applied on any of the layers: {unused_rules}")
    if check_unsharded and len(unsharded_layers) > 0:
        logger.warning(
            f"The following layers were not sharded by the {plan_label} plan: {', '.join(unsharded_layers)}"
        )


def verify_tp_sp_ep_plan(
    expected_keys: list[str],
    tp_plan: dict[str, str] | None = None,
    sp_plan: dict[str, str] | None = None,
    ep_plan: dict[str, str] | None = None,
) -> None:
    """
    Verify TP, SP, and/or EP plan entries against model parameters.

    Each non-None plan is checked independently for unused weight-sharding rules
    (colwise, grouped_gemm, packed_colwise, etc.). Module/forward-only entries
    (activation, ep_router, moe_experts_allreduce, …) are skipped.

    The "unsharded parameter" warning is emitted only for tp_plan, since SP/EP recipes
    intentionally cover subsets of the model.
    """
    if tp_plan is not None:
        _verify_plan_weight_sharding(expected_keys, tp_plan, "TP", check_unsharded=True)
    if sp_plan is not None:
        _verify_plan_weight_sharding(expected_keys, sp_plan, "SP", check_unsharded=False)
    if ep_plan is not None:
        _verify_plan_weight_sharding(expected_keys, ep_plan, "EP", check_unsharded=False)


class ColwiseParallel(TensorParallelLayer):
    """Column-wise: weight & bias → Shard(0) (Embedding: Shard(1)); input replicated, output Shard(-1)."""

    def __init__(self, *, input_layouts=None, output_layouts=None, use_local_output: bool = True):
        self.input_layouts = input_layouts or Replicate()
        self.output_layouts = output_layouts if output_layouts is not None else Shard(-1)
        self.use_local_output = use_local_output

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        placement = Shard(1) if isinstance(module, torch.nn.Embedding) else Shard(0)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        x = args[0]
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [self.input_layouts], run_check=False)
        if x.placements != (Replicate(),):
            x = x.redistribute(placements=[Replicate()], async_op=True)
        return (x,) + args[1:], kwargs  # stay DTensor into F.linear

    def transform_output_post_forward(self, module, output, mesh):
        if not isinstance(output, DTensor):
            return output
        if output.placements != (self.output_layouts,):
            output = output.redistribute(placements=[self.output_layouts], async_op=True)
        return output.to_local() if self.use_local_output else output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_layouts={self.input_layouts}, "
            f"output_layouts={self.output_layouts}, use_local_output={self.use_local_output})"
        )


class RowwiseParallel(TensorParallelLayer):
    """Row-wise: weight → Shard(1), bias → Replicate (Embedding: weight → Shard(0)).

    Linear input is sharded on the last dim; Embedding input is replicated. The module
    forward produces a Partial output which the boundary redistribute reduces to
    output_layouts (Replicate → allreduce, Shard(1) → reduce-scatter).
    """

    def __init__(self, *, input_layouts=None, output_layouts=None, use_local_output: bool = True):
        self.input_layouts = input_layouts or Shard(-1)
        self.output_layouts = output_layouts or Replicate()
        self.use_local_output = use_local_output

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        if isinstance(module, torch.nn.Embedding):
            placement = Shard(0)
        else:
            # bias is replicated (added after the row-reduce); weight shards on input dim
            placement = Replicate() if param == "bias" else Shard(1)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        # Embedding runtime sharding needs a replicated input; Linear needs Shard(-1).
        desired = Replicate() if isinstance(module, torch.nn.Embedding) else Shard(-1)
        x = args[0]
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [self.input_layouts], run_check=False)
        if x.placements != (desired,):
            x = x.redistribute(placements=[desired], async_op=True)
        return (x,) + args[1:], kwargs

    def transform_output_post_forward(self, module, output, mesh):
        if not isinstance(output, DTensor):
            return output
        if output.placements != (self.output_layouts,):
            output = output.redistribute(placements=[self.output_layouts], async_op=True)
        return output.to_local() if self.use_local_output else output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_layouts={self.input_layouts}, "
            f"output_layouts={self.output_layouts}, use_local_output={self.use_local_output})"
        )


class SequenceParallel(TensorParallelLayer):
    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = True):
        self.sequence_dim = sequence_dim
        self.use_local_output = use_local_output

    def install_forward(self, module, mesh):
        # Replicate the module's params (LayerNorm/RMSNorm ones-init → from_local is safe).
        for p_name, p in list(module.named_parameters(recurse=False)):
            module.register_parameter(
                p_name, torch.nn.Parameter(DTensor.from_local(p, mesh, [Replicate()], run_check=False))
            )
        return super().install_forward(module, mesh)

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        seq = Shard(self.sequence_dim)
        x = args[0]
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [seq], run_check=False)
        elif x.placements != (seq,):
            x = x.redistribute(placements=[seq], async_op=True)
        return (x,) + args[1:], kwargs

    def transform_output_post_forward(self, module, output, mesh):
        if isinstance(output, DTensor):
            return output.to_local() if self.use_local_output else output
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sequence_dim={self.sequence_dim}, use_local_output={self.use_local_output})"


class PrepareModuleInputOutput(TensorParallelLayer):
    """Allgather input (Shard(1) → Replicate) + local split output (Replicate → Shard(1))."""

    def __init__(self, use_local_output=True):
        self.use_local_output = use_local_output

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        x = args[0]
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [Shard(1)], run_check=False)
        x = x.redistribute(placements=[Replicate()]).to_local()
        return (x,) + args[1:], kwargs

    def transform_output_post_forward(self, module, output, mesh):
        if not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, [Replicate()], run_check=False)
        return output.redistribute(placements=[Shard(1)]).to_local()


class PrepareModuleInput(TensorParallelLayer):
    """Allgather a module input (default input_layout → desired_layout, then to-local)."""

    def __init__(self, *, input_kwarg=None, input_layout=None, desired_layout=None, use_local_output=True):
        self.input_kwarg = input_kwarg
        self.input_layout = input_layout or Shard(1)
        self.desired_layout = desired_layout or Replicate()
        self.use_local_output = use_local_output

    def _prepare(self, x, mesh):
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [self.input_layout], run_check=False)
        if x.placements != (self.desired_layout,):
            x = x.redistribute(placements=[self.desired_layout])
        return x.to_local() if self.use_local_output else x

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        if self.input_kwarg is not None:
            if kwargs.get(self.input_kwarg) is not None:
                kwargs = {**kwargs, self.input_kwarg: self._prepare(kwargs[self.input_kwarg], mesh)}
            return args, kwargs
        return (self._prepare(args[0], mesh),) + args[1:], kwargs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_kwarg={self.input_kwarg!r}, input_layout={self.input_layout}, "
            f"desired_layout={self.desired_layout}, use_local_output={self.use_local_output})"
        )


# =============================================================================
# MoE / packed-linear local-param swap (grouped_mm needs plain tensors)
# =============================================================================


def _accumulate_local_param_grad(original_param: DTensor, local_grad: torch.Tensor) -> torch.Tensor:
    """Stitch a local grad back onto the original DTensor parameter.

    Used when the forward swap detaches the local leaf (``_StridedShard`` params) because
    DTensor backward redistribute does not support that placement as a source.
    """
    tensor_meta = original_param._spec.tensor_meta
    detached_grad = local_grad.detach()
    with torch.no_grad():
        if original_param.grad is None:
            original_param.grad = DTensor.from_local(
                detached_grad,
                original_param.device_mesh,
                original_param.placements,
                run_check=False,
                shape=tensor_meta.shape,
                stride=tensor_meta.stride,
            )
        elif isinstance(original_param.grad, DTensor):
            original_param.grad._local_tensor.add_(detached_grad)
        else:
            original_param.grad.add_(detached_grad)
    return local_grad


class PackedColwiseParallel(TensorParallelLayer):
    """Column-wise parallel style for fused linear weights packed along the output dimension."""

    def __init__(
        self,
        *,
        input_layouts=None,
        use_local_output: bool = True,
        split_factor: int = 2,
    ):
        self.input_layouts = (input_layouts or Replicate(),)
        self.use_local_output = use_local_output
        self.split_factor = split_factor

    def shard_param(self, module, param, mesh):
        if not isinstance(module, torch.nn.Linear):
            raise NotImplementedError("PackedColwiseParallel currently only supports nn.Linear!")
        meta = module._parameters.get(param)
        if meta is None:
            return
        # Wrap as a DTensor placeholder. Runs on meta — distribute_tensor builds metadata only.
        placement = _StridedShard(dim=0, split_factor=self.split_factor)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        input_tensor = args[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, mesh, self.input_layouts, run_check=False)
        elif input_tensor.placements != self.input_layouts:
            input_tensor = input_tensor.redistribute(placements=self.input_layouts)
        input_tensor = input_tensor.to_local()
        return (input_tensor,) + args[1:], kwargs

    @contextlib.contextmanager
    def context_around_forward(self, module):
        # grouped_mm etc needs plain tensors, so swap the params
        to_swap_params = [
            (name, param) for name, param in module.named_parameters(recurse=False) if isinstance(param, DTensor)
        ]
        for name, param in to_swap_params:
            del module._parameters[name]
            if _find_strided_shard_placement_from_fused_params(param.placements) is not None:
                local = torch.nn.Parameter(param._local_tensor.detach(), requires_grad=param.requires_grad)
                if param.requires_grad:
                    local.register_hook(functools.partial(_accumulate_local_param_grad, param))
            else:
                local = param.to_local()
            setattr(module, name, local)
        try:
            yield
        finally:
            for name, param in to_swap_params:
                # restore the original DTensor params
                delattr(module, name)
                module.register_parameter(name, param)

    def transform_output_post_forward(self, module, output, mesh):
        if output is None or self.use_local_output:
            return output
        return DTensor.from_local(
            output, mesh, (_StridedShard(dim=-1, split_factor=self.split_factor),), run_check=False
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_layouts={self.input_layouts}, "
            f"use_local_output={self.use_local_output}, split_factor={self.split_factor})"
        )


class MoEParamShard(TensorParallelLayer):
    """Param-only TP/EP style for MoE expert weights: shard the named 3D params as DTensor
    placeholders; the matching forward comm is a separate moe_experts_allreduce entry.

    shards_expert_dim (EP grouped_gemm) shards dim 0 and updates module.num_experts to the
    per-rank local count, so the experts forward and ep_router sentinel agree.
    """

    def __init__(self, placement, *, shards_expert_dim: bool = False):
        self.placement = placement
        self.shards_expert_dim = shards_expert_dim

    def shard_param(self, module, param, mesh):
        # Wrap one expert param as a DTensor placeholder. Runs on meta —
        # distribute_tensor builds metadata only, no collective.
        meta = module._parameters.get(param)
        if meta is None:
            return
        if self.shards_expert_dim:
            # dim 0 is the expert dimension; record the per-rank local count so the
            # experts forward and the EP router agree on local expert ids/sentinel.
            module.num_experts = meta.shape[0] // mesh.size()
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [self.placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(placement={self.placement}, shards_expert_dim={self.shards_expert_dim})"


if is_torch_available() and is_torch_greater_or_equal("2.5"):

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
            dist.all_reduce(grad, group=ctx.process_group)
            return grad, None


class MoEExpertsParallel(TensorParallelLayer):
    def __init__(self, output_layouts=None):
        self.output_layouts = output_layouts or Replicate()

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh, *, is_expert_parallel=False):
        hidden_states, top_k_index, top_k_weights = args
        tp_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
        if not isinstance(hidden_states, DTensor):
            hidden_states = DTensor.from_local(hidden_states, mesh, [Replicate()], run_check=False)
        hidden_states = hidden_states.to_local()
        hidden_states = _AllReduceBackward.apply(hidden_states, tp_group)

        if isinstance(top_k_weights, DTensor):
            top_k_weights = top_k_weights.to_local()
        # Under TP the router runs replicated, so its routing-weight gradient is partial
        # across ranks and needs an all-reduce-sum in backward. Under EP the router output
        # is already sliced per-rank by `ep_router` (non-local scores zeroed), so each
        # rank's routing weights are independent — skip the all-reduce to avoid
        # double-counting the gradient.
        if not is_expert_parallel:
            top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_group)

        return (hidden_states, top_k_index, top_k_weights), kwargs

    def install_forward(self, module, mesh, *, is_expert_parallel=False):
        """Install pre / around / post transforms; ``is_expert_parallel`` is baked into the closure."""
        original_forward = module.forward

        def tp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(
                module, args, kwargs, mesh, is_expert_parallel=is_expert_parallel
            )
            with self.context_around_forward(module):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, mesh)

        module.forward = tp_forward
        return module

    @contextlib.contextmanager
    def context_around_forward(self, module):
        # grouped_mm experts forward needs plain tensors, so swap the params
        to_swap_params = [
            (name, param) for name, param in module.named_parameters(recurse=False) if isinstance(param, DTensor)
        ]
        for name, param in to_swap_params:
            del module._parameters[name]
            # Happens only when train in TP only (experts dont use grouped_gemm but rely on moe_tp_gate_up_colwise)
            if _find_strided_shard_placement_from_fused_params(param.placements) is not None:
                local = torch.nn.Parameter(param._local_tensor.detach(), requires_grad=param.requires_grad)
                if param.requires_grad:
                    local.register_hook(functools.partial(_accumulate_local_param_grad, param))
            else:
                local = param.to_local()
            setattr(module, name, local)
        try:
            yield
        finally:
            # restore the original DTensor params
            for name, param in to_swap_params:
                delattr(module, name)
                module.register_parameter(name, param)

    def transform_output_post_forward(self, module, output, mesh):
        if output is None:
            return None
        # Under TP-only each rank has a partial result; under TP+FSDP the
        # weights may be fully gathered by FSDP, making the output complete.
        has_sharded_params = any(
            isinstance(p, DTensor) and any(not pl.is_replicate() for pl in p.placements) for p in module.parameters()
        )
        source = Partial() if has_sharded_params else Replicate()
        if not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, [source], run_check=False)
        # MoE output is 2D [tokens, hidden]. For SP, Shard(1) means seq dim
        # in 3D but token dim (0) in 2D.
        target = self.output_layouts
        if output.dim() == 2 and isinstance(target, Shard) and target.dim == 1:
            target = Shard(0)
        if output.placements != (target,):
            output = output.redistribute(placements=(target,))
        return output.to_local()


class MoeIdentityParallel(TensorParallelLayer):
    """
    TP class for zero/identity experts in MoE layers.

    Under TP, the parent `MoEExpertsParallel` all-reduces the expert module
    output by summation. Identity experts produce the same output on every
    rank, so the sum gives `world_size * output`. This class divides the input
    by `world_size` to compensate.
    """

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        input_tensor = args[0]
        return (input_tensor / mesh.size(), *args[1:]), kwargs


class EpRouterParallel(TensorParallelLayer):
    """Expert-parallel router: forward-only slicing of router outputs to local experts.

    Ported from the original RouterParallel (#39501). The gate runs replicated on
    every rank and emits global (router_logits, router_scores, router_indices). Under
    EP each rank owns num_experts // ep_size experts, so this post-forward hook zeroes
    the scores of non-local experts and remaps the surviving indices into the local range,
    using num_local_experts as the sentinel for dropped (non-local) slots. The
    downstream experts forward masks that sentinel and moe_experts_allreduce sums the
    partial per-rank results.

    No parameter sharding — the gate weight stays replicated.

    Example: 128 experts, EP=8 → num_local_experts=16. Rank 0 (owns experts 0-15) keeps
    only indices in [0, 16), remaps them via fmod, and sets everything else to the
    sentinel 16; its scores are zeroed everywhere except the surviving local slots.
    """

    def transform_output_post_forward(self, module, output, mesh):
        ep_rank, ep_size = mesh.get_local_rank(), mesh.size()
        num_experts = getattr(module, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(getattr(module, "config", None), "num_experts", None)
        if num_experts is None:
            raise AttributeError(
                f"Router module {type(module).__name__} is missing `num_experts` and `config.num_experts`"
            )
        if num_experts % ep_size != 0:
            raise ValueError(f"num_experts must be divisible by ep_size: {num_experts} % {ep_size} != 0")
        num_local_experts = num_experts // ep_size

        router_logits, router_scores, router_indices = output
        non_local_mask = (router_indices // num_local_experts) != ep_rank
        router_scores = router_scores.masked_fill(non_local_mask, 0.0)
        router_indices = router_indices.masked_fill(non_local_mask, -1)
        # `-1 % 1 == 0`, so fmod only remaps correctly when there is more than one local expert.
        if num_local_experts > 1:
            router_indices = torch.fmod(router_indices, num_local_experts)
        else:
            router_indices = router_indices.masked_fill(router_indices > 0, 0).masked_fill(router_indices < 0, -1)
        router_indices = router_indices.masked_fill(router_indices == -1, num_local_experts)
        return router_logits, router_scores, router_indices


class ParallelInterface(GeneralInterface):
    """Registry of named TP styles. Configs and modeling files reference these by string name.

    Styles are split into the three parallelism groups they belong to (TENSOR_/SEQUENCE_/
    EXPERT_PARALLEL_STYLES); _global_mapping is the union of all three. Keeping them separate
    lets other code reason about a group as a whole — e.g. SEQUENCE_PARALLEL_STYLES is reused to
    validate that a user-provided tp_plan actually shards the sequence dimension when
    enable_sequence_parallel=True.

    Adding a new entry under one of the three groups is the supported way to introduce a new TP
    style. Users can also override or extend at runtime via ALL_PARALLEL_STYLES["my_style"] = ....

    Naming convention: {kind}[_{comm}][_{extra}]. The _{comm} suffix is dropped only when
    comm is "none" (no collective). All entries are eager instances, so the dicts live behind a
    torch-availability guard and are empty when torch is unavailable (this module stays
    importable; styles are only ever looked up inside apply_tensor_parallel, which needs torch).
    """

    if is_torch_available() and is_torch_greater_or_equal("2.5") and _torch_distributed_available:
        TENSOR_PARALLEL_STYLES = {
            # Column-parallel
            "colwise": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1)),
            "colwise_allgather": ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
            "colwise_loss_parallel": ColwiseParallel(
                input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False
            ),
            "packed_colwise": PackedColwiseParallel(input_layouts=Replicate()),
            # MoE intra-expert weight sharding — param-level (no forward hook). The matching forward
            # comm is declared separately via "moe_experts_allreduce" on the experts module. gate_up
            # is packed (gate||up) along dim -2 (Qwen3/Mixtral: [E, 2*inter, hidden]).
            "moe_tp_gate_up_colwise": MoEParamShard(_StridedShard(dim=-2, split_factor=2)),  # packed gate/up
            "moe_tp_down_rowwise": MoEParamShard(Shard(-1)),  # down_proj input dim
            # Row-parallel
            "rowwise_allreduce": RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
            # Vocab / embedding (rowwise sharding on vocab dim)
            "vocab_allreduce": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
        }
        SEQUENCE_PARALLEL_STYLES = {
            "rowwise_reduce_scatter": RowwiseParallel(input_layouts=Shard(-1), output_layouts=Shard(1)),
            "vocab_reduce_scatter": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            # Activation / norm (sequence-parallel passthrough). use_local_output=True: torch defaults
            # to False here, but downstream modeling code expects plain tensors, not DTensors.
            "activation": SequenceParallel(use_local_output=True),
            "activation_seq_dim_2": SequenceParallel(sequence_dim=2, use_local_output=True),
            # Module-level prepare-input (allgather seq-sharded input → replicate, to local).
            # use_local_output=True: our modeling code expects plain tensors downstream.
            "module_allgather": PrepareModuleInput(input_layout=Shard(1), desired_layout=Replicate()),
            "module_allgather_hidden_states": PrepareModuleInput(
                input_kwarg="hidden_states", input_layout=Shard(1), desired_layout=Replicate()
            ),
            "module_allgather_split": PrepareModuleInputOutput(),
        }
        EXPERT_PARALLEL_STYLES = {
            "grouped_gemm": MoEParamShard(Shard(0), shards_expert_dim=True),  # expert dim
            # Experts' forward comm (no weight sharding — that is declared per-param via "grouped_gemm"
            # / "moe_tp_*"). Installs the forward hook that all-reduces partial per-rank expert outputs.
            # Reused by dense TP-MoE as well as EP.
            "moe_experts_allreduce": MoEExpertsParallel(output_layouts=Replicate()),
            "moe_identity_expert": MoeIdentityParallel(),
            # EP router — forward-only slicing of router outputs to local experts.
            "ep_router": EpRouterParallel(),
        }
    else:
        TENSOR_PARALLEL_STYLES = SEQUENCE_PARALLEL_STYLES = EXPERT_PARALLEL_STYLES = {}

    _global_mapping = {**TENSOR_PARALLEL_STYLES, **SEQUENCE_PARALLEL_STYLES, **EXPERT_PARALLEL_STYLES}


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()


def resolve_parallel_plan(
    model, user_tp_plan: dict[str, str] | None, enable_sp: bool, enable_ep: bool
) -> dict[str, str]:
    """
    Resolve the parallel plan to apply, given enable_sequence_parallel and enable_expert_parallel.

    user_tp_plan is an explicit override (DistributedConfig.tp_plan); when provided it
    replaces the dense base recipe (_tp_plan / _sp_plan). _ep_plan is still overlaid
    when EP is on. When user_tp_plan is None the base is resolved from the model defaults:
        - user_tp_plan != None  -> user_tp_plan (∪ _ep_plan if EP)
        - SP=true,  EP=false    -> _sp_plan
        - SP=false, EP=true     -> _tp_plan ∪ _ep_plan (inference TP+EP)
        - SP=true,  EP=true     -> _sp_plan ∪ _ep_plan (training SP+EP)
        - SP=false, EP=false    -> _tp_plan (experts keep moe_tp_*)
    """

    if user_tp_plan is not None:  # either TP or SP
        base = dict(user_tp_plan)
    elif enable_sp:  # take default base_sp_plan
        base = dict(getattr(model, "_sp_plan", None) or {})  # training plan
    else:  # take default base_tp_plan
        base = dict(getattr(model, "_tp_plan", None) or {})  # inference plan

    if enable_ep:
        base = {**base, **dict(getattr(model, "_ep_plan", None) or {})}
        # Strip moe_tp_* for TP experts when EP is enabled
        base = {
            key: style
            for key, style in base.items()
            if not (
                key.endswith((".gate_up_proj", ".down_proj"))
                and style in ("moe_tp_gate_up_colwise", "moe_tp_down_rowwise")
            )
        }
    return base


def apply_tensor_parallel(model, tp_mesh):
    """Apply tensor parallelism by looking each style up by string name.

    A single named_modules() walk shards each module's direct params (param-keyed plan
    entries → shard_param) and installs its forward comm (module-keyed entries →
    install_forward). Plus the cross-cutting setup: tie-weights handling, SP position_ids
    injection, and loss_parallel activation.
    """
    enable_sp = bool(
        getattr(model.config.distributed_config, "enable_sequence_parallel", False)
        and getattr(model.config, "base_model_sp_plan", None) is not None
    )
    enable_ep = bool(
        getattr(model.config.distributed_config, "enable_expert_parallel", False)
        and getattr(model.config, "base_model_ep_plan", None) is not None
    )

    # case when enable_sequence_parallel=True and tp_plan!=None (but tp_plan has no sequence parallel styles keys) raise an error
    user_tp_plan = model.config.distributed_config.tp_plan
    if (
        enable_sp
        and user_tp_plan is not None  # user provided tp_plan manually
        and not any(style in ParallelInterface.SEQUENCE_PARALLEL_STYLES for style in user_tp_plan.values())
    ):
        raise ValueError(
            "enable_sequence_parallel=True but the provided tp_plan shards no sequence dimension "
            f"(none of {sorted(ParallelInterface.SEQUENCE_PARALLEL_STYLES)}). Provide a sequence-parallel plan "
            "(e.g. 'vocab_reduce_scatter' on the embedding) or set enable_sequence_parallel=False. "
            f"tp_plan: {user_tp_plan}"
        )

    tp_plan = resolve_parallel_plan(model, user_tp_plan, enable_sp=enable_sp, enable_ep=enable_ep)
    logger.info(f"TP plan has been resolved: {tp_plan}")
    model.tp_plan = tp_plan

    # tie_weights() replaces lm_head.weight with embed_tokens.weight after TP is applied.
    # If embed_tokens isn't in the plan, sharding lm_head as a DTensor causes tie to
    # clobber it with a plain tensor (and forward then mixes DTensor/Tensor). Skip
    # lm_head TP in that case so both ends stay plain and the tie is a real alias.
    if getattr(model.config, "tie_word_embeddings", False):
        tied_source_in_plan = any(k.endswith("embed_tokens") for k in tp_plan)
        if not tied_source_in_plan:
            tp_plan.pop("lm_head", None)

    for name, module in model.named_modules():
        # Shard each parameter directly on the module using the specified style.
        for p_name, _ in list(module.named_parameters(recurse=False)):
            full = f"{name}.{p_name}" if name else p_name
            style_name = _get_parameter_tp_plan(parameter_name=full, tp_plan=tp_plan, is_weight=True)
            if style_name is not None and style_name in ALL_PARALLEL_STYLES:
                ALL_PARALLEL_STYLES[style_name].shard_param(module, p_name, tp_mesh)
        # Install forward hooks for modules as needed by the plan.
        style_name = _get_parameter_tp_plan(parameter_name=name, tp_plan=tp_plan, is_weight=False)
        if style_name is not None and style_name in ALL_PARALLEL_STYLES:
            if style_name == "moe_experts_allreduce":
                ALL_PARALLEL_STYLES[style_name].install_forward(module, tp_mesh, is_expert_parallel=enable_ep)
            else:
                ALL_PARALLEL_STYLES[style_name].install_forward(module, tp_mesh)

    # Under SP, inputs_embeds is sequence-sharded after embed_tokens, so
    # auto-generated position_ids would use the wrong (local) seq_len.
    # Inject position_ids from the original input_ids shape before the model forward
    if enable_sp:
        base_model = getattr(model, model.base_model_prefix, model)

        def _inject_sp_metadata(mod, args, kwargs):
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if input_ids is None:
                return args, kwargs
            if "position_ids" not in kwargs or kwargs["position_ids"] is None:
                seq_len = input_ids.shape[1]
                kwargs["position_ids"] = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            return args, kwargs

        base_model.register_forward_pre_hook(_inject_sp_metadata, with_kwargs=True)

    # If the plan uses loss_parallel on lm_head, enable it globally so
    # the model's internal loss computation handles DTensor logits correctly.
    # loss_parallel patches F.cross_entropy to work with Shard(-1) logits.
    # It must be active during both forward and backward, so we enable it
    # once rather than as a context manager.
    has_loss_parallel = any(v == "colwise_loss_parallel" for v in tp_plan.values())
    if has_loss_parallel:
        from torch.distributed.tensor.parallel import loss_parallel

        model._loss_parallel_ctx = loss_parallel()
        model._loss_parallel_ctx.__enter__()

    return model
