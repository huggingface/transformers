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
import re

from ..utils import logging
from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal


if is_torch_available():
    import torch

if is_torch_available() and is_torch_greater_or_equal("2.5"):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        PrepareModuleInput,
        RowwiseParallel,
        SequenceParallel,
    )
    from torch.distributed.tensor.parallel.style import ParallelStyle
    from torch.distributed.tensor.placement_types import _StridedShard

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
    generic_param_name = replace_layer_number_by_wildcard(parameter_name)
    if generic_param_name in tp_plan:
        return tp_plan[generic_param_name]
    elif is_weight and "." in generic_param_name and (module_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
        return tp_plan[module_name]
    return None


# =============================================================================
# High-Level API Functions
# =============================================================================


def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str] | None):
    """
    Verify the TP plan of the model, log a warning if the layers that were not sharded and the rules that were not applied.

    Only weight-sharding rules (colwise, rowwise, vocab, grouped_gemm, moe_tp_gate_up_*,
    moe_tp_down_*) are checked. Module/activation entries (e.g. PrepareModuleInput,
    SequenceParallel, moe_experts_allreduce, ep_router) set up communication hooks on
    modules, not weight sharding, so they are excluded.
    """

    if tp_plan is None:
        return

    # Filter out module-level comm hooks — they don't shard weights.
    # Plan values are registry names; entries beginning with "activation"/"module" or
    # the forward-only MoE entries configure communication hooks rather than weight sharding.
    forward_only = {"activation", "moe_experts_allreduce", "ep_router"}
    weight_plan = {
        k: v for k, v in tp_plan.items() if v not in forward_only and not v.startswith(("activation_", "module_"))
    }

    generic_keys = {replace_layer_number_by_wildcard(key) for key in expected_keys}
    unsharded_layers = set(generic_keys)
    unused_rules = weight_plan.copy()

    for key in generic_keys:
        param_name = key.rsplit(".", 1)[0] if "." in key else key
        generic_param_name = re.sub(r"\d+", "*", param_name)

        if generic_param_name in weight_plan:
            unused_rules.pop(generic_param_name, None)
            unsharded_layers.discard(key)
        elif "." in generic_param_name and (parent_param_name := generic_param_name.rsplit(".", 1)[0]) in weight_plan:
            unused_rules.pop(parent_param_name, None)
            unsharded_layers.discard(key)

    if len(unused_rules) > 0:
        logger.warning(f"The following TP rules were not applied on any of the layers: {unused_rules}")
    if len(unsharded_layers) > 0:
        logger.warning(f"The following layers were not sharded: {', '.join(unsharded_layers)}")


class CustomParallelStyle(ParallelStyle):
    """Base class for custom TP styles (MoE, packed linear, etc.).

    Apply runs in two phases (subclasses should not override _apply):

      1. shard_parameters(module, mesh) — meta-time DTensor placeholders (optional)
      2. _install_forward(module, mesh) — pre / around / post forward transforms

    Unlike PyTorch registry entries (ColwiseParallel, SequenceParallel, …) which use
    distribute_module or forward hooks via their own _apply, this base wraps
    module.forward to support *args, **kwargs and an optional context manager.

    Param wrapping runs on meta (the model is on meta when apply_tensor_parallel
    is invoked); distribute_tensor on meta builds metadata only — no collective.
    Real data flows in later, async, via DtensorShardOperation during load.

    Override what you need:
      - shard_parameters(module, mesh) — wrap params as DTensor placeholders
      - transform_inputs_pre_forward(module, args, kwargs, mesh) → (args, kwargs)
      - context_around_forward(module) → context manager wrapping the call
      - transform_output_post_forward(module, output, mesh) → output
    """

    def shard_parameters(self, module, mesh):
        """Wrap selected params as DTensor placeholders. Default: no-op."""
        pass

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        return args, kwargs

    def context_around_forward(self, module):
        return contextlib.nullcontext()

    def transform_output_post_forward(self, module, output, mesh):
        return output

    def _install_forward(self, module, mesh):
        """Install pre / around / post transforms by replacing module.forward."""
        original_forward = module.forward

        def tp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(module, args, kwargs, mesh)
            with self.context_around_forward(module):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, mesh)

        module.forward = tp_forward
        return module

    def _apply(self, module, mesh):
        self.shard_parameters(module, mesh)
        return self._install_forward(module, mesh)


class PrepareModuleInputOutput(CustomParallelStyle):
    """Allgather input (Shard(1) → Replicate) + local split output (Replicate → Shard(1)).

    Used for MoE blocks with SP: the input sequence is gathered before routing,
    and the output (after expert allreduce) is split back to match the residual.
    Forward output split is a local op (no comm). Backward creates the all-gather.
    """

    def __init__(self, use_local_output=True):
        super().__init__()
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


def _accumulate_local_param_grad(original_param: DTensor, local_grad: torch.Tensor) -> torch.Tensor:
    """Stitch a local grad back onto the original DTensor parameter.

    During forward we replace the DTensor param with a detached plain-tensor
    leaf (see ``_swap_dtensor_params_for_local``) because ``grouped_mm`` / fused
    ops don't accept DTensor inputs. That swap breaks the autograd link between
    the local leaf's grad and the DTensor param's ``.grad``, so this tensor hook
    runs on the leaf and copies/accumulates the grad onto the original DTensor.

    NOTE: An autograd-aware ``param.to_local()`` swap would let backward stitch
    the grad automatically, but DTensor's backward path then redistributes the
    resulting grad — and that redistribute does not currently support
    ``_StridedShard`` placements (used by ``MoEExpertsParallel`` / ``PackedColwiseParallel``).
    """
    tensor_meta = original_param._spec.tensor_meta
    detached_grad = local_grad.detach()
    grad_dtensor = DTensor.from_local(
        detached_grad,
        original_param.device_mesh,
        original_param.placements,
        run_check=False,
        shape=tensor_meta.shape,
        stride=tensor_meta.stride,
    )
    with torch.no_grad():
        existing_grad = original_param.grad
        if existing_grad is None:
            original_param.grad = grad_dtensor
        elif isinstance(existing_grad, DTensor):
            existing_grad._local_tensor.add_(detached_grad)
        else:
            existing_grad.add_(detached_grad)
    return local_grad


@contextlib.contextmanager
def _swap_dtensor_params_for_local(module):
    """Temporarily replace DTensor params with local-shard ``Parameter``s for forward.

    ``grouped_mm`` / fused kernels don't accept DTensor inputs, so each DTensor
    param is swapped for a detached local ``Parameter``. A tensor hook
    (``_accumulate_local_param_grad``) on the local leaf copies the backward
    grad back onto the original DTensor.

    The original DTensor params are restored on exit (even on exception) so
    save_pretrained / state-dict still see sharded params.
    """
    shadows = {}
    for name, param in list(module.named_parameters(recurse=False)):
        if not isinstance(param, DTensor):
            continue
        shadows[name] = param
        local = torch.nn.Parameter(param._local_tensor.detach(), requires_grad=param.requires_grad)
        if param.requires_grad:
            local.register_hook(lambda g, p=param: _accumulate_local_param_grad(p, g))
        module._parameters.pop(name)
        setattr(module, name, local)
    try:
        yield
    finally:
        for name, param in shadows.items():
            if hasattr(module, name):
                delattr(module, name)
            module.register_parameter(name, param)


class PackedColwiseParallel(CustomParallelStyle):
    """Column-wise parallel style for fused linear weights packed along the output dimension."""

    def __init__(
        self,
        *,
        input_layouts=None,
        use_local_output: bool = True,
        split_factor: int = 2,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.use_local_output = use_local_output
        self.split_factor = split_factor

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        input_tensor = args[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, mesh, self.input_layouts, run_check=False)
        elif input_tensor.placements != self.input_layouts:
            input_tensor = input_tensor.redistribute(placements=self.input_layouts)
        input_tensor = input_tensor.to_local()
        return (input_tensor,) + args[1:], kwargs

    def context_around_forward(self, module):
        return _swap_dtensor_params_for_local(module)

    def transform_output_post_forward(self, module, output, mesh):
        if output is None or self.use_local_output:
            return output
        return DTensor.from_local(
            output, mesh, (_StridedShard(dim=-1, split_factor=self.split_factor),), run_check=False
        )

    def shard_parameters(self, module, mesh):
        if not isinstance(module, torch.nn.Linear):
            raise NotImplementedError("PackedColwiseParallel currently only supports nn.Linear!")
        # Wrap weight + bias as DTensor placeholders. Runs on meta —
        # distribute_tensor builds metadata only, no collective.
        placement = _StridedShard(dim=0, split_factor=self.split_factor)
        for name in ("weight", "bias"):
            meta = module._parameters.get(name)
            if meta is None:
                continue
            module._parameters[name] = torch.nn.Parameter(
                distribute_tensor(meta, mesh, [placement], src_data_rank=None),
                requires_grad=meta.requires_grad,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_layouts={self.input_layouts}, "
            f"use_local_output={self.use_local_output}, split_factor={self.split_factor})"
        )


class MoEParamShard(CustomParallelStyle):
    """Parameter-only TP/EP style for MoE expert weights — sharding, no forward hook.

    Wraps the named 3D expert parameters (``gate_up_proj`` / ``down_proj`` and their
    biases) as DTensor placeholders with the configured placement. The matching forward
    communication is declared *separately* in the plan via ``moe_experts_allreduce`` on
    the experts *module*. This is the param-granularity decomposition requested in review:
    weight sharding lives in configs at parameter level, module entries are forward-comm only.

    Applied by the param-level pass of ``apply_tensor_parallel`` (which walks
    ``named_parameters()`` and calls ``shard_parameters`` directly), not via ``_apply``.

    ``shards_expert_dim`` is set for the EP ``grouped_gemm`` style, which shards dim 0
    (the expert dimension). When True, ``module.num_experts`` is updated to the per-rank
    local count so the experts forward (histc / clamp / sentinel masking) and the
    ``ep_router`` sentinel index agree.
    """

    def __init__(self, placement, *, shards_expert_dim: bool = False):
        super().__init__()
        self.placement = placement
        self.shards_expert_dim = shards_expert_dim

    def shard_parameters(self, module, mesh, param_names=None):
        # Wrap the requested params as DTensor placeholders. Runs on meta —
        # distribute_tensor builds metadata only, no collective.
        if not param_names:
            return
        for name in param_names:
            meta = module._parameters.get(name)
            if meta is None:
                continue
            if self.shards_expert_dim:
                # dim 0 is the expert dimension; record the per-rank local count so the
                # experts forward and the EP router agree on local expert ids/sentinel.
                module.num_experts = meta.shape[0] // mesh.size()
            module._parameters[name] = torch.nn.Parameter(
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
        world_size, which is what DTensor's ``Replicate`` backward does incorrectly.
        """

        @staticmethod
        def forward(ctx, x, process_group):
            ctx.process_group = process_group
            return x

        @staticmethod
        def backward(ctx, grad):
            dist.all_reduce(grad, group=ctx.process_group)
            return grad, None


class MoEExpertsParallel(CustomParallelStyle):
    """Tensor-parallel forward style for MoE expert modules.

    Forward-comm only — it does NOT shard weights. Expert weight sharding is declared
    per-parameter in the config plan (``grouped_gemm`` / ``moe_tp_gate_up_*`` / ``moe_tp_down_*``)
    and applied by the param-level pass; this style only wraps the experts ``forward`` so
    that grouped_mm (which needs plain tensors) works on the already-sharded DTensor params.

    Lifecycle phases:
    1. transform_inputs_pre_forward — localize hidden_states (Replicate→local,
       gives us an all-reduce on the backward gradient for free), then fix
       routing-weight gradients (their backward is partial; use allreduce-sum,
       not divide-by-world-size).
    2. context_around_forward — swap DTensor params for local leaves so
       grouped_mm sees plain tensors; restored on exit so save_pretrained
       still sees DTensors.
    3. transform_output_post_forward — under TP-only each rank's output is
       partial (only its expert shard contributed); reduce/redistribute to
       output_layouts.
    """

    def __init__(self, output_layouts=None):
        super().__init__()
        self.output_layouts = output_layouts or Replicate()

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        hidden_states, top_k_index, top_k_weights = args
        if not isinstance(hidden_states, DTensor):
            hidden_states = DTensor.from_local(hidden_states, mesh, [Replicate()], run_check=False)
        hidden_states = hidden_states.to_local()

        if isinstance(top_k_weights, DTensor):
            top_k_weights = top_k_weights.to_local()
        # Under TP the router runs replicated, so its routing-weight gradient is partial
        # across ranks and needs an all-reduce-sum in backward. Under EP the router output
        # is already sliced per-rank by `ep_router` (non-local scores zeroed), so each
        # rank's routing weights are independent — skip the all-reduce to avoid
        # double-counting the gradient.
        if not _is_expert_parallel_enabled(module):
            tp_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
            top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_group)

        return (hidden_states, top_k_index, top_k_weights), kwargs

    def context_around_forward(self, module):
        return _swap_dtensor_params_for_local(module)

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


def _is_expert_parallel_enabled(module) -> bool:
    """Whether the model owning ``module`` was loaded with expert parallelism enabled."""
    config = getattr(module, "config", None)
    dist_config = getattr(config, "distributed_config", None)
    return bool(getattr(dist_config, "enable_expert_parallel", False))


class EpRouterParallel(CustomParallelStyle):
    """Expert-parallel router: forward-only slicing of router outputs to local experts.

    Ported from the original ``RouterParallel`` (#39501). The gate runs replicated on
    every rank and emits global ``(router_logits, router_scores, router_indices)``. Under
    EP each rank owns ``num_experts // ep_size`` experts, so this post-forward hook zeroes
    the scores of non-local experts and remaps the surviving indices into the local range,
    using ``num_local_experts`` as the sentinel for dropped (non-local) slots. The
    downstream experts forward masks that sentinel and ``moe_experts_allreduce`` sums the
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

    Adding a new entry here is the supported way to introduce a new TP style.
    Users can also override or extend at runtime via ``ALL_PARALLEL_STYLES["my_style"] = ...``.

    Naming convention: ``{kind}[_{comm}][_{extra}]``. The ``_{comm}`` suffix is dropped only when
    comm is ``"none"`` (no collective). All entries are eager instances; the dict literal lives
    behind a torch-availability guard so this module remains importable without torch.
    """

    _global_mapping = (
        {
            # Column-parallel
            "colwise": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1)),
            "colwise_allgather": ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
            "colwise_loss_parallel": ColwiseParallel(
                input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False
            ),
            "packed_colwise": PackedColwiseParallel(input_layouts=Replicate()),
            # MoE expert weight sharding — param-level (no forward hook). The matching
            # forward comm is declared separately via "moe_experts_allreduce" on the
            # experts module. gate_up is packed (gate||up) along dim -2
            # (Qwen3/Mixtral: [E, 2*inter, hidden]).
            "grouped_gemm": MoEParamShard(Shard(0), shards_expert_dim=True),  # EP — expert dim
            "moe_tp_gate_up_colwise": MoEParamShard(_StridedShard(dim=-2, split_factor=2)),  # TP — packed gate/up
            "moe_tp_down_rowwise": MoEParamShard(Shard(-1)),  # TP — down_proj input dim
            # Row-parallel
            "rowwise_allreduce": RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
            "rowwise_reduce_scatter": RowwiseParallel(input_layouts=Shard(-1), output_layouts=Shard(1)),
            # Vocab / embedding (rowwise sharding on vocab dim)
            "vocab_allreduce": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
            "vocab_reduce_scatter": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            # Activation / norm (sequence-parallel passthrough)
            # use_local_output=True: torch defaults to False here, but downstream modeling
            # code expects plain tensors, not DTensors.
            "activation": SequenceParallel(use_local_output=True),
            "activation_seq_dim_2": SequenceParallel(sequence_dim=2, use_local_output=True),
            # Module-level prepare-input. Same use_local_output=True override as above —
            # torch's default is False, our modeling code expects plain tensors downstream.
            "module_allgather": PrepareModuleInput(
                input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),), use_local_output=True
            ),
            "module_allgather_hidden_states": PrepareModuleInput(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
                use_local_output=True,
            ),
            "module_allgather_split": PrepareModuleInputOutput(),
            # MoE expert forward comm (no weight sharding — that is declared per-param via
            # the "grouped_gemm" / "moe_*" styles above on gate_up_proj / down_proj). This
            # entry only installs the forward hook that all-reduces the partial per-rank
            # expert outputs. Used for both TP and EP.
            "moe_experts_allreduce": MoEExpertsParallel(output_layouts=Replicate()),
            # EP router — forward-only slicing of router outputs to local experts.
            "ep_router": EpRouterParallel(),
        }
        if is_torch_available() and is_torch_greater_or_equal("2.5") and _torch_distributed_available
        else {}
    )


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()


# Styles that shard a *parameter* (declared at param granularity in configs) rather than
# installing a module forward hook. These are applied in a dedicated first pass that walks
# named_parameters(); any matching forward comm is a separate module-level entry.
PARAM_ONLY_STYLES = frozenset(
    {
        "grouped_gemm",
        "moe_tp_gate_up_colwise",
        "moe_tp_down_rowwise",
    }
)

MOE_TP_PARAM_STYLES = frozenset({"moe_tp_gate_up_colwise", "moe_tp_down_rowwise"})


def _merge_plans(base: dict[str, str], overlay: dict[str, str]) -> dict[str, str]:
    """Merge parallel plans; overlay entries win on duplicate keys."""
    merged = dict(base)
    merged.update(overlay)
    return merged


def _strip_moe_tp_when_ep(plan: dict[str, str], enable_ep: bool) -> dict[str, str]:
    if not enable_ep:
        return plan
    return {
        key: style
        for key, style in plan.items()
        if not (key.endswith((".gate_up_proj", ".down_proj")) and style in MOE_TP_PARAM_STYLES)
    }


def resolve_parallel_plan(model, distributed_config) -> dict[str, str]:
    """Compose SP/TP dense recipe with optional EP MoE overlay.

    When ``distributed_config.tp_plan`` is set, it is returned as-is. Otherwise the
    base plan is ``_sp_plan`` if sequence parallelism is requested and the config defines
    ``base_model_sp_plan``, else ``_tp_plan``. Expert-parallel entries from ``_ep_plan``
    are merged on top when ``enable_expert_parallel``; EP wins on key conflicts and any
    leftover intra-expert ``moe_tp_*`` entries are stripped.
    """
    if distributed_config is None:
        return dict(getattr(model, "_tp_plan", None) or {})

    if distributed_config.tp_plan is not None:
        return dict(distributed_config.tp_plan)

    enable_sp = bool(
        getattr(distributed_config, "enable_sequence_parallel", False)
        and getattr(model.config, "base_model_sp_plan", None) is not None
    )
    enable_ep = bool(getattr(distributed_config, "enable_expert_parallel", False))

    if enable_sp:
        base = dict(getattr(model, "_sp_plan", None) or {})
    else:
        base = dict(getattr(model, "_tp_plan", None) or {})

    if enable_ep:
        base = _merge_plans(base, dict(getattr(model, "_ep_plan", None) or {}))

    return _strip_moe_tp_when_ep(base, enable_ep)


def apply_tensor_parallel(model, tp_mesh, tp_plan):
    """Apply tensor parallelism in two passes, looking each style up by string name.

    Pass 1 (param-level): walk ``model.named_parameters()`` and, for keys whose plan
    style is in ``PARAM_ONLY_STYLES`` (e.g. MoE expert weights), shard the parameter
    directly as a DTensor placeholder. No forward hook is installed.

    Pass 2 (module-level): walk ``model.named_modules()``, resolve each name against
    the wildcard ``tp_plan`` and call the style's ``_apply`` (forward hooks + optional
    Linear sharding). Param sharding runs first so module forward hooks (e.g.
    ``moe_experts_allreduce``) see already-sharded DTensor params.
    """
    distributed_config = getattr(model.config, "distributed_config", None)
    sp_requested = getattr(distributed_config, "enable_sequence_parallel", False)
    sp_supported = getattr(model.config, "base_model_sp_plan", None) is not None
    enable_sp = sp_requested and sp_supported

    enable_ep = getattr(distributed_config, "enable_expert_parallel", False)

    if tp_plan is None:
        if enable_sp:
            tp_plan = dict(model._sp_plan or {})
        elif enable_ep:
            tp_plan = dict(model._ep_plan or {})
        else:
            tp_plan = dict(model._tp_plan or {})

    # tie_weights() replaces lm_head.weight with embed_tokens.weight after TP is applied.
    # If embed_tokens isn't in the plan, sharding lm_head as a DTensor causes tie to
    # clobber it with a plain tensor (and forward then mixes DTensor/Tensor). Skip
    # lm_head TP in that case so both ends stay plain and the tie is a real alias.
    if getattr(model.config, "tie_word_embeddings", False):
        tied_source_in_plan = any(k.endswith("embed_tokens") for k in tp_plan)
        if not tied_source_in_plan:
            tp_plan.pop("lm_head", None)

    # Pass 1: param-level sharding (meta DTensor placeholders). Declared per-parameter
    # in the plan (e.g. MoE expert weights). Materialize the iterator first since we
    # mutate each parent module's `_parameters` as we go.
    for param_name, _ in list(model.named_parameters()):
        style_name = _get_parameter_tp_plan(parameter_name=param_name, tp_plan=tp_plan, is_weight=True)
        if style_name not in PARAM_ONLY_STYLES:
            continue
        parent_name, short_name = param_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        ALL_PARALLEL_STYLES[style_name].shard_parameters(parent, tp_mesh, param_names=[short_name])

    # Pass 2: module-level styles (forward hooks + optional sharding for Linear modules).
    for name, submodule in model.named_modules():
        style_value = _get_parameter_tp_plan(parameter_name=name, tp_plan=tp_plan, is_weight=False)
        if style_value is None or style_value in PARAM_ONLY_STYLES:
            continue
        ALL_PARALLEL_STYLES[style_value]._apply(submodule, tp_mesh)

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
