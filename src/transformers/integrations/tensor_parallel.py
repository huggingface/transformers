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
        parallelize_module,
    )
    from torch.distributed.tensor.parallel.style import ParallelStyle
    from torch.distributed.tensor.placement_types import _StridedShard

    from ..distributed.patches import patch_dtensor_ops

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


def _to_cpu_fresh(tensor: torch.Tensor) -> torch.Tensor:
    """Plain tensor → contiguous CPU tensor with fresh storage for safetensors."""
    if tensor.device.type == "meta":
        return tensor
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to(device="cpu")
    out = torch.empty(t.shape, dtype=t.dtype, device="cpu")
    out.copy_(t)
    return out.contiguous()


def gather_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather all sharded params to full plain tensors for saving.

    Handles FSDP unshard and TP DTensor gather.
    Streams one parameter at a time to avoid holding all full tensors on GPU.
    Only rank 0 accumulates the result; other ranks return ``{}``.
    """
    tp_size = model.tp_size
    is_rank0 = dist.get_rank() == 0

    # Get state dict — FSDP unshard if needed (returns DTensors, not full tensors)
    if getattr(model, "_is_fsdp_managed_module", False):
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        state_dict = get_model_state_dict(model)
    else:
        state_dict = model.state_dict()

    # No TP — materialize on rank 0 only
    if tp_size is None:
        if is_rank0:
            return {k: _to_cpu_fresh(v) for k, v in state_dict.items()}
        return {}

    # Stream: gather one param at a time, only rank 0 keeps the CPU copy
    result = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, DTensor):
            # All ranks participate in the collective, only rank 0 keeps the result
            with torch.no_grad():
                full = _replicate_dtensor(tensor).to_local()
            if is_rank0:
                result[key] = _to_cpu_fresh(full)
            del full
        elif is_rank0:
            result[key] = _to_cpu_fresh(tensor)

    return result


def _replicate_dtensor(tensor: DTensor) -> DTensor:
    """All-gather a DTensor to fully Replicate, handling ``_StridedShard``.

    PyTorch's ``redistribute()`` does not support ``_StridedShard`` as a source::

        _StridedShard -> redistribute() -> Replicate      ❌ AssertionError
        _StridedShard -> redistribute() -> Shard           ❌ NotImplementedError
        Shard         -> redistribute() -> Replicate      ✅ works
        Replicate     -> redistribute() -> Shard           ✅ works
        Replicate     -> redistribute() -> _StridedShard  ✅ works

    So we bypass ``redistribute`` and call each placement's low-level
    ``_to_replicate_tensor`` (manual all-gather + interleaved reorder).

    We process mesh dims **right-to-left** (innermost first).  Under TP+FSDP
    the 2D mesh is ``(fsdp, tp)`` and both dims can shard the same tensor dim::

        placements = (_StridedShard(dim=0), Shard(dim=0))
        local shape = [64, 1024]   (global [256, 1024], fsdp=2, tp=2)

    Right-to-left means TP is gathered first (local grows to [128, 1024]),
    then FSDP (grows to [256, 1024]).  Each step must pass the correct
    intermediate logical shape — the global shape divided by the mesh sizes
    of dims not yet gathered (to the left).
    """
    mesh = tensor.device_mesh
    replicate_all = tuple(Replicate() for _ in range(mesh.ndim))
    with torch.no_grad():
        if any(isinstance(p, _StridedShard) for p in tensor.placements):
            local = tensor._local_tensor
            placements = tensor.placements
            for i in reversed(range(mesh.ndim)):
                p = placements[i]
                if p.is_replicate():
                    continue
                # Compute the logical shape seen at this step: dims to the left
                # (not yet gathered) still divide their tensor dimension.
                logical_shape = list(tensor.shape)
                for j in range(i):
                    pj = placements[j]
                    if not pj.is_replicate():
                        logical_shape[pj.dim] //= mesh.size(j)
                local = p._to_replicate_tensor(local, mesh, i, logical_shape)
            return DTensor.from_local(local, mesh, replicate_all, run_check=False)

        return tensor.redistribute(placements=replicate_all)


def convert_strided_to_shard(state_dict: dict) -> dict[str, tuple]:
    # Convert _StridedShard DTensors in a state dict to plain Shard for DCP compatibility.
    placement_map: dict[str, tuple] = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            nested = convert_strided_to_shard(value)
            for nk, nv in nested.items():
                placement_map[f"{key}.{nk}"] = nv
        elif isinstance(value, DTensor) and any(isinstance(p, _StridedShard) for p in value.placements):
            placement_map[key] = tuple(value.placements)
            shard_placements = tuple(Shard(p.dim) if isinstance(p, _StridedShard) else p for p in value.placements)
            state_dict[key] = _replicate_dtensor(value).redistribute(placements=shard_placements)
    return placement_map


def restore_strided_from_shard(state_dict: dict, placement_map: dict[str, tuple]) -> None:
    # Restore _StridedShard placements after dcp.load.
    def _resolve(d, dotted_key):
        parts = dotted_key.split(".", 1)
        if len(parts) == 2 and parts[0] in d and isinstance(d[parts[0]], dict):
            return _resolve(d[parts[0]], parts[1])
        return d, dotted_key

    for key, original_placements in placement_map.items():
        container, leaf_key = _resolve(state_dict, key)
        if leaf_key in container and isinstance(container[leaf_key], DTensor):
            container[leaf_key] = _replicate_dtensor(container[leaf_key]).redistribute(placements=original_placements)


def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str] | None):
    """
    Verify the TP plan of the model, log a warning if the layers that were not sharded and the rules that were not applied.

    Only weight-sharding rules (colwise, rowwise, vocab, moe_experts) are checked.
    Module/activation entries (e.g. PrepareModuleInput, SequenceParallel) set up
    communication hooks on modules, not weight sharding, so they are excluded.
    """

    if tp_plan is None:
        return

    # Filter out module-level comm hooks — they don't shard weights.
    # Plan values are registry names; entries beginning with "activation" or "module"
    # configure communication hooks rather than parameter sharding.
    weight_plan = {
        k: v for k, v in tp_plan.items() if not (v == "activation" or v.startswith(("activation_", "module_")))
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


class PrepareModuleInputOutput(ParallelStyle):
    """Allgather input (Shard(1) → Replicate) + local split output (Replicate → Shard(1)).

    Used for MoE blocks with SP: the input sequence is gathered before routing,
    and the output (after expert allreduce) is split back to match the residual.
    Forward output split is a local op (no comm). Backward creates the all-gather.
    """

    def __init__(self, use_local_output=True):
        super().__init__()
        self.use_local_output = use_local_output

    def _apply(self, module, device_mesh):
        def input_hook(mod, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if not isinstance(x, DTensor):
                x = DTensor.from_local(x, device_mesh, [Shard(1)], run_check=False)
            x = x.redistribute(placements=[Replicate()])
            x = x.to_local()
            return (x,) + (inputs[1:] if isinstance(inputs, tuple) else ())

        def output_hook(mod, inputs, output):
            if not isinstance(output, DTensor):
                output = DTensor.from_local(output, device_mesh, [Replicate()], run_check=False)
            output = output.redistribute(placements=[Shard(1)])
            return output.to_local()

        module.register_forward_pre_hook(input_hook)
        module.register_forward_hook(output_hook)
        return module


def _accumulate_local_param_grad(original_param: DTensor, local_grad: torch.Tensor) -> torch.Tensor:
    """Stitch a local grad back onto the original DTensor parameter.

    During forward we replace the DTensor param with a detached plain-tensor
    leaf (see ``_local_dtensor_params``) because ``grouped_mm`` / fused ops do
    not accept DTensor inputs. That swap breaks the autograd link between the
    local leaf's grad and the DTensor param's ``.grad``, so this tensor hook
    runs on the leaf and copies/accumulates the grad onto the original DTensor.
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
def _local_dtensor_params(module):
    """Temporarily swap DTensor params for local leaf params during one forward.

    Needed because ``grouped_mm`` / fused ops do not accept DTensor inputs: we
    forward through a detached plain-tensor leaf, then rely on
    ``_accumulate_local_param_grad`` (registered as a tensor hook on the leaf)
    to copy the backward grad onto the original DTensor param. Restores the
    DTensor params on exit (even on exception).
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


class PackedColwiseParallel(ParallelStyle):
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

    def _partition_linear_fn(self, module, device_mesh):
        if getattr(module, "weight", None) is None:
            return

        packed_shard = _StridedShard(dim=0, split_factor=self.split_factor)
        module.register_parameter(
            "weight",
            torch.nn.Parameter(
                distribute_tensor(module.weight, device_mesh, [packed_shard], src_data_rank=self.src_data_rank),
                requires_grad=module.weight.requires_grad,
            ),
        )

        if getattr(module, "bias", None) is not None:
            module.register_parameter(
                "bias",
                torch.nn.Parameter(
                    distribute_tensor(module.bias, device_mesh, [packed_shard], src_data_rank=self.src_data_rank),
                    requires_grad=module.bias.requires_grad,
                ),
            )

    def _apply(self, module, device_mesh):
        if not isinstance(module, torch.nn.Linear):
            raise NotImplementedError("PackedColwiseParallel currently only supports nn.Linear!")

        self._partition_linear_fn(module, device_mesh)

        input_layouts = self.input_layouts
        use_local_output = self.use_local_output
        split_factor = self.split_factor
        original_forward = module.forward

        def tp_forward(input_tensor, *args, **kwargs):
            if not isinstance(input_tensor, DTensor):
                input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)
            elif input_tensor.placements != input_layouts:
                input_tensor = input_tensor.redistribute(placements=input_layouts)
            input_tensor = input_tensor.to_local()

            with _local_dtensor_params(module):
                output = original_forward(input_tensor, *args, **kwargs)

            if output is None or use_local_output:
                return output
            return DTensor.from_local(
                output, device_mesh, (_StridedShard(dim=-1, split_factor=split_factor),), run_check=False
            )

        module.forward = tp_forward
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}, "
        tmpstr += f"split_factor={self.split_factor}"
        tmpstr += ")"
        return tmpstr


# Maps string tp_plan entries for MoE experts to DTensor placements.
# Used by MoEExpertsParallel._partition_fn to create DTensors from the config plan.
_STRING_TO_PLACEMENT = {
    "packed_colwise": lambda: _StridedShard(dim=-2, split_factor=2),
    "colwise": lambda: Shard(-2),
    "rowwise": lambda: Shard(-1),
}


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


class MoEExpertsParallel(ParallelStyle):
    """Tensor-parallel style for MoE expert modules.

    Shards expert weights as DTensors, then wraps the module's ``forward`` so
    that grouped_mm (which needs plain tensors) works transparently.

    The wrapped forward does four things:
    1. Localize inputs  — wrap hidden_states as Replicate DTensor then extract
       local tensor (gives us an all-reduce on the backward gradient for free).
    2. Fix routing grads — routing weights are the same on all ranks, but their
       backward gradient is partial; use allreduce-sum (not divide-by-world-size).
    3. Swap params      — temporarily replace DTensor params with local tensors
       for grouped_mm, restore them after so save_pretrained sees DTensors.
    4. Reduce output    — each rank's output is partial (only its expert shard
       contributed); all-reduce to get the complete hidden state.
    """

    def __init__(self, output_layouts=None, shard_plan: dict[str, str] | None = None):
        super().__init__()
        self.output_layouts = output_layouts or Replicate()
        self._moe_shard_plan: dict[str, str] = shard_plan or {}

    @staticmethod
    def _partition_fn(name, module, device_mesh, shard_plan):
        for param_name, param in module.named_parameters(recurse=False):
            plan_str = shard_plan.get(param_name)
            if plan_str is None:
                continue
            placement_fn = _STRING_TO_PLACEMENT.get(plan_str)
            if placement_fn is None:
                continue
            placement = placement_fn()
            dtensor = distribute_tensor(param.data, device_mesh, [placement])
            module._parameters[param_name] = torch.nn.Parameter(dtensor, requires_grad=param.requires_grad)

    def _apply(self, module, device_mesh):
        self._partition_fn(module.__class__.__name__, module, device_mesh, self._moe_shard_plan)

        output_layouts = self.output_layouts
        original_forward = module.forward
        tp_group = device_mesh.get_group() if device_mesh.ndim == 1 else device_mesh.get_group("tp")

        def tp_forward(hidden_states, top_k_index, top_k_weights):
            # --- 1. Localize hidden_states (backward all-reduce via DTensor) ---
            if not isinstance(hidden_states, DTensor):
                hidden_states = DTensor.from_local(hidden_states, device_mesh, [Replicate()], run_check=False)
            hidden_states = hidden_states.to_local()

            # --- 2. Fix routing weight gradients (allreduce-sum, not ÷ world_size) ---
            if isinstance(top_k_weights, DTensor):
                top_k_weights = top_k_weights.to_local()
            top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_group)

            # --- 3. Run forward with local params (grouped_mm needs plain tensors) ---
            with _local_dtensor_params(module):
                output = original_forward(hidden_states, top_k_index, top_k_weights)

            # --- 4. Reduce partial output ---
            if output is None:
                return None
            # Under TP-only each rank has a partial result; under TP+FSDP the
            # weights may be fully gathered by FSDP, making the output complete.
            has_sharded_params = any(
                isinstance(p, DTensor) and any(not pl.is_replicate() for pl in p.placements)
                for p in module.parameters()
            )
            source = Partial() if has_sharded_params else Replicate()
            if not isinstance(output, DTensor):
                output = DTensor.from_local(output, device_mesh, [source], run_check=False)
            # MoE output is 2D [tokens, hidden]. For SP, Shard(1) means seq dim
            # in 3D but token dim (0) in 2D.
            target = output_layouts
            if output.dim() == 2 and isinstance(target, Shard) and target.dim == 1:
                target = Shard(0)
            if output.placements != (target,):
                output = output.redistribute(placements=(target,))
            return output.to_local()

        module.forward = tp_forward
        return module


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
            # MoE — canonical shard_plan baked in (only variant in use across configs)
            "moe_experts_allreduce": MoEExpertsParallel(
                output_layouts=Replicate(),
                shard_plan={"gate_up_proj": "packed_colwise", "down_proj": "rowwise"},
            ),
        }
        if is_torch_available() and is_torch_greater_or_equal("2.5") and _torch_distributed_available
        else {}
    )


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()


def apply_tensor_parallel(model, tp_mesh, tp_plan):
    """Apply tensor parallelism using PyTorch's parallelize_module.

    Converts the wildcard tp_plan from model config into a concrete plan
    for ``parallelize_module``. Plan values are string names looked up in
    ``ALL_PARALLEL_STYLES``.
    """
    if tp_plan is None:
        return model

    if tp_plan == "auto":
        distributed_config = getattr(model.config, "distributed_config", None)
        sp_requested = getattr(distributed_config, "enable_sequence_parallel", False)
        sp_supported = getattr(model.config, "base_model_sp_plan", None) is not None

        enable_sp = sp_requested and sp_supported
        if enable_sp:
            tp_plan = dict(model._sp_plan or {})
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

    parallelize_plan = {}

    for name, _ in model.named_modules():
        style_value = _get_parameter_tp_plan(parameter_name=name, tp_plan=tp_plan, is_weight=False)
        if style_value is None:
            continue

        if not isinstance(style_value, str):
            raise TypeError(
                f"Unsupported plan value for '{name}': {style_value!r} (type {type(style_value).__name__}). "
                f"TP plan values must be strings looked up in ALL_PARALLEL_STYLES."
            )
        if style_value not in ALL_PARALLEL_STYLES:
            raise ValueError(
                f"Unknown TP style {style_value!r} for module {name!r}. Valid styles: {sorted(ALL_PARALLEL_STYLES)}"
            )
        parallelize_plan[name] = ALL_PARALLEL_STYLES[style_value]

    parallelize_module(model, tp_mesh, parallelize_plan)

    # Patch DTensor-aware operations (e.g. rotary embeddings) onto the
    # model's modeling module so modeling files stay free of DTensor code.
    patch_dtensor_ops(model)

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
