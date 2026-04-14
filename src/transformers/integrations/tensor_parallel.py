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

import re
from dataclasses import dataclass
from typing import Literal

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

from ..utils import logging
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist

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
# Tensor Sharding Utilities
# =============================================================================


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


def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str | TPStyle] | None):
    """
    Verify the TP plan of the model, log a warning if the layers that were not sharded and the rules that were not applied.

    Only weight-sharding rules (colwise, rowwise, vocab, moe_experts) are checked.
    Module/activation entries (e.g. PrepareModuleInput, SequenceParallel) set up
    communication hooks on modules, not weight sharding, so they are excluded.
    """

    if tp_plan is None:
        return

    # Filter out module-level comm hooks — they don't shard weights
    _NON_WEIGHT_KINDS = {"activation", "module"}
    weight_plan = {k: v for k, v in tp_plan.items() if not isinstance(v, TPStyle) or v.kind not in _NON_WEIGHT_KINDS}

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

    def _prepare_input_fn(self, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, self.input_layouts, run_check=False)
        elif input_tensor.placements != self.input_layouts:
            input_tensor = input_tensor.redistribute(placements=self.input_layouts)
        input_tensor = input_tensor.to_local()

        local_param_shadows = {}
        for param_name, param in list(mod.named_parameters(recurse=False)):
            if isinstance(param, DTensor):
                local_param_shadows[param_name] = param
                mod._parameters.pop(param_name)
                setattr(mod, param_name, param.to_local())
        if local_param_shadows:
            shadow_stack = getattr(mod, "_packed_local_param_shadows", None)
            if shadow_stack is None:
                shadow_stack = []
                mod._packed_local_param_shadows = shadow_stack
            shadow_stack.append(local_param_shadows)
        return (input_tensor,) + inputs[1:]

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        shadow_stack = getattr(mod, "_packed_local_param_shadows", None)
        if shadow_stack:
            for param_name, param in shadow_stack.pop().items():
                if hasattr(mod, param_name):
                    delattr(mod, param_name)
                mod.register_parameter(param_name, param)

        if outputs is None or self.use_local_output:
            return outputs
        return DTensor.from_local(
            outputs, device_mesh, (_StridedShard(dim=-1, split_factor=self.split_factor),), run_check=False
        )

    def _apply(self, module, device_mesh):
        if not isinstance(module, torch.nn.Linear):
            raise NotImplementedError("PackedColwiseParallel currently only supports nn.Linear!")

        self._partition_linear_fn(module, device_mesh)
        module.register_forward_pre_hook(lambda mod, inputs: self._prepare_input_fn(mod, inputs, device_mesh))
        module.register_forward_hook(
            lambda mod, inputs, outputs: self._prepare_output_fn(mod, outputs, device_mesh),
            always_call=True,
        )
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
    """Hybrid parallel style for MoE expert modules.

    Converts expert weights to DTensors based on the ``shard_plan`` (e.g.
    ``{"gate_up_proj": "packed_colwise", "down_proj": "rowwise"}``).
    Communication uses DTensor ``from_local``/``to_local`` on activations only —
    compatible with ``grouped_mm``.
    """

    def __init__(self, output_layouts=None):
        super().__init__()
        self.output_layouts = output_layouts or Replicate()
        self._moe_shard_plan: dict[str, str] = {}

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

    @staticmethod
    def _uses_partial_outputs(mod) -> bool:
        cached = getattr(mod, "_moe_outputs_are_partial", None)
        if cached is not None:
            return cached

        # Under TP-only the expert MLP dimension is sharded, so each rank emits a
        # partial hidden-state contribution that must be reduced. Under TP+FSDP,
        # FSDP can swap in full gathered expert weights for the current rank's
        # forward, in which case the local output is already complete.
        if hasattr(mod, "gate_up_proj"):
            gate_up_proj = mod.gate_up_proj.to_local() if isinstance(mod.gate_up_proj, DTensor) else mod.gate_up_proj
            full_expert_out = 2 * mod.intermediate_dim
            sharded_dim = -1 if getattr(mod, "is_transposed", False) else -2
            cached = gate_up_proj.shape[sharded_dim] != full_expert_out
        elif hasattr(mod, "up_proj"):
            up_proj = mod.up_proj.to_local() if isinstance(mod.up_proj, DTensor) else mod.up_proj
            full_expert_out = mod.intermediate_dim
            sharded_dim = -1 if getattr(mod, "is_transposed", False) else -2
            cached = up_proj.shape[sharded_dim] != full_expert_out
        else:
            cached = True

        mod._moe_outputs_are_partial = cached
        return cached

    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh):
        hidden_states, top_k_index, top_k_weights = inputs[0], inputs[1], inputs[2]
        # from_local([Replicate()]).to_local(): forward sees plain tensor,
        # backward graph goes through DTensor all-reduce on gradient.
        if not isinstance(hidden_states, DTensor):
            hidden_states = DTensor.from_local(hidden_states, device_mesh, [Replicate()], run_check=False)
        hidden_states = hidden_states.to_local()
        # Route weights are replicated (same on all ranks), but their backward
        # gradient is partial (each rank's contribution from its expert shard).
        # Use allreduce-sum (not Replicate's allreduce-then-divide) to aggregate.
        tp_group = device_mesh.get_group() if device_mesh.ndim == 1 else device_mesh.get_group("tp")
        if isinstance(top_k_weights, DTensor):
            top_k_weights = top_k_weights.to_local()
        top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_group)
        local_param_shadows = {}
        for param_name, param in list(mod.named_parameters(recurse=False)):
            if isinstance(param, DTensor):
                # grouped_mm expects plain tensors, but we must restore the
                # original DTensor params after the forward so save_pretrained
                # still sees the canonical sharded weights.
                local_param_shadows[param_name] = param
                mod._parameters.pop(param_name)
                setattr(mod, param_name, param.to_local())
        if local_param_shadows:
            shadow_stack = getattr(mod, "_moe_local_param_shadows", None)
            if shadow_stack is None:
                shadow_stack = []
                mod._moe_local_param_shadows = shadow_stack
            shadow_stack.append(local_param_shadows)
        return (hidden_states, top_k_index, top_k_weights)

    @staticmethod
    def _prepare_output_fn(output_layouts, mod, outputs, device_mesh):
        shadow_stack = getattr(mod, "_moe_local_param_shadows", None)
        if shadow_stack:
            for param_name, param in shadow_stack.pop().items():
                if hasattr(mod, param_name):
                    delattr(mod, param_name)
                mod.register_parameter(param_name, param)
        if outputs is None:
            return None
        # Plain TP expert weights produce partial outputs that need an all-reduce.
        # TP+FSDP can leave experts replicated across TP and sharded only across
        # experts/FSDP, in which case the local output is already complete.
        source_layout = Partial() if MoEExpertsParallel._uses_partial_outputs(mod) else Replicate()
        if not isinstance(outputs, DTensor):
            outputs = DTensor.from_local(outputs, device_mesh, [source_layout], run_check=False)
        # MoE experts output 2D [num_tokens, hidden]. For SP reduce-scatter,
        # Shard(1) means sequence dim in 3D, but in 2D the token dim is 0.
        actual_layouts = output_layouts
        if outputs.dim() == 2 and isinstance(output_layouts, Shard) and output_layouts.dim == 1:
            actual_layouts = Shard(0)
        if outputs.placements != (actual_layouts,):
            outputs = outputs.redistribute(placements=(actual_layouts,))
        return outputs.to_local()

    def _apply(self, module, device_mesh):
        # Don't use PyTorch's distribute_module — it would auto-convert all
        # params to Replicate DTensors. We create DTensors with proper Shard
        # placements in _partition_fn instead, and register hooks manually.
        self._partition_fn(module.__class__.__name__, module, device_mesh, self._moe_shard_plan)
        module.register_forward_pre_hook(lambda mod, inputs: self._prepare_input_fn(mod, inputs, device_mesh))
        module.register_forward_hook(
            lambda mod, inputs, outputs: self._prepare_output_fn(self.output_layouts, mod, outputs, device_mesh),
            always_call=True,
        )
        return module


@dataclass(frozen=True)
class TPStyle:
    kind: Literal["colwise", "packed_colwise", "rowwise", "vocab", "activation", "module", "moe_experts"]
    comm: Literal["none", "allreduce", "reduce_scatter", "allgather", "allgather_split", "loss_parallel"]
    sequence_dim: int = 1
    use_local_output: bool = True
    input_key: str | None = None
    shard_plan: dict[str, str] | None = None

    def to_dtensor_style(self) -> ParallelStyle:
        """Convert to the corresponding PyTorch DTensor ParallelStyle."""
        if self.kind == "colwise":
            match self.comm:
                case "none":
                    return ColwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=self.use_local_output
                    )
                case "allgather":
                    return ColwiseParallel(
                        input_layouts=Replicate(),
                        output_layouts=Replicate(),
                        use_local_output=self.use_local_output,
                    )
                case "loss_parallel":
                    return ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False)
        elif self.kind == "packed_colwise":
            match self.comm:
                case "none":
                    return PackedColwiseParallel(input_layouts=Replicate(), use_local_output=self.use_local_output)
        elif self.kind == "rowwise":
            match self.comm:
                case "allreduce":
                    return RowwiseParallel(
                        input_layouts=Shard(-1),
                        output_layouts=Replicate(),
                        use_local_output=self.use_local_output,
                    )
                case "reduce_scatter":
                    return RowwiseParallel(
                        input_layouts=Shard(-1),
                        output_layouts=Shard(1),
                        use_local_output=self.use_local_output,
                    )
        elif self.kind == "vocab":
            match self.comm:
                case "allreduce":
                    return RowwiseParallel(
                        input_layouts=Replicate(),
                        output_layouts=Replicate(),
                        use_local_output=self.use_local_output,
                    )
                case "reduce_scatter":
                    return RowwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=self.use_local_output
                    )
        elif self.kind == "activation":
            match self.comm:
                case "none":
                    return SequenceParallel(sequence_dim=self.sequence_dim, use_local_output=self.use_local_output)
        elif self.kind == "module":
            match self.comm:
                case "allgather":
                    if self.input_key is not None:
                        return PrepareModuleInput(
                            input_kwarg_layouts={self.input_key: Shard(1)},
                            desired_input_kwarg_layouts={self.input_key: Replicate()},
                            use_local_output=self.use_local_output,
                        )
                    return PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                        use_local_output=self.use_local_output,
                    )
                case "allgather_split":
                    return PrepareModuleInputOutput(use_local_output=self.use_local_output)
        elif self.kind == "moe_experts":
            match self.comm:
                case "allreduce":
                    return MoEExpertsParallel(output_layouts=Replicate())
                case "reduce_scatter":
                    return MoEExpertsParallel(output_layouts=Shard(1))
        raise ValueError(
            f"Invalid TPStyle({self.kind!r}, {self.comm!r}). Valid combinations:\n"
            f"  colwise:        none, allgather, loss_parallel\n"
            f"  packed_colwise: none\n"
            f"  rowwise:     allreduce, reduce_scatter\n"
            f"  vocab:       allreduce, reduce_scatter\n"
            f"  activation:  none\n"
            f"  module:      allgather, allgather_split\n"
            f"  moe_experts: allreduce, reduce_scatter"
        )

    def __str__(self):
        if self.comm == "none":
            return self.kind
        return f"{self.kind}_{self.comm}"


def apply_tensor_parallel(model, tp_mesh, tp_plan):
    """Apply tensor parallelism using PyTorch's parallelize_module.

    Converts the wildcard tp_plan from model config into a concrete plan
    for ``parallelize_module``. Plan values is a `TPStyle`` instances
    """
    if tp_plan is None:
        return model

    if tp_plan == "auto":
        enable_sp = getattr(getattr(model.config, "distributed_config", None), "enable_sequence_parallel", False)
        if enable_sp and hasattr(model.config, "base_model_sp_plan"):
            base_plan = model.config.base_model_sp_plan
        else:
            base_plan = model.config.base_model_tp_plan or {}

        # Prefix base model keys (e.g. "layers.*.q_proj" → "model.layers.*.q_proj")
        # Top-level keys like "lm_head" are kept as-is.
        base_model_prefix = model.base_model_prefix
        tp_plan = {}
        for k, v in base_plan.items():
            is_top_level = hasattr(model, k.split(".")[0])
            tp_plan[k if is_top_level else f"{base_model_prefix}.{k}"] = v

    parallelize_plan = {}

    for name, _ in model.named_modules():
        style_value = _get_parameter_tp_plan(parameter_name=name, tp_plan=tp_plan, is_weight=False)
        if style_value is None:
            continue

        if isinstance(style_value, TPStyle):
            dtensor_style = style_value.to_dtensor_style()
            parallelize_plan[name] = dtensor_style
            # For MoE modules, attach the per-parameter shard plan from TPStyle
            # so _partition_fn can create DTensors with the correct placements.
            if isinstance(dtensor_style, MoEExpertsParallel) and style_value.shard_plan:
                dtensor_style._moe_shard_plan = style_value.shard_plan
        else:
            parallelize_plan[name] = style_value

    parallelize_module(model, tp_mesh, parallelize_plan)

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
    has_loss_parallel = any(isinstance(v, TPStyle) and v.comm == "loss_parallel" for v in tp_plan.values())
    if has_loss_parallel:
        from torch.distributed.tensor.parallel import loss_parallel

        model._loss_parallel_ctx = loss_parallel()
        model._loss_parallel_ctx.__enter__()

    return model
