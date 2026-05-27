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
from typing import Any

import torch
from torch import nn

from ..utils import logging
from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_greater_or_equal


if is_torch_greater_or_equal("2.5"):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Partial, Placement, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.placement_types import _StridedShard

    # Cache this result as it's a C FFI call which can be pretty time-consuming
    _torch_distributed_available = torch.distributed.is_available()


logger = logging.get_logger(__name__)


# These functions are used to navifate the TP plan
def replace_layer_number_by_wildcard(name: str) -> str:
    """Replaces the numbers in the `name` by wildcards, only if they are in-between dots (`.`) or if they are between
    a dot (`.`) and the end of the string. This will match the use of module lists and sequences, but not parameter
    names like `"w1"` or `"w2"`. """
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def _get_parameter_tp_plan(parameter_name: str, tp_plan: dict[str, str], is_weight=True) -> str | None:
    """Gets the TP style for a parameter from the TP plan, which is a dict of parameter names (potentially with
    wildcards) to TP styles. Returns None if no match is found. If the `is_weight` flag is True, the check is widened to
    include the name of module that owns the parameter, eg. "model.1.attn.w_q" will match "model.*.attn" """
    # Try matching the generic parameter name
    generic_param_name = replace_layer_number_by_wildcard(parameter_name)
    if generic_param_name in tp_plan:
        return tp_plan[generic_param_name]
    # If that failed, and the parameter is a weight, try matching the module name
    if is_weight and "." in generic_param_name:
        module_name = generic_param_name.rsplit(".", 1)[0]  # model.*.something.some_weight -> model.*.something
        if module_name in tp_plan:
            return tp_plan[module_name]
    return None


# TODO: move and refactor
def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str] | None) -> None:
    """Verifies that all rules in the TP plan were used to distribute the model, and that the expected_keys were all
    distributed. For each offense, logs a warning."""
    # Early exit if there is nothing to verify
    if tp_plan is None:
        return None

    # Filter out the module-level communication style, because they only add hooks and they don't shard weights
    weight_plan = {
        k: v for k, v in tp_plan.items() if not (v == "activation" or v.startswith(("activation_", "module_")))
    }

    generic_keys = {replace_layer_number_by_wildcard(key) for key in expected_keys}
    unsharded_layers = set(generic_keys)
    unused_rules = weight_plan.copy()

    for key in generic_keys:
        param_name = key.rsplit(".", 1)[0] if "." in key else key
        generic_param_name = replace_layer_number_by_wildcard(param_name)

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


# These functions are used to swap between DTensor and plain tensors
def _accumulate_local_param_grad(original_param: DTensor, local_grad: torch.Tensor) -> torch.Tensor:
    """Stitch a local grad back onto the original DTensor parameter. This is needed when the forward swaps DTensors for
    plain tensors to accomodate for some kernels, which breaks the autograd link between the local leaf's .grad and the
    DTensor param's .grad. NOTE: this is only needed because we are not using the autograd-aware .to_local() swap, which
    is not compatible with _StridedShard placements, which are used by MoEExpertsParallel and PackedColwiseParallel."""
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
    """Temporarily replace DTensor params with local-shard ``Parameter``s for forward. This  isneeded for some kernels
    like grouped_mm or fused kernels that don't accept DTensor inputs. The original DTensor are restored on exit."""
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

# Below are all the TP styles that are supported by the transformers library
def matches_desired_layout(dtensor: DTensor, desired_layout: tuple[Placement, ...]) -> bool:
    """Checks if two placements are equivalent, including the case of a desired Shard with dim < 0."""
    for p, dp in zip(dtensor.placements, desired_layout, strict=True):
        if isinstance(dp, Shard) and dp.dim < 0:
            dp = Shard(dp.dim + dtensor.ndim)
        if p != dp:
            return False
    return True

class TensorParallelMixin:
    """Base class for transformers TP styles, which turn a module into a TP-aware module through the "make_tp_aware"
    method, which does the following:

    - replaces the module's forward method with a TP-aware forward method, which has a pre-forward hook, a context
    manager around the forward call, and a post-forward hook.

    - warps the module's parameters in DTensors while they are still on the meta device. This is only a CPU-side
    operation, actual data is loaded on the device later via DtensorShardOperation
    """

    def make_tp_aware(self, module: torch.nn.Module, tp_mesh: DeviceMesh) -> torch.nn.Module:
        """Make a module TP-aware by replacing its forward method with a TP-aware forward method and warping its
        parameters in DTensors based on the given tp_mesh."""
        # First, add hooks and context manager to the model's forward (this part is always the same)
        original_forward = module.forward

        def tp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(module, args, kwargs, tp_mesh)
            with self.context_around_forward(module):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, tp_mesh)

        module.forward = tp_forward
        # Second, warp the parameters in DTensors (this is the style-specific part)
        module = self.shard_meta_params(module, tp_mesh)
        return module

    def shard_meta_params(self, module: torch.nn.Module, tp_mesh: DeviceMesh) -> torch.nn.Module:
        """Shard the module's meta parameters in DTensors based on the given tp_mesh without loading the data onto the
        device, which will be done later."""
        return module

    @staticmethod
    def apply_placements_to_meta_params(
        module: torch.nn.Module, tp_mesh: DeviceMesh, placements: dict[str, tuple[Placement, ...]]
    ) -> torch.nn.Module:
        """Performs the actual wrapping of the parameters in DTensors according to the given placements."""
        for name, placement in placements.items():
            meta_tensor = module._parameters.get(name)
            if meta_tensor is None:
                continue
            meta_dtensor = distribute_tensor(meta_tensor, tp_mesh, placement, src_data_rank=None)
            module._parameters[name] = torch.nn.Parameter(meta_dtensor, requires_grad=meta_tensor.requires_grad)
        return module

    def transform_inputs_pre_forward(
        self, module: torch.nn.Module, args: tuple, kwargs: dict, tp_mesh: DeviceMesh
    ) -> tuple[tuple, dict]:
        """Transform the inputs of the module before the forward call."""
        return args, kwargs

    def context_around_forward(self, module: torch.nn.Module) -> contextlib.AbstractContextManager:
        """Context manager around the TP forward call."""
        return contextlib.nullcontext()

    def transform_output_post_forward(self, module: torch.nn.Module, output: Any, tp_mesh: DeviceMesh) -> Any:
        """Transform the output of the module after the forward call."""
        return output


class LayoutAwareTPMixin(TensorParallelMixin):
    """A base class for TP styles that need to manage input and output layouts. Most TP styles fall into that category,
    where the pre-forward and post-forward transformations are necessary to ensure the input and output are in the
    correct layouts."""

    assumed_input_layout: tuple[Placement, ...]
    desired_input_layout: tuple[Placement, ...]
    assumed_output_layout: tuple[Placement, ...]
    desired_output_layout: tuple[Placement, ...]  # can be left empty if not needed
    return_plain_output: bool = True

    def check_init(self):
        for attr in ["assumed_input_layout", "desired_input_layout", "assumed_output_layout", "desired_output_layout"]:
            # Check all layouts have been set
            if not hasattr(self, attr):
                raise AttributeError(f"Attribute {attr} is missing for {self.__class__.__name__}")
            # Check all layouts are tuples of Placement and non-empty (safe for desired_output_layout, it's optional)
            layout = getattr(self, attr)
            if not isinstance(layout, tuple) or (len(layout) == 0 and attr != "desired_output_layout"):
                raise ValueError(f"Attribute {attr} must be a non-empty tuple of Placement")

    def transform_inputs_pre_forward(
        self, module: torch.nn.Module, args: tuple, kwargs: dict, tp_mesh: DeviceMesh
    ) -> tuple[tuple, dict]:
        """Ensures the input is a DTensor with the desired input layout."""
        x = args[0]
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, tp_mesh, self.assumed_input_layout, run_check=False)
        if not matches_desired_layout(x, self.desired_input_layout):
            x = x.redistribute(placements=self.desired_input_layout, async_op=True)
        return (x,) + args[1:], kwargs

    def transform_output_post_forward(self, module: torch.nn.Module, output: Any, tp_mesh: DeviceMesh) -> Any:
        """Trnasforms the output of the module after the forward call. If the return_plain_output flag is True, the
        output is a plain tensor instead of a DTensor. If the desired_output_layout is not empty, we ensure the output
        respect that layout, even if it is later converted to a plain tensor."""
        # Early return if there are no outputs
        if output is None:
            return output

        # If there is a desired output layout, always make sure it is respected
        if self.desired_output_layout:
            if not isinstance(output, DTensor):
                output = DTensor.from_local(output, tp_mesh, self.assumed_output_layout, run_check=False)
            if not matches_desired_layout(output, self.desired_output_layout):
                output = output.redistribute(placements=self.desired_output_layout, async_op=True)

        # If we are returning a plain tensor, we need to convert the output to a plain tensor
        if self.return_plain_output and isinstance(output, DTensor):
            output = output.to_local()
        elif not self.return_plain_output and not isinstance(output, DTensor):
            output = DTensor.from_local(output, tp_mesh, self.assumed_output_layout, run_check=False)
        return output



class GatherParallel(LayoutAwareTPMixin):
    """A TP style that all gathers inputs along a given dimension. It does nothing to the outputs or the weights."""

    def __init__(self, dimension: int = 1, return_plain_output: bool = True):
        self.dimension = dimension
        self.assumed_input_layout = (Shard(dimension),)
        self.desired_input_layout = (Replicate(),)
        self.assumed_output_layout = (Placement(),)  # will never be used
        self.desired_output_layout = ()
        self.return_plain_output = return_plain_output
        self.check_init()

    def transform_output_post_forward(self, module: torch.nn.Module, output: Any, tp_mesh: DeviceMesh) -> Any:
        # this is overridden to do nothing as this class is only supposed to gather inputs
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dimension = }, {self.return_plain_output = })"


class GatherScatterSequenceParallel(LayoutAwareTPMixin):
    """A TP style that all gathers inputs along a given dimension and splits the outputs along the same dimension.
    It does nothing to the weights."""

    def __init__(self, dimension: int = 1, return_plain_output: bool = True):
        self.dimension = dimension
        self.assumed_input_layout = (Shard(dimension),)
        self.desired_input_layout = (Replicate(),)
        self.assumed_output_layout = (Replicate(),)
        self.desired_output_layout = (Shard(dimension),)
        self.return_plain_output = return_plain_output
        self.check_init()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dimension = }, {self.return_plain_output = })"


class ColwiseParallel(LayoutAwareTPMixin):
    """Column-wise parallel style for linear and embedding layers. Also supports weight packing for linear layers."""

    def __init__(
        self,
        assumed_input_layout: tuple[Placement, ...] | None = None,
        desired_output_layout: tuple[Placement, ...] | None = None,
        return_plain_output: bool = True,
        num_packed_weights: int = 1,
    ) -> None:
        """Initializes the ColwiseParallel style with:
        - assumed_input_layout: if the input tensor is a plain tensor it is converted to a DTensor with this layout.
            Defaults to `Replicate()`.
        - desired_output_layout: the layout imposed on the output. Defaults to `Shard(-1)`.
        - return_plain_output: If True, the output is a plain tensor instead of a DTensor.
        - num_packed_weights: the number of weights packed in the parameter, eg. 2 for a fused gate_up_projection,
            3 for a fused qkv_projection, ... Defaults to 1 (no packed weights)
        """
        self.assumed_input_layout = (Replicate(),) if assumed_input_layout is None else assumed_input_layout
        self.desired_input_layout = (Replicate(),)
        if num_packed_weights > 1:
            assumed_output_layout = (_StridedShard(dim=-1, split_factor=num_packed_weights),)
        else:
            assumed_output_layout = (Shard(-1),)
        self.assumed_output_layout = assumed_output_layout
        self.desired_output_layout = assumed_output_layout if desired_output_layout is None else desired_output_layout
        self.return_plain_output = return_plain_output
        self.num_packed_weights = num_packed_weights
        self.check_init()

    def shard_meta_params(self, module: torch.nn.Module, tp_mesh: DeviceMesh) -> torch.nn.Module:
        if isinstance(module, torch.nn.Linear):
            if self.num_packed_weights > 1:
                placements = {
                    "weight": (_StridedShard(dim=0, split_factor=self.num_packed_weights), ),
                    "bias": (_StridedShard(dim=0, split_factor=self.num_packed_weights),),
                }
            else:
                placements = {
                    "weight": (Shard(0),),
                    "bias": (Shard(0),),
                }
        elif isinstance(module, torch.nn.Embedding):
            if self.num_packed_weights > 1:
                raise ValueError(f"Weight packing is {self.num_packed_weights = } but it should be 1 for nn.Embedding")
            placements = {key: (Shard(1),) for key, _ in module.named_parameters()}
        else:
            raise NotImplementedError(
                f"ColwiseParallel only supports nn.Linear and nn.Embedding, but got {module.__class__.__name__}"
            )
        return self.apply_placements_to_meta_params(module, tp_mesh, placements)

    def context_around_forward(self, module: torch.nn.Module) -> contextlib.AbstractContextManager:
        """Context manager around the TP forward call: for packed weights, we need to swap the DTensors for local
        tensors because the grouped_mm operation expects plain tensors."""
        if self.num_packed_weights > 1:
            return _swap_dtensor_params_for_local(module)
        else:
            return contextlib.nullcontext()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.assumed_input_layout = }, {self.desired_output_layout = }, "
            f"{self.return_plain_output = }, {self.num_packed_weights = })"
        )


class RowwiseParallel(LayoutAwareTPMixin):
    """Row-wise parallel style for linear and embedding layers."""

    def __init__(
        self,
        assumed_input_layout: tuple[Placement, ...] = (Shard(-1),),
        desired_output_layout: tuple[Placement, ...] = (Replicate(),),
        return_plain_output: bool = True,
    ) -> None:
        """Initializes the RowwiseParallel style with:
        - assumed_input_layout: if the input tensor is a plain tensor it is converted to a DTensor with this layout.
            Defaults to `Shard(-1)`.
        - desired_output_layout: the layout imposed on the output. Defaults to `Replicate()`.
        - return_plain_output: If True, the output is a plain tensor instead of a DTensor.
        """
        self.assumed_input_layout = assumed_input_layout
        self.desired_input_layout = (Placement(),)  # placeholder for the __post_init__ check, will be set later
        self.assumed_output_layout = (Partial("sum"),)
        self.desired_output_layout = desired_output_layout
        self.return_plain_output = return_plain_output
        self.check_init()

    def shard_meta_params(self, module: torch.nn.Module, tp_mesh: DeviceMesh) -> torch.nn.Module:
        if isinstance(module, torch.nn.Linear):
            self.desired_input_layout = (Shard(-1),)
            placements = {
                "weight": (Shard(1), ),
                "bias": (Replicate(), ),
            }
        elif isinstance(module, torch.nn.Embedding):
            self.desired_input_layout = (Replicate(),)
            placements = {key: (Shard(0),) for key, _ in module.named_parameters()}
        else:
            raise NotImplementedError(
                f"RowwiseParallel only supports nn.Linear and nn.Embedding, but got {module.__class__.__name__}"
            )
        return self.apply_placements_to_meta_params(module, tp_mesh, placements)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.assumed_input_layout = }, {self.desired_output_layout = }, "
            f"{self.return_plain_output = })"
        )


class SequenceParallel(LayoutAwareTPMixin):
    """Sequence-parallel style meant for nn.LayerNorm, nn.Dropout and RMSnorm. This style can be applied to any module
    without crashing, but it does not mean that the style makes sense on any module."""

    def __init__(self, sequence_dim: int = 1, return_plain_output: bool = True):
        self.sequence_dim = sequence_dim
        self.assumed_input_layout = (Shard(sequence_dim),)
        self.desired_input_layout = (Shard(sequence_dim),)
        self.assumed_output_layout = (Shard(sequence_dim),)
        self.desired_output_layout = (Shard(sequence_dim),)
        self.return_plain_output = return_plain_output
        self.check_init()

    def shard_meta_params(self, module: torch.nn.Module, tp_mesh: DeviceMesh) -> torch.nn.Module:
        # For sequennce parallel, we replicate all params of the module across all ranks
        placements = {key: (Replicate(), ) for key, _ in module.named_parameters()}
        return self.apply_placements_to_meta_params(module, tp_mesh, placements)

    def __repr__(self) -> str:
        return ( f"{self.__class__.__name__}({self.sequence_dim = }, {self.return_plain_output = })" )


class GatherOnKwargsParallel(TensorParallelMixin):
    """A TP style that gathers the input tensor on the kwargs. Does nothing to the output or the weights."""
    def __init__(
        self,
        input_kwarg_layouts: dict[str, Placement],
        desired_input_kwarg_layouts: dict[str, Placement],
        use_local_output: bool = False,
    ) -> None:
        """Initializes the GatherOnKwargsParallel style with:
        - input_kwarg_layouts: the layouts of the input kwargs.
        - desired_input_kwarg_layouts: the desired layouts of the input kwargs.
        - use_local_output: if True, the input kwargs are converted to local tensors instead of DTensor.
        """
        self.input_kwarg_layouts = input_kwarg_layouts or {}
        self.desired_input_kwarg_layouts = desired_input_kwarg_layouts or {}
        self.use_local_output = use_local_output

    def transform_inputs_pre_forward(
        self, module: torch.nn.Module, args: tuple, kwargs: dict, tp_mesh: DeviceMesh
    ) -> tuple[tuple, dict]:
        """Enforces the desired layout for each kwargs that is present in the input_kwarg_layouts."""
        for key in self.input_kwarg_layouts.keys():
            # Skip if the key is not in the input_kwarg_layouts
            input_layout = self.input_kwarg_layouts.get(key)
            if input_layout is None:
                continue
            # Make sure the value is a DTensor
            value = kwargs[key]
            if not isinstance(value, DTensor):
                value = DTensor.from_local(value, tp_mesh, input_layout, run_check=False)
            # And enforce the desired layout if there is one
            desired_layout = self.desired_input_kwarg_layouts.get(key)
            if desired_layout is not None and value.placements != desired_layout:
                value = value.redistribute(placements=desired_layout)
            kwargs[key] = value.to_local() if self.use_local_output else value
        return args, kwargs


# TODO: reowrk these classes to fit the newest TP rewrite
class _AllReduceBackward(torch.autograd.Function):
    """Identity forward, allreduce-sum backward.

    Used for MoE routing weights: the forward value is replicated (same on all ranks), but the backward gradient is
    partial (each rank has 1/tp_size from its expert shard). We need to sum the partial gradients without dividing by
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

class MoEExpertsParallel(TensorParallelMixin):
    """Tensor-parallel style for MoE expert modules.

    Shards expert weights as DTensors, then wraps the module's ``forward`` so that grouped_mm (which needs plain
    tensors) works transparently.

    Lifecycle phases:
    1. _apply — wrap each expert weight named in shard_plan as a DTensor
       placeholder with the declared placement.
    2. transform_inputs_pre_forward — localize hidden_states (Replicate→local,
       gives us an all-reduce on the backward gradient for free), then fix
       routing-weight gradients (their backward is partial; use allreduce-sum,
       not divide-by-world-size).
    3. context_around_forward — swap DTensor params for local leaves so
       grouped_mm sees plain tensors; restored on exit so save_pretrained
       still sees DTensors.
    4. transform_output_post_forward — under TP-only each rank's output is
       partial (only its expert shard contributed); reduce/redistribute to
       output_layouts.
    """

    def __init__(self, output_layouts: Placement | None = None, shard_plan: dict[str, Placement] | None = None):
        super().__init__()
        self.output_layouts = output_layouts or Replicate()
        self._moe_shard_plan = shard_plan or {}

    def shard_meta_params(self, module: torch.nn.Module, tp_mesh: DeviceMesh) -> torch.nn.Module:
        for name, placement in self._moe_shard_plan.items():
            meta = module._parameters.get(name)
            if meta is None:
                continue
            module._parameters[name] = torch.nn.Parameter(
                distribute_tensor(meta, tp_mesh, [placement], src_data_rank=None),
                requires_grad=meta.requires_grad,
            )
        return module

    def transform_inputs_pre_forward(
        self, module: nn.Module, args: tuple, kwargs: dict, tp_mesh: DeviceMesh
    ) -> tuple[tuple, dict]:
        hidden_states, top_k_index, top_k_weights = args
        if not isinstance(hidden_states, DTensor):
            hidden_states = DTensor.from_local(hidden_states, tp_mesh, [Replicate()], run_check=False)
        hidden_states = hidden_states.to_local()

        if isinstance(top_k_weights, DTensor):
            top_k_weights = top_k_weights.to_local()
        tp_group = tp_mesh.get_group() if tp_mesh.ndim == 1 else tp_mesh.get_group("tp")
        top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_group)

        return (hidden_states, top_k_index, top_k_weights), kwargs

    def context_around_forward(self, module):
        return _swap_dtensor_params_for_local(module)

    def transform_output_post_forward(self, module: torch.nn.Module, output: Any, tp_mesh: DeviceMesh) -> Any:
        if output is None:
            return None
        # Under TP-only each rank has a partial result; under TP+FSDP the
        # weights may be fully gathered by FSDP, making the output complete.
        has_sharded_params = any(
            isinstance(p, DTensor) and any(not pl.is_replicate() for pl in p.placements) for p in module.parameters()
        )
        source = Partial() if has_sharded_params else Replicate()
        if not isinstance(output, DTensor):
            output = DTensor.from_local(output, tp_mesh, [source], run_check=False)
        # MoE output is 2D [tokens, hidden]. For SP, Shard(1) means seq dim
        # in 3D but token dim (0) in 2D.
        target = self.output_layouts
        if output.dim() == 2 and isinstance(target, Shard) and target.dim == -1:
            target = Shard(0)
        if output.placements != (target,):
            output = output.redistribute(placements=(target,))
        return output.to_local()


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
            "colwise": ColwiseParallel(),
            "colwise_allgather": ColwiseParallel(desired_output_layout=(Replicate(),)),
            "colwise_loss_parallel": ColwiseParallel(assumed_input_layout=(Shard(1),), return_plain_output=False),
            "packed_colwise": ColwiseParallel(num_packed_weights=2),
            # Row-parallel
            "rowwise_allreduce": RowwiseParallel(),
            "rowwise_reduce_scatter": RowwiseParallel(desired_output_layout=(Shard(1),)),
            # Vocab / embedding (rowwise sharding on vocab dim)
            "vocab_allreduce": RowwiseParallel(assumed_input_layout=(Replicate(),)),
            "vocab_reduce_scatter": RowwiseParallel(
                assumed_input_layout=(Replicate(),), desired_output_layout=(Shard(1),)
            ),
            # Activation / norm (sequence-parallel passthrough)
            # use_local_output=True: torch defaults to False here, but downstream modeling
            # code expects plain tensors, not DTensors.
            "activation": SequenceParallel(return_plain_output=True),
            "activation_seq_dim_2": SequenceParallel(sequence_dim=2, return_plain_output=True),
            # Module-level prepare-input. Same use_local_output=True override as above —
            # torch's default is False, our modeling code expects plain tensors downstream.
            "module_allgather": GatherParallel(dimension=1, return_plain_output=True),
            "module_allgather_hidden_states": GatherOnKwargsParallel(
                input_kwarg_layouts={"hidden_states": (Shard(1), )},
                desired_input_kwarg_layouts={"hidden_states": (Replicate(),)},
                use_local_output=True,
            ),
            "module_allgather_split": GatherScatterSequenceParallel(dimension=1, return_plain_output=True),
            # MoE — canonical shard_plan baked in (only variant in use across configs).
            # gate_up_proj is packed (gate||up along output dim) so we use _StridedShard
            # to interleave; down_proj is plain rowwise on its input dim.
            "moe_experts_allreduce": MoEExpertsParallel(
                output_layouts=Replicate(),
                shard_plan={
                    "gate_up_proj": _StridedShard(dim=-2, split_factor=2),
                    "down_proj": Shard(-1),
                },
            ),
        }
        if is_torch_greater_or_equal("2.5") and _torch_distributed_available
        else {}
    )


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()


def parallelize_model(model: torch.nn.Module, tp_mesh: DeviceMesh, tp_plan: dict[str, str] | None = None):
    """Applies parallelism (TP or SP) to a model by walking the model's submodules and trying to match them to a pattern
    in the tp_plan. If such a match is found, the submodule is made TP-aware by the ``make_tp_aware`` method of the
    matched TP style."""
    distributed_config = getattr(model.config, "distributed_config", None)
    sp_requested = getattr(distributed_config, "enable_sequence_parallel", False)
    sp_supported = getattr(model.config, "base_model_sp_plan", None) is not None
    enable_sp = sp_requested and sp_supported

    if tp_plan is None:
        if enable_sp:
            tp_plan = dict(model._sp_plan or {})
        else:
            tp_plan = dict(model._tp_plan or {})

    # tie_weights() replaces lm_head.weight with embed_tokens.weight after TP is applied.
    # If embed_tokens isn't in the plan, sharding lm_head as a DTensor causes tie to
    # replace it with a plain tensor (and forward then mixes DTensor/Tensor). Skip
    # lm_head TP in that case so both ends stay plain and the tie is a real alias.
    if getattr(model.config, "tie_word_embeddings", False):
        tied_source_in_plan = any(k.endswith("embed_tokens") for k in tp_plan)
        if not tied_source_in_plan:
            tp_plan.pop("lm_head", None)

    for name, submodule in model.named_modules():
        style_value = _get_parameter_tp_plan(parameter_name=name, tp_plan=tp_plan, is_weight=False)
        if style_value is None:
            continue
        ALL_PARALLEL_STYLES[style_value].make_tp_aware(submodule, tp_mesh)

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
