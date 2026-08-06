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
from ..utils.import_utils import is_torch_available, is_torch_distributed_available


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.placement_types import _StridedShard


def replace_layer_number_by_wildcard(name: str) -> str:
    """
    Replace the numbers in the `name` by wildcards, only if they are in-between dots (`.`) or if they are between
    a dot (`.`) and the end of the string.
    This matches how modules are named/numbered when using a nn.ModuleList or nn.Sequential, but will NOT match
    numbers in a parameter name itself, e.g. if the param is named `"w1"` or `"w2"`.
    """
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str] | None):
    """
    Verify the TP plan of the model, log a warning if the layers that were not sharded and the rules that were not applied.
    """

    if tp_plan is None:
        return

    generic_keys = {replace_layer_number_by_wildcard(key) for key in expected_keys}
    unsharded_layers = set(generic_keys)
    unused_rules = tp_plan.copy()

    for key in generic_keys:
        param_name = key.rsplit(".", 1)[0] if "." in key else key
        generic_param_name = re.sub(r"\d+", "*", param_name)

        if generic_param_name in tp_plan:
            unused_rules.pop(generic_param_name, None)
            unsharded_layers.discard(key)
        elif "." in generic_param_name and (parent_param_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
            unused_rules.pop(parent_param_name, None)
            unsharded_layers.discard(key)

    if len(unused_rules) > 0:
        logger.warning(f"The following TP rules were not applied on any of the layers: {unused_rules}")
    if len(unsharded_layers) > 0:
        logger.warning(f"The following layers were not sharded: {', '.join(unsharded_layers)}")


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


class TensorParallelLayer:
    def should_use_local_tensors(self, module):
        """Whether this module's forward requires local inputs and parameters."""
        return False

    def validate_param(self, module, param, mesh, parameter_name=None):
        """Validate a parameter before applying this TP style."""
        pass

    def shard_param(self, module, param, mesh):
        """Wrap ONE parameter as a DTensor placeholder. Default: no-op."""
        pass

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        return args, kwargs

    def context_around_forward(self, module, mesh):
        if self.should_use_local_tensors(module):
            return _use_local_dtensor_params(module)
        return contextlib.nullcontext()

    def transform_output_post_forward(self, module, output, mesh):
        return output

    def install_forward(self, module, mesh):
        """Install pre / around / post transforms by replacing module.forward."""
        original_forward = module.forward

        def tp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(module, args, kwargs, mesh)
            with self.context_around_forward(module, mesh):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, mesh)

        module.forward = tp_forward
        return module


class ColwiseParallel(TensorParallelLayer):
    """Column-wise: weight & bias → Shard(0) (Embedding: Shard(1)); input replicated, output Shard(-1)."""

    def __init__(self, *, input_layouts=None, output_layouts=None, use_local_output: bool = True):
        self.input_layouts = input_layouts or Replicate()
        self.output_layouts = output_layouts if output_layouts is not None else Shard(-1)
        self.use_local_output = use_local_output

    def should_use_local_tensors(self, module):
        use_local_quantized_path = getattr(module, "_hf_quantized_needs_local_tp", False)
        uses_local_inference_kernel = isinstance(module, torch.nn.Linear) and not torch.is_grad_enabled()
        return use_local_quantized_path or uses_local_inference_kernel

    def validate_param(self, module, param, mesh, parameter_name=None):
        meta = module._parameters.get(param)
        gathers_output = isinstance(self.output_layouts, Replicate)
        if meta is None or not gathers_output:
            return

        shard_dim = 1 if isinstance(module, torch.nn.Embedding) else meta.ndim - 2
        output_size = meta.shape[shard_dim]
        tp_size = mesh.size()
        if output_size % tp_size != 0:
            parameter_name = parameter_name or param
            layer_name = parameter_name.rsplit(".", 1)[0]
            raise ValueError(
                f"The output size of `{layer_name}` ({output_size}) must be divisible by the tensor parallel size "
                f"({tp_size}) when gathering a colwise output."
            )

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        placement = Shard(1) if isinstance(module, torch.nn.Embedding) else Shard(meta.ndim - 2)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        x = args[0]
        is_local_input = not isinstance(x, DTensor)

        # regular Linear inference fast path: the input is already replicated, the
        # local weight owns a shard of the output features, and no backward is needed.
        if is_local_input and isinstance(module, torch.nn.Linear) and not torch.is_grad_enabled():
            return args, kwargs

        # Plain-local quantized fast path. The input is already replicated, so avoid
        # wrapping it as a DTensor. If its gradient is needed, synchronize the partial
        # input-gradient contributions explicitly.
        if is_local_input and getattr(module, "_hf_quantized_needs_local_tp", False):
            if torch.is_grad_enabled() and x.requires_grad:
                process_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
                x = _AllReduceBackward.apply(x, process_group)
            return (x,) + args[1:], kwargs

        # DTensor path: handle regular training and layout redistribution.
        if is_local_input:
            x = DTensor.from_local(x, mesh, [self.input_layouts], run_check=False)
        if x.placements != (Replicate(),):
            x = x.redistribute(placements=[Replicate()], async_op=True)
        if self.should_use_local_tensors(module):
            x = x.to_local(grad_placements=[Partial()])
        return (x,) + args[1:], kwargs

    def transform_output_post_forward(self, module, output, mesh):
        # The local forward produced this rank's shard of the output features (last dim).
        output_is_local_shard = (
            not isinstance(output, DTensor)
            and isinstance(self.output_layouts, Shard)
            and self.output_layouts.dim in (-1, output.dim() - 1)
        )
        if self.should_use_local_tensors(module) and self.use_local_output and output_is_local_shard:
            return output
        if not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, [Shard(-1)], run_check=False)
        if output.placements != (self.output_layouts,):
            output = output.redistribute(placements=[self.output_layouts], async_op=True)
        return output.to_local() if self.use_local_output else output


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

    def should_use_local_tensors(self, module):
        use_local_quantized_path = getattr(module, "_hf_quantized_needs_local_tp", False)
        uses_local_inference_kernel = isinstance(module, torch.nn.Linear) and not torch.is_grad_enabled()
        return use_local_quantized_path or uses_local_inference_kernel

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        if isinstance(module, torch.nn.Embedding):
            placement = Shard(0)
        else:
            # bias is replicated (added after the row-reduce); weight shards on input dim (-1)
            placement = Replicate() if param == "bias" else Shard(-1)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        # Embedding runtime sharding needs a replicated input; Linear needs Shard(-1).
        desired = Replicate() if isinstance(module, torch.nn.Embedding) else Shard(-1)
        x = args[0]
        # A local kernel can use a plain input when the previous layer already split it.
        #  avoid a redundant Tensor -> DTensor -> Tensor round trip.
        input_has_desired_layout = self.input_layouts == desired
        if self.should_use_local_tensors(module) and input_has_desired_layout and not isinstance(x, DTensor):
            return args, kwargs
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [self.input_layouts], run_check=False)
        if x.placements != (desired,):
            x = x.redistribute(placements=[desired], async_op=True)
        if self.should_use_local_tensors(module):
            x = x.to_local()
        return (x,) + args[1:], kwargs

    @contextlib.contextmanager
    def context_around_forward(self, module, mesh):
        if not self.should_use_local_tensors(module):
            yield
        else:
            # A rowwise local forward must produce only its partial matmul. If we don't hide
            # the bias, we will be adding the bias x world_size times which is not correct.
            # We should add it once after the all_reduce (redistribute).
            bias = module._parameters.get("bias")
            if bias is not None:
                module._parameters["bias"] = None
            try:
                with _use_local_dtensor_params(module):
                    yield
            finally:
                if bias is not None:
                    module._parameters["bias"] = bias

    def transform_output_post_forward(self, module, output, mesh):
        use_local_inference_path = (
            isinstance(module, torch.nn.Linear)
            and not isinstance(output, DTensor)
            and isinstance(self.output_layouts, Replicate)
            and not output.requires_grad
        )
        if use_local_inference_path:
            process_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
            dist.all_reduce(output, group=process_group)
            if (bias := module._parameters.get("bias")) is not None:
                output = output + (bias.to_local() if isinstance(bias, DTensor) else bias)
        else:
            # Dtensor tracks whether the result is partial, sharded, or replicated.
            if not isinstance(output, DTensor):
                output = DTensor.from_local(output, mesh, [Partial()], run_check=False)
            if output.placements != (self.output_layouts,):
                output = output.redistribute(placements=[self.output_layouts], async_op=True)
            if self.should_use_local_tensors(module) and (bias := module._parameters.get("bias")) is not None:
                output = output + bias
            if self.use_local_output:
                output = output.to_local()

        return output


class ReplicatedWithGradAllReduce(TensorParallelLayer):
    """Replicated parameter whose gradient is partial.

    For norms that sit between a colwise and a rowwise layer and normalize along a sharded
    axis — e.g. Qwen3's per-head ``q_norm``/``k_norm``, which only see this rank's heads. The
    forward needs no collective (the param is replicated and the activation is already local),
    but each rank's parameter gradient only covers its own heads, so the gradients have to be
    summed across the mesh.
    """

    def install_forward(self, module, mesh):
        # A module hook rather than `param.register_hook`: params are replaced during weight
        # loading, which happens after TP is applied, and would drop a param-level hook.
        def _all_reduce_grads(mod, grad_input, grad_output):
            for param in mod.parameters(recurse=False):
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=mesh.get_group())

        module.register_full_backward_hook(_all_reduce_grads)
        return module


class AllReduceParallel(TensorParallelLayer):
    """All-reduce a module's partial forward output across the TP mesh."""

    def transform_output_post_forward(self, module, output, mesh):
        if output is None:
            return None
        if not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, [Partial()], run_check=False)
        if output.placements != (Replicate(),):
            output = output.redistribute(placements=[Replicate()])
        return output.to_local()


class MlaKvAProjParallel(TensorParallelLayer):
    """All-reduce the rotary branch gradient of an MLA KV-A projection."""

    def transform_output_post_forward(self, module, output, mesh):
        rope_dim = module.config.qk_rope_head_dim
        pass_output, rope_output = output.split([output.shape[-1] - rope_dim, rope_dim], dim=-1)
        rope_output = _AllReduceBackward.apply(rope_output, mesh.get_group())
        return torch.cat([pass_output, rope_output], dim=-1)


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


# =============================================================================
# MoE / packed-linear (grouped_mm needs plain tensors)
# =============================================================================


class PackedColwiseParallel(TensorParallelLayer):
    """Column-wise parallel style for fused linear weights packed along the output dimension."""

    def __init__(
        self,
        *,
        use_local_output: bool = True,
        split_factor: int = 2,
    ):
        self.input_layouts = (Replicate(),)
        self.use_local_output = use_local_output
        self.split_factor = split_factor
        # Same as ColwiseParallel: replicated input, output-dim sharded weight, so the input
        # gradient is partial.
        self.input_grad_placements = [Partial()]

    def should_use_local_tensors(self, module):
        return True

    def _packed_output_shard_dim(self, param_ndim: int) -> int:
        """Dimension holding packed gate/up features: dim 0 for 2D Linear, dim 1 for 3D MoE experts."""
        if param_ndim == 1:
            return -1
        return param_ndim - 2

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        shard_dim = self._packed_output_shard_dim(meta.ndim)
        # Wrap as a DTensor placeholder. Runs on meta — distribute_tensor builds metadata only.
        if meta.ndim == 1:
            placement = Shard(shard_dim)
        else:
            placement = _StridedShard(dim=shard_dim, split_factor=self.split_factor)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        input_tensor = args[0]
        # Ensure the input is a Replicate DTensor on the TP mesh.
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, mesh, self.input_layouts, run_check=False)
        elif input_tensor.placements != self.input_layouts:
            input_tensor = input_tensor.redistribute(placements=self.input_layouts)

        # The packed kernels runs on local tensors, so Dtensor cannot infer the layout of the
        # gradient produced by the kernel. The kernel sees replicated input + partial weights
        # which means input gradient will be partial as well
        input_tensor = input_tensor.to_local(grad_placements=[Partial()])
        return (input_tensor,) + args[1:], kwargs

    def transform_output_post_forward(self, module, output, mesh):
        if output is None or self.use_local_output:
            return output
        return DTensor.from_local(
            output, mesh, (_StridedShard(dim=-1, split_factor=self.split_factor),), run_check=False
        )


class PackedRowwiseParallel(TensorParallelLayer):
    """Parameter style for fused weights packed along the final dimension."""

    def __init__(self, *, split_factor: int = 2):
        self.split_factor = split_factor

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        placement = Replicate() if meta.ndim == 1 else _StridedShard(dim=-1, split_factor=self.split_factor)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )


class MoEParamShard(TensorParallelLayer):
    """Param-only EP style for MoE expert weights (``grouped_gemm``).

    Shards dim 0 and updates module.num_experts to the per-rank local count so the
    experts forward and ep_router sentinel agree.
    """

    def __init__(self, placement, *, shards_expert_dim: bool = False):
        self.placement = placement
        self.shards_expert_dim = shards_expert_dim

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        if self.shards_expert_dim and hasattr(module, "num_experts"):
            module.num_experts = meta.shape[0] // mesh.size()
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [self.placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
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


class MoEExpertsParallel(TensorParallelLayer):
    def should_use_local_tensors(self, module):
        return True

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh, *, is_expert_parallel=False):
        hidden_states, *remaining_args = args
        tp_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.to_local()
        hidden_states = _AllReduceBackward.apply(hidden_states, tp_group)

        if len(remaining_args) >= 2:
            top_k_weights = remaining_args[1]
            if isinstance(top_k_weights, DTensor):
                top_k_weights = top_k_weights.to_local()
            if not is_expert_parallel:
                top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_group)
            remaining_args[1] = top_k_weights

        return (hidden_states, *remaining_args), kwargs

    def install_forward(self, module, mesh, *, is_expert_parallel=False):
        """Install the transforms but pass `is_expert_parallel` in the forward call."""
        original_forward = module.forward
        output_source = (
            Partial()
            if any(
                isinstance(param, DTensor) and any(not placement.is_replicate() for placement in param.placements)
                for param in module.parameters()
            )
            else Replicate()
        )

        def tp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(
                module, args, kwargs, mesh, is_expert_parallel=is_expert_parallel
            )
            with self.context_around_forward(module, mesh):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, mesh, source=output_source)

        module.forward = tp_forward
        return module

    def transform_output_post_forward(self, module, output, mesh, source=None):
        if output is None:
            return None

        has_sharded_parameters = any(
            isinstance(param, DTensor) and any(not placement.is_replicate() for placement in param.placements)
            for param in module.parameters()
        )
        if not has_sharded_parameters:
            return output

        process_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
        return _AllReduceForward.apply(output, process_group)


class MoeIdentityParallel(TensorParallelLayer):
    """Compensate for moe_tp_experts summing identity-expert outputs across ranks."""

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        input_tensor = args[0]
        return (input_tensor / mesh.size(), *args[1:]), kwargs


class EpRouterParallel(TensorParallelLayer):
    """Expert-parallel router: forward-only slicing of router outputs to local experts."""

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
        if num_local_experts > 1:
            router_indices = torch.fmod(router_indices, num_local_experts)
        else:
            router_indices = router_indices.masked_fill(router_indices > 0, 0).masked_fill(router_indices < 0, -1)
        router_indices = router_indices.masked_fill(router_indices == -1, num_local_experts)
        return router_logits, router_scores, router_indices


class RouterParallelMegaMoe(EpRouterParallel):
    """Router TP plan used with DeepGEMM Mega MoE.

    Mega MoE handles EP dispatch inside the kernel and wants raw global expert ids
    with unmasked routing weights, so the router doesn't pre-shard per EP rank like
    ``EpRouterParallel`` does.
    """

    def transform_output_post_forward(self, module, output, mesh):
        return output


class MoeTensorParalellMegaMoeExperts(MoEExpertsParallel):
    """TP layer for DeepGEMM Mega MoE experts.

    Mega MoE is inference-only (the kernel has no backward) and handles EP dispatch +
    combine + per-rank token sharding internally — so we skip the gradient-sync hooks
    that ``MoEExpertsParallel`` would apply, and we forward the EP ``process_group``
    into the module so the symm-buffer rendezvous can run on first forward.
    """

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh, *, is_expert_parallel=False):
        hidden_states, top_k_index, top_k_weights = args[0], args[1], args[2]
        return (hidden_states, top_k_index, top_k_weights, mesh.get_group()), kwargs

    def context_around_forward(self, module, mesh):
        return _use_local_dtensor_params(module)

    def transform_output_post_forward(self, module, output, mesh):
        return output


class ParallelInterface(GeneralInterface):
    """Registry of named TP styles for the DTensor backend."""

    _global_mapping = (
        {
            "embedding_rowwise": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
            "colwise_gather_output": ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
            "colwise": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1)),
            "rowwise": RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
            "rowwise_split_input": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
            "packed_colwise": PackedColwiseParallel(),
            "packed_rowwise": PackedRowwiseParallel(),
            "sequence_parallel": SequenceParallel(use_local_output=True),
            "grouped_gemm": MoEParamShard(Shard(0), shards_expert_dim=True),
            "ep_router": EpRouterParallel(),
            "megamoe_router": RouterParallelMegaMoe(),
            "moe_tp_experts": MoEExpertsParallel(),
            "megamoe_experts": MoeTensorParalellMegaMoeExperts(),
            "moe_identity_expert": MoeIdentityParallel(),
            "replicated_with_grad_allreduce": ReplicatedWithGradAllReduce(),
            "mla_kv_a_proj": MlaKvAProjParallel(),
            "all_reduce": AllReduceParallel(),
        }
        if is_torch_distributed_available()
        else {}
    )


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()


def apply_tensor_parallelism(model, tp_mesh):
    """DTensor backend: shard params as placeholders and install TP forward hooks."""

    for name, module in model.named_modules():
        for p_name, _ in list(module.named_parameters(recurse=False)):
            full = f"{name}.{p_name}" if name else p_name
            style_name = _get_parameter_tp_plan(parameter_name=full, tp_plan=model.tp_plan, is_weight=True)
            if style_name is not None and style_name in ALL_PARALLEL_STYLES:
                style = ALL_PARALLEL_STYLES[style_name]
                style.validate_param(module, p_name, tp_mesh, parameter_name=full)
                style.shard_param(module, p_name, tp_mesh)
        style_name = _get_parameter_tp_plan(parameter_name=name, tp_plan=model.tp_plan, is_weight=False)
        if style_name is not None and style_name in ALL_PARALLEL_STYLES:
            if style_name == "mla_kv_a_proj":
                module.config = model.config.get_text_config()
            ALL_PARALLEL_STYLES[style_name].install_forward(module, tp_mesh)
        module._is_hooked = True

    return model


def gather_state_dict_for_save(
    state_dict: dict[str, torch.Tensor],
    _tp_plan: dict[str, str],
    _device_mesh,
    _tp_size: int,
) -> dict[str, torch.Tensor]:
    """Gather TP-sharded ``DTensor`` parameters to full CPU tensors for checkpoint saving.

    Every rank must call this function so ``DTensor.full_tensor()`` collectives complete.
    """
    gathered = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            gathered[key] = tensor.detach().cpu().contiguous()
        else:
            gathered[key] = tensor
    return gathered
