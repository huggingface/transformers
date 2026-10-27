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
from __future__ import annotations

import contextlib
import re
import warnings
from itertools import chain
from typing import TYPE_CHECKING

from ..utils import logging
from ..utils.import_utils import is_torch_available, is_torch_distributed_available
from .parallel_utils import (
    ALL_PARALLEL_STYLES,
    ParallelInterface,  # noqa: F401
    _use_local_dtensor_params,
)
from .utils import _get_torch_distributed_rank


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from .configuration_utils import DistributedConfig


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.placement_types import _StridedShard

    from .parallel_utils import _AllReduceBackward, _AllReduceForward


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


def get_kv_heads_per_rank(tp_plan: dict[str, str], num_heads: int | list[int], tp_size: int) -> int | list[int] | None:
    """Returns the KV heads each rank holds under `tp_plan`, or `None` if they cannot be evenly sharded."""
    # Projections whose output holds the attention keys, fused or not
    key_projections = ("k_proj", "qkv_proj", "query_key_value")
    if not any(key.rsplit(".", 1)[-1] in key_projections and style == "colwise" for key, style in tp_plan.items()):
        # Unsharded heads: no plan, a gathered output, or experts only
        return num_heads
    layer_heads = [num_heads] if isinstance(num_heads, int) else num_heads
    if any(heads % tp_size for heads in layer_heads):
        return None
    # A scalar must stay scalar: `early_initialization` broadcasts it, but wants one entry per layer in a list
    return num_heads // tp_size if isinstance(num_heads, int) else [h // tp_size for h in layer_heads]


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

    @staticmethod
    def _from_local_last_dim_shard(local, mesh, size):
        """Create a `DTensor` for a tensor sharded across its last dimension.
        The local shard can have a different size from the other ranks when `size` is not divisible by the TP size."""
        shape = (*local.shape[:-1], size)
        return DTensor.from_local(
            local, mesh, [Shard(-1)], run_check=False, shape=shape, stride=torch.empty(shape, device="meta").stride()
        )

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
    """Column-wise for linear layers: weight & bias → Shard(0); input replicated, output Shard(-1).

    The output width (`out_features`) need not divide by the TP size: the last rank then holds a shorter slice."""

    def __init__(self, *, input_layouts=None, output_layouts=None, use_local_output: bool = True):
        self.input_layouts = input_layouts or Replicate()
        self.output_layouts = output_layouts if output_layouts is not None else Shard(-1)
        self.use_local_output = use_local_output

    def should_use_local_tensors(self, module):
        use_local_quantized_path = getattr(module, "_hf_quantized_needs_local_tp", False)
        uses_local_inference_kernel = isinstance(module, torch.nn.Linear) and not torch.is_grad_enabled()
        return use_local_quantized_path or uses_local_inference_kernel

    def install_forward(self, module, mesh):
        if not hasattr(module, "out_features"):
            raise ValueError(f"{type(module).__name__} has no `out_features`: colwise shards linear layers.")
        return super().install_forward(module, mesh)

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        placement = Shard(meta.ndim - 2)
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
            x = x.redistribute(placements=[Replicate()])
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
            # `out_features`, not the weight's shape: a packed or quantized weight doesn't show the full width
            output = self._from_local_last_dim_shard(output, mesh, module.out_features)
        if output.placements != (self.output_layouts,):
            output = output.redistribute(placements=[self.output_layouts])
        return output.to_local() if self.use_local_output else output


class RowwiseParallel(TensorParallelLayer):
    """Row-wise for linear layers: weight → Shard(1), bias → Replicate; input sharded on the last dim.

    The module forward produces a Partial output which the boundary redistribute reduces to
    output_layouts (Replicate → allreduce, Shard(1) → reduce-scatter). The input width (`in_features`) need not
    divide by the TP size.
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
        # bias is replicated (added after the row-reduce); weight shards on input dim (-1)
        placement = Replicate() if param == "bias" else Shard(-1)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def install_forward(self, module, mesh):
        if not hasattr(module, "in_features"):
            raise ValueError(f"{type(module).__name__} has no `in_features`: rowwise shards linear layers.")
        return super().install_forward(module, mesh)

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        desired = Shard(-1)
        x = args[0]
        # A local kernel can use a plain input when the previous layer already split it.
        #  avoid a redundant Tensor -> DTensor -> Tensor round trip.
        input_has_desired_layout = self.input_layouts == desired
        if self.should_use_local_tensors(module) and input_has_desired_layout and not isinstance(x, DTensor):
            return args, kwargs
        if not isinstance(x, DTensor):
            if isinstance(self.input_layouts, Shard):
                # the input is this rank's slice of the module's declared `in_features`
                x = self._from_local_last_dim_shard(x, mesh, module.in_features)
            else:
                x = DTensor.from_local(x, mesh, [self.input_layouts], run_check=False)
        if x.placements != (desired,):
            x = x.redistribute(placements=[desired])
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
                output = output.redistribute(placements=[self.output_layouts])
            if self.should_use_local_tensors(module) and (bias := module._parameters.get("bias")) is not None:
                output = output + bias
            if self.use_local_output:
                output = output.to_local()

        return output


class EmbeddingParallel(TensorParallelLayer):
    """Embedding with the token ids replicated on every rank. `shard_dim=0` is vocab-parallel: each rank holds a slice
    of the vocabulary, and the ranks' partial lookups are summed. `shard_dim=1` shards the hidden dim: each rank looks
    up a slice of every vector, and the slices are gathered. The sharded dim need not divide by the TP size."""

    def __init__(self, *, shard_dim: int = 0, use_local_output: bool = True):
        self.shard_dim = shard_dim
        self.use_local_output = use_local_output

    def should_use_local_tensors(self, module):
        return getattr(module, "_hf_quantized_needs_local_tp", False)

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [Shard(self.shard_dim)], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        x = args[0]
        if self.should_use_local_tensors(module) and not isinstance(x, DTensor):
            return args, kwargs
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [Replicate()], run_check=False)
        elif x.placements != (Replicate(),):
            x = x.redistribute(placements=[Replicate()])
        if self.should_use_local_tensors(module):
            x = x.to_local()
        return (x,) + args[1:], kwargs

    def transform_output_post_forward(self, module, output, mesh):
        if not isinstance(output, DTensor):
            if self.shard_dim == 0:
                output = DTensor.from_local(output, mesh, [Partial()], run_check=False)
            else:
                output = self._from_local_last_dim_shard(output, mesh, module.embedding_dim)
        if output.placements != (Replicate(),):
            output = output.redistribute(placements=[Replicate()])
        return output.to_local() if self.use_local_output else output


class ReplicatedWithGradAllReduce(TensorParallelLayer):
    """Replicated parameter whose gradient is partial.

    For norms that sit between a colwise and a rowwise layer and normalize along a sharded
    axis — e.g. Qwen3's per-head ``q_norm``/``k_norm``, which only see this rank's heads. The
    forward needs no collective (the param is replicated and the activation is already local),
    but each rank's parameter gradient only covers its own heads, so the gradients have to be
    summed across the mesh.
    """

    def install_forward(self, module, mesh):
        original_forward = module.forward
        # Module-level backward hooks create graph breaks when using torch.compile but not with parameter-level
        # hooks. We can't add them during setup time as the parameters are empty placeholders that are going to be
        # replaced with the loaded weights, discarding all previously attached hooks. So they are attached only on
        # first forward pass. A dict tracks what's already hooked.
        is_already_hooked: dict[str, torch.nn.Parameter] = {}

        def _all_reduce_grad(param):
            dist.all_reduce(param.grad, group=mesh.get_group())

        def tp_forward(*args, **kwargs):
            if torch.is_grad_enabled():
                for name, param in module.named_parameters(recurse=False):
                    if param.requires_grad and is_already_hooked.get(name) is not param:
                        param.register_post_accumulate_grad_hook(_all_reduce_grad)
                        is_already_hooked[name] = param
            return original_forward(*args, **kwargs)

        module.forward = tp_forward
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
    """
    For MLA attention used in DeepSeek-V2 style models (deepseek_v2, longcat_flash, glm_moe_dsa, glm4_moe_lite):
    kv_a_proj_with_mqa output is [kv_lora_rank + qk_rope_head_dim] (can have different naming but important thing
    to understand is that it is split)
    Example below (from modeling_longcat_flash.py):

    kv_a_proj_with_mqa
            |
            split
            /    \
        k_pass    k_rot  <-- "bypasses kv_b_proj"
        |          |        (goes straight to attention,
    kv_a_layernorm |         never touches kv_b_proj)
        |          |
    kv_b_proj      |
    (colwise)      |
        |          |
        k_pass     k_rot
            \\      /
               cat
                |
            key_states

    k_pass is passed to kv_b_proj (colwise) which has built-in all_reduce_backward so we don't have a partial gradient for it.
    However, k_rot goes straight to attention, never touches kv_b_proj. So we need to average gradient across all ranks otherwise we only get gradient for one rank (partial gradient).
    """

    def transform_output_post_forward(self, module, output, mesh):
        if not hasattr(module.config, "qk_rope_head_dim"):
            raise AttributeError(
                f"Config for {type(module).__name__} does not have `qk_rope_head_dim`. "
                "MlaKvAProjParallel requires `qk_rope_head_dim` to be defined in the model config. "
                "Please add it to the model's config or update the TP plan mapping."
            )
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
            x = x.redistribute(placements=[seq])
        return (x,) + args[1:], kwargs

    def transform_output_post_forward(self, module, output, mesh):
        if isinstance(output, DTensor):
            return output.to_local() if self.use_local_output else output
        return output


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


class MoeExpertsParallel(TensorParallelLayer):
    def should_use_local_tensors(self, module):
        return True

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh, *, is_expert_parallel=False):
        hidden_states, *routing_args = args
        tp_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.to_local()
        hidden_states = _AllReduceBackward.apply(hidden_states, tp_group)

        if len(routing_args) >= 2:
            top_k_index, top_k_weights, *extra_args = routing_args
            if isinstance(top_k_weights, DTensor):
                top_k_weights = top_k_weights.to_local()
            if not is_expert_parallel:
                top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_group)
            routing_args = [top_k_index, top_k_weights, *extra_args]

        return (hidden_states, *routing_args), kwargs

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
    """
    Used in longcat_flash zero_experts (nn.Identity) which return the same value on every GPU, but the
    moe_tp_experts will sums across GPUs. Therefore we pre-divide the identity input by tp_size to cancel the extra scaling.
    """

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        input_tensor = args[0]
        return (input_tensor / mesh.size(), *args[1:]), kwargs


if is_torch_distributed_available():
    ALL_PARALLEL_STYLES.register("embedding_rowwise", EmbeddingParallel(shard_dim=0))
    ALL_PARALLEL_STYLES.register("embedding_colwise", EmbeddingParallel(shard_dim=1))
    ALL_PARALLEL_STYLES.register(
        "colwise_gather_output", ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate())
    )
    ALL_PARALLEL_STYLES.register("colwise_rep", ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()))
    ALL_PARALLEL_STYLES.register("colwise", ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1)))
    ALL_PARALLEL_STYLES.register("rowwise", RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()))
    ALL_PARALLEL_STYLES.register(
        "rowwise_split_input", RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate())
    )
    ALL_PARALLEL_STYLES.register("rowwise_rep", RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()))
    ALL_PARALLEL_STYLES.register("packed_colwise", PackedColwiseParallel())
    ALL_PARALLEL_STYLES.register("packed_rowwise", PackedRowwiseParallel())
    ALL_PARALLEL_STYLES.register("sequence_parallel", SequenceParallel(use_local_output=True))
    ALL_PARALLEL_STYLES.register("replicated_with_grad_allreduce", ReplicatedWithGradAllReduce())
    ALL_PARALLEL_STYLES.register("mla_kv_a_proj", MlaKvAProjParallel())
    ALL_PARALLEL_STYLES.register("all_reduce", AllReduceParallel())
    ALL_PARALLEL_STYLES.register("moe_tp_experts", MoeExpertsParallel())
    ALL_PARALLEL_STYLES.register("moe_identity_expert", MoeIdentityParallel())
    # Every rank keeps the full param, e.g. a per-tensor quantization scale
    ALL_PARALLEL_STYLES.register("replicate", TensorParallelLayer())


def _validate_parallel_plan_styles(plan: dict[str, str] | None) -> None:
    unsupported_styles = {style for style in (plan or {}).values() if style not in ALL_PARALLEL_STYLES}
    if unsupported_styles:
        raise ValueError(
            f"Unsupported parallel styles: {unsupported_styles}. Supported styles are {list(ALL_PARALLEL_STYLES.keys())}"
        )


def resolve_parallel_plans(
    model: PreTrainedModel, distributed_config: DistributedConfig
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge the `DistributedConfig` overrides into the model's plans and split them between TP and EP.

    Returns the TP plan to apply to the dense modules and the EP plan to apply to the experts. Each plan is empty
    when its parallel size is 1. EP owns every module it names, so TP rules for those modules and their children
    are dropped: expert weights are sharded once, by the EP plan.
    """
    # Reject invalid paths before merging, e.g. "layers.*" when the model uses "model.layers.*".
    names = {replace_layer_number_by_wildcard(n) for n, _ in chain(model.named_modules(), model.named_parameters())}
    for plan_name in ("tp_plan", "ep_plan"):
        override = getattr(distributed_config, plan_name)
        if isinstance(override, dict):
            plan = getattr(model, plan_name)
            if unknown := override.keys() - names - plan.keys():
                raise ValueError(f"`{plan_name}` keys {sorted(unknown)} match nothing in {type(model).__name__}.")
            setattr(model, f"_{plan_name}", {} if plan_name == "tp_plan" and not override else plan | override)

    tp_plan = dict(model.tp_plan) if distributed_config.tp_size > 1 else {}
    ep_plan = dict(model.ep_plan) if distributed_config.ep_size > 1 else {}
    if distributed_config.ep_size > 1 and not ep_plan:
        raise ValueError(
            f"Expert parallelism was requested (`ep_size={distributed_config.ep_size}`), but `{type(model).__name__}` "
            "does not define an expert-parallel plan. Pass `ep_plan` in `DistributedConfig`, add a "
            "`base_model_ep_plan` to the model's config, or disable expert parallelism."
        )

    if "ep_dispatch_experts" in ep_plan.values() and "ep_router" in ep_plan.values():
        raise ValueError("`ep_dispatch_experts` routes tokens itself; remove the `ep_router` rules from `ep_plan`.")

    # EP rules take precedence: drop TP rules on EP modules and their children.
    is_expert = re.compile(rf"(?:{'|'.join(map(re.escape, ep_plan))})(?:\..+)?").fullmatch
    tp_plan = {name: style for name, style in tp_plan.items() if not is_expert(name)}

    _validate_parallel_plan_styles(tp_plan)
    _validate_parallel_plan_styles(ep_plan)
    return tp_plan, ep_plan


def apply_tensor_parallelism(model, tp_mesh, tp_plan=None):
    """Apply parameter sharding and forward hooks on the TP mesh."""
    tp_plan = model.tp_plan if tp_plan is None else tp_plan
    _validate_parallel_plan_styles(tp_plan)

    for name, module in model.named_modules():
        # Create DTensor placeholders so the loader knows which shard belongs to this rank.
        for p_name, _ in list(module.named_parameters(recurse=False)):
            full = f"{name}.{p_name}" if name else p_name
            style_name = _get_parameter_plan(parameter_name=full, plan=tp_plan, is_weight=True)
            if style_name is not None and style_name in ALL_PARALLEL_STYLES:
                style = ALL_PARALLEL_STYLES[style_name]
                style.validate_param(module, p_name, tp_mesh, parameter_name=full)
                style.shard_param(module, p_name, tp_mesh)

        # Install the input/output transforms required by this module's TP style.
        style_name = _get_parameter_plan(parameter_name=name, plan=tp_plan, is_weight=False)
        if style_name is not None and style_name in ALL_PARALLEL_STYLES:
            if style_name == "mla_kv_a_proj":
                # MLA needs to know the qk_rope_head_dim to split the projection output into KV and RoPE parts.
                # TODO: Store qk_rope_head_dim on MLA projection modules when the models initialize them.
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


# BC: these moved to expert_parallel.py in v5.21, keep the old import path with a warning until v5.23.
# Lazy import because expert_parallel.py imports from this file (circular import otherwise).
_MOVED_TO_EXPERT_PARALLEL = (
    "MoEParamShard",
    "EpRouterParallel",
    "RouterParallelMegaMoe",
    "EpDispatchExpertsParallel",
    "MoeTensorParalellMegaMoeExperts",
    "apply_expert_parallelism",
)


def __getattr__(name: str):
    if name in _MOVED_TO_EXPERT_PARALLEL:
        from . import expert_parallel

        if _get_torch_distributed_rank() == 0:
            warnings.warn(
                f"`transformers.distributed.tensor_parallel.{name}` is deprecated and will be removed in v5.23: "
                f"import it from `transformers.distributed.expert_parallel` instead.",
                FutureWarning,
                stacklevel=2,
            )
        return getattr(expert_parallel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
