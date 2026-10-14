# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Parameter-sharded execution of logical stages, including depth loops and autograd communication."""

from functools import partial
from inspect import signature

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from ..utils import is_torch_distributed_available


if is_torch_distributed_available():
    from torch.distributed import (
        ReduceOp,
        all_gather_object,
        all_reduce,
        broadcast,
        broadcast_object_list,
        get_backend,
        get_global_rank,
        get_rank,
        get_world_size,
    )


class _BroadcastState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, group, average, *tensors):
        rank = get_rank(group)
        metadata = [[(tuple(value.shape), value.dtype) for value in tensors] if rank == owner else None]
        source = get_global_rank(group, owner)
        broadcast_object_list(metadata, src=source, group=group)
        ctx.owner, ctx.group, ctx.average = owner, group, average
        ctx.inputs = [(tuple(value.shape), value.device, value.dtype) for value in tensors]
        ctx.outputs = metadata[0]
        result = []
        for index, (shape, dtype) in enumerate(ctx.outputs):
            device = tensors[0].device
            value = (
                tensors[index].detach().clone() if rank == owner else torch.empty(shape, dtype=dtype, device=device)
            )
            wire = value.cpu() if get_backend(group) == "gloo" else value
            broadcast(wire, src=source, group=group)
            result.append(wire.to(device))
        return tuple(result)

    @staticmethod
    def backward(ctx, *gradients):
        result = []
        for gradient, (input_shape, device, dtype), (output_shape, output_dtype) in zip(
            gradients, ctx.inputs, ctx.outputs
        ):
            if gradient is None:
                gradient = torch.zeros(output_shape, dtype=output_dtype, device=device)
            wire = (
                gradient.detach().contiguous().cpu() if get_backend(ctx.group) == "gloo" else gradient.detach().clone()
            )
            all_reduce(wire, group=ctx.group)
            if ctx.average:
                wire.div_(get_world_size(ctx.group))
            result.append(
                wire.to(device)
                if get_rank(ctx.group) == ctx.owner
                else torch.zeros(input_shape, dtype=dtype, device=device)
            )
        return None, None, None, *result


def broadcast_state(state, stage, owner, average=False):
    """Communicate a tensor pytree as one autograd node so KV and hidden gradients use the same collective order."""
    rank = stage.pp_rank
    leaves, structure = tree_flatten(state)
    constants = {index: value for index, value in enumerate(leaves) if not isinstance(value, torch.Tensor)}
    spec = [(structure, constants) if rank == owner else None]
    broadcast_object_list(spec, src=get_global_rank(stage.pp_group, owner), group=stage.pp_group)
    assert spec[0] is not None
    structure, constants = spec[0]
    tensors = [value for value in leaves if isinstance(value, torch.Tensor)]
    if rank != owner:
        # Match the producer's tensor arity even when it published a new cross-layer dependency.
        tensors = [tensors[0] * 0 for _ in range(structure.num_leaves - len(constants))]
    values = iter(_BroadcastState.apply(owner, stage.pp_group, average, *tensors))
    leaves = [constants[index] if index in constants else next(values) for index in range(structure.num_leaves)]
    return tree_unflatten(leaves, structure)


class _AverageCapturedGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, size):
        ctx.size = size
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient / ctx.size, None


def layer_owner(stage, source_index, num_layers):
    return next(
        rank
        for rank in range(stage.pp_size)
        if stage.layer_range_for_rank(rank, num_layers)[0]
        <= source_index
        < stage.layer_range_for_rank(rank, num_layers)[1]
    )


def _sync_replicated_gradients(model, previous_gradients):
    stage = model._pp_stage
    parameters = dict(model.named_parameters())
    for name in stage.replicated_parameters:
        parameter = parameters.get(name)
        reference = stage.parameter_metadata[name]
        device = next(model.parameters()).device
        used = torch.tensor(int(parameter is not None and parameter.grad is not None), device=device)
        wire_device = torch.device("cpu") if stage.comm_on_cpu else device
        used = used.to(wire_device)
        all_reduce(used, group=stage.pp_group)
        if not used.item():
            continue
        gradient = (
            parameter.grad.detach()
            if parameter is not None and parameter.grad is not None
            else torch.zeros(reference[0], dtype=reference[1], device=device)
        )
        previous = previous_gradients.get(name)
        if previous is not None:
            gradient = gradient - previous
        wire = gradient.clone().to(wire_device)
        all_reduce(wire, group=stage.pp_group)
        if parameter is not None:
            parameter.grad = wire.to(parameter.device)
            if previous is not None:
                parameter.grad.add_(previous)
    stage.gradient_sync_queued = False


def _pipeline_forward(model, *args, **kwargs):
    stage = model._pp_stage
    plans = [None] * stage.pp_size
    all_gather_object(plans, model.get_layer_execution_plan(), group=stage.pp_group)
    if any(plan != plans[0] for plan in plans):
        raise ValueError("All pipeline ranks must use the same execution plan.")
    original = model._layer_execution_pp_original_forward
    if args:
        bound = signature(original).bind(*args, **kwargs)
        kwargs = dict(bound.arguments)
        for name, parameter in signature(original).parameters.items():
            if parameter.kind == parameter.VAR_KEYWORD:
                kwargs.update(kwargs.pop(name, {}))
    labels = kwargs.pop("labels", None)
    requested_tuple = kwargs.get("return_dict", model.config.return_dict) is False
    kwargs["return_dict"] = True
    output = original(**kwargs)
    output.logits = broadcast_state(output.logits, stage, stage.pp_size - 1, average=True)
    captured_outputs = []
    for name in output:
        if name not in ("logits", "loss", "past_key_values", "cache_params"):
            output[name] = tree_map(
                lambda value: _AverageCapturedGradient.apply(value, stage.pp_size)
                if isinstance(value, torch.Tensor) and value.requires_grad
                else value,
                output[name],
            )
            captured_outputs.append(output[name])
    if labels is not None:
        loss = model.loss_function(
            logits=output.logits,
            labels=labels,
            vocab_size=model.config.get_text_config(decoder=True).vocab_size,
            **kwargs,
        )
        output = type(output)(**{**output, "loss": loss})
    if model.training and output.logits.requires_grad and stage.replicated_parameters:

        def queue_sync(gradient):
            if not stage.gradient_sync_queued:
                stage.gradient_sync_queued = True
                parameters = dict(model.named_parameters())
                previous = {
                    name: parameters[name].grad.detach().clone()
                    for name in stage.replicated_parameters
                    if name in parameters and parameters[name].grad is not None
                }
                torch.autograd.Variable._execution_engine.queue_callback(
                    partial(_sync_replicated_gradients, model, previous)
                )
            return gradient

        leaves, _ = tree_flatten((output.logits, captured_outputs))
        for value in leaves:
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.register_hook(queue_sync)
    return output.to_tuple() if requested_tuple else output


def configure_pipeline(model, stage):
    """Install the loop-aware runtime after physical parameter ownership has been assigned."""
    stage.layer_execution = True
    stage.gradient_sync_queued = False
    parameters = dict(model.named_parameters())
    names = [None] * stage.pp_size
    all_gather_object(names, tuple(parameters), group=stage.pp_group)
    counts = {name: sum(name in owned for owned in names) for name in set().union(*map(set, names))}
    stage.parameter_copies = counts
    stage.replicated_parameters = sorted(name for name, count in counts.items() if count > 1)
    metadata = {name: (tuple(value.shape), value.dtype) for name, value in parameters.items()}
    all_metadata = [None] * stage.pp_size
    all_gather_object(all_metadata, metadata, group=stage.pp_group)
    stage.parameter_metadata = {name: meta for item in all_metadata for name, meta in item.items()}
    model._layer_execution_pp_original_forward = model.forward
    model.forward = partial(_pipeline_forward, model)
    setattr(model.forward, "__signature__", signature(model._layer_execution_pp_original_forward))
    model.is_parallelizable = True
    model.model_parallel = True


def gather_pipeline_state_dict(model, state_dict):
    """Gather one tensor at a time, so checkpoint saving does not materialize the full model on every accelerator."""
    stage = model._pp_stage
    metadata = {name: (tuple(value.shape), value.dtype) for name, value in state_dict.items()}
    descriptions = [None] * stage.pp_size
    all_gather_object(descriptions, metadata, group=stage.pp_group)
    result = {}
    device = torch.device("cpu") if stage.comm_on_cpu else next(model.parameters()).device
    for name in sorted(set().union(*map(set, descriptions))):
        owner = next(index for index, description in enumerate(descriptions) if name in description)
        shape, dtype = descriptions[owner][name]
        value = (
            state_dict[name].detach().to(device)
            if stage.pp_rank == owner
            else torch.empty(shape, dtype=dtype, device=device)
        )
        broadcast(value, src=get_global_rank(stage.pp_group, owner), group=stage.pp_group)
        if stage.pp_is_first_stage:
            result[name] = value.cpu().clone()
    if stage.pp_is_first_stage:
        for destination, source in stage.original_tied_weights_keys.items():
            if destination in result and source in result:
                result[destination] = result[source]
    return result


def pipeline_gradient_norm(model, maximum):
    """Clip a global norm, counting replicated boundary weights once."""
    stage = model._pp_stage
    device = torch.device("cpu") if stage.comm_on_cpu else next(model.parameters()).device
    squared = torch.zeros((), dtype=torch.float32, device=device)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum().to(device) / stage.parameter_copies[name]
    all_reduce(squared, group=stage.pp_group)
    norm = squared.sqrt()
    if maximum != float("inf"):
        coefficient = (maximum / (norm + 1e-6)).clamp(max=1)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient.to(parameter.device))
    return norm


def unscale_pipeline_gradients(accelerator, optimizer, stage):
    """Unscale locally, then make every stage agree whether the optimizer must skip an overflowing update."""
    accelerator.unscale_gradients(optimizer)
    scaler = accelerator.scaler
    if scaler is None or not scaler.is_enabled():
        return
    underlying = getattr(optimizer, "optimizer", optimizer)
    found = scaler._per_optimizer_states[id(underlying)]["found_inf_per_device"]
    device = torch.device("cpu") if stage.comm_on_cpu else accelerator.device
    overflow = sum((value.to(device) for value in found.values()), torch.zeros((), device=device))
    all_reduce(overflow, op=ReduceOp.MAX, group=stage.pp_group)
    for value in found.values():
        value.copy_(overflow.to(value.device))
