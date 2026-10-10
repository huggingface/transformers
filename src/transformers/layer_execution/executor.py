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

"""Execute a plan through the original decoder modules and their existing hooks."""

from collections.abc import Sequence
from functools import partial
from inspect import Parameter, signature
from typing import TypeVar

import torch
from typing_extensions import Unpack

from ..cache_utils import Cache
from ..utils import ModelOutput, TransformersKwargs, is_peft_available
from ..utils.generic import merge_with_config_defaults
from ..utils.output_capturing import _active_collector, capture_outputs
from .adapters import _get_adapter_class
from .cache import LayerExecutionCache, _ExecutionCacheView
from .plan import LayerExecutionPlan, RepeatRange
from .state import state_signature


_ModelT = TypeVar("_ModelT", bound=torch.nn.Module)


def _reject_disabled_plan_cache(decoder, args, kwargs):
    if decoder.config.layer_execution_plan is None:
        if any(isinstance(value, LayerExecutionCache) for value in (*args, *kwargs.values())):
            raise ValueError(
                "LayerExecutionCache requires an enabled execution plan; clear the cache after disabling it."
            )


def _dispatch_decoder(decoder, *args, **kwargs):
    # Keep each model's native positional argument order, including nonstandard cache argument names.
    if args:
        forward_signature = decoder._layer_execution_forward_signature
        bound = forward_signature.bind(*args, **kwargs)
        kwargs = dict(bound.arguments)
        for name, parameter in forward_signature.parameters.items():
            if parameter.kind == Parameter.VAR_KEYWORD:
                kwargs.update(kwargs.pop(name, {}))
            elif parameter.kind == Parameter.VAR_POSITIONAL and kwargs.pop(name, ()):
                raise ValueError("Layer execution requires named decoder forward parameters.")
    return _execute_decoder(decoder, **kwargs)


def _bind_execution_forward(decoder):
    # A partial survives pickle-based process spawning, while preserving the native signature for inspection.
    forward = partial(_dispatch_decoder, decoder)
    setattr(forward, "__signature__", decoder._layer_execution_forward_signature)
    return forward


def _get_model_and_decoder(model: torch.nn.Module):
    from ..modeling_utils import PreTrainedModel, unwrap_model

    unwrapped = unwrap_model(model)
    if not isinstance(unwrapped, PreTrainedModel) and is_peft_available():
        from peft import PeftModel

        if isinstance(unwrapped, PeftModel):
            unwrapped = unwrap_model(unwrapped.get_base_model())
    if not isinstance(unwrapped, PreTrainedModel):
        raise TypeError("Layer execution requires a PreTrainedModel or its distributed/PEFT wrapper.")
    # Unwrap only the containers being configured. Keep wrappers and hooks around individual blocks intact.
    decoder = unwrap_model(unwrapped.get_decoder())
    visited = set()
    while hasattr(decoder, "get_decoder"):
        if id(decoder) in visited:
            raise ValueError("get_decoder returned a cycle instead of a decoder stack.")
        visited.add(id(decoder))
        nested_decoder = unwrap_model(decoder.get_decoder())
        if nested_decoder is decoder:
            break
        decoder = nested_decoder
    if not isinstance(decoder, torch.nn.Module):
        raise TypeError("get_decoder must return a decoder module.")
    return unwrapped, decoder


def get_layer_execution_plan(model: torch.nn.Module) -> LayerExecutionPlan | None:
    """Return an immutable snapshot of a model's current decoder execution plan.

    Args:
        model (`torch.nn.Module`): Model, optionally wrapped in a distributed or PEFT container.
    """
    _, decoder = _get_model_and_decoder(model)
    order = decoder.config.layer_execution_plan
    return LayerExecutionPlan.from_config(decoder.config) if order is not None else None


def set_layer_execution_plan(
    model: _ModelT,
    plan: LayerExecutionPlan | Sequence[int] | None = None,
    *,
    repeats: Sequence[RepeatRange] | None = None,
) -> _ModelT:
    """Enable or disable a shared-parameter execution plan on a supported decoder-only text stack.

    Args:
        model (`torch.nn.Module`): Loaded model, optionally wrapped in a distributed or PEFT container.
        plan (`LayerExecutionPlan`, `Sequence[int]` or `None`, *optional*): Explicit execution order. Omitting both
            `plan` and `repeats` restores the original forward.
        repeats (`Sequence[RepeatRange]`, *optional*): Compile repeat ranges using the decoder's source layer count.
            Pass either `plan` or `repeats`.

    Returns:
        `torch.nn.Module`: The same model or wrapper, preserving registered parameters and state-dict keys.
    """
    unwrapped, decoder = _get_model_and_decoder(model)
    manager = getattr(unwrapped, "_cached_continuous_batching_manager", None)
    if manager is not None:
        if manager.is_running():
            raise ValueError("Stop continuous batching before changing the execution plan.")
        manager.destroy()
        if getattr(unwrapped, "_cached_continuous_batching_manager", None) is manager:
            del unwrapped._cached_continuous_batching_manager
    if unwrapped.config.is_encoder_decoder:
        raise ValueError("Layer execution supports decoder-only models, not encoder-decoder models.")
    if repeats is not None:
        if plan is not None:
            raise ValueError("Pass either plan or repeats, not both.")
        plan = LayerExecutionPlan.from_repeats(decoder.config.num_hidden_layers, repeats)
    elif plan is not None and not isinstance(plan, LayerExecutionPlan):
        plan = LayerExecutionPlan(plan)
    if plan is None:
        if hasattr(decoder, "_layer_execution_original_forward"):
            decoder.forward = decoder._layer_execution_original_forward
            del decoder._layer_execution_original_forward
            del decoder._layer_execution_forward_signature
            del decoder._layer_execution_adapter
        decoder.config.layer_execution_plan = None
        decoder.config.layer_execution_options = None
        if hasattr(decoder, "_layer_execution_steps"):
            del decoder._layer_execution_steps
        if unwrapped.config is not decoder.config:
            unwrapped.config.layer_execution_plan = None
        pipeline = getattr(unwrapped, "_pp_stage", getattr(decoder, "_pp_stage", None))
        if pipeline is not None and getattr(pipeline, "layer_execution", False):
            pipeline.layer_execution = False
            unwrapped.forward = unwrapped._pp_native_forward
        if callable(getattr(model, "zero_optimization_stage", None)):
            from .deepspeed import configure_deepspeed_layer_execution

            configure_deepspeed_layer_execution(model)
        return model
    adapter_class = _get_adapter_class(decoder.config.model_type)
    if adapter_class is None:
        raise ValueError(f"No decoder-only layer execution adapter is registered for {decoder.config.model_type}.")
    adapter = adapter_class()
    layers = adapter.get_layers(decoder)
    plan.validate(len(layers))
    if len(layers) != decoder.config.num_hidden_layers:
        raise ValueError("The source layer count must match num_hidden_layers.")
    adapter.validate(decoder)
    steps = plan.compile(adapter.kv_dependencies(decoder))
    pipeline = getattr(unwrapped, "_pp_stage", getattr(decoder, "_pp_stage", None))
    if (
        pipeline is not None
        and not hasattr(unwrapped, "_pp_native_forward")
        and not getattr(pipeline, "layer_execution", False)
    ):
        raise ValueError("pipeline parallelism metadata requires an initialized pipeline runtime.")
    if not hasattr(decoder, "_layer_execution_original_forward"):
        decoder._layer_execution_forward_signature = signature(decoder.forward)
        decoder._layer_execution_original_forward = decoder.forward
    if not hasattr(decoder, "_layer_execution_cache_guard"):
        decoder._layer_execution_cache_guard = decoder.register_forward_pre_hook(
            _reject_disabled_plan_cache, with_kwargs=True
        )
    decoder._layer_execution_adapter = adapter
    decoder._layer_execution_steps = steps
    decoder.config.layer_execution_plan = list(plan.layer_order)
    decoder.config.layer_execution_options = plan.options() or None
    if unwrapped.config is not decoder.config:
        unwrapped.config.layer_execution_plan = None
    decoder.forward = _bind_execution_forward(decoder)
    if pipeline is not None and not getattr(pipeline, "layer_execution", False):
        from .pipeline import configure_pipeline

        unwrapped.forward = unwrapped._pp_original_forward
        configure_pipeline(unwrapped, pipeline)
    if callable(getattr(model, "zero_optimization_stage", None)):
        from .deepspeed import configure_deepspeed_layer_execution

        configure_deepspeed_layer_execution(model)
    return model


@merge_with_config_defaults
def _execute_decoder(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> ModelOutput:
    adapter = self._layer_execution_adapter
    if adapter.cache_name != "past_key_values":
        named_cache = kwargs.pop(adapter.cache_name, None)
        if named_cache is not None:
            if past_key_values is not None:
                raise ValueError(f"Pass only {adapter.cache_name} or past_key_values, not both.")
            past_key_values = named_cache
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
    # RL trainers can generate under no_grad while leaving the policy in training mode.
    if self.training and torch.is_grad_enabled():
        if past_key_values is not None:
            raise ValueError("Training a layer execution plan requires past_key_values=None.")
        use_cache = False
    if inputs_embeds is None:
        assert input_ids is not None
        pipeline = getattr(self, "_pp_stage", None)
        if pipeline is not None and pipeline.layer_execution:
            from .pipeline import broadcast_state

            if pipeline.pp_is_first_stage:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = torch.zeros(
                    (*input_ids.shape, pipeline.embedding_width),
                    dtype=self.dtype,
                    device=input_ids.device,
                    requires_grad=self.training,
                )
            inputs_embeds = broadcast_state(inputs_embeds, pipeline, 0)
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
    plan = LayerExecutionPlan.from_config(self.config)
    if past_key_values is not None:
        if not isinstance(past_key_values, LayerExecutionCache) or past_key_values.plan != plan:
            raise ValueError(
                "Use a LayerExecutionCache created for the current plan; clear the cache after changing plans."
            )
    elif use_cache:
        past_key_values = LayerExecutionCache(self.config, steps=self._layer_execution_steps)
    if past_key_values is not None:
        past_key_values._batch_size = inputs_embeds.shape[0]
    if past_key_values is not None and past_key_values._record_past and inputs_embeds.shape[1] > 1:
        # Candidate verification must leave a recoverable state at every possible accepted-token boundary.
        outputs = []
        for token_index in range(inputs_embeds.shape[1]):
            token_kwargs = adapter.slice_token_kwargs(kwargs, token_index, inputs_embeds.shape[1])
            if input_ids is not None:
                token_kwargs["input_ids_for_adapter"] = input_ids[:, token_index : token_index + 1]
            token_positions = position_ids[..., token_index : token_index + 1] if position_ids is not None else None

            def slice_mask(mask):
                if isinstance(mask, dict):
                    return {name: slice_mask(value) for name, value in mask.items()}
                if isinstance(mask, torch.Tensor):
                    if mask.ndim == 2:
                        return mask[:, : int(past_key_values.get_seq_length()) + 1]
                    if mask.ndim == 4:
                        query = slice(token_index, token_index + 1) if mask.shape[-2] > 1 else slice(None)
                        return mask[..., query, : int(past_key_values.get_seq_length()) + 1]
                return mask

            token_mask = slice_mask(attention_mask)
            outputs.append(
                _execute_decoder(
                    self,
                    inputs_embeds=inputs_embeds[:, token_index : token_index + 1],
                    attention_mask=token_mask,
                    position_ids=token_positions,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **token_kwargs,
                )
            )

        def merge(values):
            first = values[0]
            if isinstance(first, torch.Tensor):
                if first.ndim == 4 and len({value.shape[-1] for value in values}) > 1:
                    maximum = max(value.shape[-1] for value in values)
                    values = [torch.nn.functional.pad(value, (0, maximum - value.shape[-1])) for value in values]
                return torch.cat(values, dim=-2)
            if isinstance(first, tuple):
                return tuple(merge([value[index] for value in values]) for index in range(len(first)))
            return first

        return type(outputs[0])(**{name: merge([output[name] for output in outputs]) for name in outputs[0]})
    kwargs.setdefault("input_ids_for_adapter", input_ids)
    return _execute_stack(self, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache, **kwargs)


@capture_outputs
def _execute_stack(self, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache, **kwargs):
    adapter = self._layer_execution_adapter
    prepare_kwargs = {name: value for name, value in kwargs.items() if name != "cache"}
    hidden_states, context = adapter.prepare(
        self, inputs_embeds, attention_mask, position_ids, past_key_values, **prepare_kwargs
    )
    hidden_shape = state_signature(hidden_states)
    layers = adapter.get_layers(self)
    shared_states = {}
    paged_results = {}
    pipeline = getattr(self, "_pp_stage", None)
    owners = [None] * len(self._layer_execution_steps)
    collector, capture_keys = None, ()
    if pipeline is not None and pipeline.layer_execution:
        from .pipeline import broadcast_state, layer_owner

        owners = [
            layer_owner(pipeline, step.source_index, self.config.num_hidden_layers)
            for step in self._layer_execution_steps
        ]
        collector = _active_collector.get()
        capture_keys = tuple(key for key in (collector or {}) if not key.startswith("_"))
    for step in self._layer_execution_steps:
        capture_lengths = {key: len(collector[key]) for key in capture_keys}
        execution_index, source_index = step.execution_index, step.source_index
        cache_view = (
            _ExecutionCacheView(past_key_values, source_index, execution_index)
            if past_key_values is not None
            else None
        )
        owner = owners[execution_index]
        if pipeline is None or owner is None or owner == pipeline.pp_rank:
            layer_kwargs = {**kwargs, **adapter.step_kwargs(self, step, context, shared_states)}
            layer_kwargs.pop("input_ids_for_adapter", None)
            if kwargs.get("cache") is not None:
                from ..generation.continuous_batching.cache import PagedAttentionCache
                from .paged import _PagedExecutionCacheView

                if isinstance(kwargs["cache"], PagedAttentionCache):
                    layer_kwargs["cache"] = _PagedExecutionCacheView(kwargs["cache"], step, paged_results)
            if past_key_values is not None:
                past_key_values.before_step(execution_index)
            layer_output = adapter.execute_step(
                layers[source_index], hidden_states, layer_kwargs, cache_view, use_cache
            )
            hidden_states = adapter.extract_state(layer_output, hidden_states)
            adapter.finish_step(step, layer_output, layer_kwargs, shared_states)
            if past_key_values is not None:
                past_key_values.after_step(execution_index)
        else:
            from torch.utils._pytree import tree_map

            hidden_states = tree_map(lambda value: value * 0, hidden_states)
        if (
            pipeline is not None
            and owner is not None
            and (capture_keys or execution_index + 1 == len(owners) or owners[execution_index + 1] != owner)
        ):
            updates = (
                {key: tuple(collector[key][capture_lengths[key] :]) for key in capture_keys}
                if owner == pipeline.pp_rank
                else {}
            )
            hidden_states, shared_states, updates = broadcast_state(
                (hidden_states, shared_states, updates), pipeline, owner
            )
            for key in capture_keys:
                collector[key][capture_lengths[key] :] = updates[key]
        if state_signature(hidden_states) != hidden_shape:
            raise ValueError("Every layer in an execution plan must preserve the hidden-state shape.")
    if past_key_values is not None:
        past_key_values.advance(inputs_embeds.shape[1])
    output = adapter.finalize(self, hidden_states, past_key_values, context)
    if pipeline is not None and capture_keys:
        output.last_hidden_state = broadcast_state(output.last_hidden_state, pipeline, pipeline.pp_size - 1)
    return output
