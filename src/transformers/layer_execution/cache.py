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

"""Independent state for every logical decoder execution."""

import copy
from typing import TYPE_CHECKING

import torch
from torch.utils._pytree import tree_flatten, tree_map

from ..cache_utils import Cache, CacheLayerMixin, DynamicLayer, LinearAttentionLayer, get_layer_types_and_kwargs
from .backends import create_cache_layer
from .plan import LayerExecutionPlan


if TYPE_CHECKING:
    from ..configuration_utils import PreTrainedConfig


class LayerExecutionCache(Cache):
    """Independent state for every logical execution, using native dynamic, static, quantized or offloaded storage.

    Args:
        config (`PreTrainedConfig`): Model or text configuration containing `layer_execution_plan`.
        cache_implementation (`str`, *optional*, defaults to `"dynamic"`): Storage backend: `"dynamic"`, `"static"`,
            `"offloaded"`, `"offloaded_static"`, or `"quantized"`.
        max_cache_len (`int`, *optional*): Required positive token capacity for static storage.
        cache_config (`dict`, *optional*): Backend options, including quantizer backend and precision.
        steps (`Sequence[LayerExecutionStep]`, *optional*): Compiled adapter dependencies. Generation supplies these
            automatically; custom adapters with native KV dependencies should supply them for manual caches.
    """

    def __init__(
        self,
        config: "PreTrainedConfig",
        cache_implementation="dynamic",
        max_cache_len=None,
        cache_config=None,
        *,
        steps=None,
    ):
        config = config.get_text_config(decoder=True)
        if config.layer_execution_plan is None:
            raise ValueError("LayerExecutionCache requires a layer_execution_plan in the text configuration.")
        plan = LayerExecutionPlan.from_config(config)
        plan.validate(config.num_hidden_layers)
        if getattr(config, "num_kv_shared_layers", 0) and plan.kv_sharing != "native":
            raise ValueError("Native cross-layer KV sharing requires kv_sharing='native'.")
        if cache_implementation not in ("dynamic", "static", "offloaded", "offloaded_static", "quantized"):
            raise ValueError(f"Unsupported layer execution cache implementation: {cache_implementation!r}.")
        self.cache_implementation = cache_implementation
        self.plan = plan
        if steps is None and getattr(config, "num_kv_shared_layers", 0):
            first_shared = config.num_hidden_layers - config.num_kv_shared_layers
            producers = {kind: index for index, kind in enumerate(config.layer_types[:first_shared])}
            dependencies = {
                index: producers[kind] for index, kind in enumerate(config.layer_types) if index >= first_shared
            }
            steps = plan.compile(dependencies)
        self.steps = steps or plan.compile()
        layer_types, layer_kwargs = get_layer_types_and_kwargs(config)
        if config.model_type == "recurrent_gemma":
            layer_types = [
                "linear_attention" if kind == "recurrent" else "sliding_attention" for kind in config.layers_block_type
            ]
            layer_kwargs = [
                {"sliding_window": config.sliding_window} if kind == "sliding_attention" else {}
                for kind in layer_types
            ]
        layers = [
            LinearAttentionLayer()
            if step.kv_producer is not None
            else create_cache_layer(
                layer_types[step.source_index],
                layer_kwargs[step.source_index],
                cache_implementation,
                max_cache_len,
                cache_config or {},
            )
            for step in self.steps
        ]
        # Offloading is managed at execution boundaries, also covering reads without an attention-cache update.
        Cache.__init__(self, layers=layers)
        self.offloading = cache_implementation in ("offloaded", "offloaded_static")
        self._offload_streams = {}
        self._state_devices = {}
        self.layer_order = plan.layer_order
        self.num_source_layers = config.num_hidden_layers
        self._seen_tokens = torch.zeros((), dtype=torch.long) if cache_implementation == "static" else 0
        self._has_attention_cache = any(isinstance(layer, CacheLayerMixin) for layer in layers)
        self.layer_states = [{} for _ in layers]
        self._record_past = False
        self._history = {}
        self._batch_size = -1

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Depth recurrence does not advance token positions. Advance only once after a complete decoder forward.
        return self._seen_tokens

    @property
    def batch_size(self):
        result = super().batch_size
        if result == -1:
            leaves, _ = tree_flatten(self.layer_states)
            result = next((value.shape[0] for value in leaves if isinstance(value, torch.Tensor)), -1)
        return result if result != -1 else self._batch_size

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        # Generation can still prepare an unused full-attention mask when a plan selects only recurrent layers.
        if not self._has_attention_cache:
            return self._seen_tokens + query_length, 0
        layer = self.layers[layer_idx]
        if isinstance(layer, CacheLayerMixin) and not layer.is_initialized:
            # Non-owned pipeline slots have metadata but no tensors; masks still span the request's actual history.
            if self.cache_implementation in ("dynamic", "offloaded", "quantized"):
                window = getattr(layer, "sliding_window", None)
                if window is not None:
                    return min(self._seen_tokens, window - 1) + query_length, max(self._seen_tokens - window + 1, 0)
                return self._seen_tokens + query_length, 0
        return super().get_mask_sizes(query_length, layer_idx)

    @property
    def is_compileable(self) -> bool:
        return self.cache_implementation == "static" and not self._record_past

    @property
    def requires_explicit_mask(self) -> bool:
        """Static capacity includes unwritten positions, even when transfers prevent compilation."""
        return self.cache_implementation in ("static", "offloaded_static")

    def before_step(self, execution_index):
        if not self.offloading:
            return
        layer = self.layers[execution_index]
        device = getattr(layer, "device", None) or self._state_devices.get(execution_index)
        if device is None or device.type != "cuda":
            return
        if device not in self._offload_streams:
            self._offload_streams[device] = torch.cuda.Stream(device=device)
        stream = self._offload_streams[device]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            layer.prefetch()
            self.layer_states[execution_index] = tree_map(
                lambda value: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value,
                self.layer_states[execution_index],
            )
        torch.cuda.current_stream(device).wait_stream(stream)

    def after_step(self, execution_index):
        if self.offloading:
            layer = self.layers[execution_index]
            leaves, _ = tree_flatten(self.layer_states[execution_index])
            device = getattr(layer, "device", None) or next(
                (value.device for value in leaves if isinstance(value, torch.Tensor)), None
            )
            if device is not None and device.type == "cuda":
                self._state_devices[execution_index] = device
                layer.offload()
                self.layer_states[execution_index] = tree_map(
                    lambda value: value.to("cpu", non_blocking=True) if isinstance(value, torch.Tensor) else value,
                    self.layer_states[execution_index],
                )

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # Cache.update's physical-next-layer prefetch is inappropriate for virtual stages and shared KV consumers.
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def activate_past_recording(self):
        # Snapshot complete execution state so rejected speculative tokens also roll back recurrent and custom state.
        self._record_past = True
        self.record_snapshot()

    def record_snapshot(self):
        if self._record_past:
            self._history[int(self._seen_tokens)] = self._copy_execution_state()

    def _copy_execution_state(self):
        for device in set(self._state_devices.values()):
            torch.cuda.current_stream(device).synchronize()
        layers = []
        for original in self.layers:
            layer = copy.copy(original)
            for name, value in vars(original).items():
                # Dynamic KV updates concatenate into fresh storage. Only in-place recurrent/static state needs copying.
                if isinstance(original, DynamicLayer) and name in ("keys", "values", "indexer_keys"):
                    continue
                setattr(
                    layer, name, tree_map(lambda item: item.clone() if isinstance(item, torch.Tensor) else item, value)
                )
            layers.append(layer)
        states = tree_map(lambda item: item.clone() if isinstance(item, torch.Tensor) else item, self.layer_states)
        return layers, states

    def commit_past(self, retain_tokens=1):
        """Release accepted speculative history, keeping the last boundary needed by assistant continuation."""
        earliest = int(self._seen_tokens) - retain_tokens
        self._history = {length: state for length, state in self._history.items() if length >= earliest}

    def advance(self, tokens):
        if isinstance(self._seen_tokens, torch.Tensor):
            self._seen_tokens.add_(tokens)
        else:
            self._seen_tokens += tokens
        self.record_snapshot()

    def reset(self):
        for device in set(self._state_devices.values()):
            torch.cuda.current_stream(device).synchronize()
        super().reset()
        if isinstance(self._seen_tokens, torch.Tensor):
            self._seen_tokens.zero_()
        else:
            self._seen_tokens = 0
        self.layer_states = [{} for _ in self.layers]
        self._history.clear()

    def crop(self, tokens_to_remove: int):
        current_length = int(self._seen_tokens)
        new_length = (
            max(0, current_length + tokens_to_remove)
            if tokens_to_remove <= 0
            else min(current_length, tokens_to_remove)
        )
        if self._record_past:
            if new_length not in self._history:
                raise ValueError("No recorded execution state at the requested rollback position.")
            self.layers, self.layer_states = self._history[new_length]
            self.layers, self.layer_states = self._copy_execution_state()
            self._history = {length: state for length, state in self._history.items() if length <= new_length}
        else:
            super().crop(tokens_to_remove)
        if isinstance(self._seen_tokens, torch.Tensor):
            self._seen_tokens.fill_(new_length)
        else:
            self._seen_tokens = new_length

    def reorder_cache(self, beam_idx):
        self._batch_size = len(beam_idx)
        for index, layer in enumerate(self.layers):
            self.before_step(index)
            layer.reorder_cache(beam_idx)
            if hasattr(layer, "batch_size"):
                layer.batch_size = len(beam_idx)
            self.layer_states[index] = tree_map(
                lambda value: value.index_select(0, beam_idx.to(value.device))
                if isinstance(value, torch.Tensor)
                else value,
                self.layer_states[index],
            )
            self.after_step(index)
        self._history.clear()
        self.record_snapshot()

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats):
        if self.batch_size != -1:
            self.reorder_cache(torch.arange(self.batch_size).repeat_interleave(repeats))

    @classmethod
    def stack(cls, caches):
        """Batch equal-length dynamic request caches without sharing their mutable storage."""
        first = caches[0]
        if any(
            cache.plan != first.plan or int(cache.get_seq_length()) != int(first.get_seq_length()) for cache in caches
        ):
            raise ValueError("Only matching plans and sequence lengths can share a decode batch.")
        if any(cache.cache_implementation != "dynamic" for cache in caches):
            raise ValueError("Cache stacking uses dynamic storage; other backends can run as separate request groups.")
        result = copy.copy(first)
        result.layers = []
        for layer_index, source in enumerate(first.layers):
            layer = copy.copy(source)
            for name in ("keys", "values", "indexer_keys", "conv_states", "recurrent_states"):
                if hasattr(source, name):
                    values = [getattr(cache.layers[layer_index], name) for cache in caches]
                    setattr(
                        layer,
                        name,
                        tree_map(
                            lambda *items: torch.cat(items, dim=0) if isinstance(items[0], torch.Tensor) else items[0],
                            *values,
                        ),
                    )
            if hasattr(source, "batch_size"):
                layer.batch_size = sum(cache.batch_size for cache in caches)
            result.layers.append(layer)
        result.layer_states = tree_map(
            lambda *items: torch.cat(items, dim=0) if isinstance(items[0], torch.Tensor) else items[0],
            *[cache.layer_states for cache in caches],
        )
        result._history = {}
        result._batch_size = sum(cache.batch_size for cache in caches)
        return result

    def unstack(self):
        """Return one independent cache per batch row, including custom recurrent state."""
        result = []
        for index in range(self.batch_size):
            cache = copy.copy(self)
            cache.layers = copy.deepcopy(self.layers)
            cache.layer_states = copy.deepcopy(self.layer_states)
            cache._history = {}
            cache.reorder_cache(torch.tensor([index]))
            result.append(cache)
        return result


class _ExecutionCacheLayers:
    """Expose only the source layer's state, remapped to its current execution position."""

    def __init__(self, cache: LayerExecutionCache, source_index: int, execution_index: int):
        self.cache = cache
        self.source_index = source_index
        self.execution_index = execution_index

    def __len__(self):
        return self.cache.num_source_layers

    def __getitem__(self, index):
        if index != self.source_index:
            raise ValueError("A layer execution cannot access another source layer's cache.")
        return self.cache.layers[self.execution_index]


class _ExecutionCacheView(Cache):
    """Per-call routing without changing the shared module's layer_idx."""

    def __init__(self, cache: LayerExecutionCache, source_index: int, execution_index: int):
        self.cache = cache
        self.source_index = source_index
        self.execution_index = execution_index
        self.layers = _ExecutionCacheLayers(cache, source_index, execution_index)
        self.layer_class_to_replicate = None
        self.offloading = cache.offloading

    @property
    def state(self):
        """Execution-local custom state; no mutable state is stored on the shared parameter module."""
        return self.cache.layer_states[self.execution_index]

    def _check_index(self, layer_idx):
        if layer_idx is not None and layer_idx != self.source_index:
            raise ValueError("A layer execution cannot access another source layer's cache.")

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self._check_index(layer_idx)
        return self.cache.update(key_states, value_states, self.execution_index, *args, **kwargs)

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kwargs):
        self._check_index(layer_idx)
        return self.cache.update_conv_state(conv_states, self.execution_index, state_idx, **kwargs)

    def update_recurrent_state(self, recurrent_states, layer_idx, state_idx=0, **kwargs):
        self._check_index(layer_idx)
        return self.cache.update_recurrent_state(recurrent_states, self.execution_index, state_idx, **kwargs)

    def has_previous_state(self, layer_idx=None, state_idx=None):
        self._check_index(layer_idx)
        return self.cache.has_previous_state(self.execution_index, state_idx)

    def get_seq_length(self, layer_idx=0):
        return self.cache.get_seq_length()

    def get_max_length(self, layer_idx=None):
        self._check_index(layer_idx)
        return self.cache.get_max_length(self.execution_index)

    def get_mask_sizes(self, query_length, layer_idx):
        self._check_index(layer_idx)
        return self.cache.get_mask_sizes(query_length, self.execution_index)

    @property
    def batch_size(self):
        return self.cache.batch_size

    @property
    def is_compileable(self):
        return self.cache.is_compileable

    @property
    def requires_explicit_mask(self):
        return self.cache.requires_explicit_mask

    @property
    def is_initialized(self):
        layer = self.cache.layers[self.execution_index]
        return isinstance(layer, CacheLayerMixin) and layer.supports_early_init and layer.is_initialized

    @property
    def is_croppable(self):
        return self.cache.layers[self.execution_index].is_croppable

    @property
    def is_sliding(self):
        flags = [False] * self.cache.num_source_layers
        flags[self.source_index] = self.cache.is_sliding[self.execution_index]
        return flags

    @property
    def is_linear(self):
        flags = [False] * self.cache.num_source_layers
        flags[self.source_index] = self.cache.is_linear[self.execution_index]
        return flags
