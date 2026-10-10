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

"""Route native paged attention storage by logical execution position."""

import copy

from ..generation.continuous_batching.cache import PagedAttentionCache
from .plan import LayerExecutionPlan


def execution_cache_config(config):
    """Expand only cache metadata; preserve the original model config, registered layer count, and weight names."""
    plan = LayerExecutionPlan.from_config(config)
    result = copy.deepcopy(config)
    types = getattr(config, "layer_types", None)
    result.num_hidden_layers = len(plan.layer_order)
    if types is not None:
        result.layer_types = [types[index] for index in plan.layer_order]
    dependencies = {}
    shared = getattr(config, "num_kv_shared_layers", 0)
    if shared:
        cutoff = config.num_hidden_layers - shared
        producers = {kind: index for index, kind in enumerate(types[:cutoff])}
        dependencies = {index: producers[kind] for index, kind in enumerate(types) if index >= cutoff}
    result._layer_execution_cache_owners = [
        step.execution_index for step in plan.compile(dependencies) if step.kv_producer is None
    ]
    result.layer_execution_plan = None
    result.layer_execution_options = None
    return result


class _PagedExecutionCacheView(PagedAttentionCache):
    # Subclassing retains native attention integrations' isinstance checks without allocating a second cache.
    def __init__(self, cache, step, step_results):
        self._parent = cache
        self._step = step
        self._step_results = step_results

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def update(self, key_states, value_states, layer_idx, kwargs):
        if layer_idx != self._step.source_index:
            raise ValueError("A paged execution can only access its own source layer.")
        if self._step.kv_producer is not None:
            states, routed_kwargs = self._step_results[self._step.kv_producer]
            kwargs.update(routed_kwargs)
            return states
        result = self._parent.update(key_states, value_states, self._step.execution_index, kwargs)
        self._step_results[self._step.execution_index] = (
            result,
            {
                name: kwargs[name]
                for name in ("cu_seq_lens_k", "max_length_k", "block_table", "k_cache", "v_cache")
                if name in kwargs
            },
        )
        return result

    def get_cache_for_block_table(self, layer_idx):
        owner = self._step.kv_producer if self._step.kv_producer is not None else self._step.execution_index
        return self._parent.get_cache_for_block_table(owner)
