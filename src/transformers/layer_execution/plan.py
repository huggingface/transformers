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

"""Immutable plans for shared-parameter decoder execution."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RepeatRange:
    """Repeat a half-open, zero-based layer range a total of `total_passes` times.

    Args:
        start (`int`): First source layer, inclusive.
        stop (`int`): Last source layer, exclusive.
        total_passes (`int`, *optional*, defaults to 2): Total number of executions of the range.
    """

    start: int
    stop: int
    total_passes: int = 2

    def __post_init__(self):
        if any(type(value) is not int for value in (self.start, self.stop, self.total_passes)):
            raise ValueError("Range boundaries and total_passes must be integers.")
        if self.start < 0 or self.stop <= self.start or self.total_passes < 1:
            raise ValueError("A repeat range requires 0 <= start < stop and total_passes >= 1.")


@dataclass(frozen=True)
class LayerExecutionPlan:
    """An immutable sequence of zero-based source layer indices, with weights shared between repeated visits.

    Args:
        layer_order (`Sequence[int]`): Source layer index for each logical execution position.
        kv_sharing (`str`, *optional*, defaults to `"independent"`): Set to `"native"` to retain required cross-layer
            KV dependencies. Repeated producer visits still own independent histories.
        kv_dependencies (`Sequence[int | None]`, *optional*): One producer execution index per logical position.
            `None` binds a native consumer to the latest earlier visit of its producer source layer.
    """

    layer_order: tuple[int, ...]
    kv_sharing: str = "independent"
    kv_dependencies: tuple[int | None, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "layer_order", tuple(self.layer_order))
        if not self.layer_order or any(type(index) is not int or index < 0 for index in self.layer_order):
            raise ValueError("layer_order must be a non-empty sequence of non-negative integer indices.")
        if self.kv_sharing not in ("independent", "native"):
            raise ValueError("kv_sharing must be 'independent' or 'native'.")
        object.__setattr__(self, "kv_dependencies", tuple(self.kv_dependencies))
        if self.kv_dependencies and len(self.kv_dependencies) != len(self.layer_order):
            raise ValueError("kv_dependencies must contain one entry per execution position.")

    def validate(self, num_hidden_layers: int) -> None:
        """Check that all source layers exist in a stack of `num_hidden_layers` layers."""
        if type(num_hidden_layers) is not int or num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be a positive integer.")
        if max(self.layer_order) >= num_hidden_layers:
            raise ValueError(f"layer_order contains an index outside the {num_hidden_layers} source layers.")

    def compile(self, dependencies: dict[int, int] | None = None) -> tuple["LayerExecutionStep", ...]:
        """Bind native KV consumers to earlier producer executions, without changing registered modules.

        Native sharing is explicit. By default a consumer reads the most recent execution of its producer; an
        entry in `kv_dependencies` can select a particular earlier execution instead. Every producer execution
        retains an independent cache, including producers revisited by a loop.
        """
        dependencies = dependencies or {}
        if dependencies and self.kv_sharing != "native":
            raise ValueError("Cross-layer KV sharing requires a plan with kv_sharing='native'.")
        latest = {}
        steps = []
        for execution_index, source_index in enumerate(self.layer_order):
            producer_source = dependencies.get(source_index)
            producer = self.kv_dependencies[execution_index] if self.kv_dependencies else None
            if producer_source is not None:
                producer = latest.get(producer_source) if producer is None else producer
                if (
                    type(producer) is not int
                    or not 0 <= producer < execution_index
                    or self.layer_order[producer] != producer_source
                ):
                    raise ValueError(
                        f"Execution {execution_index} requires an earlier producer layer {producer_source}."
                    )
            elif producer is not None:
                raise ValueError(f"Execution {execution_index} is not a native KV consumer.")
            steps.append(LayerExecutionStep(execution_index, source_index, producer))
            latest[source_index] = execution_index
        return tuple(steps)

    @classmethod
    def from_config(cls, config) -> "LayerExecutionPlan":
        """Restore the execution order and optional dependency bindings saved with the original model config."""
        return cls(config.layer_execution_plan, **(config.layer_execution_options or {}))

    def options(self) -> dict:
        """Return JSON-compatible options, omitting defaults for existing checkpoints."""
        result = {}
        if self.kv_sharing != "independent":
            result["kv_sharing"] = self.kv_sharing
        if self.kv_dependencies:
            result["kv_dependencies"] = list(self.kv_dependencies)
        return result

    @classmethod
    def from_repeats(cls, num_hidden_layers: int, repeats: Sequence[RepeatRange]) -> "LayerExecutionPlan":
        """Compile non-overlapping repeat ranges, retaining all other layers in their original order."""
        ranges = sorted(repeats, key=lambda repeat: repeat.start)
        order = []
        cursor = 0
        for repeat in ranges:
            if repeat.start < cursor or repeat.stop > num_hidden_layers:
                raise ValueError("Repeat ranges must not overlap and must lie within the source layer stack.")
            order.extend(range(cursor, repeat.start))
            order.extend(list(range(repeat.start, repeat.stop)) * repeat.total_passes)
            cursor = repeat.stop
        order.extend(range(cursor, num_hidden_layers))
        plan = cls(tuple(order))
        plan.validate(num_hidden_layers)
        return plan


@dataclass(frozen=True)
class LayerExecutionStep:
    """One immutable logical execution; `source_index` selects weights and `execution_index` selects state."""

    execution_index: int
    source_index: int
    kv_producer: int | None = None
