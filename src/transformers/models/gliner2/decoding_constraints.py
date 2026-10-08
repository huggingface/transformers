# Copyright 2026 The HuggingFace Inc. team and the GLiNER2 authors. All rights reserved.
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
"""Inference-only constrained classification decoder."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from types import MappingProxyType
from typing import Any, Protocol, final, runtime_checkable


_RESERVED = (
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "(",
    ")",
)
_ACTIVATIONS = ("auto", "sigmoid", "softmax")
_DECODERS = ("auto", "independent", "exact", "beam")
_ON_INFEASIBLE = ("relax", "min_violations", "raise")
_AGGREGATIONS = ("max", "mean", "first")
_MODEL_KEYS = (
    "json_structures",
    "classifications",
    "entities",
    "relations",
    "json_descriptions",
    "entity_descriptions",
)
_L = "[L]"


class SchemaError(ValueError):
    """Invalid schema, task, label, or constraint."""


class InfeasibleError(RuntimeError):
    """No assignment satisfies the constraints."""

    def __init__(self, message: str, violations=()):
        super().__init__(message)
        self.violations = tuple(violations)


def _clean(value: str, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{what} must be a non-empty string")
    for token in _RESERVED:
        if token in value:
            raise SchemaError(
                f"{what} may not contain {token!r}: label and prompt strings are "
                f"injected into the model prompt verbatim, and this would corrupt "
                f"logit-to-label alignment"
            )
    return value


def _prob(value: float | None, field_name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise SchemaError(f"{field_name} must be in [0, 1]")


def _number(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def probability_to_logit(probability: float) -> float:
    """Return log-odds, accepting exact zero and one."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if probability == 0.0:
        return -math.inf
    if probability == 1.0:
        return math.inf
    return math.log(probability) - math.log1p(-probability)


def center_logit(logit: float, threshold: float = 0.5) -> float:
    """Center a logit so positive utility means passing ``threshold``."""
    return float(logit) - probability_to_logit(threshold)


def sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid."""
    value = float(value)
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(values: Sequence[float]) -> list:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [item / total for item in exps]


def _geometric_mean(values) -> float | None:
    values = list(values)
    if not values:
        return None
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("confidence components must be probabilities in [0, 1]")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def label_names_from_tokens(schema_tokens: Sequence[str]) -> tuple:
    """Recover label order from the ``[L]`` tokens that were encoded."""
    return tuple(schema_tokens[i + 1] for i in range(len(schema_tokens) - 1) if schema_tokens[i] == _L)


def recover_task_name(schema_tokens: Sequence[str], known: Sequence[str]) -> str:
    """Resolve the owning task by boundary-aware longest match."""
    prompt_str = schema_tokens[2] if len(schema_tokens) > 2 else ""
    best = None
    for name in known:
        if prompt_str.startswith(name):
            rest = prompt_str[len(name) :]
            if rest == "" or rest[0] in (":", " "):
                if best is None or len(name) > len(best):
                    best = name
    if best is not None:
        return best
    return prompt_str.split(" [DESCRIPTION] ", 1)[0].split(":", 1)[0]


@dataclass(frozen=True)
class LabelSpec:
    """One classification label."""

    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        _clean(self.name, "label name")
        if self.description is not None:
            _clean(self.description, f"description of label {self.name!r}")


@dataclass(frozen=True)
class TaskSpec:
    """Cardinality, order, and calibration for one classification task."""

    name: str
    labels: tuple
    min_labels: int = 0
    max_labels: int | None = None
    ordered: bool = False
    threshold: float = 0.5
    candidate_threshold: float | None = None
    activation: str = "auto"
    temperature: float = 1.0
    default: str | None = None
    instruction: str | None = None
    examples: tuple = ()

    def __post_init__(self) -> None:
        _clean(self.name, "task name")
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "examples", tuple(tuple(pair) for pair in self.examples))
        if not self.labels:
            raise SchemaError(f"task {self.name!r} must declare at least one label")
        names = [label.name for label in self.labels]
        if len(set(names)) != len(names):
            raise SchemaError(f"task {self.name!r} has duplicate label names")
        if not isinstance(self.min_labels, int) or self.min_labels < 0:
            raise SchemaError(f"task {self.name!r}: min_labels must be a non-negative int")
        if self.max_labels is not None:
            if not isinstance(self.max_labels, int) or self.max_labels < 0:
                raise SchemaError(f"task {self.name!r}: max_labels must be a non-negative int")
            if self.max_labels > len(self.labels):
                raise SchemaError(
                    f"task {self.name!r}: max_labels ({self.max_labels}) exceeds the "
                    f"number of labels ({len(self.labels)})"
                )
        if self.default is not None:
            if self.default not in names:
                raise SchemaError(f"task {self.name!r}: default {self.default!r} is not one of its labels")
            if self.min_labels < 1:
                object.__setattr__(self, "min_labels", 1)
        if self.max_labels is not None and self.min_labels > self.max_labels:
            raise SchemaError(
                f"task {self.name!r}: min_labels ({self.min_labels}) exceeds max_labels ({self.max_labels})"
            )
        if self.ordered and len(self.labels) < 2:
            raise SchemaError(f"ordered task {self.name!r} requires at least two labels")
        if not isinstance(self.threshold, (int, float)) or not 0 < self.threshold < 1:
            raise SchemaError(f"task {self.name!r}: threshold must be in (0, 1)")
        _prob(self.candidate_threshold, f"task {self.name!r}: candidate_threshold")
        if self.activation not in _ACTIVATIONS:
            raise SchemaError(f"task {self.name!r}: activation must be one of {_ACTIVATIONS}")
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0:
            raise SchemaError(f"task {self.name!r}: temperature must be positive")
        if self.instruction is not None:
            _clean(self.instruction, f"instruction of task {self.name!r}")
        for pair in self.examples:
            if len(pair) != 2:
                raise SchemaError(f"task {self.name!r}: each example must be a (text, label) pair")
            _, label = pair
            if label not in names:
                raise SchemaError(f"task {self.name!r}: example label {label!r} is not one of its labels")

    @property
    def label_names(self) -> tuple:
        return tuple(label.name for label in self.labels)

    @property
    def is_exclusive(self) -> bool:
        return self.min_labels == 1 and self.max_labels == 1

    def effective_max_labels(self) -> int:
        return len(self.labels) if self.max_labels is None else self.max_labels


def _coerce_labels(labels: Any) -> tuple:
    if isinstance(labels, Mapping):
        return tuple(LabelSpec(name, desc) for name, desc in labels.items())
    if isinstance(labels, str):
        raise SchemaError("labels must be a list or a {label: description} mapping, not a str")
    coerced = []
    for item in labels:
        if isinstance(item, LabelSpec):
            coerced.append(item)
        elif isinstance(item, str):
            coerced.append(LabelSpec(item))
        else:
            raise SchemaError(f"invalid label entry {item!r}")
    return tuple(coerced)


def _partition(constraints, active):
    keep, pure_drop, mixed = [], [], []
    for constraint in constraints:
        refs = frozenset(constraint.references())
        if refs <= active:
            keep.append(constraint)
        elif refs & active:
            mixed.append(constraint)
        else:
            pure_drop.append(constraint)
    return keep, pure_drop, mixed


class ClassificationSchema:
    """Mutable builder for tasks and hard constraints."""

    def __init__(self) -> None:
        self._tasks: dict = {}
        self._constraints: list = []

    @property
    def task_specs(self) -> tuple:
        return tuple(self._tasks.values())

    @property
    def task_order(self) -> tuple:
        return tuple(self._tasks.keys())

    @property
    def constraints(self) -> tuple:
        return tuple(self._constraints)

    def task_spec(self, name: str) -> TaskSpec:
        try:
            return self._tasks[name]
        except KeyError:
            raise SchemaError(f"unknown task {name!r}") from None

    def task(
        self,
        name,
        labels,
        *,
        min_labels=0,
        max_labels=None,
        ordered=False,
        threshold=0.5,
        candidate_threshold=None,
        activation="auto",
        temperature=1.0,
        default=None,
        instruction=None,
        examples=(),
    ):
        if name in self._tasks:
            raise SchemaError(f"task {name!r} is already defined")
        self._tasks[name] = TaskSpec(
            name=name,
            labels=_coerce_labels(labels),
            min_labels=min_labels,
            max_labels=max_labels,
            ordered=ordered,
            threshold=threshold,
            candidate_threshold=candidate_threshold,
            activation=activation,
            temperature=temperature,
            default=default,
            instruction=instruction,
            examples=tuple(examples),
        )
        return self

    def single(self, name, labels, **kwargs):
        kwargs.setdefault("min_labels", 1)
        kwargs.setdefault("max_labels", 1)
        return self.task(name, labels, **kwargs)

    def multi(self, name, labels, **kwargs):
        return self.task(name, labels, **kwargs)

    def ordinal(self, name, labels, **kwargs):
        kwargs.setdefault("min_labels", 1)
        kwargs.setdefault("max_labels", 1)
        kwargs["ordered"] = True
        return self.task(name, labels, **kwargs)

    def constrain(self, *expressions):
        """Append constraints validated against tasks declared so far."""
        declared = set(self._tasks)
        for expr in expressions:
            if not hasattr(expr, "references"):
                raise SchemaError(
                    f"constraint {expr!r} is not a constraint expression; build one with the constraints DSL"
                )
            missing = frozenset(expr.references()) - declared
            if missing:
                name = sorted(missing)[0]
                raise SchemaError(
                    f"constraint references undeclared task {name!r}; declare task {name!r} before constraining it"
                )
            self._constraints.append(expr)
        return self

    def subset(self, *names, keep_constraints="strict"):
        """Return a new schema containing only ``names``."""
        return self._narrow(self._resolve_names(names), keep_constraints)

    def drop(self, *names, keep_constraints="strict"):
        """Return a new schema with ``names`` removed."""
        removed = self._resolve_names(names)
        return self._narrow(frozenset(self._tasks) - removed, keep_constraints)

    def _resolve_names(self, names) -> frozenset:
        requested = frozenset(names)
        unknown = requested - set(self._tasks)
        if unknown:
            raise SchemaError(f"unknown task(s): {sorted(unknown)}")
        return requested

    def _narrow(self, active: frozenset, keep_constraints: str):
        if keep_constraints not in ("strict", "prune", "error_on_mixed"):
            raise SchemaError("keep_constraints must be 'strict', 'prune' or 'error_on_mixed'")
        keep, pure_drop, mixed = _partition(self._constraints, active)
        if mixed:
            raise SchemaError(
                "cannot narrow: constraint(s) reference both kept and removed tasks; "
                "a half-applied invariant is never safe"
            )
        if keep_constraints == "strict" and pure_drop:
            raise SchemaError(
                "cannot narrow under keep_constraints='strict': would drop "
                f"{len(pure_drop)} constraint(s); use 'prune' or 'error_on_mixed'"
            )
        if keep_constraints == "prune" and pure_drop:
            warnings.warn(
                f"subset/drop dropped {len(pure_drop)} constraint(s) referencing only "
                f"removed tasks; the prompt changes, so remaining scores are a new "
                f"measurement",
                stacklevel=3,
            )
        out = ClassificationSchema()
        for name, spec in self._tasks.items():
            if name in active:
                out._tasks[name] = spec
        out._constraints = list(keep)
        return out

    def to_dict(self) -> dict:
        return {
            "version": 3,
            "tasks": {name: _task_to_dict(spec) for name, spec in self._tasks.items()},
            "constraints": [constraint.to_dict() for constraint in self._constraints],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping) -> ClassificationSchema:
        schema = cls()
        for name, spec in data.get("tasks", {}).items():
            schema.task(name, **_task_kwargs_from_dict(spec))
        for raw in data.get("constraints", ()):
            schema.constrain(constraint_from_dict(raw))
        return schema

    @classmethod
    def from_json(cls, value: str) -> ClassificationSchema:
        return cls.from_dict(json.loads(value))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ClassificationSchema):
            return NotImplemented
        return self._tasks == other._tasks and list(self._constraints) == list(other._constraints)

    def __repr__(self) -> str:
        return f"ClassificationSchema(tasks={list(self._tasks)}, constraints={len(self._constraints)})"


def _task_to_dict(spec: TaskSpec) -> dict:
    descs = {label.name: label.description for label in spec.labels if label.description is not None}
    labels: Any = {label.name: label.description for label in spec.labels} if descs else list(spec.label_names)
    out: dict = {"labels": labels}
    if spec.min_labels:
        out["min_labels"] = spec.min_labels
    if spec.max_labels is not None:
        out["max_labels"] = spec.max_labels
    if spec.ordered:
        out["ordered"] = spec.ordered
    if spec.threshold != 0.5:
        out["threshold"] = spec.threshold
    if spec.candidate_threshold is not None:
        out["candidate_threshold"] = spec.candidate_threshold
    if spec.activation != "auto":
        out["activation"] = spec.activation
    if spec.temperature != 1.0:
        out["temperature"] = spec.temperature
    if spec.default is not None:
        out["default"] = spec.default
    if spec.instruction is not None:
        out["instruction"] = spec.instruction
    if spec.examples:
        out["examples"] = [list(pair) for pair in spec.examples]
    return out


def _task_kwargs_from_dict(spec: Mapping) -> dict:
    kwargs = dict(spec)
    kwargs["labels"] = kwargs.pop("labels")
    examples = kwargs.pop("examples", None)
    if examples is not None:
        kwargs["examples"] = tuple(tuple(pair) for pair in examples)
    return kwargs


@runtime_checkable
class Assignment(Protocol):
    """What a constraint reads from a partial assignment."""

    def is_decided(self, task: str) -> bool: ...
    def selected(self, task: str): ...
    def domain(self, task: str): ...
    def holds(self, task: str, label: str): ...
    def levels(self, task: str): ...
    def index(self, task: str, label: str) -> int: ...
    def default(self, task: str): ...


def _task_spec(schema, task):
    getter = getattr(schema, "task_spec", None)
    if getter is not None:
        return getter(task)
    return schema.task(task)


class DictAssignment:
    """Selected labels plus domains, with no search state."""

    def __init__(self, schema, selected, decided, domains=None):
        self._schema = schema
        self._selected = {key: frozenset(value) for key, value in (selected or {}).items()}
        self._decided = frozenset(decided or ())
        self._domains = {key: frozenset(value) for key, value in (domains or {}).items()}

    def is_decided(self, task):
        return task in self._decided

    def selected(self, task):
        return self._selected.get(task, frozenset())

    def domain(self, task):
        if task in self._decided:
            return self._selected.get(task, frozenset())
        if task in self._domains:
            return self._domains[task]
        return frozenset(_task_spec(self._schema, task).label_names)

    def holds(self, task, label):
        if label in self.selected(task):
            return True
        if label in self.domain(task):
            return False if self.is_decided(task) else None
        return False

    def levels(self, task):
        spec = _task_spec(self._schema, task)
        index = {name: i for i, name in enumerate(spec.label_names)}
        return frozenset(index[label] for label in self.domain(task) if label in index)

    def index(self, task, label):
        return _task_spec(self._schema, task).label_names.index(label)

    def default(self, task):
        return _task_spec(self._schema, task).default


def _selectable(assignment, task: str):
    return assignment.selected(task) | assignment.domain(task)


def k_not(value):
    return None if value is None else (not value)


def k_and(values):
    seen_none = False
    for value in values:
        if value is False:
            return False
        if value is None:
            seen_none = True
    return None if seen_none else True


def k_or(values):
    seen_none = False
    for value in values:
        if value is True:
            return True
        if value is None:
            seen_none = True
    return None if seen_none else False


def k_implies(left, right):
    return k_or([k_not(left), right])


def k_iff(left, right):
    if left is None or right is None:
        return None
    return left == right


class Constraint(ABC):
    """Hard constraint with Kleene evaluation."""

    @abstractmethod
    def evaluate(self, assignment):
        """Return True, False, or None when still undetermined."""

    @abstractmethod
    def references(self):
        """Task names this node reads."""

    def label_references(self):
        return frozenset()

    def set_references(self):
        return frozenset()

    def count_references(self):
        return frozenset()

    def coupling_references(self):
        return self.set_references() | self.count_references()

    def check_schema(self, schema) -> None:
        return None

    @final
    def still_satisfiable(self, assignment) -> bool:
        return self.evaluate(assignment) is not False

    @final
    def satisfied(self, assignment) -> bool:
        return self.evaluate(assignment) is True

    @abstractmethod
    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class LabelRef(Constraint):
    """A single (task, label) literal."""

    task: str
    label: str

    def evaluate(self, assignment):
        return assignment.holds(self.task, self.label)

    def references(self):
        return frozenset({self.task})

    def label_references(self):
        return frozenset({(self.task, self.label)})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        if self.label not in spec.label_names:
            raise SchemaError(f"{self.label!r} is not a label of {self.task!r}")

    def to_dict(self):
        return {"type": "LabelRef", "task": self.task, "label": self.label}


@dataclass(frozen=True)
class AnySelected(Constraint):
    """At least one label of ``task`` is selected."""

    task: str

    def evaluate(self, assignment):
        return k_or([assignment.holds(self.task, label) for label in _selectable(assignment, self.task)])

    def references(self):
        return frozenset({self.task})

    def set_references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        _task_spec(schema, self.task)

    def to_dict(self):
        return {"type": "AnySelected", "task": self.task}


@dataclass(frozen=True)
class AnyOtherSelected(Constraint):
    """At least one non-default label of ``task`` is selected."""

    task: str

    def evaluate(self, assignment):
        default = assignment.default(self.task)
        return k_or(
            [assignment.holds(self.task, label) for label in _selectable(assignment, self.task) if label != default]
        )

    def references(self):
        return frozenset({self.task})

    def set_references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        if spec.default is None:
            raise SchemaError(f"any_other_selected requires task {self.task!r} to declare a default")

    def to_dict(self):
        return {"type": "AnyOtherSelected", "task": self.task}


@dataclass(frozen=True)
class IsDefault(Constraint):
    """The declared default label is selected."""

    task: str

    def evaluate(self, assignment):
        default = assignment.default(self.task)
        if default is None:
            return False
        return assignment.holds(self.task, default)

    def references(self):
        return frozenset({self.task})

    def set_references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        if spec.default is None:
            raise SchemaError(f"is_default requires task {self.task!r} to declare a default")

    def to_dict(self):
        return {"type": "IsDefault", "task": self.task}


@dataclass(frozen=True)
class Cardinality(Constraint):
    """Selection count lies in ``[minimum, maximum]``."""

    task: str
    minimum: int = 0
    maximum: int | None = None

    def evaluate(self, assignment):
        selected = assignment.selected(self.task)
        domain = assignment.domain(self.task)
        lo = len(selected)
        hi = len(selected | domain)
        maximum = self.maximum if self.maximum is not None else hi
        if lo > maximum:
            return False
        if hi < self.minimum:
            return False
        if lo >= self.minimum and hi <= maximum:
            return True
        return None

    def references(self):
        return frozenset({self.task})

    def count_references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        count = len(spec.label_names)
        if not 0 <= self.minimum <= count:
            raise SchemaError(f"cardinality minimum {self.minimum} outside [0, {count}] for {self.task!r}")
        if self.maximum is not None:
            if not 0 <= self.maximum <= count:
                raise SchemaError(f"cardinality maximum {self.maximum} outside [0, {count}] for {self.task!r}")
            if self.minimum > self.maximum:
                raise SchemaError(
                    f"cardinality minimum {self.minimum} exceeds maximum {self.maximum} for {self.task!r}"
                )

    def to_dict(self):
        return {
            "type": "Cardinality",
            "task": self.task,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


class _OrdinalNode(Constraint):
    task: str
    level: str

    def references(self):
        return frozenset({self.task})

    def check_schema(self, schema):
        spec = _task_spec(schema, self.task)
        if not spec.ordered:
            raise SchemaError(f"ordinal op requires an ordered task; {self.task!r} is unordered")
        if self.level not in spec.label_names:
            raise SchemaError(f"{self.level!r} is not a label of {self.task!r}")


@dataclass(frozen=True)
class MinLevel(_OrdinalNode):
    """Selected level is at least ``level``."""

    task: str
    level: str

    def evaluate(self, assignment):
        floor = assignment.index(self.task, self.level)
        levels = assignment.levels(self.task)
        if not levels:
            return False
        if min(levels) >= floor:
            return True
        if max(levels) < floor:
            return False
        return None

    def to_dict(self):
        return {"type": "MinLevel", "task": self.task, "level": self.level}


@dataclass(frozen=True)
class MaxLevel(_OrdinalNode):
    """Selected level is at most ``level``."""

    task: str
    level: str

    def evaluate(self, assignment):
        ceil = assignment.index(self.task, self.level)
        levels = assignment.levels(self.task)
        if not levels:
            return False
        if max(levels) <= ceil:
            return True
        if min(levels) > ceil:
            return False
        return None

    def to_dict(self):
        return {"type": "MaxLevel", "task": self.task, "level": self.level}


@dataclass(frozen=True)
class AtLevel(_OrdinalNode):
    """Selected level is exactly ``level``."""

    task: str
    level: str

    def evaluate(self, assignment):
        target = assignment.index(self.task, self.level)
        levels = assignment.levels(self.task)
        if not levels:
            return False
        if levels == {target}:
            return True
        if target not in levels:
            return False
        return None

    def to_dict(self):
        return {"type": "AtLevel", "task": self.task, "level": self.level}


@dataclass(frozen=True)
class Not(Constraint):
    """Negation of ``child``."""

    child: Constraint

    def evaluate(self, assignment):
        return k_not(self.child.evaluate(assignment))

    def references(self):
        return self.child.references()

    def label_references(self):
        return self.child.label_references()

    def set_references(self):
        return self.child.set_references()

    def count_references(self):
        return self.child.count_references()

    def check_schema(self, schema):
        self.child.check_schema(schema)

    def to_dict(self):
        return {"type": "Not", "child": self.child.to_dict()}


class _NaryNode(Constraint):
    children: tuple

    def references(self):
        out = frozenset()
        for child in self.children:
            out |= child.references()
        return out

    def label_references(self):
        out = frozenset()
        for child in self.children:
            out |= child.label_references()
        return out

    def set_references(self):
        out = frozenset()
        for child in self.children:
            out |= child.set_references()
        return out

    def count_references(self):
        out = frozenset()
        for child in self.children:
            out |= child.count_references()
        return out

    def check_schema(self, schema):
        for child in self.children:
            child.check_schema(schema)


@dataclass(frozen=True)
class And(_NaryNode):
    """Conjunction."""

    children: tuple

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))

    def evaluate(self, assignment):
        return k_and([child.evaluate(assignment) for child in self.children])

    def to_dict(self):
        return {"type": "And", "children": [child.to_dict() for child in self.children]}


@dataclass(frozen=True)
class Or(_NaryNode):
    """Disjunction."""

    children: tuple

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))

    def evaluate(self, assignment):
        return k_or([child.evaluate(assignment) for child in self.children])

    def to_dict(self):
        return {"type": "Or", "children": [child.to_dict() for child in self.children]}


@dataclass(frozen=True)
class ExactlyOneOf(_NaryNode):
    """Exactly one child holds."""

    children: tuple

    def __post_init__(self):
        object.__setattr__(self, "children", tuple(self.children))

    def evaluate(self, assignment):
        vals = [child.evaluate(assignment) for child in self.children]
        trues = sum(1 for value in vals if value is True)
        nones = sum(1 for value in vals if value is None)
        if trues >= 2:
            return False
        if trues == 1:
            return True if nones == 0 else None
        return False if nones == 0 else None

    def to_dict(self):
        return {"type": "ExactlyOneOf", "children": [child.to_dict() for child in self.children]}


class _BinaryNode(Constraint):
    left: Constraint
    right: Constraint

    def references(self):
        return self.left.references() | self.right.references()

    def label_references(self):
        return self.left.label_references() | self.right.label_references()

    def set_references(self):
        return self.left.set_references() | self.right.set_references()

    def count_references(self):
        return self.left.count_references() | self.right.count_references()

    def check_schema(self, schema):
        self.left.check_schema(schema)
        self.right.check_schema(schema)


@dataclass(frozen=True)
class Implies(_BinaryNode):
    """``cond`` implies ``then``."""

    cond: Constraint
    then: Constraint

    @property
    def left(self):
        return self.cond

    @property
    def right(self):
        return self.then

    def evaluate(self, assignment):
        return k_implies(self.cond.evaluate(assignment), self.then.evaluate(assignment))

    def to_dict(self):
        return {"type": "Implies", "cond": self.cond.to_dict(), "then": self.then.to_dict()}


@dataclass(frozen=True)
class Iff(_BinaryNode):
    """Both sides are equivalent."""

    left: Constraint
    right: Constraint

    def evaluate(self, assignment):
        return k_iff(self.left.evaluate(assignment), self.right.evaluate(assignment))

    def to_dict(self):
        return {"type": "Iff", "left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass(frozen=True)
class Excludes(_BinaryNode):
    """The two sides cannot both hold."""

    left: Constraint
    right: Constraint

    def evaluate(self, assignment):
        return k_not(k_and([self.left.evaluate(assignment), self.right.evaluate(assignment)]))

    def to_dict(self):
        return {"type": "Excludes", "left": self.left.to_dict(), "right": self.right.to_dict()}


_TYPES = {
    cls.__name__: cls
    for cls in (
        LabelRef,
        Not,
        And,
        Or,
        Implies,
        Iff,
        Excludes,
        ExactlyOneOf,
        Cardinality,
        AtLevel,
        MinLevel,
        MaxLevel,
        IsDefault,
        AnySelected,
        AnyOtherSelected,
    )
}


def constraint_from_dict(data: Mapping) -> Constraint:
    """Rebuild a constraint node from its ``to_dict`` payload."""
    values = dict(data)
    kind = values.pop("type", None)
    try:
        cls = _TYPES[kind]
    except KeyError as exc:
        raise SchemaError(f"unknown constraint type {kind!r}") from exc
    return cls(**{key: _rebuild(value) for key, value in values.items()})


def _rebuild(value):
    if isinstance(value, Mapping) and "type" in value:
        return constraint_from_dict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_rebuild(item) for item in value)
    return value


def _expr(value) -> Constraint:
    if isinstance(value, Constraint):
        return value
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, str) for item in value):
        return LabelRef(value[0], value[1])
    raise SchemaError(
        f"cannot interpret {value!r} as a constraint expression; use a ('task', 'label') tuple or a DSL constructor"
    )


def _int(value, what) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError(f"{what} must be a non-negative int")
    return value


def _task_name(value, what) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{what} must be a task name string")
    return value


def all_of(*exprs) -> Constraint:
    return And(tuple(_expr(item) for item in exprs))


def any_of(*exprs) -> Constraint:
    return Or(tuple(_expr(item) for item in exprs))


def not_(expr) -> Constraint:
    return Not(_expr(expr))


def implies(cond, then) -> Constraint:
    return Implies(_expr(cond), _expr(then))


def iff(left, right) -> Constraint:
    return Iff(_expr(left), _expr(right))


def excludes(left, right) -> Constraint:
    return Excludes(_expr(left), _expr(right))


def exactly_one_of(*exprs) -> Constraint:
    return ExactlyOneOf(tuple(_expr(item) for item in exprs))


def at_least(task, count) -> Constraint:
    return Cardinality(_task_name(task, "at_least task"), _int(count, "at_least count"), None)


def at_most(task, count) -> Constraint:
    return Cardinality(_task_name(task, "at_most task"), 0, _int(count, "at_most count"))


def exactly(task, count) -> Constraint:
    number = _int(count, "exactly count")
    return Cardinality(_task_name(task, "exactly task"), number, number)


def at_level(task, level) -> Constraint:
    return AtLevel(_task_name(task, "at_level task"), level)


def min_level(task, level) -> Constraint:
    return MinLevel(_task_name(task, "min_level task"), level)


def max_level(task, level) -> Constraint:
    return MaxLevel(_task_name(task, "max_level task"), level)


def between_level(task, lo, hi) -> Constraint:
    name = _task_name(task, "between_level task")
    return And((MinLevel(name, lo), MaxLevel(name, hi)))


def any_selected(task) -> Constraint:
    return AnySelected(_task_name(task, "any_selected task"))


def any_other_selected(task) -> Constraint:
    return AnyOtherSelected(_task_name(task, "any_other_selected task"))


def is_default(task) -> Constraint:
    return IsDefault(_task_name(task, "is_default task"))


def label(task, name) -> Constraint:
    return LabelRef(_task_name(task, "label task"), name)


@dataclass(frozen=True)
class CompiledClassificationSchema:
    """Model schema dict plus the constraint set the decoder searches."""

    model_schema: dict
    task_specs: tuple
    constraints: tuple
    task_order: tuple
    fingerprint: str

    def task(self, name) -> TaskSpec:
        for spec in self.task_specs:
            if spec.name == name:
                return spec
        raise SchemaError(f"unknown task {name!r}")

    def task_spec(self, name) -> TaskSpec:
        return self.task(name)

    def build(self) -> dict:
        return self.model_schema


def _classification_entry(spec: TaskSpec) -> dict:
    entry = {
        "task": spec.name,
        "labels": list(spec.label_names),
        "multi_label": not spec.is_exclusive,
        "cls_threshold": spec.threshold,
        "class_act": spec.activation,
    }
    if spec.instruction:
        entry["prompt"] = spec.instruction
    descs = {label.name: label.description for label in spec.labels if label.description}
    if descs:
        entry["label_descriptions"] = descs
    if spec.examples:
        entry["examples"] = [list(pair) for pair in spec.examples]
    return entry


def _has_reserved(value: str) -> bool:
    return any(token in value for token in _RESERVED)


def _assert_model_schema(model: dict) -> None:
    for key in _MODEL_KEYS:
        if key not in model:
            raise SchemaError(f"compiled model schema is missing key {key!r}")
    if not isinstance(model["classifications"], list):
        raise SchemaError("compiled 'classifications' must be a list")
    for entry in model["classifications"]:
        task = entry.get("task")
        if not isinstance(task, str) or not task:
            raise SchemaError("classification entry has an invalid 'task'")
        for required in ("labels", "multi_label", "cls_threshold", "class_act"):
            if required not in entry:
                raise SchemaError(f"classification task {task!r} is missing {required!r}")
        if not isinstance(entry["labels"], list) or not entry["labels"]:
            raise SchemaError(f"classification task {task!r} has invalid 'labels'")
        if not isinstance(entry["multi_label"], bool):
            raise SchemaError(f"classification task {task!r} has non-bool 'multi_label'")
        strings = [task] + list(entry["labels"])
        if "prompt" in entry:
            strings.append(entry["prompt"])
        strings += list(entry.get("label_descriptions", {}).keys())
        strings += list(entry.get("label_descriptions", {}).values())
        for pair in entry.get("examples", []):
            strings += [str(item) for item in pair]
        for value in strings:
            if isinstance(value, str) and _has_reserved(value):
                raise SchemaError(
                    f"classification task {task!r} emitted a reserved marker token in "
                    f"{value!r}; this would corrupt logit-to-label alignment"
                )


def _check_prefix_collisions(task_order) -> None:
    for left in task_order:
        for right in task_order:
            if left == right:
                continue
            if right.startswith(left) and len(right) > len(left) and right[len(left)] in (":", " "):
                raise SchemaError(
                    f"task name {left!r} is a boundary-prefix of {right!r}; the prompt "
                    f"resolver could confuse them. Rename one."
                )


def _static_feasibility(schema, task_specs) -> None:
    constraints = schema.constraints
    if not constraints:
        return
    undetermined = DictAssignment(schema, selected={}, decided=())
    for constraint in constraints:
        if constraint.evaluate(undetermined) is False:
            raise SchemaError("constraint set is unsatisfiable on the declared label sets")
    for spec in task_specs:
        if not spec.is_exclusive:
            continue
        touching = [constraint for constraint in constraints if spec.name in constraint.references()]
        if not touching:
            continue
        reachable = False
        for label_name in spec.label_names:
            assignment = DictAssignment(schema, {spec.name: {label_name}}, decided=[spec.name])
            if all(constraint.evaluate(assignment) is not False for constraint in touching):
                reachable = True
                break
        if not reachable:
            raise SchemaError(
                f"no label of exclusive task {spec.name!r} satisfies its constraints; "
                f"the task is pinned to the empty set"
            )


def _lower_defaults(schema, task_specs) -> tuple:
    constraints = list(schema.constraints)
    for spec in task_specs:
        if spec.default is not None:
            constraints.append(Iff(IsDefault(spec.name), Not(AnyOtherSelected(spec.name))))
    return tuple(constraints)


def _fingerprint(schema: ClassificationSchema) -> str:
    payload = json.dumps(schema.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_schema(schema) -> CompiledClassificationSchema:
    """Compile tasks and constraints into a decoder problem."""
    if isinstance(schema, CompiledClassificationSchema):
        return schema
    if not isinstance(schema, ClassificationSchema):
        raise SchemaError(f"compile_schema expects a ClassificationSchema, got {type(schema).__name__}")
    task_specs = schema.task_specs
    if not task_specs:
        raise SchemaError("cannot compile a schema with no tasks")
    for constraint in schema.constraints:
        constraint.check_schema(schema)
    _check_prefix_collisions(schema.task_order)
    _static_feasibility(schema, task_specs)
    model = {
        "json_structures": [],
        "classifications": [_classification_entry(spec) for spec in task_specs],
        "entities": {},
        "relations": [],
        "json_descriptions": {},
        "entity_descriptions": {},
    }
    _assert_model_schema(model)
    return CompiledClassificationSchema(
        model_schema=model,
        task_specs=tuple(task_specs),
        constraints=_lower_defaults(schema, task_specs),
        task_order=tuple(schema.task_order),
        fingerprint=_fingerprint(schema),
    )


def task_utilities(spec, logits) -> dict:
    """Centered log-odds. Positive means the label clears the threshold."""
    return {label: center_logit(value / spec.temperature, spec.threshold) for label, value in logits.items()}


def task_probabilities(spec, logits) -> dict:
    """Sigmoid retention probabilities. Presentation may still be softmax."""
    return {label: sigmoid(value / spec.temperature) for label, value in logits.items()}


def retain(spec, logits, *, candidate_threshold, cap, rescued) -> frozenset:
    """Keep labels above the retention floor, plus every rescued label."""
    rescued = frozenset(rescued)
    floor = spec.candidate_threshold if spec.candidate_threshold is not None else candidate_threshold
    finite = {label: value for label, value in logits.items() if math.isfinite(value)}
    probs = task_probabilities(spec, finite)
    utils = task_utilities(spec, finite)
    keep = {label for label, prob in probs.items() if prob >= floor} | (rescued & set(finite))
    if len(keep) > cap:
        ranked = sorted(keep, key=lambda label: (label not in rescued, -utils[label], label))
        keep = set(ranked[: max(cap, len(rescued & set(finite)))])
    return frozenset(keep)


@dataclass(frozen=True)
class LocalAssignment:
    """One cardinality-feasible label set for a single task."""

    task: str
    labels: frozenset
    utility: float

    def __post_init__(self):
        object.__setattr__(self, "labels", frozenset(self.labels))


def _utility_of(labels, utilities) -> float:
    return float(sum(utilities[label] for label in labels))


def _all_subsets(items):
    items = sorted(items)
    for size in range(len(items) + 1):
        for combo in combinations(items, size):
            yield frozenset(combo)


def _sorted_locals(locals_):
    return sorted(locals_, key=lambda local: (-local.utility, tuple(sorted(local.labels))))


def enumerate_locals(spec, retained, utilities, *, bound_labels, set_coupled=False, count_coupled=False) -> list:
    """Enumerate cardinality-feasible locals, highest utility first."""
    retained = frozenset(retained)
    min_labels = spec.min_labels
    max_labels = min(spec.effective_max_labels(), len(retained))
    if spec.is_exclusive:
        locals_ = [LocalAssignment(spec.name, frozenset({label}), utilities[label]) for label in retained]
        return _sorted_locals(locals_)
    if set_coupled:
        locals_ = []
        for base in _all_subsets(retained):
            if min_labels <= len(base) <= max_labels:
                locals_.append(LocalAssignment(spec.name, base, _utility_of(base, utilities)))
        return _sorted_locals(locals_)
    bound = retained & frozenset(bound_labels)
    free = retained - bound
    free_ranked = sorted(free, key=lambda label: (-utilities[label], label))
    locals_ = []
    for base in _all_subsets(bound):
        if len(base) > max_labels:
            continue
        if count_coupled:
            lo = max(min_labels, len(base))
            hi = min(max_labels, len(base) + len(free))
            for count in range(lo, hi + 1):
                need = count - len(base)
                selected = frozenset(base) | frozenset(free_ranked[:need])
                if len(selected) == count:
                    locals_.append(
                        LocalAssignment(
                            spec.name,
                            selected,
                            _utility_of(selected, utilities),
                        )
                    )
            continue
        selected = set(base)
        for free_label in free_ranked:
            if len(selected) >= max_labels:
                break
            if utilities[free_label] > 0:
                selected.add(free_label)
        if len(selected) < min_labels:
            for free_label in free_ranked:
                if len(selected) >= min_labels:
                    break
                if free_label not in selected:
                    selected.add(free_label)
        if not (min_labels <= len(selected) <= max_labels):
            continue
        locals_.append(LocalAssignment(spec.name, frozenset(selected), _utility_of(selected, utilities)))
    return _sorted_locals(locals_)


@dataclass(frozen=True)
class ClassificationScores:
    """Raw per-label logits plus the schema fingerprint."""

    text: str
    tasks: Mapping
    fingerprint: str
    specs: Mapping

    def __post_init__(self):
        frozen = MappingProxyType({task: MappingProxyType(dict(values)) for task, values in self.tasks.items()})
        object.__setattr__(self, "tasks", frozen)

    def _spec(self, task):
        try:
            return self.specs[task]
        except KeyError:
            raise SchemaError(f"unknown task {task!r}") from None

    def logit(self, task: str, label: str) -> float:
        return self.tasks[task][label]

    def probability(self, task: str, label: str) -> float:
        """Temperature then sigmoid, or softmax for an exclusive task."""
        spec = self._spec(task)
        temp = spec.temperature
        activation = spec.activation
        if activation == "auto":
            activation = "sigmoid" if not spec.is_exclusive else "softmax"
        if activation == "softmax":
            names = list(self.tasks[task])
            scaled = [self.tasks[task][name] / temp for name in names]
            probs = _softmax(scaled)
            return probs[names.index(label)]
        return sigmoid(self.tasks[task][label] / temp)

    def utility(self, task: str, label: str) -> float:
        """Temperature then threshold centering. This is the search objective."""
        spec = self._spec(task)
        return center_logit(self.tasks[task][label] / spec.temperature, spec.threshold)

    def top(self, task: str, k: int = 5) -> tuple:
        items = [(label, self.probability(task, label)) for label in self.tasks[task]]
        items.sort(key=lambda pair: (-pair[1], pair[0]))
        return tuple(items[:k])


@dataclass
class Solution:
    """Chosen locals, objective, and any violated constraints."""

    assignments: dict
    score: float
    violations: tuple = ()
    exact: bool = True
    decoder: str = "independent"

    @property
    def feasible(self) -> bool:
        return not self.violations


class SearchAssignment:
    """Kleene assignment over decided locals and undecided domains."""

    def __init__(self, problem, chosen, decided):
        self._problem = problem
        self._chosen = chosen
        self._decided = frozenset(decided)

    def is_decided(self, task):
        return task in self._decided

    def selected(self, task):
        if task in self._decided:
            return self._chosen[task].labels
        return self._problem.always[task]

    def domain(self, task):
        if task in self._decided:
            return self._chosen[task].labels
        return self._problem.possible[task]

    def holds(self, task, label):
        if label in self.selected(task):
            return True
        if label in self.domain(task):
            return False if self.is_decided(task) else None
        return False

    def levels(self, task):
        index = self._problem.index_map[task]
        return frozenset(index[label] for label in self.domain(task) if label in index)

    def index(self, task, label):
        return self._problem.index_map[task][label]

    def default(self, task):
        return self._problem.schema.task(task).default


class DecodeProblem:
    """Per-task locals and the constraints that couple them."""

    def __init__(self, schema, task_order, locals_map, constraints):
        self.schema = schema
        self.task_order = tuple(task_order)
        self.locals = locals_map
        self.constraints = tuple(constraints)
        self.possible = {}
        self.always = {}
        self.index_map = {}
        for task in self.task_order:
            local_list = self.locals[task]
            union = frozenset().union(*[local.labels for local in local_list]) if local_list else frozenset()
            inter = frozenset.intersection(*[local.labels for local in local_list]) if local_list else frozenset()
            self.possible[task] = union
            self.always[task] = inter
            names = schema.task(task).label_names
            self.index_map[task] = {name: i for i, name in enumerate(names)}
        self._touch_cache = {}

    def constraints_touching(self, task):
        cached = self._touch_cache.get(task)
        if cached is None:
            cached = tuple(constraint for constraint in self.constraints if task in constraint.references())
            self._touch_cache[task] = cached
        return cached

    def assignment(self, chosen, decided):
        return SearchAssignment(self, chosen, decided)

    def violations_of(self, chosen):
        assignment = SearchAssignment(self, chosen, self.task_order)
        return tuple(constraint for constraint in self.constraints if constraint.evaluate(assignment) is False)


def _fallback_locals(spec, retained, utils):
    ranked = sorted(retained, key=lambda label: (-utils[label], label))
    max_labels = min(spec.effective_max_labels(), len(ranked))
    count = max(spec.min_labels, 1)
    count = min(count, max_labels) if max_labels else 0
    chosen = frozenset(ranked[:count])
    return [LocalAssignment(spec.name, chosen, float(sum(utils[label] for label in chosen)))]


def build_problem(compiled, scores, config, *, active=None, full_retention_tasks=()):
    """Retain candidates, enumerate locals, and drop constraints outside ``active``."""
    order = compiled.task_order
    if active is not None:
        active_set = set(active)
        order = tuple(task for task in order if task in active_set)
    active_set = set(order)
    full_retention = set(full_retention_tasks)
    constraints = tuple(constraint for constraint in compiled.constraints if constraint.references() <= active_set)
    label_refs: dict = {}
    set_tasks: set = set()
    count_tasks: set = set()
    for constraint in constraints:
        for task, label_name in constraint.label_references():
            label_refs.setdefault(task, set()).add(label_name)
        set_tasks |= set(constraint.set_references())
        count_tasks |= set(constraint.count_references())
    locals_map: dict = {}
    for task in order:
        spec = compiled.task(task)
        logits = scores.tasks[task]
        rescued = label_refs.get(task, set())
        if task in full_retention:
            retained = frozenset(spec.label_names)
        else:
            retained = retain(
                spec,
                logits,
                candidate_threshold=config.candidate_threshold,
                cap=config.max_candidates_per_task,
                rescued=rescued,
            )
        if len(retained) < spec.min_labels:
            retained = frozenset(spec.label_names)
        utils = task_utilities(spec, {label: logits[label] for label in retained})
        set_coupled = task in set_tasks
        count_coupled = (task in count_tasks) and not set_coupled
        locals_ = enumerate_locals(
            spec,
            retained,
            utils,
            bound_labels=label_refs.get(task, set()),
            set_coupled=set_coupled,
            count_coupled=count_coupled,
        )
        if not locals_:
            locals_ = _fallback_locals(spec, retained, utils)
        locals_map[task] = locals_
    return DecodeProblem(compiled, order, locals_map, constraints)


class IndependentDecoder:
    """Pick each task's best local. Single-task constraints still apply."""

    name = "independent"

    def decode(self, problem) -> Solution:
        chosen: dict = {}
        for task in problem.task_order:
            touching = problem.constraints_touching(task)
            picked = None
            for local in problem.locals[task]:
                assignment = problem.assignment({task: local}, [task])
                if all(constraint.evaluate(assignment) is not False for constraint in touching):
                    picked = local
                    break
            if picked is None:
                picked = problem.locals[task][0]
            chosen[task] = picked
        score = sum(local.utility for local in chosen.values())
        return Solution(
            assignments=dict(chosen),
            score=score,
            violations=problem.violations_of(chosen),
            exact=True,
            decoder=self.name,
        )


class _BudgetExceeded(Exception):
    pass


def _order(problem):
    return sorted(
        problem.task_order,
        key=lambda task: (-len(problem.constraints_touching(task)), len(problem.locals[task]), task),
    )


def _suffix_max(order, problem):
    suffix = [0.0] * (len(order) + 1)
    for index in range(len(order) - 1, -1, -1):
        locals_ = problem.locals[order[index]]
        best = max((local.utility for local in locals_), default=0.0)
        suffix[index] = best + suffix[index + 1]
    return suffix


class ExactDecoder:
    """DFS with branch and bound over utility-sorted locals."""

    name = "exact"

    def decode(self, problem, *, budget: int = 200_000) -> Solution | None:
        order = _order(problem)
        suffix = _suffix_max(order, problem)
        best = {"assign": None, "score": -math.inf}
        nodes = {"n": 0}

        def dfs(index, chosen, score):
            nodes["n"] += 1
            if nodes["n"] > budget:
                raise _BudgetExceeded
            if score + suffix[index] <= best["score"]:
                return
            if index == len(order):
                best["assign"] = dict(chosen)
                best["score"] = score
                return
            task = order[index]
            touching = problem.constraints_touching(task)
            for local in problem.locals[task]:
                if score + local.utility + suffix[index + 1] <= best["score"]:
                    break
                chosen[task] = local
                assignment = problem.assignment(chosen, order[: index + 1])
                if all(constraint.evaluate(assignment) is not False for constraint in touching):
                    dfs(index + 1, chosen, score + local.utility)
                del chosen[task]

        dfs(0, {}, 0.0)
        if best["assign"] is None:
            return None
        return Solution(
            assignments=best["assign"],
            score=best["score"],
            violations=(),
            exact=True,
            decoder=self.name,
        )


class MinViolationsDecoder:
    """Same DFS, lexicographic on fewest violations then utility."""

    name = "min_violations"

    def decode(self, problem, *, budget: int = 200_000) -> Solution:
        order = _order(problem)
        best = {"assign": None, "weight": math.inf, "score": -math.inf}
        nodes = {"n": 0}

        def dfs(index, chosen, score):
            nodes["n"] += 1
            if nodes["n"] > budget:
                raise _BudgetExceeded
            if index == len(order):
                violated = problem.violations_of(chosen)
                weight = float(len(violated))
                key = (weight, -score)
                current = (best["weight"], -best["score"])
                if key < current:
                    best["assign"] = dict(chosen)
                    best["weight"] = weight
                    best["score"] = score
                return
            task = order[index]
            for local in problem.locals[task]:
                chosen[task] = local
                dfs(index + 1, chosen, score + local.utility)
                del chosen[task]

        exhausted = False
        try:
            dfs(0, {}, 0.0)
        except _BudgetExceeded:
            exhausted = True
        assign = best["assign"] or {task: problem.locals[task][0] for task in problem.task_order}
        return Solution(
            assignments=dict(assign),
            score=sum(local.utility for local in assign.values()),
            violations=problem.violations_of(assign),
            exact=not exhausted,
            decoder=self.name,
        )


def _signature(chosen, order):
    return tuple((task, tuple(sorted(chosen[task].labels))) for task in order if task in chosen)


class BeamDecoder:
    """Bounded search. An empty beam is infeasible, not a violating answer."""

    name = "beam"

    def decode(self, problem, *, beam_size: int = 16) -> Solution | None:
        order = sorted(
            problem.task_order,
            key=lambda task: (-len(problem.constraints_touching(task)), len(problem.locals[task]), task),
        )
        beams = [(0.0, {})]
        for index, task in enumerate(order):
            touching = problem.constraints_touching(task)
            expanded = []
            seen = set()
            for score, chosen in beams:
                for local in problem.locals[task]:
                    nxt = dict(chosen)
                    nxt[task] = local
                    assignment = problem.assignment(nxt, order[: index + 1])
                    if all(constraint.evaluate(assignment) is not False for constraint in touching):
                        expanded.append((score + local.utility, nxt))
            expanded.sort(key=lambda item: (-item[0], _signature(item[1], order)))
            beams = []
            for score, chosen in expanded:
                sig = _signature(chosen, order)
                if sig in seen:
                    continue
                seen.add(sig)
                beams.append((score, chosen))
                if len(beams) >= beam_size:
                    break
            if not beams:
                return None
        score, chosen = beams[0]
        return Solution(
            assignments=dict(chosen),
            score=score,
            violations=problem.violations_of(chosen),
            exact=False,
            decoder=self.name,
        )


def select_decoder(problem, requested: str) -> str:
    """Use exact search when any constraint crosses tasks."""
    if requested != "auto":
        return requested
    cross_task = any(len(constraint.references()) > 1 for constraint in problem.constraints)
    return "exact" if cross_task else "independent"


def _primary(problem, config):
    decoder = select_decoder(problem, config.decoder)
    if decoder == "independent":
        solution = IndependentDecoder().decode(problem)
        return solution if solution.feasible else None
    if decoder == "beam":
        solution = BeamDecoder().decode(problem, beam_size=config.beam_size)
        return solution if (solution is not None and solution.feasible) else None
    try:
        solution = ExactDecoder().decode(problem, budget=config.exact_node_budget)
    except _BudgetExceeded:
        solution = BeamDecoder().decode(problem, beam_size=config.beam_size)
    return solution if (solution is not None and solution.feasible) else None


def decode(problem, config, *, widen=None) -> Solution:
    """Run the decoder, then relax, minimize violations, or raise."""
    solution = _primary(problem, config)
    if solution is not None:
        return solution
    mode = config.on_infeasible
    working = problem
    if mode == "relax" and widen is not None:
        relaxed = widen()
        recovered = _primary(relaxed, config)
        if recovered is not None:
            return recovered
        working = relaxed
    if mode in ("relax", "min_violations"):
        return MinViolationsDecoder().decode(working, budget=config.exact_node_budget)
    diagnosis = MinViolationsDecoder().decode(working, budget=config.exact_node_budget)
    raise InfeasibleError(
        "no assignment satisfies the classification constraints",
        violations=diagnosis.violations,
    )


@dataclass(frozen=True)
class Violation:
    """One constraint the returned assignment breaks."""

    constraint: object
    tasks: tuple
    weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))

    def __str__(self) -> str:
        tasks = ", ".join(self.tasks)
        return f"violated {self.constraint} (tasks: {tasks}; weight {self.weight})"


@dataclass(frozen=True)
class TaskResult:
    """Selected labels and presentation probabilities for one task."""

    task: str
    labels: tuple
    probabilities: MappingProxyType
    utilities: MappingProxyType
    confidence: float | None
    exclusive: bool
    ordered: bool
    level: int | None

    def __post_init__(self):
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "probabilities", MappingProxyType(dict(self.probabilities)))
        object.__setattr__(self, "utilities", MappingProxyType(dict(self.utilities)))

    @property
    def label(self) -> str | None:
        if not self.exclusive:
            raise SchemaError(f"task {self.task!r} is not single-label; use .labels, not .label")
        return self.labels[0] if self.labels else None


@dataclass(frozen=True)
class ClassificationResult:
    """Frozen multi-task assignment."""

    text: str
    tasks: MappingProxyType
    feasible: bool
    violations: tuple
    objective: float
    decoder: str
    exact: bool
    include_confidence: bool = True

    def __post_init__(self):
        object.__setattr__(self, "tasks", MappingProxyType(dict(self.tasks)))
        object.__setattr__(self, "violations", tuple(self.violations))

    def __getitem__(self, task: str) -> TaskResult:
        try:
            return self.tasks[task]
        except KeyError:
            raise SchemaError(f"unknown task {task!r}") from None

    def value(self, task: str):
        result = self[task]
        return result.label if result.exclusive else result.labels

    def selected(self, task: str) -> tuple:
        return self[task].labels

    def confidence(self, task: str) -> float | None:
        return self[task].confidence

    def utility(self, task: str, label: str) -> float:
        return self[task].utilities[label]

    def probabilities(self, task: str):
        return self[task].probabilities

    def to_dict(self, *, include_confidence: bool | None = None) -> dict:
        include = self.include_confidence if include_confidence is None else include_confidence
        out: dict = {}
        for name, result in self.tasks.items():
            value = result.label if result.exclusive else list(result.labels)
            if include:
                out[name] = {
                    "value": value,
                    "confidence": result.confidence,
                    "probabilities": dict(result.probabilities),
                }
            else:
                out[name] = value
        if include or not self.feasible:
            out["_meta"] = {
                "feasible": self.feasible,
                "decoder": self.decoder,
                "exact": self.exact,
                "objective": self.objective,
                "violations": [str(item) for item in self.violations],
            }
        return out


class ResultBuilder:
    """Assemble a frozen result from a solution and its scores."""

    def build(self, compiled, scores, solution, *, active_order=None, include_confidence=True):
        order = active_order or [task for task in compiled.task_order if task in solution.assignments]
        task_results = {}
        for task in order:
            spec = compiled.task(task)
            selected = solution.assignments[task].labels
            labels = tuple(label for label in spec.label_names if label in selected)
            probs = {label: scores.probability(task, label) for label in spec.label_names}
            utils = {label: scores.utility(task, label) for label in spec.label_names}
            level = spec.label_names.index(labels[0]) if spec.ordered and labels else None
            confidence = self._confidence(spec, labels, probs) if include_confidence else None
            task_results[task] = TaskResult(
                task=task,
                labels=labels,
                probabilities=probs,
                utilities=utils,
                confidence=confidence,
                exclusive=spec.is_exclusive,
                ordered=spec.ordered,
                level=level,
            )
        violations = tuple(
            Violation(constraint, tuple(sorted(constraint.references()))) for constraint in solution.violations
        )
        return ClassificationResult(
            text=scores.text,
            tasks=task_results,
            feasible=solution.feasible,
            violations=violations,
            objective=float(solution.score),
            decoder=solution.decoder,
            exact=solution.exact,
            include_confidence=include_confidence,
        )

    @staticmethod
    def _confidence(spec, labels, probs) -> float | None:
        if not labels:
            return 1.0 if spec.default is not None else None
        if spec.is_exclusive:
            return probs[labels[0]]
        selected = set(labels)
        components = [probs[label] if label in selected else 1.0 - probs[label] for label in spec.label_names]
        return _geometric_mean(components)


@dataclass(frozen=True)
class ClassificationConfig:
    """Decoder and retention controls for one call."""

    decoder: str = "auto"
    exact_node_budget: int = 200_000
    beam_size: int = 16
    candidate_threshold: float = 0.5
    max_candidates_per_task: int = 64
    include_confidence: bool = True
    on_infeasible: str = "relax"

    def __post_init__(self):
        if self.decoder not in _DECODERS:
            raise ValueError(f"decoder must be one of {_DECODERS}")
        if self.on_infeasible not in _ON_INFEASIBLE:
            raise ValueError(f"on_infeasible must be one of {_ON_INFEASIBLE}")
        if self.exact_node_budget <= 0:
            raise ValueError("exact_node_budget must be positive")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if not 0 <= self.candidate_threshold <= 1:
            raise ValueError("candidate_threshold must be in [0, 1]")
        if self.max_candidates_per_task <= 0:
            raise ValueError("max_candidates_per_task must be positive")


def _aggregate(values, mode):
    if mode == "max":
        return max(values)
    if mode == "mean":
        return sum(values) / len(values)
    return values[0]


def aggregate_classification_logits(chunk_logits, schema, mode: str = "max") -> dict:
    """Aggregate per-chunk label logits so decoding runs once."""
    if mode not in _AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {_AGGREGATIONS}")
    if not chunk_logits:
        raise ValueError("cannot aggregate an empty list of chunk scores")
    compiled = compile_schema(_coerce_schema(schema))
    aligned = [_align_logits(item, compiled) for item in chunk_logits]
    tasks = {}
    for spec in compiled.task_specs:
        tasks[spec.name] = {
            label: _aggregate([row[spec.name][label] for row in aligned], mode) for label in spec.label_names
        }
    return tasks


def _schema_from_model_dict(data: Mapping) -> ClassificationSchema:
    schema = ClassificationSchema()
    for entry in data.get("classifications", ()):
        labels = entry["labels"]
        descriptions = entry.get("label_descriptions") or {}
        if descriptions:
            labels = {name: descriptions.get(name) for name in labels}
        multi = bool(entry.get("multi_label", False))
        kwargs = {
            "threshold": entry.get("cls_threshold", 0.5),
            "activation": entry.get("class_act", "auto"),
            "instruction": entry.get("prompt"),
            "examples": tuple(tuple(pair) for pair in entry.get("examples", ())),
        }
        if multi:
            schema.multi(entry["task"], labels, **kwargs)
        else:
            schema.single(entry["task"], labels, **kwargs)
    return schema


def _coerce_schema(schema):
    if isinstance(schema, CompiledClassificationSchema):
        return schema
    if isinstance(schema, ClassificationSchema):
        return schema
    if isinstance(schema, Mapping):
        if "tasks" in schema:
            return ClassificationSchema.from_dict(schema)
        if "classifications" in schema:
            return _schema_from_model_dict(schema)
    if hasattr(schema, "to_dict") and hasattr(schema, "task_order"):
        payload = schema.to_dict()
        if isinstance(payload, Mapping) and "tasks" in payload:
            return ClassificationSchema.from_dict(payload)
    raise SchemaError(f"expected a classification schema, got {type(schema).__name__}")


def _effective_temperature(spec: TaskSpec, temperature) -> float:
    if isinstance(temperature, Mapping):
        value = float(temperature.get(spec.name, spec.temperature))
    else:
        value = float(spec.temperature) * float(temperature)
    if not value > 0:
        raise ValueError("temperature must be positive")
    return value


def _vector(values) -> list:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "tolist") and not isinstance(values, (str, bytes, Mapping)):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise SchemaError("label logits must be a mapping or a sequence")
    return list(values)


def _label_logit_map(values, names) -> dict:
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        if missing:
            raise SchemaError(f"logits missing labels {missing}")
        unknown = [name for name in values if name not in names]
        if unknown:
            raise SchemaError(f"logits have unknown labels {unknown}")
        return {name: _number(values[name]) for name in names}
    row = _vector(values)
    finite_tail = row
    if len(row) > len(names) and all(not math.isfinite(_number(item)) for item in row[len(names) :]):
        finite_tail = row[: len(names)]
    if len(finite_tail) != len(names):
        raise SchemaError(f"expected {len(names)} label logits, got {len(row)}")
    return {name: _number(value) for name, value in zip(names, finite_tail)}


def _rows(logits, count):
    shape = getattr(logits, "shape", None)
    if shape is not None and len(tuple(shape)) == 2:
        if int(shape[0]) != count:
            raise SchemaError(f"expected {count} task rows, got {int(shape[0])}")
        return [logits[index] for index in range(count)]
    if isinstance(logits, (list, tuple)):
        if len(logits) != count:
            raise SchemaError(f"expected {count} task logit rows, got {len(logits)}")
        return list(logits)
    raise SchemaError("logits must be a task mapping or one row per task")


def _align_encoded(payload, compiled) -> dict:
    tokens = payload.get("schema_tokens_list", payload.get("schema_tokens"))
    raw = payload["logits"]
    if not tokens:
        raise SchemaError("schema_tokens is empty")
    if isinstance(tokens[0], str):
        groups = [tokens]
        rows = [raw]
    else:
        groups = list(tokens)
        rows = _rows(raw, len(groups))
    known = compiled.task_order
    found = {task: {} for task in known}
    for group, row in zip(groups, rows):
        names = label_names_from_tokens(group)
        task = recover_task_name(group, known)
        if task not in known:
            raise SchemaError(f"encoded tokens name unknown task {task!r}")
        spec = compiled.task(task)
        mapped = _label_logit_map(row, names)
        unknown = [name for name in mapped if name not in spec.label_names]
        if unknown:
            raise SchemaError(f"task {task!r} logits recovered unknown labels {unknown}")
        found[task].update(mapped)
    out = {}
    for spec in compiled.task_specs:
        if not found[spec.name]:
            raise SchemaError(f"logits missing task {spec.name!r}")
        out[spec.name] = {label: found[spec.name].get(label, float("-inf")) for label in spec.label_names}
    return out


def _align_logits(logits, compiled) -> dict:
    if isinstance(logits, Mapping) and ("schema_tokens" in logits or "schema_tokens_list" in logits):
        return _align_encoded(logits, compiled)
    payload = logits
    if (
        isinstance(payload, Mapping)
        and "tasks" in payload
        and not any(task in payload for task in compiled.task_order)
    ):
        payload = payload["tasks"]
    if isinstance(payload, Mapping):
        missing = [task for task in compiled.task_order if task not in payload]
        if missing:
            raise SchemaError(f"logits missing tasks {missing}")
        return {task: _label_logit_map(payload[task], compiled.task(task).label_names) for task in compiled.task_order}
    if len(compiled.task_order) == 1:
        shape = getattr(payload, "shape", None)
        if shape is None or len(tuple(shape)) == 1:
            names = compiled.task(compiled.task_order[0]).label_names
            return {compiled.task_order[0]: _label_logit_map(payload, names)}
    rows = _rows(payload, len(compiled.task_order))
    return {
        task: _label_logit_map(row, compiled.task(task).label_names) for task, row in zip(compiled.task_order, rows)
    }


def _with_temperature(compiled, temperature):
    """Scale every task temperature by the call temperature before search."""
    specs = []
    changed = False
    for spec in compiled.task_specs:
        temp = _effective_temperature(spec, temperature)
        if temp == spec.temperature:
            specs.append(spec)
        else:
            specs.append(replace(spec, temperature=temp))
            changed = True
    if not changed:
        return compiled
    return replace(compiled, task_specs=tuple(specs))


def _scores_from_logits(logits, compiled, text: str) -> ClassificationScores:
    if hasattr(logits, "probability") and hasattr(logits, "utility") and hasattr(logits, "tasks"):
        if getattr(logits, "fingerprint", compiled.fingerprint) != compiled.fingerprint:
            raise SchemaError(
                "scores were produced for a different schema (fingerprint mismatch); re-score before decoding"
            )
        return logits
    return ClassificationScores(
        text=text,
        tasks=_align_logits(logits, compiled),
        fingerprint=compiled.fingerprint,
        specs={spec.name: spec for spec in compiled.task_specs},
    )


def decode_constrained_classification(
    logits,
    schema,
    temperature,
    *,
    text: str = "",
    active: Sequence[str] | None = None,
    decoder: str = "auto",
    exact_node_budget: int = 200_000,
    beam_size: int = 16,
    candidate_threshold: float = 0.5,
    max_candidates_per_task: int = 64,
    include_confidence: bool = True,
    on_infeasible: str = "relax",
) -> dict:
    """Decode label logits under hard cross-task constraints.

    ``temperature`` scales logits together with each task's schema temperature.
    Pass ``1.0`` when ``logits`` are already temperature-scaled. Returns the
    ``ClassificationResult.to_dict()`` mapping.
    """
    config = ClassificationConfig(
        decoder=decoder,
        exact_node_budget=exact_node_budget,
        beam_size=beam_size,
        candidate_threshold=candidate_threshold,
        max_candidates_per_task=max_candidates_per_task,
        include_confidence=include_confidence,
        on_infeasible=on_infeasible,
    )
    compiled = compile_schema(_coerce_schema(schema))
    prebuilt = hasattr(logits, "probability") and hasattr(logits, "utility") and hasattr(logits, "tasks")
    if not prebuilt:
        compiled = _with_temperature(compiled, temperature)
    scores = _scores_from_logits(logits, compiled, text)
    problem = build_problem(compiled, scores, config, active=active)

    def widen():
        return build_problem(
            compiled,
            scores,
            config,
            active=active,
            full_retention_tasks=problem.task_order,
        )

    solution = decode(problem, config, widen=widen)
    result = ResultBuilder().build(
        compiled,
        scores,
        solution,
        active_order=problem.task_order,
        include_confidence=config.include_confidence,
    )
    return result.to_dict()
