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

"""Shared AST helper functions used across mlinter rule modules."""

import ast
from dataclasses import dataclass
from pathlib import Path


MODELS_ROOT = Path("src/transformers/models")


@dataclass(frozen=True)
class Violation:
    file_path: Path
    line_number: int
    message: str
    rule_id: str | None = None


def full_name(node: ast.AST):
    """Return full dotted name from an Attribute or Name node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return full_name(node.value) + "." + node.attr
    else:
        raise ValueError("Not a Name or Attribute node")


def _simple_name(name: str) -> str:
    return name.split(".")[-1]


def _model_dir_name(file_path: Path) -> str | None:
    try:
        relative = file_path.resolve().relative_to(MODELS_ROOT.resolve())
    except ValueError:
        try:
            relative = file_path.relative_to(MODELS_ROOT)
        except ValueError:
            return None
    if len(relative.parts) < 2:
        return None
    return relative.parts[0]


def _known_model_dirs() -> set[str]:
    return {path.name for path in MODELS_ROOT.iterdir() if path.is_dir()}


def _has_rule_suppression(lines: list[str], rule_id: str, line_number: int) -> bool:
    if line_number <= 0:
        return False
    token = f"trf-ignore: {rule_id}".lower()
    candidate_indexes = (line_number - 1, line_number - 2)
    for idx in candidate_indexes:
        if 0 <= idx < len(lines) and token in lines[idx].lower():
            return True
    return False


def _collect_class_bases(tree: ast.Module) -> dict[str, list[str]]:
    class_to_bases: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = []
        for base in node.bases:
            try:
                base_names.append(full_name(base))
            except ValueError:
                continue
        class_to_bases[node.name] = base_names
    return class_to_bases


def _inherits_pretrained_model(
    class_name: str, class_to_bases: dict[str, list[str]], visiting: set[str] | None = None
) -> bool:
    if visiting is None:
        visiting = set()
    if class_name in visiting:
        return False
    visiting.add(class_name)

    for base_name in class_to_bases.get(class_name, []):
        simple_base_name = _simple_name(base_name)
        if simple_base_name.endswith("PreTrainedModel"):
            return True
        if simple_base_name in class_to_bases and _inherits_pretrained_model(
            simple_base_name, class_to_bases, visiting
        ):
            return True
    return False


def iter_pretrained_classes(tree: ast.Module, source_lines: list[str], rule_id: str) -> list[ast.ClassDef]:
    """Yield ClassDef nodes that inherit from PreTrainedModel (transitively), skipping suppressed ones."""
    class_to_bases = _collect_class_bases(tree)
    results = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _inherits_pretrained_model(node.name, class_to_bases):
            continue
        if _has_rule_suppression(source_lines, rule_id, node.lineno):
            continue
        results.append(node)
    return results


def _get_class_assignments(class_node: ast.ClassDef) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for item in class_node.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            assignments[item.targets[0].id] = item.value
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
            assignments[item.target.id] = item.value
    return assignments


def _class_methods(class_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {item.name: item for item in class_node.body if isinstance(item, ast.FunctionDef)}


def _function_argument_names(function_node: ast.FunctionDef) -> set[str]:
    names = {arg.arg for arg in function_node.args.args}
    names.update(arg.arg for arg in function_node.args.kwonlyargs)
    if function_node.args.vararg is not None:
        names.add(function_node.args.vararg.arg)
    if function_node.args.kwarg is not None:
        names.add(function_node.args.kwarg.arg)
    return names


def _function_uses_name(function_node: ast.FunctionDef, name: str) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        for node in ast.walk(function_node)
    )


def is_self_method_call(node: ast.AST, method: str) -> bool:
    """Check if `node` is a method call on `self`, such as `self.method(...)`"""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method
    )


def is_super_method_call(node: ast.AST, method: str) -> bool:
    """Check if `node` is a call to `super().method(...)`"""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        and node.func.attr == method
    )


def _is_direct_pretrained_config_subclass(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        try:
            if _simple_name(full_name(base)) in {"PreTrainedConfig", "PretrainedConfig"}:
                return True
        except ValueError:
            continue
    return False


def _has_strict_decorator(class_node: ast.ClassDef) -> bool:
    for decorator in class_node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "strict":
            return True

    return False
