"""Detect which components are kernelizable (`@use_kernel_forward_from_hub`) and their Hub repos.

Some modules are decorated with ``@use_kernel_forward_from_hub("<LayerName>")`` so their forward can be
swapped for an optimized kernel from the Hub (the ``kernels`` integration). The decorator's layer-name
argument is the useful fact; the ``kernel_layer_name`` instance attribute is only set after ``kernelize()``,
so it is not readable off a meta model. And the ``layer_name -> repo`` registry accessor is gated behind
``is_kernels_available()`` (often False, and it needs the optional ``kernels`` package).

So — like ``modular.py`` — we read both facts straight from source with the stdlib ``ast``: which classes are
decorated (from each model's ``modeling_*.py``) and which Hub repos each layer maps to (from the hardcoded
``_KERNEL_MAPPING`` in ``transformers/integrations/hub_kernels.py``). No torch, no network, no extra dependency.
"""

from __future__ import annotations

import ast
import os
from functools import lru_cache

from .modular import _parse
from .modular_graph import models_root


_DECORATOR = "use_kernel_forward_from_hub"


def _decorator_layer_name(decorator: ast.expr) -> str | None:
    """The string arg of a ``@use_kernel_forward_from_hub("X")`` decorator, or None."""
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name != _DECORATOR or not decorator.args:
        return None
    arg = decorator.args[0]
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


def detect_kernel_layers(model_dir: str) -> dict[str, str]:
    """Map ``{class_name: kernel_layer_name}`` for classes decorated in the model's ``modeling_*.py``."""
    mapping: dict[str, str] = {}
    if not os.path.isdir(model_dir):
        return mapping
    for name in sorted(os.listdir(model_dir)):
        if not (name.startswith("modeling_") and name.endswith(".py")):
            continue
        module = _parse(os.path.join(model_dir, name))
        if module is None:
            continue
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    layer = _decorator_layer_name(decorator)
                    if layer is not None:
                        mapping[node.name] = layer
                        break
    return mapping


@lru_cache(maxsize=1)
def kernel_repositories() -> dict[str, list[str]]:
    """Map ``{kernel_layer_name: [hub_repo_id, ...]}`` parsed from ``hub_kernels.py``'s ``_KERNEL_MAPPING``.

    Read from source (the accessor is gated behind ``is_kernels_available()``); best-effort → ``{}`` on failure.
    """
    path = os.path.join(os.path.dirname(models_root().rstrip(os.sep)), "integrations", "hub_kernels.py")
    module = _parse(path)
    if module is None:
        return {}

    mapping_dict: ast.Dict | None = None
    for node in ast.walk(module):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        )
        if any(isinstance(t, ast.Name) and t.id == "_KERNEL_MAPPING" for t in targets) and isinstance(
            node.value, ast.Dict
        ):
            mapping_dict = node.value
            break
    if mapping_dict is None:
        return {}

    repos: dict[str, list[str]] = {}
    for key, value in zip(mapping_dict.keys, mapping_dict.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        found: list[str] = []
        for sub in ast.walk(value):
            if isinstance(sub, ast.keyword) and sub.arg == "repo_id":
                repo = sub.value
                if isinstance(repo, ast.Constant) and isinstance(repo.value, str) and repo.value not in found:
                    found.append(repo.value)
        repos[key.value] = found
    return repos
