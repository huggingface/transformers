"""Recover *what a modular model changes* relative to its parent(s), from source.

Many Transformers models are defined as a ``modular_<name>.py`` whose classes inherit from
another model (``class GemmaModel(LlamaModel)``). ``utils/modular_model_converter.py`` exists
to *generate* the standalone modeling file from that; here we do the opposite — we mirror its
inheritance semantics with the stdlib ``ast`` to recover the *diff payload* directly:

- a method/attr present in BOTH the modular class and the named parent class   -> overridden
- present ONLY in the modular class                                            -> added
- a deletion sentinel (``attr = AttributeError(...)`` / ``def f(): raise AttributeError``) -> deleted

The parent model of a class is read from the modular file's
``from ..<model>.modeling_<model> import <Class>`` imports — the same import-following the
converter performs. A clean modular model yields a tiny diff; a bloated one yields a large
diff. ``diff_size`` is therefore an automatic, per-model measure of how modular a model is:
it is the signal this module exists to surface. No torch, no imports of the model itself.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from functools import lru_cache

from .recognizers import classify_class_name


# ---------------------------------------------------------------------------- data model


@dataclass
class ClassChange:
    """The change set of a single class defined in a modular file, vs its parent."""

    name: str
    relation: str  # "inherits" | "new"
    parent: str | None = None
    parent_model: str | None = None
    overridden_methods: list[str] = field(default_factory=list)
    added_methods: list[str] = field(default_factory=list)
    deleted_methods: list[str] = field(default_factory=list)
    overridden_attrs: list[str] = field(default_factory=list)
    added_attrs: list[str] = field(default_factory=list)
    deleted_attrs: list[str] = field(default_factory=list)

    @property
    def n_changes(self) -> int:
        return (
            len(self.overridden_methods)
            + len(self.added_methods)
            + len(self.deleted_methods)
            + len(self.overridden_attrs)
            + len(self.added_attrs)
            + len(self.deleted_attrs)
        )

    @property
    def is_trivial(self) -> bool:
        """A pure ``class Foo(Bar): pass`` rename — inherits everything, changes nothing."""
        return self.relation == "inherits" and self.n_changes == 0


@dataclass
class Modularity:
    """Modular-diff summary for one model_type."""

    model_type: str
    is_modular: bool
    parent_model: str | None = None
    parent_models: list[str] = field(default_factory=list)
    changes: list[ClassChange] = field(default_factory=list)
    note: str | None = None

    @property
    def totals(self) -> dict[str, int]:
        t = {"overridden": 0, "added": 0, "deleted": 0, "new_classes": 0, "trivial": 0}
        for c in self.changes:
            t["overridden"] += len(c.overridden_methods) + len(c.overridden_attrs)
            t["added"] += len(c.added_methods) + len(c.added_attrs)
            t["deleted"] += len(c.deleted_methods) + len(c.deleted_attrs)
            if c.relation == "new":
                t["new_classes"] += 1
            if c.is_trivial:
                t["trivial"] += 1
        return t

    @property
    def diff_size(self) -> int:
        """Single-number modularity metric: smaller = more of the parent is reused as-is.

        New classes are weighted (×3): adding a whole class is a bigger deviation from the
        parent than tweaking a member. This matches the metric used by the community gallery.
        """
        t = self.totals
        return t["overridden"] + t["added"] + t["deleted"] + 3 * t["new_classes"]


# --------------------------------------------------------------------------------- ast bits


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _parent_model_from_import(node: ast.ImportFrom) -> str | None:
    module = node.module or ""
    if ".modeling_" not in module:
        return None
    return module.split(".modeling_")[0].rsplit(".", 1)[-1] or None


def _members(klass: ast.ClassDef) -> tuple[dict[str, ast.AST], dict[str, ast.AST]]:
    methods: dict[str, ast.AST] = {}
    attrs: dict[str, ast.AST] = {}
    for node in klass.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    attrs[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            attrs[node.target.id] = node
    return methods, attrs


def _is_attr_deletion(node: ast.AST) -> bool:
    # `foo = AttributeError(...)` is the converter's sentinel for deleting an inherited attr.
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        return isinstance(node.value.func, ast.Name) and node.value.func.id == "AttributeError"
    return False


def _is_method_deletion(fn: ast.AST) -> bool:
    # `def foo(self): raise AttributeError(...)` signals removal of an inherited method.
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for stmt in fn.body:
        if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
            if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "AttributeError":
                return True
    return False


@lru_cache(maxsize=512)
def _parse(path: str) -> ast.Module | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return ast.parse(handle.read())
    except (OSError, SyntaxError):
        return None


def _top_classes(module: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}


@lru_cache(maxsize=512)
def _parent_classes(models_root: str, parent_model: str) -> dict[str, ast.ClassDef]:
    """Top-level classes across ``models/<parent_model>/modeling_*.py``."""
    directory = os.path.join(models_root, parent_model)
    if not os.path.isdir(directory):
        return {}
    classes: dict[str, ast.ClassDef] = {}
    for name in sorted(os.listdir(directory)):
        if name.startswith("modeling_") and name.endswith(".py"):
            module = _parse(os.path.join(directory, name))
            if module is not None:
                # First definition wins; modeling_<model>.py is visited first by sort order.
                for cls_name, node in _top_classes(module).items():
                    classes.setdefault(cls_name, node)
    return classes


# --------------------------------------------------------------------------------- public API


def _modular_path(model_dir: str) -> str | None:
    if not os.path.isdir(model_dir):
        return None
    for name in sorted(os.listdir(model_dir)):
        if name.startswith("modular_") and name.endswith(".py"):
            return os.path.join(model_dir, name)
    return None


def compute_modularity(model_dir: str, model_type: str) -> Modularity:
    """Compute the modular diff for the model whose source lives in ``model_dir``.

    ``model_dir`` is the package directory (e.g. ``.../transformers/models/gemma``). Standalone
    models (no ``modular_*.py``) return ``is_modular=False`` with an empty change set.
    """
    path = _modular_path(model_dir)
    if path is None:
        return Modularity(model_type=model_type, is_modular=False, note="standalone model (no modular file)")

    module = _parse(path)
    if module is None:
        return Modularity(model_type=model_type, is_modular=True, note="could not parse modular file")

    models_root = os.path.dirname(model_dir.rstrip(os.sep))
    import_model: dict[str, str] = {}
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            parent_model = _parent_model_from_import(node)
            if parent_model is not None:
                for alias in node.names:
                    import_model[alias.name] = parent_model

    changes: list[ClassChange] = []
    parent_vote: dict[str, int] = {}

    for cls_name, node in _top_classes(module).items():
        bases = [_base_name(base) for base in node.bases]
        inherited = next((b for b in bases if b in import_model), None)
        child_methods, child_attrs = _members(node)

        if inherited is None:
            # Brand-new class (inherits nn.Module / a config base / a local class).
            changes.append(
                ClassChange(
                    name=cls_name,
                    relation="new",
                    parent=bases[0] if bases else None,
                    added_methods=sorted(child_methods),
                    added_attrs=sorted(child_attrs),
                )
            )
            continue

        parent_model = import_model[inherited]
        parent_vote[parent_model] = parent_vote.get(parent_model, 0) + 1
        parent_cls = _parent_classes(models_root, parent_model).get(inherited)
        cc = ClassChange(name=cls_name, relation="inherits", parent=inherited, parent_model=parent_model)

        if parent_cls is None:
            # Parent source not found: we cannot resolve override-vs-add, so treat all as overrides.
            cc.overridden_methods = sorted(child_methods)
            cc.overridden_attrs = sorted(child_attrs)
            changes.append(cc)
            continue

        parent_methods, parent_attrs = _members(parent_cls)
        for name, fn in child_methods.items():
            if _is_method_deletion(fn):
                cc.deleted_methods.append(name)
            elif name in parent_methods:
                cc.overridden_methods.append(name)
            else:
                cc.added_methods.append(name)
        for name, stmt in child_attrs.items():
            if _is_attr_deletion(stmt):
                cc.deleted_attrs.append(name)
            elif name in parent_attrs:
                cc.overridden_attrs.append(name)
            else:
                cc.added_attrs.append(name)
        for bucket in (
            cc.overridden_methods,
            cc.added_methods,
            cc.deleted_methods,
            cc.overridden_attrs,
            cc.added_attrs,
            cc.deleted_attrs,
        ):
            bucket.sort()
        changes.append(cc)

    parent_model = max(parent_vote, key=parent_vote.get) if parent_vote else None
    parent_models = sorted(parent_vote, key=lambda m: (-parent_vote[m], m))
    return Modularity(
        model_type=model_type,
        is_modular=True,
        parent_model=parent_model,
        parent_models=parent_models,
        changes=changes,
    )


def modularity_payload(modularity: Modularity) -> dict:
    """Project a ``Modularity`` into the artifact's ``modularity`` summary + ``patches`` list.

    Returns ``{"extends", "modularity", "patches"}``. ``patches`` are the per-class change sets
    projected onto semantic component kinds (``component_kind``), so a consumer can read
    "this model overrides its attention" without knowing Python class names.
    """
    summary = {
        "is_modular": modularity.is_modular,
        "parent_model": modularity.parent_model,
        "parent_models": modularity.parent_models,
        "diff_size": modularity.diff_size,
        "totals": modularity.totals,
    }
    if modularity.note:
        summary["note"] = modularity.note

    patches = []
    for c in modularity.changes:
        patches.append(
            {
                "relation": c.relation,
                "target_class": c.name,
                "component_kind": classify_class_name(c.name),
                "parent_class": c.parent,
                "parent_model": c.parent_model,
                "overridden": {"methods": c.overridden_methods, "attrs": c.overridden_attrs},
                "added": {"methods": c.added_methods, "attrs": c.added_attrs},
                "deleted": {"methods": c.deleted_methods, "attrs": c.deleted_attrs},
                "n_changes": c.n_changes,
            }
        )

    return {"extends": modularity.parent_model, "modularity": summary, "patches": patches}
