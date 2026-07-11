"""The interesting part: extract *what a modular model changes* relative to its parent(s).

We do not re-run the full ``utils/modular_model_converter.py`` (it exists to *generate*
standalone modeling files). Instead we mirror its inheritance semantics with libcst to
recover the diff payload directly:

- a method/attr present in BOTH the modular class and the named parent class  -> overridden
- present ONLY in the modular class                                            -> added
- a deletion sentinel (``attr = AttributeError(...)`` or ``raise AttributeError``) -> deleted

These are exactly the rules ``replace_class_node`` applies when it merges a child class
onto its parent. The parent model a class inherits from is read from the modular file's
``from ..<model>.modeling_<model> import <Class>`` imports -- the same import-following the
converter does. A clean modular model yields a tiny ``ArchDiff``; a bloated one yields a
large diff, which is the signal we want to surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import libcst as cst

from .discover import models_root


# ---------------------------------------------------------------------------- data model


@dataclass
class ClassChange:
    """The change set for a single class defined in a modular file."""

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
        """A pure ``class Foo(Bar): pass`` rename -- inherits everything, changes nothing."""
        return self.relation == "inherits" and self.n_changes == 0


@dataclass
class ArchDiff:
    model: str
    is_modular: bool
    parent_model: str | None
    parent_models: list[str]
    changes: list[ClassChange]
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


# ---------------------------------------------------------------------------- libcst bits


def _dotted(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Attribute):
        return _dotted(node.value) + "." + node.attr.value
    if isinstance(node, cst.Name):
        return node.value
    return ""


def _base_name(base: cst.Arg) -> str:
    v = base.value
    if isinstance(v, cst.Name):
        return v.value
    if isinstance(v, cst.Attribute):
        return v.attr.value
    return _dotted(v)


class _Collector(cst.CSTVisitor):
    """Collect imports and top-level class definitions from a parsed module."""

    def __init__(self) -> None:
        # imported name -> source model dir (parsed from `..<model>.modeling_<model>`)
        self.import_model: dict[str, str] = {}
        self.classes: dict[str, cst.ClassDef] = {}
        self._depth = 0

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        module = _dotted(node.module) if node.module is not None else ""
        if ".modeling_" not in module:
            return
        parent_model = module.split(".modeling_")[0].rsplit(".", 1)[-1]
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            self.import_model[alias.name.value] = parent_model

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self._depth == 0:
            self.classes[node.name.value] = node
        self._depth += 1
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._depth -= 1


def _methods_and_attrs(klass: cst.ClassDef) -> tuple[dict[str, cst.FunctionDef], dict[str, cst.CSTNode]]:
    methods: dict[str, cst.FunctionDef] = {}
    attrs: dict[str, cst.CSTNode] = {}
    for node in klass.body.body:
        if isinstance(node, cst.FunctionDef):
            methods[node.name.value] = node
        elif isinstance(node, cst.SimpleStatementLine) and node.body:
            stmt = node.body[0]
            if isinstance(stmt, cst.Assign) and stmt.targets:
                target = stmt.targets[0].target
                if isinstance(target, cst.Name):
                    attrs[target.value] = stmt
            elif isinstance(stmt, cst.AnnAssign) and isinstance(stmt.target, cst.Name):
                attrs[stmt.target.value] = stmt
    return methods, attrs


def _is_attr_deletion(stmt: cst.CSTNode) -> bool:
    # `foo = AttributeError(...)` is the converter's sentinel for deleting an inherited attr.
    if isinstance(stmt, cst.Assign) and isinstance(stmt.value, cst.Call):
        return isinstance(stmt.value.func, cst.Name) and stmt.value.func.value == "AttributeError"
    return False


def _is_method_deletion(fn: cst.FunctionDef) -> bool:
    # `def foo(self): raise AttributeError(...)` (or a bare `...`/del) signals removal.
    body = fn.body.body if isinstance(fn.body, cst.IndentedBlock) else []
    for node in body:
        if isinstance(node, cst.SimpleStatementLine):
            for s in node.body:
                if isinstance(s, cst.Raise) and isinstance(s.exc, cst.Call):
                    if isinstance(s.exc.func, cst.Name) and s.exc.func.value == "AttributeError":
                        return True
    return False


@lru_cache(maxsize=512)
def _parse_module_file(path: str) -> cst.Module | None:
    try:
        return cst.parse_module(open(path, encoding="utf-8").read())
    except Exception:
        return None


@lru_cache(maxsize=512)
def _parent_classes(parent_model: str) -> dict[str, cst.ClassDef]:
    """Top-level classes defined in ``models/<parent_model>/modeling_<parent_model>.py``."""
    path = os.path.join(models_root(), parent_model, f"modeling_{parent_model}.py")
    if not os.path.exists(path):
        # some models ship multiple modeling_*.py; scan them all
        d = os.path.join(models_root(), parent_model)
        if not os.path.isdir(d):
            return {}
        collector = _Collector()
        for f in sorted(os.listdir(d)):
            if f.startswith("modeling_") and f.endswith(".py"):
                mod = _parse_module_file(os.path.join(d, f))
                if mod is not None:
                    mod.visit(collector)
        return collector.classes
    mod = _parse_module_file(path)
    if mod is None:
        return {}
    collector = _Collector()
    mod.visit(collector)
    return collector.classes


# ---------------------------------------------------------------------------- public API


def _modular_path(model: str) -> str | None:
    d = os.path.join(models_root(), model)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.startswith("modular_") and f.endswith(".py"):
            return os.path.join(d, f)
    return None


def resolve_parent(model: str) -> str | None:
    """The dominant parent model of ``model`` (the one most classes inherit from), or None."""
    d = diff(model)
    return d.parent_model


def diff(model: str) -> ArchDiff:
    """Compute the modular diff for ``model``.

    For a standalone model (no ``modular_*.py``), returns ``is_modular=False`` with a note.
    """
    path = _modular_path(model)
    if path is None:
        return ArchDiff(
            model=model,
            is_modular=False,
            parent_model=None,
            parent_models=[],
            changes=[],
            note="standalone model (no modular parent)",
        )

    mod = _parse_module_file(path)
    if mod is None:
        return ArchDiff(model, True, None, [], [], note="could not parse modular file")

    collector = _Collector()
    mod.visit(collector)

    changes: list[ClassChange] = []
    parent_vote: dict[str, int] = {}

    for cls_name, node in collector.classes.items():
        bases = [_base_name(b) for b in node.bases]
        # find a base that maps to another transformers model
        inherited_parent = next((b for b in bases if b in collector.import_model), None)

        if inherited_parent is None:
            # brand-new class (inherits nn.Module / PreTrainedConfig / a local class)
            methods, attrs = _methods_and_attrs(node)
            changes.append(
                ClassChange(
                    name=cls_name,
                    relation="new",
                    parent=bases[0] if bases else None,
                    added_methods=sorted(methods),
                    added_attrs=sorted(attrs),
                )
            )
            continue

        parent_model = collector.import_model[inherited_parent]
        parent_vote[parent_model] = parent_vote.get(parent_model, 0) + 1
        parent_cls = _parent_classes(parent_model).get(inherited_parent)

        child_methods, child_attrs = _methods_and_attrs(node)
        if parent_cls is None:
            # could not locate parent source: treat all child members as overrides we cannot resolve
            changes.append(
                ClassChange(
                    name=cls_name,
                    relation="inherits",
                    parent=inherited_parent,
                    parent_model=parent_model,
                    overridden_methods=sorted(child_methods),
                    overridden_attrs=sorted(child_attrs),
                )
            )
            continue

        parent_methods, parent_attrs = _methods_and_attrs(parent_cls)
        cc = ClassChange(name=cls_name, relation="inherits", parent=inherited_parent, parent_model=parent_model)
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
        for lst in (
            cc.overridden_methods,
            cc.added_methods,
            cc.deleted_methods,
            cc.overridden_attrs,
            cc.added_attrs,
            cc.deleted_attrs,
        ):
            lst.sort()
        changes.append(cc)

    parent_model = max(parent_vote, key=parent_vote.get) if parent_vote else None
    parent_models = sorted(parent_vote, key=lambda m: (-parent_vote[m], m))
    return ArchDiff(
        model=model,
        is_modular=True,
        parent_model=parent_model,
        parent_models=parent_models,
        changes=changes,
    )
