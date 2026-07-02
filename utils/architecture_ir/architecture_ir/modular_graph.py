"""Library-wide modular inheritance graph — a quick reference of which architectures are linked.

Each model's per-artifact ``extends`` only records its own dominant parent. This module aggregates
those links across the whole library into a single forest: for every model that participates in a
modular relationship (is modular, or is the parent of one) it records its parent, children, the
transitive ``root`` ancestor, depth, and ``diff_size``. Two models that share a ``root`` are in the
same lineage — the pairs whose semantic IDs align cleanly and are worth comparing.

Built purely by ``ast``-parsing ``modular_*.py`` files and their parents' modeling files (via
``modular.compute_modularity``): no torch, no model build, no config instantiation, so the entire
forest computes in seconds independently of which artifacts are generated.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable

from .modular import compute_modularity


GRAPH_SCHEMA_VERSION = "modular-graph-v0"

_NON_MODELS = {"auto", "__pycache__"}


def models_root() -> str:
    """Locate ``transformers/models`` inside the installed source tree."""
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed `transformers` package.")
    return os.path.join(list(spec.submodule_search_locations)[0], "models")


def _is_model_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return any(
        name.startswith(("modeling_", "modular_", "configuration_")) and name.endswith(".py")
        for name in os.listdir(path)
    )


def _candidate_names(root: str) -> list[str]:
    names = []
    for name in sorted(os.listdir(root)):
        if name in _NON_MODELS or name.startswith((".", "_")):
            continue
        if _is_model_dir(os.path.join(root, name)):
            names.append(name)
    return names


def build_modular_graph(root: str | None = None, only: Iterable[str] | None = None) -> dict:
    """Build the modular inheritance forest.

    ``only`` restricts the seed models (ancestors are still resolved transitively so ``root`` is
    correct); by default every model directory is considered. Isolated standalone models — not
    modular and never a parent — are pruned, so the graph is the modular universe plus its spines.
    """
    root = root or models_root()
    seeds = list(only) if only is not None else _candidate_names(root)

    # Compute modularity for each seed and, transitively, every referenced parent.
    computed: dict[str, object] = {}
    worklist = list(seeds)
    while worklist:
        name = worklist.pop()
        if name in computed:
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            computed[name] = None  # referenced parent with no discoverable source
            continue
        modularity = compute_modularity(path, name)
        computed[name] = modularity
        if modularity.parent_model and modularity.parent_model not in computed:
            worklist.append(modularity.parent_model)

    nodes: dict[str, dict] = {}
    for name, modularity in computed.items():
        if modularity is None:
            nodes[name] = {
                "extends": None,
                "parents": [],
                "children": [],
                "is_modular": False,
                "diff_size": None,
            }
        else:
            nodes[name] = {
                "extends": modularity.parent_model,
                "parents": list(modularity.parent_models),
                "children": [],
                "is_modular": modularity.is_modular,
                "diff_size": modularity.diff_size if modularity.is_modular else None,
            }

    for name, node in nodes.items():
        parent = node["extends"]
        if parent and parent in nodes:
            nodes[parent]["children"].append(name)
    for node in nodes.values():
        node["children"].sort()

    # Transitive root + depth (cycle-safe).
    for name, node in nodes.items():
        chain, current, seen = 0, name, set()
        while nodes.get(current, {}).get("extends") and current not in seen:
            seen.add(current)
            current = nodes[current]["extends"]
            chain += 1
        node["root"] = current
        node["depth"] = chain

    # Prune isolated standalones (no parent, no children, not modular).
    nodes = {
        name: node
        for name, node in nodes.items()
        if node["is_modular"] or node["children"] or node["extends"] is not None
    }
    # A pruned node may still be listed as a parent's child or another node's root; that is fine —
    # roots/children only reference surviving spine nodes here because a referenced parent always
    # has children and is therefore retained.

    roots = sorted(
        (name for name, node in nodes.items() if node["extends"] is None and node["children"]),
        key=lambda name: (-len(nodes[name]["children"]), name),
    )

    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "roots": roots,
        "nodes": dict(sorted(nodes.items())),
        "provenance": {
            "generator": "utils/architecture_ir",
            "source": "ast_modular_files",
            "schema_version": GRAPH_SCHEMA_VERSION,
        },
    }
