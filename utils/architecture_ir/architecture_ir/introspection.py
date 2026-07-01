"""Generic module-tree introspection for architecture IR artifacts."""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from .dataflow import capture_dataflow, symbolize_shape
from .enrich import architecture_facts, attention_spec, moe_spec, norm_type
from .modular import compute_modularity, modularity_payload
from .recognizers import build_edges, classify_component, detect_repeats, summarize_semantic_components
from .semantic_model import SCHEMA_VERSION, ArchitectureArtifact, Component, Edge, Repeat


def build_architecture_artifact(model_type: str, resolved: Any) -> dict[str, Any]:
    components = inspect_module_tree(resolved.model)
    repeats = detect_repeats(components, resolved.config)
    edges = build_edges(components)
    artifact = _build_compact_artifact(model_type, resolved, components, repeats, edges)
    _enrich_artifact(artifact, model_type, resolved.config)
    artifact.update(_modular_payload(model_type, resolved))
    _attach_dataflow(artifact, resolved)
    return artifact


def _enrich_artifact(artifact: dict[str, Any], model_type: str, config: Any) -> None:
    """Add semantic-depth facts: model-level ``architecture`` block, per-component attributes,
    and the cache/route edges that decoder attention and MoE routing imply."""
    facts = architecture_facts(model_type, config)
    artifact["architecture"] = facts
    view = facts.get("view")

    attn = attention_spec(config)
    moe = moe_spec(config)
    for component in [*artifact["components"], *artifact["templates"]]:
        kind = component["kind"]
        if kind in {"attention", "cross_attention"} and attn:
            component["attributes"] = dict(attn)
        elif kind == "normalization":
            component["attributes"] = {"norm_type": norm_type(component["class_name"])}
        elif kind == "moe" and moe:
            component["attributes"] = dict(moe)

    _attach_cache_edges(artifact, view)
    _attach_route_edges(artifact)


def _attach_cache_edges(artifact: dict[str, Any], view: str | None) -> None:
    """Emit cache_read/cache_write for self-attention that maintains a KV cache (decoder side)."""
    existing = {(edge["source"], edge["target"], edge["kind"]) for edge in artifact["edges"]}
    for component in [*artifact["components"], *artifact["templates"]]:
        if component["kind"] != "attention":
            continue
        on_decoder_side = view == "decoder" or (view == "enc_dec" and "decoder" in component.get("path_pattern", ""))
        if not on_decoder_side:
            continue
        for source, target, kind in (
            ("state:kv_cache", component["id"], "cache_read"),
            (component["id"], "state:kv_cache", "cache_write"),
        ):
            if (source, target, kind) not in existing:
                existing.add((source, target, kind))
                artifact["edges"].append({"source": source, "target": target, "kind": kind})


def _attach_route_edges(artifact: dict[str, Any]) -> None:
    """Emit a route edge from each MoE block to its experts container, when detectable."""
    existing = {(edge["source"], edge["target"], edge["kind"]) for edge in artifact["edges"]}
    ids = {c["id"] for c in [*artifact["components"], *artifact["templates"]]}
    for component in [*artifact["components"], *artifact["templates"]]:
        if component["kind"] != "moe":
            continue
        for child_id in component.get("children", []):
            if child_id in ids and ("expert" in child_id.lower()):
                key = (component["id"], child_id, "route")
                if key not in existing:
                    existing.add(key)
                    artifact["edges"].append({"source": component["id"], "target": child_id, "kind": "route"})


def _attach_dataflow(artifact: dict[str, Any], resolved: Any) -> None:
    """Ground the top-level flow in an observed meta forward: attach shapes + observed edges.

    Best-effort: models whose forward can't run on meta keep their structural edges and simply
    get no ``dataflow`` block.
    """
    try:
        observed = capture_dataflow(resolved.model, resolved.config)
    except Exception:
        observed = None
    if observed is None:
        return

    stage_id = _stage_id_lookup(artifact)
    config = resolved.config
    stages = []
    for stage in observed["stages"]:
        component_id = stage_id.get(f"model.{stage['name']}")
        stages.append(
            {
                "id": component_id,
                "module_name": stage["name"],
                "class_name": stage["class_name"],
                "in_shape": symbolize_shape(stage["in_shape"], config),
                "out_shape": symbolize_shape(stage["out_shape"], config),
            }
        )

    artifact["dataflow"] = {
        "source": "observed_forward_meta",
        "input": {
            "name": observed["input"]["name"],
            "shape": symbolize_shape(observed["input"]["shape"], config),
        },
        "output": {"shape": symbolize_shape(observed["output"]["shape"], config)},
        "stages": stages,
    }

    # Add any observed consecutive-stage data edges missing from the structural set. Position
    # modules (e.g. rotary embeddings) execute on the main path but are not data-path carriers —
    # they reach attention via the "position" edge — so exclude them from the data chain.
    existing = {(edge["source"], edge["target"], edge["kind"]) for edge in artifact["edges"]}
    kind_by_id = _kind_by_id(artifact)
    mapped_ids = [
        stage["id"]
        for stage in stages
        if stage["id"] is not None and kind_by_id.get(stage["id"]) not in {"position", "mask"}
    ]
    for source, target in zip(mapped_ids, mapped_ids[1:]):
        if source == target or (source, target, "data") in existing:
            continue
        existing.add((source, target, "data"))
        artifact["edges"].append(
            {"source": source, "target": target, "kind": "data", "provenance": "observed_forward"}
        )


def _stage_id_lookup(artifact: dict[str, Any]) -> dict[str, str]:
    """Map a top-level symbolic path (``model.<child>``) to its semantic component/repeat id."""
    lookup: dict[str, str] = {}
    for component in artifact["components"]:
        lookup[component["path_pattern"]] = component["id"]
    for repeat in artifact["repeats"]:
        lookup[repeat["container_path_pattern"]] = repeat["id"]
    return lookup


def _kind_by_id(artifact: dict[str, Any]) -> dict[str, str]:
    kinds = {component["id"]: component["kind"] for component in artifact["components"]}
    kinds.update({template["id"]: template["kind"] for template in artifact["templates"]})
    kinds.update({repeat["id"]: "symbolic_repeat" for repeat in artifact["repeats"]})
    return kinds


def _modular_payload(model_type: str, resolved: Any) -> dict[str, Any]:
    """Compute the modular-diff payload (extends / modularity / patches) from source.

    Best-effort and never fatal: on any failure we still emit a ``modularity`` summary marking
    the model non-modular with a note, so the artifact shape stays stable.
    """
    try:
        model_dir = os.path.dirname(inspect.getfile(resolved.config.__class__))
        return modularity_payload(compute_modularity(model_dir, model_type))
    except Exception as error:
        return {
            "extends": None,
            "modularity": {
                "is_modular": False,
                "parent_model": None,
                "parent_models": [],
                "diff_size": 0,
                "totals": {"overridden": 0, "added": 0, "deleted": 0, "new_classes": 0, "trivial": 0},
                "note": f"modularity unavailable: {error.__class__.__name__}",
            },
            "patches": [],
        }


def build_expanded_architecture_artifact(model_type: str, resolved: Any) -> ArchitectureArtifact:
    components = inspect_module_tree(resolved.model)
    repeats = detect_repeats(components, resolved.config)
    edges = build_edges(components)
    return ArchitectureArtifact(
        schema_version=SCHEMA_VERSION,
        model_type=model_type,
        entrypoints=resolved.entrypoints,
        config=_config_payload(resolved.config, [repeat.expression for repeat in repeats]),
        components=components,
        semantic_components=summarize_semantic_components(components),
        hierarchy=_hierarchy_payload(components),
        repeats=repeats,
        edges=edges,
        provenance={
            "generator": "utils/architecture_ir",
            "source": "local_transformers_checkout",
            "resolution": "AutoConfig.for_model + AutoModel.from_config",
            "instantiation": "config_only_meta_device_no_init_weights",
            "schema_version": SCHEMA_VERSION,
            "model_type": model_type,
            "config_class": resolved.config.__class__.__name__,
            "config_class_module": resolved.config.__class__.__module__,
            "model_class": resolved.model.__class__.__name__,
            "model_class_module": resolved.model.__class__.__module__,
        },
        warnings=list(resolved.warnings),
    )


def inspect_module_tree(model: Any) -> list[Component]:
    components: list[Component] = []
    path_to_id: dict[str, str] = {}

    for path, module in model.named_modules():
        component_id = _component_id(path)
        path_to_id[path] = component_id
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        parent_id = path_to_id[parent_path] if path else None
        class_name = module.__class__.__name__
        component = Component(
            id=component_id,
            name=path.rsplit(".", 1)[-1] if path else "<root>",
            path=path,
            parent=parent_id,
            kind=classify_component(path, class_name),
            class_name=class_name,
            module=module.__class__.__module__,
            children=[],
            attributes=_module_attributes(module),
            provenance={
                "component_id": component_id,
                "module_path": module.__class__.__module__,
                "class_name": class_name,
                "source": "module_tree",
            },
        )
        components.append(component)

    by_path = {component.path: component for component in components}
    for component in components:
        if component.parent is None:
            continue
        parent_path = component.path.rsplit(".", 1)[0] if "." in component.path else ""
        by_path[parent_path].children.append(component.id)

    return components


def artifact_to_json_dict(artifact: ArchitectureArtifact | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(artifact, Mapping):
        return _json_safe(artifact)
    if is_dataclass(artifact):
        return _json_safe(asdict(artifact))
    raise TypeError(f"Unsupported architecture artifact type: {type(artifact)}")


def _build_compact_artifact(
    model_type: str,
    resolved: Any,
    components: list[Component],
    repeats: list[Repeat],
    edges: list[Edge],
) -> dict[str, Any]:
    repeat_templates = _repeat_templates(repeats)
    component_id_map = _canonical_component_id_map(components, repeat_templates)
    compact_components = _compact_components(components, repeat_templates, component_id_map)
    compact_templates = _compact_templates(components, repeat_templates, component_id_map)
    compact_repeats = _compact_repeats(repeats, repeat_templates)
    known_ids = {component["id"] for component in compact_components}
    known_ids.update(template["id"] for template in compact_templates)
    known_ids.update(repeat["id"] for repeat in compact_repeats)
    _prune_children(compact_components, known_ids)
    _prune_children(compact_templates, known_ids)
    kind_by_id = {component["id"]: component["kind"] for component in compact_components}
    kind_by_id.update({template["id"]: template["kind"] for template in compact_templates})
    kind_by_id.update({repeat["id"]: "repeat" for repeat in compact_repeats})

    referenced_fields = _referenced_config_field_names([repeat.get("count_expr") for repeat in compact_repeats])

    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "metadata": {
            "level": "architecture_template",
            "format": "compact_template",
            "description": (
                "Config-parametric ArchitectureTemplate with symbolic repeats and reusable component templates. "
                "Resolve against a checkpoint config.json to obtain a ResolvedGraph."
            ),
        },
        "entrypoints": resolved.entrypoints,
        "config": _config_payload(resolved.config, referenced_fields),
        "components": compact_components,
        "templates": compact_templates,
        "repeats": compact_repeats,
        "edges": _compact_edges(edges, component_id_map, known_ids, kind_by_id),
        "provenance": {
            "generator": "utils/architecture_ir",
            "source": "local_transformers_checkout",
            "resolution": "AutoConfig.for_model + AutoModel.from_config",
            "instantiation": "config_only_meta_device_no_init_weights",
            "schema_version": SCHEMA_VERSION,
            "model_type": model_type,
            "config_class": resolved.config.__class__.__name__,
            "config_class_module": resolved.config.__class__.__module__,
            "model_class": resolved.model.__class__.__name__,
            "model_class_module": resolved.model.__class__.__module__,
            "path_style": "symbolic_patterns",
        },
        "warnings": list(resolved.warnings),
    }


def _repeat_templates(repeats: list[Repeat]) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    used_body_ids: set[str] = set()
    used_repeat_ids: set[str] = set()

    for repeat in repeats:
        body_id = _unique_id(_repeat_body_id(repeat), used_body_ids)
        repeat_id = _unique_id(_pluralize(body_id), used_repeat_ids)
        first_child_path = _path_from_component_id(repeat.repeated_component_ids[0])
        templates[repeat.id] = {
            "body_id": body_id,
            "repeat_id": repeat_id,
            "index_symbol": "i",
            "container_path": repeat.path,
            "container_path_pattern": _model_path_pattern(repeat.path),
            "item_path": first_child_path,
            "item_path_pattern": _path_with_symbolic_index(repeat.path, first_child_path),
        }
    return templates


def _compact_components(
    components: list[Component],
    repeat_templates: dict[str, dict[str, Any]],
    component_id_map: dict[str, str],
) -> list[dict[str, Any]]:
    by_path = {component.path: component for component in components}
    compact: list[dict[str, Any]] = []

    for component in components:
        if _repeat_for_descendant_path(component.path, repeat_templates) is not None:
            continue
        if _repeat_for_container_path(component.path, repeat_templates) is not None:
            continue
        if not _include_canonical_component(component):
            continue

        compact.append(
            _component_payload(
                component,
                component_id_map[component.id],
                _symbolic_children(component, by_path, repeat_templates, component_id_map),
                _model_path_pattern(component.path),
            )
        )

    return compact


def _compact_templates(
    components: list[Component],
    repeat_templates: dict[str, dict[str, Any]],
    component_id_map: dict[str, str],
) -> list[dict[str, Any]]:
    by_path = {component.path: component for component in components}
    compact: list[dict[str, Any]] = []

    for template in repeat_templates.values():
        item_path = template["item_path"]
        body_components = [
            component
            for component in components
            if component.path == item_path or component.path.startswith(f"{item_path}.")
        ]
        for component in body_components:
            if not _include_canonical_component(component):
                continue
            compact.append(
                _component_payload(
                    component,
                    component_id_map[component.id],
                    _symbolic_children(component, by_path, repeat_templates, component_id_map),
                    _path_with_symbolic_index(template["container_path"], component.path),
                )
            )

    return compact


def _compact_repeats(repeats: list[Repeat], repeat_templates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for repeat in repeats:
        template = repeat_templates[repeat.id]
        compact.append(
            {
                "id": template["repeat_id"],
                "kind": "symbolic_repeat",
                "body": template["body_id"],
                "count": repeat.count,
                "count_expr": repeat.expression,
                "count_source": repeat.count_source,
                "index_symbol": template["index_symbol"],
                "container_path_pattern": template["container_path_pattern"],
                "item_path_pattern": template["item_path_pattern"],
                "repeated_class_name": repeat.repeated_class_name,
                "provenance": {
                    "source": "module_tree_repeat_collapse",
                    "container_path_pattern": template["container_path_pattern"],
                    "item_path_pattern": template["item_path_pattern"],
                    "class_name": repeat.repeated_class_name,
                },
            }
        )
    return compact


def _compact_edges(
    edges: list[Edge],
    component_id_map: dict[str, str],
    known_ids: set[str],
    kind_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for edge in edges:
        source = component_id_map.get(edge.source, edge.source)
        target = component_id_map.get(edge.target, edge.target)
        if source == target:
            continue
        if not _is_known_edge_endpoint(source, known_ids) or not _is_known_edge_endpoint(target, known_ids):
            continue
        if edge.kind == "data" and not _is_coarse_data_edge(source, target, kind_by_id):
            continue

        key = (source, target, edge.kind)
        if key in seen:
            continue
        seen.add(key)
        compact.append(
            {
                "source": source,
                "target": target,
                "kind": edge.kind,
            }
        )

    return compact


def _canonical_component_id_map(
    components: list[Component],
    repeat_templates: dict[str, dict[str, Any]],
) -> dict[str, str]:
    by_path = {component.path: component for component in components}
    mapping: dict[str, str] = {}

    for component in components:
        template = _repeat_for_container_path(component.path, repeat_templates)
        if template is not None:
            mapping[component.id] = template["repeat_id"]
            continue

        repeated_template = _repeat_for_descendant_path(component.path, repeat_templates)
        if repeated_template is not None:
            relative_path = _relative_repeated_path(repeated_template["container_path"], component.path)
            mapping[component.id] = repeated_template["body_id"] + (f".{relative_path}" if relative_path else "")
            continue

        mapping[component.id] = _component_id_for_path(component.path, by_path)

    return mapping


def _component_payload(
    component: Component,
    component_id: str,
    children: list[str],
    path_pattern: str,
) -> dict[str, Any]:
    return {
        "id": component_id,
        "kind": component.kind,
        "class_name": component.class_name,
        "module": component.module,
        "path_pattern": path_pattern,
        "children": children,
    }


def _symbolic_children(
    component: Component,
    by_path: dict[str, Component],
    repeat_templates: dict[str, dict[str, Any]],
    component_id_map: dict[str, str],
) -> list[str]:
    children: list[str] = []
    seen: set[str] = set()
    for child_id in component.children:
        child_path = _path_from_component_id(child_id)
        child = by_path[child_path]
        template = _repeat_for_container_path(child.path, repeat_templates)
        symbolic_id = template["repeat_id"] if template is not None else component_id_map[child.id]
        if symbolic_id not in seen:
            children.append(symbolic_id)
            seen.add(symbolic_id)
    return children


def _include_canonical_component(component: Component) -> bool:
    class_name = component.class_name.lower()
    if class_name == "linear" or class_name == "dropout" or class_name.endswith("activation"):
        return False

    return component.kind in {
        "model",
        "embedding",
        "position",
        "attention",
        "cross_attention",
        "feed_forward",
        "moe",
        "normalization",
        "pooler",
        "transformer_block",
        "encoder",
        "decoder",
        "stack",
        "repeated_container",
    }


def _prune_children(components: list[dict[str, Any]], known_ids: set[str]) -> None:
    for component in components:
        component["children"] = [child_id for child_id in component["children"] if child_id in known_ids]


def _repeat_for_container_path(path: str, repeat_templates: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for template in repeat_templates.values():
        if path == template["container_path"]:
            return template
    return None


def _repeat_for_descendant_path(path: str, repeat_templates: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    matches = []
    for template in repeat_templates.values():
        container_path = template["container_path"]
        if path == template["item_path"] or re.match(rf"^{re.escape(container_path)}\.\d+(\.|$)", path):
            matches.append(template)
    if not matches:
        return None
    return max(matches, key=lambda template: len(template["container_path"]))


def _path_from_component_id(component_id: str) -> str:
    return "" if component_id == "root" else component_id.removeprefix("module:")


def _component_id_for_path(path: str, by_path: dict[str, Component]) -> str:
    if path == "":
        return "model"

    component = by_path[path]
    if component.kind == "model":
        return "model"
    return path


def _repeat_body_id(repeat: Repeat) -> str:
    path_lower = repeat.path.lower()
    class_lower = repeat.repeated_class_name.lower()

    if "decoder" in path_lower or "decoder" in class_lower:
        context = "decoder"
    elif "encoder" in path_lower or "encoder" in class_lower:
        context = "encoder"
    else:
        context = None

    if "block" in path_lower or class_lower.endswith("block"):
        unit = "block"
    elif "layer" in path_lower or "layer" in class_lower:
        unit = "layer"
    else:
        unit = _snake_case(repeat.repeated_class_name)

    return f"{context}_{unit}" if context is not None else unit


def _unique_id(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        used.add(candidate)
        return candidate

    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    value = f"{candidate}_{suffix}"
    used.add(value)
    return value


def _pluralize(value: str) -> str:
    if value.endswith("s"):
        return value
    if value.endswith("y"):
        return f"{value[:-1]}ies"
    return f"{value}s"


def _model_path_pattern(path: str) -> str:
    return "model" if path == "" else f"model.{path}"


def _path_with_symbolic_index(container_path: str, path: str) -> str:
    if path == container_path:
        return _model_path_pattern(path)
    return re.sub(
        rf"^model\.{re.escape(container_path)}\.\d+",
        f"model.{container_path}.{{i}}",
        _model_path_pattern(path),
        count=1,
    )


def _relative_repeated_path(container_path: str, path: str) -> str:
    relative = re.sub(rf"^{re.escape(container_path)}\.\d+\.?", "", path, count=1)
    return relative


def _is_known_edge_endpoint(endpoint: str, known_ids: set[str]) -> bool:
    return endpoint in known_ids or endpoint.startswith("input:")


def _is_coarse_data_edge(source: str, target: str, kind_by_id: dict[str, str]) -> bool:
    coarse_kinds = {
        "model",
        "embedding",
        "encoder",
        "decoder",
        "stack",
        "transformer_block",
        "attention",
        "cross_attention",
        "feed_forward",
        "normalization",
        "pooler",
        "repeat",
    }
    return kind_by_id.get(source) in coarse_kinds and kind_by_id.get(target) in coarse_kinds


def _snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.lower()


# Architecture-defining scalar config fields worth carrying in the compact template.
# Deliberately a small whitelist of shape/behaviour knobs: we do NOT dump the full config
# (label maps, token ids, runtime flags, version strings, nested dicts) into the artifact.
_SALIENT_CONFIG_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_hidden_layers",
    "num_layers",
    "num_decoder_layers",
    "max_position_embeddings",
    "sliding_window",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "n_routed_experts",
    "hidden_act",
    "is_encoder_decoder",
    "tie_word_embeddings",
)

_CONFIG_EXPR_FIELD_RE = re.compile(r"\bconfig\.([A-Za-z_][A-Za-z0-9_]*)")


def _referenced_config_field_names(expressions: list[str | None]) -> list[str]:
    """Config field names referenced by config expressions in the artifact (e.g. count_expr)."""
    names: list[str] = []
    seen: set[str] = set()
    for expression in expressions:
        if not isinstance(expression, str):
            continue
        for name in _CONFIG_EXPR_FIELD_RE.findall(expression):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _config_payload(config: Any, referenced_field_names: list[str]) -> dict[str, Any]:
    fields = config.to_dict() if hasattr(config, "to_dict") else dict(config.__dict__)

    referenced = {name: _json_safe(fields[name]) for name in referenced_field_names if name in fields}
    salient = {
        name: _json_safe(fields[name])
        for name in _SALIENT_CONFIG_FIELDS
        if name in fields and _is_simple_value(fields[name])
    }
    return {
        "class_name": config.__class__.__name__,
        "module": config.__class__.__module__,
        "model_type": getattr(config, "model_type", None),
        # Config knobs the artifact is parameterized by (referenced by config expressions),
        # with their default values as observed at generation time.
        "referenced_fields": referenced,
        # Curated architecture-defining scalar defaults; NOT the full config.
        "salient_fields": salient,
    }


def _hierarchy_payload(components: list[Component]) -> dict[str, Any]:
    return {
        "root": "root",
        "children": {component.id: component.children for component in components if component.children},
    }


def _component_id(path: str) -> str:
    return "root" if path == "" else f"module:{path}"


def _module_attributes(module: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    class_name = module.__class__.__name__

    if class_name == "Embedding":
        attributes.update(
            {
                "num_embeddings": getattr(module, "num_embeddings", None),
                "embedding_dim": getattr(module, "embedding_dim", None),
                "padding_idx": getattr(module, "padding_idx", None),
            }
        )
    elif class_name == "Linear":
        attributes.update(
            {
                "in_features": getattr(module, "in_features", None),
                "out_features": getattr(module, "out_features", None),
                "bias": getattr(module, "bias", None) is not None,
            }
        )
    elif class_name == "Dropout":
        attributes["p"] = getattr(module, "p", None)

    if hasattr(module, "__len__") and class_name in {"ModuleList", "Sequential"}:
        attributes["length"] = len(module)

    for attr_name in (
        "hidden_size",
        "num_heads",
        "num_key_value_heads",
        "head_dim",
        "eps",
        "normalized_shape",
        "is_decoder",
    ):
        if hasattr(module, attr_name):
            value = getattr(module, attr_name)
            if _is_simple_value(value):
                attributes[attr_name] = value

    return _json_safe(attributes)


def _json_safe(value: Any) -> Any:
    if _is_simple_value(value):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return repr(value)


def _is_simple_value(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)
