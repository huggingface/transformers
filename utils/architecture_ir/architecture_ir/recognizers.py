"""Reusable semantic recognizers for module-tree architecture IR."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .semantic_model import Component, Edge, Repeat


# Edge kinds defined by architecture-template-v0 / resolved-graph-v0. The v0
# generator emits the first five for llama/bert/t5; "route" (MoE dispatch) and
# "cache_read"/"cache_write" (KV-cache dataflow) are part of the contract and
# reserved for recognizers that will emit them for MoE and cached-attention models.
EDGE_KINDS = (
    "data",
    "residual",
    "mask",
    "position",
    "cross_attention",
    "route",
    "cache_read",
    "cache_write",
)

_COUNT_FIELD_PREFERENCES = {
    "decoder": (
        "num_decoder_layers",
        "decoder_layers",
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    ),
    "encoder": (
        "num_encoder_layers",
        "encoder_layers",
        "num_layers",
        "num_hidden_layers",
        "n_layer",
    ),
    "default": (
        "num_hidden_layers",
        "num_layers",
        "n_layer",
        "num_decoder_layers",
        "encoder_layers",
        "decoder_layers",
    ),
}


def classify_component(path: str, class_name: str) -> str:
    path_lower = path.lower()
    class_lower = class_name.lower()
    semantic_key = f"{path_lower}.{class_lower}"

    if path == "":
        return "model"
    if class_name == "ModuleList":
        return "repeated_container"
    if "crossattention" in semantic_key or "cross_attention" in semantic_key or "encdecattention" in semantic_key:
        return "cross_attention"
    if any(token in semantic_key for token in ("position", "rotary", "relative_attention_bias")):
        return "position"
    if "embedding" in class_lower or path_lower.endswith("embed_tokens") or path_lower == "shared":
        return "embedding"
    if "attention" in class_lower or path_lower.endswith("attention") or path_lower.endswith("self_attn"):
        return "attention"
    if _looks_like_feed_forward(path_lower, class_lower):
        return "feed_forward"
    if "layernorm" in class_lower or "rmsnorm" in class_lower or class_lower.endswith("norm"):
        return "normalization"
    if class_lower == "dropout":
        return "dropout"
    if class_lower in {"linear", "conv1d", "conv2d"} or class_lower.endswith("linear"):
        return "projection"
    if "activation" in class_lower or path_lower.endswith("act_fn"):
        return "activation"
    if "pooler" in class_lower or path_lower.endswith("pooler"):
        return "pooler"
    if _looks_like_transformer_block(class_lower):
        return "transformer_block"
    if path_lower.endswith("encoder") or class_lower.endswith("encoder"):
        return "encoder"
    if path_lower.endswith("decoder") or class_lower.endswith("decoder"):
        return "decoder"
    if class_lower.endswith("stack"):
        return "stack"
    return "module"


def detect_repeats(components: list[Component], config: Any) -> list[Repeat]:
    by_id = {component.id: component for component in components}
    config_counts = _config_count_fields(config)
    repeats: list[Repeat] = []

    for component in components:
        if component.kind != "repeated_container" or len(component.children) < 2:
            continue

        child_classes = [by_id[child_id].class_name for child_id in component.children]
        if len(set(child_classes)) != 1:
            continue

        count = len(component.children)
        field_name = _infer_count_field(component.path, count, config_counts)
        expression = f"config.{field_name}" if field_name is not None else str(count)
        count_source = "config" if field_name is not None else "module_tree"

        repeats.append(
            Repeat(
                id=f"repeat:{component.path or 'root'}",
                component_id=component.id,
                path=component.path,
                repeated_class_name=child_classes[0],
                count=count,
                expression=expression,
                count_source=count_source,
                repeated_component_ids=list(component.children),
                provenance={
                    "component_id": component.id,
                    "module_path": component.module,
                    "class_name": component.class_name,
                    "source": "module_tree",
                },
            )
        )

    return repeats


def build_edges(components: list[Component]) -> list[Edge]:
    by_id = {component.id: component for component in components}
    edges: list[Edge] = []
    seen: set[tuple[str, str, str]] = set()

    def add_edge(source: str, target: str, kind: str, description: str, component: Component | None = None) -> None:
        if kind not in EDGE_KINDS:
            raise ValueError(f"Unsupported edge kind: {kind}")
        key = (source, target, kind)
        if key in seen:
            return
        seen.add(key)
        provenance = {"source": "semantic_recognizer"}
        if component is not None:
            provenance.update(
                {
                    "component_id": component.id,
                    "module_path": component.module,
                    "class_name": component.class_name,
                }
            )
        edges.append(
            Edge(
                id=f"edge:{len(edges)}",
                source=source,
                target=target,
                kind=kind,
                description=description,
                provenance=provenance,
            )
        )

    for component in components:
        ordered_children = [
            by_id[child_id] for child_id in component.children if by_id[child_id].kind not in {"position", "dropout"}
        ]
        for source, target in zip(ordered_children, ordered_children[1:]):
            add_edge(
                source.id,
                target.id,
                "data",
                "Coarse data edge inferred from module child order.",
                component,
            )

    attention_components = _components_with_kinds(components, {"attention", "cross_attention"})
    for component in attention_components:
        add_edge(
            "input:attention_mask",
            component.id,
            "mask",
            "Attention modules consume an attention mask when the model forward provides one.",
            component,
        )

    for block in _components_with_kinds(components, {"transformer_block"}):
        descendants = _descendants(block, by_id)
        for target in _first_by_kind(descendants, {"attention", "cross_attention", "feed_forward"}):
            add_edge(
                block.id,
                target.id,
                "residual",
                "Residual path inferred from transformer block semantics.",
                target,
            )

    for position_component in _components_with_kinds(components, {"position"}):
        for target in _position_targets(position_component, components):
            add_edge(
                position_component.id,
                target.id,
                "position",
                "Position information is consumed by embeddings or attention.",
                position_component,
            )

    encoder = _first_component(components, {"encoder"})
    for component in _components_with_kinds(components, {"cross_attention"}):
        add_edge(
            encoder.id if encoder is not None else "input:encoder_hidden_states",
            component.id,
            "cross_attention",
            "Decoder cross-attention consumes encoder hidden states.",
            component,
        )

    return edges


def summarize_semantic_components(components: Iterable[Component]) -> list[dict[str, Any]]:
    return [
        {
            "id": component.id,
            "kind": component.kind,
            "path": component.path,
            "class_name": component.class_name,
        }
        for component in components
        if component.kind != "module"
    ]


def _looks_like_transformer_block(class_lower: str) -> bool:
    return (
        class_lower.endswith("block")
        or class_lower.endswith("layer")
        or class_lower.endswith("decoderlayer")
        or class_lower.endswith("encoderlayer")
    )


def _looks_like_feed_forward(path_lower: str, class_lower: str) -> bool:
    tokens = ("mlp", "feedforward", "feed_forward", "layerff", "denseactdense", "ffn", "intermediate")
    if any(token in path_lower or token in class_lower for token in tokens):
        return True
    return path_lower.endswith(".output") and ".attention." not in path_lower


def _config_count_fields(config: Any) -> dict[str, int]:
    fields: dict[str, int] = {}
    if hasattr(config, "to_dict"):
        for key, value in config.to_dict().items():
            if isinstance(value, int) and not isinstance(value, bool):
                fields[key] = value

    for field_names in _COUNT_FIELD_PREFERENCES.values():
        for field_name in field_names:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                if isinstance(value, int) and not isinstance(value, bool):
                    fields[field_name] = value
    return fields


def _infer_count_field(path: str, count: int, config_counts: dict[str, int]) -> str | None:
    path_lower = path.lower()
    if "decoder" in path_lower:
        groups = ("decoder", "default")
    elif "encoder" in path_lower:
        groups = ("encoder", "default")
    else:
        groups = ("default",)

    for group in groups:
        for field_name in _COUNT_FIELD_PREFERENCES[group]:
            if config_counts.get(field_name) == count:
                return field_name

    for field_name, value in sorted(config_counts.items()):
        if value == count and field_name.startswith("num_") and "layer" in field_name:
            return field_name
    return None


def _components_with_kinds(components: Iterable[Component], kinds: set[str]) -> list[Component]:
    return [component for component in components if component.kind in kinds]


def _first_component(components: Iterable[Component], kinds: set[str]) -> Component | None:
    for component in components:
        if component.kind in kinds:
            return component
    return None


def _descendants(component: Component, by_id: dict[str, Component]) -> list[Component]:
    descendants: list[Component] = []
    stack = list(component.children)
    while stack:
        child_id = stack.pop(0)
        child = by_id[child_id]
        descendants.append(child)
        stack.extend(child.children)
    return descendants


def _first_by_kind(components: Iterable[Component], kinds: set[str]) -> list[Component]:
    found: dict[str, Component] = {}
    for component in components:
        if component.kind in kinds and component.kind not in found:
            found[component.kind] = component
    return [found[kind] for kind in ("attention", "cross_attention", "feed_forward") if kind in found]


def _position_targets(position_component: Component, components: list[Component]) -> list[Component]:
    ancestors = [
        component
        for component in components
        if position_component.path.startswith(f"{component.path}.")
        and component.kind in {"attention", "cross_attention"}
    ]
    if ancestors:
        return [max(ancestors, key=lambda component: len(component.path))]

    parents = [
        component
        for component in components
        if position_component.parent == component.id and component.kind == "embedding"
    ]
    if parents:
        return parents

    if position_component.path.startswith("embeddings."):
        embedding = _first_component(
            (component for component in components if component.path == "embeddings"),
            {"embedding"},
        )
        if embedding is not None:
            return [embedding]

    return _components_with_kinds(components, {"attention", "cross_attention"})
