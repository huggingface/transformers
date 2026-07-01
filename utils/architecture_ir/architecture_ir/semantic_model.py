"""Dataclasses for the experimental architecture IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "architecture-artifact-v0"


@dataclass
class Component:
    id: str
    name: str
    path: str
    parent: str | None
    kind: str
    class_name: str
    module: str
    children: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Repeat:
    id: str
    component_id: str
    path: str
    repeated_class_name: str
    count: int
    expression: str
    count_source: str
    repeated_component_ids: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    id: str
    source: str
    target: str
    kind: str
    description: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureArtifact:
    schema_version: str
    model_type: str
    entrypoints: dict[str, Any]
    config: dict[str, Any]
    components: list[Component]
    semantic_components: list[dict[str, Any]]
    hierarchy: dict[str, Any]
    repeats: list[Repeat]
    edges: list[Edge]
    provenance: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
