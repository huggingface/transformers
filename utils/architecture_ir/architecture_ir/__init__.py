"""Experimental local architecture IR generation helpers."""

from .introspection import build_architecture_artifact, build_expanded_architecture_artifact
from .resolver import resolve_architecture
from .semantic_model import SCHEMA_VERSION


__all__ = [
    "SCHEMA_VERSION",
    "build_architecture_artifact",
    "build_expanded_architecture_artifact",
    "resolve_architecture",
]
