"""Experimental local architecture IR generation helpers."""

from .introspection import build_architecture_artifact, build_expanded_architecture_artifact
from .modular_graph import GRAPH_SCHEMA_VERSION, build_modular_graph
from .resolver import (
    ConfigExpressionError,
    evaluate_config_expression,
    resolve_architecture,
    resolve_template_to_graph,
)
from .semantic_model import (
    RESOLVED_GRAPH_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TEMPLATE_SCHEMA_VERSION,
)


__all__ = [
    "SCHEMA_VERSION",
    "TEMPLATE_SCHEMA_VERSION",
    "RESOLVED_GRAPH_SCHEMA_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "ConfigExpressionError",
    "build_architecture_artifact",
    "build_expanded_architecture_artifact",
    "build_modular_graph",
    "evaluate_config_expression",
    "resolve_architecture",
    "resolve_template_to_graph",
]
