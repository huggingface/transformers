"""Resolve model types to local config/model classes without checkpoint loading.

This module hosts the two distinct "resolve" steps of the IR pipeline:

* :func:`resolve_architecture` maps a ``model_type`` to its local config/model
  classes and instantiates the model from configuration only. It feeds the
  *generator* that emits the ``ArchitectureTemplate``.
* :func:`resolve_template_to_graph` combines an ``ArchitectureTemplate`` with a
  checkpoint ``config.json`` to produce a ``ResolvedGraph`` with evaluated
  configuration values. It feeds downstream *consumers* (viewers, diffs, docs).
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .semantic_model import RESOLVED_GRAPH_SCHEMA_VERSION


@dataclass
class ResolvedArchitecture:
    model_type: str
    config: Any
    model: Any
    entrypoints: dict[str, Any]
    warnings: list[str]


def resolve_architecture(model_type: str) -> ResolvedArchitecture:
    try:
        import torch

        from transformers.initialization import no_init_weights
        from transformers.models.auto.configuration_auto import AutoConfig
        from transformers.models.auto.modeling_auto import AutoModel
    except Exception as error:
        raise RuntimeError(
            "Architecture IR generation requires a local Transformers development environment with torch available."
        ) from error

    config = AutoConfig.for_model(model_type)

    with torch.device("meta"), no_init_weights():
        model = AutoModel.from_config(config)

    return ResolvedArchitecture(
        model_type=model_type,
        config=config,
        model=model,
        entrypoints={
            "config_class": {
                "name": config.__class__.__name__,
                "module": config.__class__.__module__,
            },
            "model_class": {
                "name": model.__class__.__name__,
                "module": model.__class__.__module__,
            },
            "auto_config": "transformers.models.auto.configuration_auto.AutoConfig.for_model",
            "auto_model": "transformers.models.auto.modeling_auto.AutoModel.from_config",
        },
        warnings=[],
    )


class ConfigExpressionError(ValueError):
    """Raised when a config expression is not a supported v0 expression."""


# Binary/unary operators allowed inside config expressions. Config expressions are
# a small arithmetic DSL over ``config.<field>`` references and numeric literals;
# they are intentionally NOT arbitrary Python (no calls, names, subscripts, ...).
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def evaluate_config_expression(expression: str, config: Mapping[str, Any]) -> Any:
    """Safely evaluate a v0 config expression string against a config mapping.

    Supported grammar: ``config.<field>`` references, integer/float literals, and
    the arithmetic operators ``+ - * / // % **`` with parentheses. Anything else
    (function calls, arbitrary names, attribute chains, subscripts) is rejected.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ConfigExpressionError(f"Invalid config expression: {expression!r}") from error
    return _eval_node(tree.body, expression, config)


def _eval_node(node: ast.AST, expression: str, config: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config":
        if node.attr not in config:
            raise ConfigExpressionError(f"Config field {node.attr!r} referenced by {expression!r} is missing.")
        return config[node.attr]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, expression, config)
        right = _eval_node(node.right, expression, config)
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, expression, config))
    raise ConfigExpressionError(f"Unsupported config expression: {expression!r}")


def resolve_template_to_graph(
    template: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_source: str | None = None,
) -> dict[str, Any]:
    """Resolve an ``ArchitectureTemplate`` against a checkpoint config into a ``ResolvedGraph``.

    The resolved graph stays compact and symbolic: repeats keep a single body and
    are *not* expanded into one component per instance. Only config-parametric
    values (currently repeat counts) are evaluated to concrete numbers. The
    originating ``count_expr`` is preserved for provenance.
    """
    warnings: list[str] = []
    referenced_fields: dict[str, Any] = {}

    resolved_repeats = []
    for repeat in template.get("repeats", []):
        resolved_repeat = dict(repeat)
        expression = repeat.get("count_expr")
        if expression is not None:
            try:
                value = evaluate_config_expression(expression, config)
                resolved_repeat["count"] = value
                resolved_repeat["count_resolved"] = True
                for field_name in _referenced_config_fields(expression):
                    if field_name in config:
                        referenced_fields[field_name] = config[field_name]
            except ConfigExpressionError as error:
                resolved_repeat["count_resolved"] = False
                warnings.append(str(error))
        resolved_repeats.append(resolved_repeat)

    model_type = template.get("model_type")
    return {
        "schema_version": RESOLVED_GRAPH_SCHEMA_VERSION,
        "model_type": model_type,
        "metadata": {
            "level": "resolved_graph",
            "format": "compact_resolved",
            "description": (
                "ResolvedGraph produced by evaluating an ArchitectureTemplate against a checkpoint config.json. "
                "Repeats remain symbolic; only config-parametric values are evaluated."
            ),
        },
        "template_ref": {
            "schema_version": template.get("schema_version"),
            "model_type": model_type,
        },
        "config": {
            "source": config_source,
            "model_type": config.get("model_type", model_type),
            "referenced_fields": referenced_fields,
        },
        "components": list(template.get("components", [])),
        "templates": list(template.get("templates", [])),
        "repeats": resolved_repeats,
        "edges": list(template.get("edges", [])),
        "provenance": {
            "generator": "utils/architecture_ir",
            "resolution": "resolve_template_to_graph",
            "template_schema_version": template.get("schema_version"),
            "schema_version": RESOLVED_GRAPH_SCHEMA_VERSION,
        },
        "warnings": warnings,
    }


def _referenced_config_fields(expression: str) -> list[str]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return []
    fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config":
            fields.append(node.attr)
    return fields
