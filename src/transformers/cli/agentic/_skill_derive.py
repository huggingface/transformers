# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auto-generate a ``skill.json`` manifest by introspecting the CLI argument
parsers already defined in ``src/transformers/cli/agentic/``.

Each command emits its result via a huggingface_hub ``Output`` primitive
(``table``/``result``/``dict``) or the local ``answer()`` helper. With
``--format json``, stdout is the JSON shape documented in
``output_schemas.OUTPUT_SCHEMAS`` for each capability.

Usage::

    from transformers.cli.agentic._skill_derive import derive_skill_from_cli
    manifest = derive_skill_from_cli()

    # or:
    python -m transformers.cli.agentic._skill_derive > skill.json
"""

import inspect
import json
import sys
from typing import Annotated, Any, get_args, get_origin

import typer

from .app import register_agentic_commands
from .output_schemas import OUTPUT_SCHEMAS


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json(py_type: type) -> dict:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = get_origin(py_type)

    if origin is Annotated:
        return _python_type_to_json(get_args(py_type)[0])

    if origin is not None:
        args = get_args(py_type)
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        if len(non_none) == 1:
            inner = _python_type_to_json(non_none[0])
            if has_none and isinstance(inner.get("type"), str):
                inner["type"] = [inner["type"], "null"]
            return inner
        if origin is list:
            return {"type": "array", "items": _python_type_to_json(args[0])} if args else {"type": "array"}

    json_type = _TYPE_MAP.get(py_type)
    if json_type:
        return {"type": json_type}
    return {"type": "string", "description": f"Unhandled Python type: {py_type}"}


def _extract_param_schema(func) -> dict[str, Any]:
    """Extract typed parameters from a Typer command function."""
    inputs: dict[str, Any] = {}
    for name, param in inspect.signature(func).parameters.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            inputs[name] = {"type": "string", "description": "GAP: untyped parameter"}
            continue

        schema = _python_type_to_json(annotation)
        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default

        if get_origin(annotation) is Annotated:
            for arg in get_args(annotation)[1:]:
                if isinstance(arg, (typer.models.OptionInfo, typer.models.ArgumentInfo)) and arg.help:
                    schema["description"] = arg.help

        inputs[name] = schema
    return inputs


def _invocation(cli_name: str, func) -> str:
    parts = [f"transformers {cli_name}"]
    for name, param in inspect.signature(func).parameters.items():
        dashed = name.replace("_", "-")
        if param.default is inspect.Parameter.empty:
            parts.append(f"<{dashed}>")
        else:
            parts.append(f"--{dashed} <{name}>")
    return " ".join(parts)


def derive_skill_from_cli() -> dict[str, Any]:
    """Introspect the agentic CLI commands and produce a skill manifest."""
    app = typer.Typer()
    register_agentic_commands(app)

    capabilities: list[dict[str, Any]] = []
    for reg in app.registered_commands:
        func = reg.callback
        if func is None:
            continue
        cli_name = reg.name or func.__name__.replace("_", "-")
        doc = inspect.getdoc(func) or ""
        description = doc.split("\n", 1)[0].strip()
        outputs = OUTPUT_SCHEMAS.get(cli_name, {"type": "object", "description": "GAP: output schema not defined"})
        capabilities.append(
            {
                "id": cli_name,
                "description": description,
                "inputs": _extract_param_schema(func),
                "outputs": outputs,
                "invocation": {"cli": _invocation(cli_name, func)},
            }
        )

    return {
        "name": "transformers",
        "description": (
            "Agent-invokable Transformers commands. Pass `--format json` at the top level "
            "(e.g. `transformers --format json classify ...`) to receive the structured output "
            "documented in each capability's `outputs` schema."
        ),
        "capabilities": capabilities,
    }


def main():
    manifest = derive_skill_from_cli()
    gaps = [
        cap["id"]
        for cap in manifest["capabilities"]
        if cap["outputs"].get("description", "").startswith("GAP:")
        or any(v.get("description", "").startswith("GAP:") for v in cap["inputs"].values())
    ]
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if gaps:
        sys.stderr.write(f"\n{len(gaps)} capability/capabilities with gaps: {', '.join(gaps)}\n")


if __name__ == "__main__":
    main()
