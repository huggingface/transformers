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

Usage (programmatic)::

    from transformers.cli.agentic._skill_derive import derive_skill_from_cli
    manifest = derive_skill_from_cli()

Usage (CLI)::

    python -m transformers.cli.agentic._skill_derive

The function inspects every command registered in ``app.py`` via Typer,
extracts typed parameters and docstrings, and produces a JSON manifest
conforming to the skill-envelope schema.
"""

import inspect
import json
from pathlib import Path
from typing import Annotated, Any, get_args, get_origin

import typer

from .app import register_agentic_commands


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_COMMON_OUTPUTS = {
    "classify": {
        "type": "object",
        "properties": {
            "labels": {"type": "array", "items": {"type": "string"}},
            "scores": {"type": "array", "items": {"type": "number"}},
        },
    },
    "ner": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "entity_group": {"type": "string"},
                "score": {"type": "number"},
                "word": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
        },
    },
    "token-classify": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "score": {"type": "number"},
                "word": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
        },
    },
    "qa": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "number"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
        },
    },
    "table-qa": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "coordinates": {"type": "array"},
            "cells": {"type": "array", "items": {"type": "string"}},
            "aggregator": {"type": "string"},
        },
    },
    "summarize": {
        "type": "array",
        "items": {"type": "object", "properties": {"summary_text": {"type": "string"}}},
    },
    "translate": {
        "type": "array",
        "items": {"type": "object", "properties": {"translation_text": {"type": "string"}}},
    },
    "fill-mask": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "token": {"type": "integer"},
                "token_str": {"type": "string"},
                "sequence": {"type": "string"},
            },
        },
    },
    "generate": {"type": "string", "description": "Generated text."},
    "detect-watermark": {
        "type": "object",
        "properties": {
            "prediction": {"type": "string"},
            "confidence": {"type": "number"},
        },
    },
    "image-classify": {
        "type": "array",
        "items": {"type": "object", "properties": {"label": {"type": "string"}, "score": {"type": "number"}}},
    },
    "detect": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "score": {"type": "number"},
                "box": {
                    "type": "object",
                    "properties": {
                        "xmin": {"type": "number"},
                        "ymin": {"type": "number"},
                        "xmax": {"type": "number"},
                        "ymax": {"type": "number"},
                    },
                },
            },
        },
    },
    "segment": {
        "type": "object",
        "description": "Semantic segmentation class ratios or SAM mask metadata.",
    },
    "depth": {
        "type": "object",
        "properties": {
            "size": {"type": "string", "description": "Depth map dimensions, e.g. '480x640'."},
            "output_path": {"type": "string", "description": "Path to saved PNG if --output was given."},
        },
    },
    "keypoints": {"type": "object", "description": "Keypoint matching result (schema depends on pipeline output)."},
    "video-classify": {
        "type": "array",
        "items": {"type": "object", "properties": {"label": {"type": "string"}, "score": {"type": "number"}}},
    },
    "transcribe": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    },
    "audio-classify": {
        "type": "array",
        "items": {"type": "object", "properties": {"label": {"type": "string"}, "score": {"type": "number"}}},
    },
    "speak": {
        "type": "object",
        "properties": {"output_path": {"type": "string"}},
    },
    "audio-generate": {
        "type": "object",
        "properties": {"output_path": {"type": "string"}},
    },
    "vqa": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    },
    "document-qa": {
        "type": "object",
        "properties": {"answer": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
    },
    "caption": {
        "type": "object",
        "properties": {"caption": {"type": "string"}},
    },
    "ocr": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    },
    "multimodal-chat": {"type": "string", "description": "Generated text from multimodal conversation."},
    "embed": {
        "type": "object",
        "properties": {
            "shape": {"type": "array", "items": {"type": "integer"}},
            "preview": {"type": "array", "items": {"type": "number"}},
        },
    },
    "tokenize": {
        "type": "object",
        "properties": {
            "tokens": {"type": "array", "items": {"type": "string"}},
            "token_ids": {"type": "array", "items": {"type": "integer"}},
            "num_tokens": {"type": "integer"},
        },
    },
    "inspect": {"type": "object", "description": "Model configuration dictionary."},
    "inspect-forward": {
        "type": "object",
        "description": "Attention and hidden-state shape/stat summary per layer.",
    },
    "benchmark-quantization": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "tokens_per_sec": {"type": "number"},
                "time_sec": {"type": "number"},
                "peak_memory_mb": {"type": "number"},
                "output_preview": {"type": "string"},
            },
        },
    },
    "train": {
        "type": "object",
        "properties": {
            "output_dir": {"type": "string"},
            "metrics": {"type": "object", "description": "Training metrics from the Trainer."},
        },
    },
    "quantize": {
        "type": "object",
        "properties": {"output_path": {"type": "string"}},
    },
    "export": {
        "type": "object",
        "properties": {"output_path": {"type": "string"}, "format": {"type": "string"}},
    },
}


def _python_type_to_json(py_type: type) -> dict:
    """Convert a Python type annotation to a JSON Schema fragment.

    Handles ``str | None`` (Optional), plain types, and ``Annotated`` wrappers.
    """
    origin = get_origin(py_type)

    if origin is Annotated:
        inner = get_args(py_type)[0]
        return _python_type_to_json(inner)

    if origin is type(None):
        return {"type": "null"}

    # Handle Union (e.g. str | None -> Optional[str])
    if origin is not None:
        args = get_args(py_type)
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        if len(non_none) == 1:
            schema = _python_type_to_json(non_none[0])
            if has_none:
                schema["nullable"] = True
            return schema
        if len(non_none) > 1:
            return {"type": "string", "nullable": has_none}

    # Handle list types
    if origin is list:
        args = get_args(py_type)
        if args:
            return {"type": "array", "items": _python_type_to_json(args[0])}
        return {"type": "array"}

    json_type = _TYPE_MAP.get(py_type)
    if json_type:
        return {"type": json_type}

    return {"type": "string", "description": f"Unhandled Python type: {py_type}"}


def _extract_param_schema(func: callable) -> dict[str, Any]:
    """Extract typed parameters from a Typer command function.

    Returns a dict of ``{param_name: json_schema}`` suitable for the
    ``inputs`` field of a skill capability.
    """
    sig = inspect.signature(func)
    inputs: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name == "output_json":
            continue

        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            if param.default is not inspect.Parameter.empty:
                inputs[name] = {"type": "string", "nullable": True, "default": param.default}
            else:
                inputs[name] = {"type": "string", "description": "GAP: untyped parameter"}
            continue

        schema = _python_type_to_json(annotation)

        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default

        # Extract help text from Annotated metadata (typer.Option)
        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            for arg in args[1:]:
                if isinstance(arg, typer.models.OptionInfo):
                    if arg.help:
                        schema["description"] = arg.help
                elif isinstance(arg, typer.models.ArgumentInfo):
                    if arg.help:
                        schema["description"] = arg.help

        inputs[name] = schema

    return inputs


def _get_cli_name(func: callable, registered_name: str | None = None) -> str:
    return registered_name or func.__name__.replace("_", "-")


def _get_description(func: callable) -> str:
    doc = inspect.getdoc(func)
    if doc:
        first_line = doc.split("\n")[0].strip()
        if first_line:
            return first_line
    return ""


def _get_invocation(cli_name: str, func: callable) -> str:
    sig = inspect.signature(func)
    parts = [f"transformers {cli_name}"]

    for name, param in sig.parameters.items():
        if name == "output_json":
            continue
        if param.default is not inspect.Parameter.empty:
            parts.append(f"--{name.replace('_', '-')} <{name}>")
        else:
            parts.append(f"<{name.replace('_', '-')}>")

    return " ".join(parts)


def derive_skill_from_cli() -> dict[str, Any]:
    """Introspect the agentic CLI commands and produce a skill manifest.

    The manifest follows the envelope::

        {
          "name": "transformers",
          "capabilities": [
            {
              "id": "...",
              "description": "...",
              "inputs": {...},
              "outputs": {...},
              "invocation": {"cli": "..."}
            }
          ]
        }

    Gaps in typing are flagged explicitly in the ``description`` field of
    the affected input schema with the prefix ``"GAP:"`` so they are easy
    to find and fix.
    """
    app = typer.Typer()
    register_agentic_commands(app)

    capabilities: list[dict[str, Any]] = []

    # We introspect the original functions from the agentic submodules
    # by iterating the registered commands on the Typer app.
    for registration in app.registered_commands:
        func = registration.callback
        if func is None:
            continue

        cli_name = registration.name or func.__name__.replace("_", "-")
        description = _get_description(func)
        inputs = _extract_param_schema(func)
        outputs = _COMMON_OUTPUTS.get(cli_name, {"type": "object", "description": "GAP: output schema not defined"})
        invocation = _get_invocation(cli_name, func)

        capabilities.append(
            {
                "id": cli_name,
                "description": description,
                "inputs": inputs,
                "outputs": outputs,
                "invocation": {"cli": invocation},
            }
        )

    return {
        "name": "transformers",
        "capabilities": capabilities,
    }


def main():
    out_dir = Path(__file__).resolve().parent.parent.parent / ".agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "skill.json"

    manifest = derive_skill_from_cli()
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"skill.json written to {out_path}")

    # Report gaps
    gap_count = 0
    for cap in manifest["capabilities"]:
        for key, schema in cap["inputs"].items():
            desc = schema.get("description", "")
            if desc.startswith("GAP:"):
                gap_count += 1
                print(f"  GAP: {cap['id']}.{key} — {desc}")
        out_desc = cap["outputs"].get("description", "")
        if out_desc.startswith("GAP:"):
            gap_count += 1
            print(f"  GAP: {cap['id']}.outputs — {out_desc}")

    if gap_count:
        print(f"\n{gap_count} gap(s) found. These should be fixed in the CLI definitions.")
    else:
        print("No typing gaps detected.")


if __name__ == "__main__":
    main()
