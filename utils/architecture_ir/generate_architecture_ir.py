#!/usr/bin/env python
"""Generate experimental local architecture IR artifacts for Transformers model types."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from architecture_ir import (  # noqa: E402
    SCHEMA_VERSION,
    build_architecture_artifact,
    build_expanded_architecture_artifact,
    build_modular_graph,
    resolve_architecture,
)
from architecture_ir.serializer import write_artifact, write_json  # noqa: E402


MANIFEST_SCHEMA_VERSION = "architecture-ir-manifest-v0"


def generate_architecture_ir(
    architectures: list[str],
    output_dir: str | Path,
    *,
    debug_expanded: bool = False,
    modular_graph: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    artifacts_path = output_path / "artifacts"
    artifacts_path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifacts_dir": "artifacts",
        "architectures": [],
        "failures": [],
    }

    for model_type in architectures:
        try:
            resolved = resolve_architecture(model_type)
            artifact = build_architecture_artifact(model_type, resolved)
            artifact_name = f"{_safe_artifact_stem(model_type)}.json"
            artifact_path = artifacts_path / artifact_name
            write_artifact(artifact, artifact_path)
            entry = {
                "model_type": model_type,
                "status": "ok",
                "artifact": f"artifacts/{artifact_name}",
                "config_class": resolved.config.__class__.__name__,
                "model_class": resolved.model.__class__.__name__,
            }
            if debug_expanded:
                expanded_artifact = build_expanded_architecture_artifact(model_type, resolved)
                expanded_path = output_path / "debug" / "expanded" / artifact_name
                write_artifact(expanded_artifact, expanded_path)
                entry["debug_expanded_artifact"] = f"debug/expanded/{artifact_name}"
            manifest["architectures"].append(entry)
        except Exception as error:
            manifest["failures"].append(
                {
                    "model_type": model_type,
                    "status": "failed",
                    "error_type": error.__class__.__name__,
                    "error": str(error),
                }
            )

    if modular_graph:
        # Library-wide inheritance forest, independent of the requested architectures.
        write_json(output_path / "modular_graph.json", build_modular_graph())
        manifest["modular_graph"] = "modular_graph.json"

    write_json(output_path / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        required=True,
        nargs="+",
        help="One or more Transformers model_type values, for example: llama bert t5.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where manifest and artifacts will be written.")
    parser.add_argument(
        "--debug-expanded",
        action="store_true",
        help="Also write the full expanded module dump under debug/expanded/.",
    )
    parser.add_argument(
        "--modular-graph",
        action="store_true",
        help="Also write modular_graph.json: the library-wide modular inheritance forest.",
    )
    args = parser.parse_args(argv)

    manifest = generate_architecture_ir(
        args.architectures,
        args.output_dir,
        debug_expanded=args.debug_expanded,
        modular_graph=args.modular_graph,
    )
    manifest_path = Path(args.output_dir) / "manifest.json"
    print(f"Wrote {manifest_path}")
    if "modular_graph" in manifest:
        print(f"Wrote {Path(args.output_dir) / manifest['modular_graph']}")

    for entry in manifest["architectures"]:
        print(f"Wrote {Path(args.output_dir) / entry['artifact']}")
        if "debug_expanded_artifact" in entry:
            print(f"Wrote {Path(args.output_dir) / entry['debug_expanded_artifact']}")

    if manifest["failures"]:
        for failure in manifest["failures"]:
            print(f"Failed {failure['model_type']}: {failure['error']}", file=sys.stderr)
        return 1
    return 0


def _safe_artifact_stem(model_type: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_type).strip("._") or "architecture"


if __name__ == "__main__":
    raise SystemExit(main())
