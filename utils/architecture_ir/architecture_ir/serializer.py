"""JSON serialization for architecture IR artifacts and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .introspection import artifact_to_json_dict


def write_artifact(artifact: Any, path: Path) -> None:
    write_json(path, artifact_to_json_dict(artifact))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
