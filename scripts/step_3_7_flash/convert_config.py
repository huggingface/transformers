"""Convert a Step-3.7-Flash config.json to the standardized hub format.

Handles two legacy patterns found in original checkpoints:

1. moe_layers_enum (comma-separated string or list of ints) → mlp_layer_types
   (per-layer list of "dense" / "sparse" strings)

2. Per-layer lists that are longer or shorter than num_hidden_layers — some
   checkpoints include extra entries for MTP/speculative-decode layers after the
   decoder stack; others omit trailing entries and rely on repetition of the last
   value. Both are normalised to exactly num_hidden_layers entries.

The script operates directly on config.json and handles both the flat format
(text-config fields at the top level, model_type "step3p5") and the nested
format (text-config under "text_config", model_type "step3p7").

Usage:
    # in-place update
    python scripts/step_3_7_flash/convert_config.py --config-dir /path/to/model

    # write to a separate directory
    python scripts/step_3_7_flash/convert_config.py \\
        --config-dir /path/to/model --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

PER_LAYER_FIELDS = [
    "layer_types",
    "mlp_layer_types",
    "swiglu_limits",
    "swiglu_limits_shared",
    "use_rope_layers",
    "yarn_only_types",
]


def _normalize(values: list[Any], n: int) -> list[Any]:
    if len(values) < n:
        values = values + [values[-1]] * (n - len(values))
    return values[:n]


def convert_text_cfg(cfg: dict) -> dict:
    cfg = dict(cfg)
    n = cfg.get("num_hidden_layers", 45)

    # moe_layers_enum → mlp_layer_types
    if "moe_layers_enum" in cfg and "mlp_layer_types" not in cfg:
        raw = cfg.pop("moe_layers_enum")
        if isinstance(raw, str):
            moe_set = {int(i) for i in raw.split(",") if i.strip()}
        else:
            moe_set = {int(i) for i in raw}
        cfg["mlp_layer_types"] = ["sparse" if i in moe_set else "dense" for i in range(n)]
    else:
        cfg.pop("moe_layers_enum", None)

    # normalise per-layer lists
    for field in PER_LAYER_FIELDS:
        if field in cfg and cfg[field]:
            cfg[field] = _normalize(cfg[field], n)

    return cfg


def convert(config_path: Path) -> dict:
    raw = json.loads(config_path.read_text())

    if "text_config" in raw:
        raw["text_config"] = convert_text_cfg(raw["text_config"])
    else:
        raw = convert_text_cfg(raw)

    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Step-3.7-Flash config to standardised format")
    parser.add_argument("--config-dir", type=Path, required=True, help="Directory containing config.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write converted config here (default: update in-place)",
    )
    args = parser.parse_args()

    src = args.config_dir / "config.json"
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = args.output_dir or args.config_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_dir and args.output_dir != args.config_dir:
        for f in args.config_dir.iterdir():
            if f.name != "config.json":
                shutil.copy2(f, out_dir / f.name)

    converted = convert(src)
    dst = out_dir / "config.json"
    dst.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
