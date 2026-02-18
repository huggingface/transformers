# Copyright 2026 The HuggingFace Team and NVIDIA CORPORATION. All rights reserved.
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

"""Convert legacy OmniVinci/VILA checkpoints to a standard HF-loadable layout.

This conversion script:
1) rewrites legacy VILA class strings to canonical OmniVinci names,
2) normalizes top-level config fields for local HF loading,
3) merges component safetensors into a top-level `model.safetensors`.

The destination is treated as an export directory and receives only model artifacts
(config/tokenizer/processor/chat-template metadata + merged weights). Source files
under the original repository are never copied verbatim as Python modules.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors.torch import safe_open, save_file


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_SRC_PATH = Path("/fs/nexus-projects/JSALT_workshop/lasha/Dev/omnivinci")
DEFAULT_DST_PATH = Path("/fs/nexus-projects/JSALT_workshop/lasha/Dev/comni")

JSON_FILES_TO_REWRITE = (
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)

TOP_LEVEL_METADATA_FILES = {
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
}

COMPONENT_TO_PREFIX = {
    "llm": "llm",
    "vision_tower": "vision_tower.vision_tower",
    "mm_projector": "mm_projector",
    "sound_tower": "sound_tower.audio_tower",
    "sound_mm_projector": "sound_mm_projector",
}

CONFIG_FIELD_TO_COMPONENT = {
    "llm_cfg": "llm",
    "vision_tower_cfg": "vision_tower",
    "mm_projector_cfg": "mm_projector",
    "sound_tower_cfg": "sound_tower",
    "sound_mm_projector_cfg": "sound_mm_projector",
}

OPTIONAL_COMPONENT_FIELDS = {"sound_tower_cfg", "sound_mm_projector_cfg"}

WEIGHT_FILE_PATTERNS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".msgpack",
)

STRING_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmodeling_vila\.VILAConfig\b"), "configuration_omnivinci.OmniVinciConfig"),
    (
        re.compile(r"\bmodeling_vila\.VILAForCausalLM\b"),
        "modeling_omnivinci.OmniVinciForCausalLM",
    ),
    (
        re.compile(r"\bmodeling_omnivinci\.VILAForCausalLM\b"),
        "modeling_omnivinci.OmniVinciForCausalLM",
    ),
    (re.compile(r"\bconfiguration_omnivinci\.VILAConfig\b"), "configuration_omnivinci.OmniVinciConfig"),
    (
        re.compile(r"\bauto_processor\.VILAProcessor\b"),
        "processing_omnivinci.OmniVinciProcessor",
    ),
    (
        re.compile(r"\bprocessing_omnivinci\.VILAProcessor\b"),
        "processing_omnivinci.OmniVinciProcessor",
    ),
    (re.compile(r"\bVILAProcessorKwargs\b"), "OmniVinciProcessorKwargs"),
    (re.compile(r"\bVILAProcessor\b"), "OmniVinciProcessor"),
    (re.compile(r"\bVILAForCausalLM\b"), "OmniVinciForCausalLM"),
    (re.compile(r"\bVILAConfig\b"), "OmniVinciConfig"),
)


def _is_weight_file(name: str) -> bool:
    return name.endswith(WEIGHT_FILE_PATTERNS) or name == "model.safetensors.index.json"


def _is_top_level_metadata_file(name: str) -> bool:
    return name in TOP_LEVEL_METADATA_FILES or name.endswith(".jinja")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _rewrite_string(value: str) -> str:
    out = value
    for pattern, replacement in STRING_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    return out


def _deep_rewrite(obj: Any) -> Any:
    if isinstance(obj, str):
        return _rewrite_string(obj)
    if isinstance(obj, list):
        return [_deep_rewrite(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _deep_rewrite(value) for key, value in obj.items()}
    return obj


def _rewrite_json_file(path: Path) -> bool:
    if not path.exists():
        return False

    original = _load_json(path)
    rewritten = _deep_rewrite(original)

    if rewritten == original:
        logger.info("No changes needed: %s", path)
        return False

    _save_json(path, rewritten)
    logger.info("Rewrote metadata: %s", path)
    return True


def _copy_tree_metadata_only(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        return

    for item in src_dir.rglob("*"):
        if "__pycache__" in item.parts or ".git" in item.parts:
            continue

        rel = item.relative_to(src_dir)
        out = dst_dir / rel

        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue

        if _is_weight_file(item.name):
            continue

        if item.suffix in {".py", ".pyc", ".pyo", ".pyi"}:
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, out)


def _prepare_destination_tree(src_root: Path, dst_root: Path, clean_dst: bool = True) -> None:
    if clean_dst and dst_root.exists() and dst_root != src_root:
        logger.info("Cleaning destination directory: %s", dst_root)
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    for item in src_root.iterdir():
        if not item.is_file():
            continue
        if not _is_top_level_metadata_file(item.name):
            continue
        shutil.copy2(item, dst_root / item.name)

    for component in COMPONENT_TO_PREFIX:
        _copy_tree_metadata_only(src_root / component, dst_root / component)


def _resolve_component_dir(dirpath: Path):
    if not dirpath.is_dir():
        return None

    idx = dirpath / "model.safetensors.index.json"
    mono = dirpath / "model.safetensors"

    if idx.exists():
        wm = _load_json(idx).get("weight_map") or {}
        by_shard: dict[str, list[str]] = defaultdict(list)
        for key, shard in wm.items():
            by_shard[shard].append(key)
        return ("sharded", dirpath, {shard: sorted(keys) for shard, keys in sorted(by_shard.items())})

    if mono.exists():
        return ("file", mono)

    cands = sorted([x for x in dirpath.iterdir() if x.suffix == ".safetensors"])
    if len(cands) == 1:
        return ("file", cands[0])

    return None


def _collect_component_state(src_root: Path) -> dict[str, Any]:
    state: dict[str, Any] = {}

    for component, out_prefix in COMPONENT_TO_PREFIX.items():
        comp = _resolve_component_dir(src_root / component)
        if not comp:
            logger.info("No weights found for optional component: %s", component)
            continue

        if comp[0] == "file":
            fp: Path = comp[1]
            with safe_open(str(fp), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key == "__metadata__":
                        continue
                    state[f"{out_prefix}.{key}"] = f.get_tensor(key)
        else:
            base: Path = comp[1]
            shard_map: dict[str, list[str]] = comp[2]
            for shard, keys in shard_map.items():
                sp = base / shard
                with safe_open(str(sp), framework="pt", device="cpu") as f:
                    for key in keys:
                        state[f"{out_prefix}.{key}"] = f.get_tensor(key)

        logger.info("Collected %s weights under prefix '%s'", component, out_prefix)

    return state


def _normalize_top_level_config(dst_root: Path) -> None:
    cfg_path = dst_root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing required top-level config: {cfg_path}")

    cfg = _load_json(cfg_path)
    cfg = _deep_rewrite(cfg)

    for field, component in CONFIG_FIELD_TO_COMPONENT.items():
        component_cfg_path = dst_root / component / "config.json"
        if component_cfg_path.exists():
            cfg[field] = _load_json(component_cfg_path)
        elif field in OPTIONAL_COMPONENT_FIELDS:
            cfg[field] = None

    cfg["architectures"] = ["OmniVinciForCausalLM"]
    cfg["resume_path"] = None

    auto_map = cfg.get("auto_map") or {}
    auto_map.update(
        {
            "AutoConfig": "configuration_omnivinci.OmniVinciConfig",
            "AutoProcessor": "processing_omnivinci.OmniVinciProcessor",
            "AutoModel": "modeling_omnivinci.OmniVinciForCausalLM",
            "AutoModelForCausalLM": "modeling_omnivinci.OmniVinciForCausalLM",
        }
    )
    cfg["auto_map"] = auto_map

    _save_json(cfg_path, cfg)
    logger.info("Normalized top-level config: %s", cfg_path)


def _rewrite_metadata_jsons(dst_root: Path) -> tuple[list[Path], list[Path]]:
    touched = []
    missing = []

    for name in JSON_FILES_TO_REWRITE:
        path = dst_root / name
        if not path.exists():
            missing.append(path)
            continue
        if _rewrite_json_file(path):
            touched.append(path)

    return touched, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert legacy OmniVinci/VILA checkpoints to HF-loadable format.")
    parser.add_argument(
        "--src_path",
        type=Path,
        default=DEFAULT_SRC_PATH,
        help=f"Source model directory (default: {DEFAULT_SRC_PATH}).",
    )
    parser.add_argument(
        "--dst_path",
        type=Path,
        default=DEFAULT_DST_PATH,
        help=f"Destination export directory (default: {DEFAULT_DST_PATH}).",
    )
    # Backward-compatible aliases.
    parser.add_argument("--model_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip_weights",
        action="store_true",
        help="Skip writing top-level model.safetensors.",
    )
    parser.add_argument(
        "--keep_dst",
        action="store_true",
        help="Do not clean destination directory before writing artifacts.",
    )
    parser.add_argument(
        "--allow_inplace",
        action="store_true",
        help="Allow dst_path == src_path (modifies source). Disabled by default.",
    )
    return parser.parse_args()


def convert_omnivinci_to_hf(
    model_dir: Path,
    output_dir: Path | None = None,
    skip_weights: bool = False,
    clean_dst: bool = True,
) -> Path:
    src_root = model_dir.expanduser().resolve()
    dst_root = output_dir.expanduser().resolve() if output_dir else src_root

    if not src_root.is_dir():
        raise NotADirectoryError(f"--src_path must be a directory, got: {src_root}")

    if dst_root != src_root:
        logger.info("Preparing destination metadata tree: %s", dst_root)
        _prepare_destination_tree(src_root, dst_root, clean_dst=clean_dst)

    touched, missing = _rewrite_metadata_jsons(dst_root)
    _normalize_top_level_config(dst_root)

    if not skip_weights:
        state = _collect_component_state(src_root)
        if not state:
            raise FileNotFoundError("No component safetensors found under legacy component directories.")

        weights_out = dst_root / "model.safetensors"
        save_file(state, str(weights_out))
        logger.info("Wrote merged top-level weights: %s", weights_out)

    if touched:
        logger.info("Converted %d metadata file(s).", len(touched))
    else:
        logger.info("No metadata rewrite changes were required.")

    if missing:
        logger.info("Skipped %d missing metadata file(s).", len(missing))
        for path in missing:
            logger.info("  - %s", path)

    return dst_root


def main() -> None:
    args = parse_args()

    src_path = (args.model_dir or args.src_path).expanduser().resolve()
    dst_path = (args.output_dir or args.dst_path).expanduser().resolve()

    if src_path == dst_path and not args.allow_inplace:
        raise ValueError(
            f"Refusing in-place conversion for safety: src_path == dst_path == {src_path}. "
            "Use a different --dst_path (recommended) or pass --allow_inplace explicitly."
        )

    convert_omnivinci_to_hf(
        src_path,
        output_dir=dst_path,
        skip_weights=args.skip_weights,
        clean_dst=not args.keep_dst,
    )


if __name__ == "__main__":
    main()
