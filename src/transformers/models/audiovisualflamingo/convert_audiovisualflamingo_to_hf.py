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

"""Convert legacy AudioVisualFlamingo/VILA checkpoints to native HF AudioVisualFlamingo artifacts.

This conversion script:
1) rewrites legacy VILA class strings to canonical AudioVisualFlamingo names,
2) normalizes a single top-level config for local HF loading,
3) loads the native HF model/processor and saves with `save_pretrained`.

The destination is treated as an export directory and contains only root-level
artifacts (weights/config/tokenizer/processor/chat-template). Python source files
and component subfolder configs are not copied.
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

import torch
from safetensors.torch import safe_open

from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    GenerationConfig,
    AudioVisualFlamingoConfig,
    AudioVisualFlamingoForConditionalGeneration,
    AudioVisualFlamingoProcessor,
    WhisperFeatureExtractor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_SRC_PATH = Path("/fs/nexus-projects/JSALT_workshop/lasha/Dev/audiovisualflamingo")
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
    (re.compile(r"\bmodeling_vila\.VILAConfig\b"), "configuration_audiovisualflamingo.AudioVisualFlamingoConfig"),
    (
        re.compile(r"\bmodeling_vila\.VILAForCausalLM\b"),
        "modeling_audiovisualflamingo.AudioVisualFlamingoForConditionalGeneration",
    ),
    (
        re.compile(r"\bmodeling_vila\.VILAForConditionalGeneration\b"),
        "modeling_audiovisualflamingo.AudioVisualFlamingoForConditionalGeneration",
    ),
    (
        re.compile(r"\bmodeling_audiovisualflamingo\.VILAForCausalLM\b"),
        "modeling_audiovisualflamingo.AudioVisualFlamingoForConditionalGeneration",
    ),
    (
        re.compile(r"\bmodeling_audiovisualflamingo\.VILAForConditionalGeneration\b"),
        "modeling_audiovisualflamingo.AudioVisualFlamingoForConditionalGeneration",
    ),
    (
        re.compile(r"\bmodeling_audiovisualflamingo\.AudioVisualFlamingoForCausalLM\b"),
        "modeling_audiovisualflamingo.AudioVisualFlamingoForConditionalGeneration",
    ),
    (
        re.compile(r"\bconfiguration_audiovisualflamingo\.VILAConfig\b"),
        "configuration_audiovisualflamingo.AudioVisualFlamingoConfig",
    ),
    (
        re.compile(r"\bauto_processor\.VILAProcessor\b"),
        "processing_audiovisualflamingo.AudioVisualFlamingoProcessor",
    ),
    (
        re.compile(r"\bprocessing_audiovisualflamingo\.VILAProcessor\b"),
        "processing_audiovisualflamingo.AudioVisualFlamingoProcessor",
    ),
    (re.compile(r"\bVILAProcessorKwargs\b"), "AudioVisualFlamingoProcessorKwargs"),
    (re.compile(r"\bVILAProcessor\b"), "AudioVisualFlamingoProcessor"),
    (re.compile(r"\bVILAForCausalLM\b"), "AudioVisualFlamingoForConditionalGeneration"),
    (re.compile(r"\bVILAForConditionalGeneration\b"), "AudioVisualFlamingoForConditionalGeneration"),
    (re.compile(r"\bAudioVisualFlamingoForCausalLM\b"), "AudioVisualFlamingoForConditionalGeneration"),
    (re.compile(r"\bVILAConfig\b"), "AudioVisualFlamingoConfig"),
)


AUDIO_PREPROCESSOR_KEYS = (
    "feature_extractor_type",
    "feature_size",
    "sampling_rate",
    "chunk_length",
    "hop_length",
    "n_fft",
    "n_samples",
    "nb_max_frames",
    "padding_side",
    "padding_value",
    "return_attention_mask",
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


def _copy_top_level_metadata(src_root: Path, dst_root: Path) -> None:
    for item in src_root.iterdir():
        if not item.is_file():
            continue
        if _is_top_level_metadata_file(item.name):
            shutil.copy2(item, dst_root / item.name)


def _copy_llm_metadata_to_root(src_root: Path, dst_root: Path) -> None:
    llm_dir = src_root / "llm"
    if not llm_dir.is_dir():
        return

    for item in llm_dir.iterdir():
        if not item.is_file():
            continue
        if _is_weight_file(item.name):
            continue
        if item.suffix in {".py", ".pyc", ".pyo", ".pyi"}:
            continue
        if item.name == "config.json":
            continue
        # Legacy AudioVisualFlamingo loads generation defaults from Python/runtime, not llm/generation_config.json.
        # We export the effective runtime config explicitly in `_export_effective_generation_config`.
        if item.name == "generation_config.json":
            continue
        shutil.copy2(item, dst_root / item.name)


def _copy_merged_preprocessor_config(src_root: Path, dst_root: Path) -> None:
    target_preprocessor = dst_root / "preprocessor_config.json"
    root_preprocessor = src_root / "preprocessor_config.json"
    vision_preprocessor = src_root / "vision_tower" / "preprocessor_config.json"

    if vision_preprocessor.exists():
        merged_preprocessor = _load_json(vision_preprocessor)
    elif root_preprocessor.exists():
        merged_preprocessor = _load_json(root_preprocessor)
    else:
        return

    if root_preprocessor.exists():
        audio_preprocessor = _load_json(root_preprocessor)
        for key in AUDIO_PREPROCESSOR_KEYS:
            if key in audio_preprocessor:
                merged_preprocessor[key] = audio_preprocessor[key]

    if "feature_size" not in merged_preprocessor:
        sound_tower_cfg = src_root / "sound_tower" / "config.json"
        if sound_tower_cfg.exists():
            num_mel_bins = _load_json(sound_tower_cfg).get("num_mel_bins")
            if num_mel_bins is not None:
                merged_preprocessor["feature_size"] = int(num_mel_bins)

    if "feature_size" in merged_preprocessor and "feature_extractor_type" not in merged_preprocessor:
        merged_preprocessor["feature_extractor_type"] = "WhisperFeatureExtractor"

    _save_json(target_preprocessor, merged_preprocessor)


def _ensure_processor_config(dst_root: Path, config: dict[str, Any] | None = None) -> None:
    processor_path = dst_root / "processor_config.json"
    payload = {}
    if processor_path.exists():
        payload = _load_json(processor_path)

    payload["processor_class"] = "AudioVisualFlamingoProcessor"
    if config is not None:
        payload["config"] = config
    _save_json(processor_path, payload)


def _resolve_tokenizer_source_dir(src_root: Path, dst_root: Path) -> Path:
    llm_dir = src_root / "llm"
    if (llm_dir / "tokenizer_config.json").exists():
        return llm_dir
    if (src_root / "tokenizer_config.json").exists():
        return src_root
    if (dst_root / "tokenizer_config.json").exists():
        return dst_root
    raise FileNotFoundError(
        "Could not locate tokenizer files in src_root/llm, src_root, or dst_root. Expected tokenizer_config.json."
    )


def _resolve_image_processor_source_dir(src_root: Path, dst_root: Path) -> Path:
    candidates = (src_root / "vision_tower", dst_root, src_root)
    for candidate in candidates:
        if (candidate / "preprocessor_config.json").exists():
            return candidate
    raise FileNotFoundError("Could not locate image processor files in src_root/vision_tower, dst_root, or src_root.")


def _resolve_feature_extractor_source_dir(src_root: Path, dst_root: Path) -> Path:
    candidates = (dst_root, src_root)
    for candidate in candidates:
        if (candidate / "preprocessor_config.json").exists():
            return candidate
    raise FileNotFoundError("Could not locate preprocessor_config.json for WhisperFeatureExtractor loading.")


def _collect_encoder_boundary_tokens(config: dict[str, Any]) -> list[str]:
    token_keys = {"start_tokens", "end_tokens", "sep_tokens"}
    collected = []
    seen = set()

    def _maybe_add(token):
        if not isinstance(token, str) or token == "None" or token in seen:
            return
        seen.add(token)
        collected.append(token)

    def _visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in token_keys:
                    _maybe_add(value)
                _visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _visit(item)

    # Keep parity with processor default.
    _maybe_add("\n")

    for attr in ("image_encoder", "video_encoder", "sound_encoder"):
        encoder_config = config.get(attr)
        if isinstance(encoder_config, str):
            try:
                encoder_config = json.loads(encoder_config)
            except Exception:
                continue
        _visit(encoder_config)

    return collected


def _populate_token_id_fields(cfg: dict[str, Any], src_root: Path, dst_root: Path) -> None:
    tokenizer_src = _resolve_tokenizer_source_dir(src_root, dst_root)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src), use_fast=True)

    media_tokens = cfg.get("media_tokens") or {"image": "<image>", "video": "<vila/video>", "sound": "<sound>"}
    cfg["media_tokens"] = media_tokens
    media_token_ids = {}
    for name, token in media_tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            tokenized = tokenizer(token, add_special_tokens=False).input_ids
            if len(tokenized) != 1:
                raise ValueError(f"Media token `{token}` must map to a single tokenizer id.")
            token_id = tokenized[0]
        media_token_ids[name] = int(token_id)
    cfg["media_token_ids"] = media_token_ids

    cfg["encoder_text_token_ids"] = {
        token_text: [int(token_id) for token_id in tokenizer(token_text).input_ids]
        for token_text in _collect_encoder_boundary_tokens(cfg)
    }


def _export_effective_generation_config(src_root: Path, dst_root: Path) -> None:
    """
    Export a minimal generation config for AudioVisualFlamingo.

    Keep this intentionally small and rely on HF `GenerationConfig` defaults
    (greedy decoding unless users override sampling/beam settings).
    """

    tokenizer_src = _resolve_tokenizer_source_dir(src_root, dst_root)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src), use_fast=True)

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define `eos_token_id` to build generation config.")

    pad_token_id = tokenizer.pad_token_id or eos_token_id
    bos_token_id = tokenizer.bos_token_id or eos_token_id

    generation_config = GenerationConfig(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    generation_config.save_pretrained(str(dst_root))
    logger.info("Exported generation config via GenerationConfig.save_pretrained to %s", dst_root)


def _prepare_destination_tree(src_root: Path, dst_root: Path, clean_dst: bool = True) -> None:
    if clean_dst and dst_root.exists() and dst_root != src_root:
        logger.info("Cleaning destination directory: %s", dst_root)
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    _copy_top_level_metadata(src_root, dst_root)
    _copy_llm_metadata_to_root(src_root, dst_root)
    _copy_merged_preprocessor_config(src_root, dst_root)
    _ensure_processor_config(dst_root)
    _export_effective_generation_config(src_root, dst_root)


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


def _normalize_top_level_config(dst_root: Path, src_root: Path) -> dict[str, Any]:
    cfg_path = dst_root / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing required top-level config: {cfg_path}")

    cfg = _load_json(cfg_path)
    cfg = _deep_rewrite(cfg)

    for field, component in CONFIG_FIELD_TO_COMPONENT.items():
        component_cfg_path = src_root / component / "config.json"
        if component_cfg_path.exists():
            cfg[field] = _deep_rewrite(_load_json(component_cfg_path))
        elif field in OPTIONAL_COMPONENT_FIELDS:
            cfg[field] = None

    cfg["model_type"] = "audiovisualflamingo"
    cfg["architectures"] = ["AudioVisualFlamingoForConditionalGeneration"]
    cfg["_name_or_path"] = str(dst_root)
    cfg["resume_path"] = None
    _populate_token_id_fields(cfg, src_root, dst_root)

    # Native integration is now in-tree via CONFIG/MODEL/PROCESSOR auto mappings.
    # Keep exported configs clean and avoid remote-code prompts by dropping legacy auto_map entries.
    cfg.pop("auto_map", None)

    _ensure_processor_config(dst_root, config=cfg)

    _save_json(cfg_path, cfg)
    logger.info("Normalized top-level config: %s", cfg_path)
    return cfg


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


def _save_processor(
    src_root: Path,
    dst_root: Path,
    config_payload: dict[str, Any],
) -> AudioVisualFlamingoProcessor:
    tokenizer_src = _resolve_tokenizer_source_dir(src_root, dst_root)
    image_processor_src = _resolve_image_processor_source_dir(src_root, dst_root)
    feature_extractor_src = _resolve_feature_extractor_source_dir(src_root, dst_root)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src), use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(str(image_processor_src), use_fast=False)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(feature_extractor_src))

    config = AudioVisualFlamingoConfig(**config_payload)
    processor = AudioVisualFlamingoProcessor(
        image_processor=image_processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        config=config,
    )
    processor.save_pretrained(str(dst_root))
    logger.info("Saved processor via save_pretrained: %s", dst_root)
    return processor


def _save_model_from_state(
    dst_root: Path,
    config_payload: dict[str, Any],
    state_dict: dict[str, Any],
) -> AudioVisualFlamingoForConditionalGeneration:
    config = AudioVisualFlamingoConfig(**config_payload)
    model = AudioVisualFlamingoForConditionalGeneration(config).to(dtype=torch.bfloat16)

    load_res = model.load_state_dict(state_dict, strict=True)
    if load_res.missing_keys:
        missing = load_res.missing_keys
        raise ValueError(f"Missing keys when loading converted AudioVisualFlamingo checkpoint: {missing[:10]}")
    if load_res.unexpected_keys:
        unexpected = load_res.unexpected_keys
        raise ValueError(f"Unexpected keys when loading converted AudioVisualFlamingo checkpoint: {unexpected[:10]}")

    generation_config_path = dst_root / "generation_config.json"
    if generation_config_path.exists():
        model.generation_config = GenerationConfig.from_pretrained(str(dst_root))

    model.save_pretrained(str(dst_root), safe_serialization=True)
    logger.info("Saved model via save_pretrained: %s", dst_root)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy AudioVisualFlamingo/VILA checkpoints to HF-loadable format."
    )
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
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Optional Hub repo id to push converted assets, e.g. `username/audiovisualflamingo`.",
    )
    return parser.parse_args()


def convert_audiovisualflamingo_to_hf(
    model_dir: Path,
    output_dir: Path | None = None,
    skip_weights: bool = False,
    clean_dst: bool = True,
    push_to_hub: str | None = None,
) -> Path:
    src_root = model_dir.expanduser().resolve()
    dst_root = output_dir.expanduser().resolve() if output_dir else src_root

    if not src_root.is_dir():
        raise NotADirectoryError(f"--src_path must be a directory, got: {src_root}")

    if dst_root != src_root:
        logger.info("Preparing destination metadata tree: %s", dst_root)
        _prepare_destination_tree(src_root, dst_root, clean_dst=clean_dst)

    touched, missing = _rewrite_metadata_jsons(dst_root)
    config_payload = _normalize_top_level_config(dst_root, src_root)
    processor = _save_processor(src_root, dst_root, config_payload)

    model = None
    if not skip_weights:
        state = _collect_component_state(src_root)
        if not state:
            raise FileNotFoundError("No component safetensors found under legacy component directories.")
        model = _save_model_from_state(dst_root, config_payload, state)

    if touched:
        logger.info("Converted %d metadata file(s).", len(touched))
    else:
        logger.info("No metadata rewrite changes were required.")

    if missing:
        logger.info("Skipped %d missing metadata file(s).", len(missing))
        for path in missing:
            logger.info("  - %s", path)

    if push_to_hub:
        logger.info("Pushing processor to the Hub: %s", push_to_hub)
        processor.push_to_hub(push_to_hub)
        if model is not None:
            logger.info("Pushing model to the Hub: %s", push_to_hub)
            model.push_to_hub(push_to_hub)

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

    convert_audiovisualflamingo_to_hf(
        src_path,
        output_dir=dst_path,
        skip_weights=args.skip_weights,
        clean_dst=not args.keep_dst,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
