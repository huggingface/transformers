# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

"""Convert AudioFlamingo3 checkpoints into a Hugging Face repository layout."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from safetensors.torch import safe_open
from transformers import AutoConfig, AudioFlamingo3ForConditionalGeneration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        raise FileNotFoundError(f"Missing JSON: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Dict[str, Any], p: Path) -> None:
    _ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _strip_keys(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    ks = set(keys)
    return {k: v for k, v in d.items() if k not in ks}


def write_generation_config(src_root: Path, dst_root: Path) -> None:
    data = _load_json(src_root / "llm" / "generation_config.json")
    data["max_new_tokens"] = 1024
    _save_json(data, dst_root / "generation_config.json")
    logger.info("generation_config.json")


_BASE_MAIN_CONFIG: Dict[str, Any] = {
    "model_type": "audioflamingo3",
    "model_dtype": "torch.float16",
    "hidden_size": 3584,
    "sound_hidden_size": 1280,
    "bos_token_id": 151670,
    "eos_token_id": 151645,
    "sound_token_id": 151669,
    "pad_token_id": 151671,
    "model_max_length": 8192,
    "end_newline_token_id": 198,
    "padding_side": "right",
    "media_tokens": {"sound": "<sound>"},
}


def write_main_config(src_root: Path, dst_root: Path) -> None:
    top_cfg = _load_json(src_root / "config.json")
    final_cfg = dict(_BASE_MAIN_CONFIG)
    for key in ("model_dtype", "hidden_size", "sound_hidden_size"):
        if key in top_cfg:
            final_cfg[key] = top_cfg[key]

    text_cfg = _strip_keys(
        _load_json(src_root / "llm" / "config.json"),
        keys=("_name_or_path", "architectures"),
    )
    enc_cfg = _strip_keys(
        _load_json(src_root / "sound_tower" / "config.json"),
        keys=("_name_or_path", "architectures"),
    )
    enc_cfg["model_type"] = "audioflamingo3_encoder"

    final_cfg["text_config"] = text_cfg
    final_cfg["encoder_config"] = enc_cfg

    _save_json(final_cfg, dst_root / "config.json")
    logger.info("config.json")


def _has_sound_token_in_list(lst) -> bool:
    if not isinstance(lst, list):
        return False
    for item in lst:
        if item == "<sound>":
            return True
        if isinstance(item, dict):
            for k in ("content", "id", "token", "special_token"):
                if item.get(k) == "<sound>":
                    return True
    return False


def write_tokenizer(src_root: Path, dst_root: Path) -> None:
    llm_dir = src_root / "llm"
    tok_json_src = llm_dir / "tokenizer.json"
    tok_cfg_src = llm_dir / "tokenizer_config.json"
    if not tok_json_src.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json: {tok_json_src}")
    if not tok_cfg_src.is_file():
        raise FileNotFoundError(f"Missing tokenizer_config.json: {tok_cfg_src}")

    _ensure_dir(dst_root)
    (dst_root / "tokenizer.json").write_bytes(tok_json_src.read_bytes())

    cfg = _load_json(tok_cfg_src)
    specials = cfg.get("additional_special_tokens")
    if specials is None:
        specials = ["<sound>"]
    else:
        if not isinstance(specials, list):
            specials = [specials]
        if not _has_sound_token_in_list(specials):
            specials.append("<sound>")
    cfg["additional_special_tokens"] = specials

    _save_json(cfg, dst_root / "tokenizer_config.json")
    _save_json({"additional_special_tokens": ["<sound>"]}, dst_root / "special_tokens_map.json")
    logger.info("tokenizer.json, tokenizer_config.json, special_tokens_map.json")


_PREPROCESSOR_CONFIG: Dict[str, Any] = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "AudioFlamingo3Processor",
    "return_attention_mask": True,
    "sampling_rate": 16000,
}


def write_preprocessor_config(dst_root: Path) -> None:
    _save_json(_PREPROCESSOR_CONFIG, dst_root / "preprocessor_config.json")
    logger.info("preprocessor_config.json")


COMPONENTS = ("llm", "sound_tower", "sound_mm_projector")


def _resolve_component_dir(dirpath: Path):
    if not dirpath.is_dir():
        return None
    idx = dirpath / "model.safetensors.index.json"
    mono = dirpath / "model.safetensors"
    if idx.exists():
        wm = _load_json(idx).get("weight_map") or {}
        by_shard: Dict[str, List[str]] = defaultdict(list)
        for k, shard in wm.items():
            by_shard[shard].append(k)
        return ("sharded", dirpath, {k: sorted(v) for k, v in sorted(by_shard.items())})
    if mono.exists():
        return ("file", mono)
    cands = sorted([x for x in dirpath.iterdir() if x.suffix == ".safetensors"])
    return ("file", cands[0]) if len(cands) == 1 else None


def merge_and_shard_weights(src_root: Path, dst_root: Path) -> None:
    state: Dict[str, Any] = {}
    for tag in COMPONENTS:
        comp = _resolve_component_dir(src_root / tag)
        if not comp:
            continue
        if comp[0] == "file":
            fp: Path = comp[1]
            with safe_open(str(fp), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k == "__metadata__":
                        continue
                    state[f"{tag}.{k}"] = f.get_tensor(k)
        else:
            base: Path = comp[1]
            shard_map: Dict[str, List[str]] = comp[2]
            for shard, keys in shard_map.items():
                sp = base / shard
                with safe_open(str(sp), framework="pt", device="cpu") as f:
                    for k in keys:
                        state[f"{tag}.{k}"] = f.get_tensor(k)

    if not state:
        raise FileNotFoundError("No tensors found in llm/, sound_tower/, or sound_mm_projector/.")

    cfg = AutoConfig.from_pretrained(dst_root)
    with torch.device("meta"):
        model = AudioFlamingo3ForConditionalGeneration(cfg)

    model.save_pretrained(
        save_directory=str(dst_root),
        state_dict=state
    )
    logger.info("model.safetensors index and shards")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert AudioFlamingo3 to Hugging Face format.")
    ap.add_argument("src_dir", help="Source model root directory")
    ap.add_argument("dst_dir", help="Destination directory for converted model")
    args = ap.parse_args()

    src_root = Path(args.src_dir).resolve()
    dst_root = Path(args.dst_dir).resolve()

    write_generation_config(src_root, dst_root)
    write_main_config(src_root, dst_root)
    write_tokenizer(src_root, dst_root)
    write_preprocessor_config(dst_root)
    merge_and_shard_weights(src_root, dst_root)


if __name__ == "__main__":
    main()
