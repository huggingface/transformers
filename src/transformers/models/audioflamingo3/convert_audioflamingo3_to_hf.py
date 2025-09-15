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
from typing import Any, Iterable

import torch
from safetensors.torch import safe_open

from transformers import (
    AudioFlamingo3ForConditionalGeneration,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    WhisperFeatureExtractor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(p: Path) -> dict[str, Any]:
    if not p.is_file():
        raise FileNotFoundError(f"Missing JSON: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: dict[str, Any], p: Path) -> None:
    _ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _strip_keys(d: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    ks = set(keys)
    return {k: v for k, v in d.items() if k not in ks}


def get_generation_config(src_root: Path) -> dict[str, Any]:
    data = _load_json(src_root / "llm" / "generation_config.json")
    logger.info("generation_config.json")
    return data


_BASE_MAIN_CONFIG: dict[str, Any] = {
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
    "ignore_index": -100,
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


def write_processor(src_root: Path, dst_root: Path) -> None:
    llm_dir = src_root / "llm"

    AutoTokenizer.from_pretrained(str(llm_dir)).save_pretrained(
        str(dst_root), legacy_format=False, save_jinja_files=False
    )
    WhisperFeatureExtractor(
        feature_size=128,
        return_attention_mask=True,
        processor_class="AudioFlamingo3Processor",
    ).save_pretrained(str(dst_root))

    logger.info("processor (tokenizer + preprocessor)")


COMPONENTS = ("llm", "sound_tower", "sound_mm_projector")


def _resolve_component_dir(dirpath: Path):
    if not dirpath.is_dir():
        return None
    idx = dirpath / "model.safetensors.index.json"
    mono = dirpath / "model.safetensors"
    if idx.exists():
        wm = _load_json(idx).get("weight_map") or {}
        by_shard: dict[str, list[str]] = defaultdict(list)
        for k, shard in wm.items():
            by_shard[shard].append(k)
        return ("sharded", dirpath, {k: sorted(v) for k, v in sorted(by_shard.items())})
    if mono.exists():
        return ("file", mono)
    cands = sorted([x for x in dirpath.iterdir() if x.suffix == ".safetensors"])
    return ("file", cands[0]) if len(cands) == 1 else None


def merge_and_shard_weights(src_root: Path, dst_root: Path, gen_cfg_dict: dict[str, Any]) -> None:
    state: dict[str, Any] = {}
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
            shard_map: dict[str, list[str]] = comp[2]
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

    gen_cfg = GenerationConfig(**gen_cfg_dict)
    gen_cfg._from_model_config = False
    model.generation_config = gen_cfg

    model.save_pretrained(save_directory=str(dst_root), state_dict=state)
    logger.info("model.safetensors index and shards")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert AudioFlamingo3 to Hugging Face format.")
    ap.add_argument("src_dir", help="Source model root directory")
    ap.add_argument("dst_dir", help="Destination directory for converted model")
    args = ap.parse_args()

    src_root = Path(args.src_dir).resolve()
    dst_root = Path(args.dst_dir).resolve()

    gen_cfg_dict = get_generation_config(src_root)
    write_main_config(src_root, dst_root)
    write_processor(src_root, dst_root)
    merge_and_shard_weights(src_root, dst_root, gen_cfg_dict)


if __name__ == "__main__":
    main()
