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
from typing import Any

import accelerate
from safetensors.torch import safe_open

from transformers import (
    AudioFlamingo3Config,
    AudioFlamingo3ForConditionalGeneration,
    AutoTokenizer,
    GenerationConfig,
    Qwen2Config,
    WhisperFeatureExtractor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_json(p: Path):
    if not p.is_file():
        raise FileNotFoundError(f"Missing JSON: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_processor(src_root: Path, dst_root: Path):
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


PREFIX_MAP = {
    "llm": "language_model",
    "sound_tower": "audio_tower",
    "sound_mm_projector": "multi_modal_projector",
}


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


def merge_and_shard_weights(src_root: Path, dst_root: Path):
    state: dict[str, Any] = {}
    for tag in PREFIX_MAP.keys():
        comp = _resolve_component_dir(src_root / tag)
        if not comp:
            continue

        out_prefix = PREFIX_MAP.get(tag, tag)

        if comp[0] == "file":
            fp: Path = comp[1]
            with safe_open(str(fp), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k == "__metadata__":
                        continue
                    state[f"{out_prefix}.{k}"] = f.get_tensor(k)
        else:
            base: Path = comp[1]
            shard_map: dict[str, list[str]] = comp[2]
            for shard, keys in shard_map.items():
                sp = base / shard
                with safe_open(str(sp), framework="pt", device="cpu") as f:
                    for k in keys:
                        state[f"{out_prefix}.{k}"] = f.get_tensor(k)

    if not state:
        raise FileNotFoundError("No tensors found in llm/, sound_tower/, or sound_mm_projector/.")

    text_config = Qwen2Config(
        bos_token_id=151643,
        dtype="bfloat16",
        eos_token_id=151645,
        hidden_size=3584,
        intermediate_size=18944,
        model_max_length=8192,
        num_attention_heads=28,
        num_hidden_layers=28,
        num_key_value_heads=4,
        rope_theta=1000000.0,
        use_cache=False,
        vocab_size=151672,
    )
    config = AudioFlamingo3Config(text_config=text_config)
    with accelerate.init_empty_weights():
        model = AudioFlamingo3ForConditionalGeneration(config)

    model.save_pretrained(save_directory=str(dst_root), state_dict=state)
    logger.info("model.safetensors index and shards")


def write_generation_config(dst_root: Path) -> None:
    generation_config = GenerationConfig(
        bos_token_id=151643,
        do_sample=True,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        repetition_penalty=1.05,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
    )
    generation_config.save_pretrained(dst_root)
    logger.info("generation_config.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert AudioFlamingo3 to Hugging Face format.")
    ap.add_argument("src_dir", help="Source model root directory")
    ap.add_argument("dst_dir", help="Destination directory for converted model")
    args = ap.parse_args()

    src_root = Path(args.src_dir).resolve()
    dst_root = Path(args.dst_dir).resolve()

    write_processor(src_root, dst_root)
    merge_and_shard_weights(src_root, dst_root)
    write_generation_config(dst_root)


if __name__ == "__main__":
    main()
