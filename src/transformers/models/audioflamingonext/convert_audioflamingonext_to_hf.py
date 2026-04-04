# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
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

"""Convert AudioFlamingoNext checkpoints into a Hugging Face repository layout."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open

from transformers import (
    AudioFlamingoNextConfig,
    AudioFlamingoNextForConditionalGeneration,
    AudioFlamingoNextProcessor,
    AutoTokenizer,
    GenerationConfig,
    Qwen2Config,
    WhisperFeatureExtractor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


SYSTEM_PROMPT = (
    "You are Audio Flamingo-Next, a multimodal assistant for language and audio. "
    "On each turn you receive an optional audio clip which may contain speech, music, or ambient sounds "
    "and optional text, you will receive at least one or both; use your world knowledge and reasoning "
    "to help the user with any task. Interpret the entirety of the content of any input audio—regardless "
    "of whether the user calls it audio, speech, music, or sound."
)

PREFIX_MAP = {
    "llm": "language_model",
    "sound_tower": "audio_tower",
    "sound_mm_projector": "multi_modal_projector",
}


def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_audio_config(src_root: Path) -> dict[str, Any]:
    config = _load_json(src_root / "sound_tower" / "config.json")
    return {
        "model_type": "audioflamingo3_encoder",
        "hidden_size": config["d_model"],
        "intermediate_size": config["encoder_ffn_dim"],
        "num_hidden_layers": config["encoder_layers"],
        "num_attention_heads": config["encoder_attention_heads"],
        "num_mel_bins": config["num_mel_bins"],
        "max_source_positions": config["max_source_positions"],
        "scale_embedding": config.get("scale_embedding", False),
        "activation_function": config.get("activation_function", "gelu"),
        "dropout": config.get("dropout", 0.0),
        "attention_dropout": config.get("attention_dropout", 0.0),
        "activation_dropout": config.get("activation_dropout", 0.0),
        "layerdrop": config.get("encoder_layerdrop", config.get("layerdrop", 0.0)),
        "initializer_range": config.get("init_std", 0.02),
    }


def write_processor(src_root: Path, dst_root: Path):
    llm_dir = src_root / "llm"

    # fmt: off
    processor_chat_template = (
        "{% if messages[0]['role'] != 'system' %}"
            "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "{% endif %}"
        "{% for m in messages if m['content'] is not none %}"
            "<|im_start|>{{ m['role'] }}\n"
            "{% if m['content'] is string %}"
                "{{ m['content'] }}"
            "{% else %}"
                "{% set audio = namespace(found=False) %}"
                "{% set text_buf = namespace(v='') %}"
                "{% for c in m['content'] %}"
                    "{% if c.get('type') == 'audio' or 'audio' in c %}"
                        "{% set audio.found = True %}"
                    "{% elif c.get('type') == 'text' or 'text' in c %}"
                        "{% set text_buf.v = text_buf.v + c['text'] %}"
                    "{% endif %}"
                "{% endfor %}"
                "{% if audio.found %}{{ '<sound>' }}{% endif %}{{ text_buf.v }}"
            "{% endif %}"
            "<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
        "{% endif %}"
    )
    # fmt: on

    tokenizer = AutoTokenizer.from_pretrained(str(llm_dir), use_fast=True)
    processor = AudioFlamingoNextProcessor(
        feature_extractor=WhisperFeatureExtractor(feature_size=128, return_attention_mask=True),
        tokenizer=tokenizer,
        chat_template=processor_chat_template,
        max_audio_len=1800,
    )
    processor.save_pretrained(str(dst_root))

    logger.info("processor (tokenizer + preprocessor)")
    return processor


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
        return ("sharded", dirpath, {key: sorted(value) for key, value in sorted(by_shard.items())})
    if mono.exists():
        return ("file", mono)
    candidates = sorted(path for path in dirpath.iterdir() if path.suffix == ".safetensors")
    return ("file", candidates[0]) if len(candidates) == 1 else None


def merge_and_shard_weights(src_root: Path, dst_root: Path, processor: AudioFlamingoNextProcessor):
    state: dict[str, Any] = {}
    for tag in PREFIX_MAP:
        component = _resolve_component_dir(src_root / tag)
        if not component:
            continue

        out_prefix = PREFIX_MAP[tag]
        if component[0] == "file":
            file_path: Path = component[1]
            with safe_open(str(file_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key == "__metadata__":
                        continue
                    state[f"{out_prefix}.{key}"] = f.get_tensor(key)
        else:
            base: Path = component[1]
            shard_map: dict[str, list[str]] = component[2]
            for shard, keys in shard_map.items():
                shard_path = base / shard
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    for key in keys:
                        state[f"{out_prefix}.{key}"] = f.get_tensor(key)

    if not state:
        raise FileNotFoundError("No tensors found in llm/, sound_tower/, or sound_mm_projector/.")

    llm_dir = src_root / "llm"
    tokenizer = processor.tokenizer
    text_config = Qwen2Config.from_pretrained(str(llm_dir))
    text_config.bos_token_id = tokenizer.bos_token_id
    text_config.eos_token_id = tokenizer.eos_token_id
    text_config.pad_token_id = tokenizer.pad_token_id
    text_config.vocab_size = len(tokenizer)
    text_config.use_cache = False

    vocab = tokenizer.get_vocab()
    config = AudioFlamingoNextConfig(
        text_config=text_config,
        audio_config=_load_audio_config(src_root),
        audio_token_id=vocab["<sound>"],
        audio_bos_token_id=vocab.get("<|sound_bos|>"),
        audio_eos_token_id=vocab.get("<|sound_eos|>"),
        rope_parameters={"rope_type": "default", "rope_theta": 1800, "partial_rotary_factor": 0.2},
    )
    model = AudioFlamingoNextForConditionalGeneration(config).to(dtype=torch.bfloat16)

    projector_key_mapping = {
        "multi_modal_projector.layers.0.weight": "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.layers.0.bias": "multi_modal_projector.linear_1.bias",
        "multi_modal_projector.layers.2.weight": "multi_modal_projector.linear_2.weight",
        "multi_modal_projector.layers.2.bias": "multi_modal_projector.linear_2.bias",
    }
    for old_key, new_key in projector_key_mapping.items():
        if old_key in state:
            state[new_key] = state.pop(old_key)

    state.pop("audio_tower.sound_tower.pos_emb.freqs", None)

    load_res = model.load_state_dict(state, strict=True)
    if getattr(load_res, "missing_keys", None) and load_res.missing_keys:
        missing_keys = load_res.missing_keys
        raise ValueError(f"Missing keys when loading: {missing_keys[:10]}{' ...' if len(missing_keys) > 10 else ''}")
    if getattr(load_res, "unexpected_keys", None) and load_res.unexpected_keys:
        unexpected_keys = load_res.unexpected_keys
        raise ValueError(
            f"Unexpected keys when loading: {unexpected_keys[:10]}{' ...' if len(unexpected_keys) > 10 else ''}"
        )

    try:
        model.generation_config = GenerationConfig.from_pretrained(str(llm_dir))
    except OSError:
        model.generation_config = GenerationConfig(
            bos_token_id=text_config.bos_token_id,
            eos_token_id=text_config.eos_token_id,
            pad_token_id=text_config.pad_token_id,
        )

    model.save_pretrained(save_directory=str(dst_root))
    logger.info("model.safetensors index and shards")
    return model


"""
Reproducible Usage
==================

1) Download the original AudioFlamingoNext weights from NVIDIA (requires Git LFS and private repo access):

```
git lfs install
git clone https://huggingface.co/nvidia/audio-flamingo-next
```

This creates a folder `audio-flamingo-next/` containing the original components:
`llm/`, `sound_tower/`, and `sound_mm_projector/`.

2) Convert to the Hugging Face Transformers format (locally):

```
python src/transformers/models/audioflamingonext/convert_audioflamingonext_to_hf.py \
  --src_dir audio-flamingo-next \
  --dst_dir audio-flamingo-next-hf
```

3) Convert and push directly to the Hub (requires `huggingface-cli login` or `HF_TOKEN`):

```
python src/transformers/models/audioflamingonext/convert_audioflamingonext_to_hf.py \
  --src_dir audio-flamingo-next \
  --dst_dir audio-flamingo-next-hf \
  --push_to_hub <username-or-org>/audio-flamingo-next-hf
```
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert AudioFlamingoNext to Hugging Face format.")
    ap.add_argument("--src_dir", required=True, help="Source model root directory")
    ap.add_argument("--dst_dir", required=True, help="Destination directory for converted model")
    ap.add_argument(
        "--push_to_hub",
        default=None,
        type=str,
        help=(
            "Optional repository ID to push the converted assets to the Hugging Face Hub, "
            "e.g. 'username/audio-flamingo-next-hf'."
        ),
    )
    args = ap.parse_args()

    src_root = Path(args.src_dir).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    dst_root = Path(args.dst_dir).resolve()
    if dst_root.exists():
        raise FileExistsError(f"Destination already exists: {dst_root}")

    processor = write_processor(src_root, dst_root)
    model = merge_and_shard_weights(src_root, dst_root, processor)

    if args.push_to_hub:
        logger.info("Pushing processor to the Hub ...")
        processor.push_to_hub(args.push_to_hub)
        logger.info("Pushing model to the Hub ...")
        model.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
