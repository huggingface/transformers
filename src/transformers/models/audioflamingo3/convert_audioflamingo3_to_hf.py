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

from safetensors.torch import safe_open

from transformers import (
    AudioFlamingo3Config,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3Processor,
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

    # fmt: off
    tokenizer_chat_template = (
        "{% if messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
        "{% endif %}"
        "{% for message in messages if message['content'] is not none %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    )
    # fmt: on

    # fmt: off
    processor_chat_template = (
        "{% if messages[0]['role'] != 'system' %}"
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
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

    processor = AudioFlamingo3Processor(
        feature_extractor=WhisperFeatureExtractor(feature_size=128, return_attention_mask=True),
        tokenizer=AutoTokenizer.from_pretrained(str(llm_dir), chat_template=tokenizer_chat_template, use_fast=True),
        chat_template=processor_chat_template,
    )
    processor.save_pretrained(str(dst_root))

    logger.info("processor (tokenizer + preprocessor)")
    return processor


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
    model = AudioFlamingo3ForConditionalGeneration(config)

    # Update state dict to new key names if necessary
    # TODO (ebezzam) double-check
    projector_key_mapping = {
        "multi_modal_projector.layers.0.weight": "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.layers.0.bias": "multi_modal_projector.linear_1.bias",
        "multi_modal_projector.layers.2.weight": "multi_modal_projector.linear_2.weight",
        "multi_modal_projector.layers.2.bias": "multi_modal_projector.linear_2.bias",
    }
    for old_key, new_key in projector_key_mapping.items():
        if old_key in state:
            state[new_key] = state.pop(old_key)

    # Load weights into the instantiated model so we can push via `push_to_hub` later.
    load_res = model.load_state_dict(state, strict=True)
    # Enforce a clean load
    if getattr(load_res, "missing_keys", None) and load_res.missing_keys:
        mk = load_res.missing_keys
        raise ValueError(f"Missing keys when loading: {mk[:10]}{' ...' if len(mk) > 10 else ''}")
    if getattr(load_res, "unexpected_keys", None) and load_res.unexpected_keys:
        uk = load_res.unexpected_keys
        raise ValueError(f"Unexpected keys when loading: {uk[:10]}{' ...' if len(uk) > 10 else ''}")

    model.save_pretrained(save_directory=str(dst_root))
    logger.info("model.safetensors index and shards")
    return model


def write_generation_config(dst_root: Path) -> GenerationConfig:
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
    return generation_config


"""
Reproducible Usage
==================

1) Download the original AudioFlamingo-3 weights from NVIDIA (requires Git LFS):

```
git lfs install
git clone https://huggingface.co/nvidia/audio-flamingo-3
```

This will create a folder `audio-flamingo-3/` containing the original components:
`llm/`, `sound_tower/`, and `sound_mm_projector/`.

2) Convert to the Hugging Face Transformers format (locally):

```
python src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py \
  --src_dir audio-flamingo-3 \
  --dst_dir audio-flamingo-3-hf
```

3) Convert and push directly to the Hub (requires `huggingface-cli login` or `HF_TOKEN`):

```
python src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py \
  --src_dir audio-flamingo-3 \
  --dst_dir audio-flamingo-3-hf \
  --push_to_hub <username-or-org>/audio-flamingo-3
```

This command uploads both the processor (tokenizer + feature extractor) and the converted
model (sharded safetensors + configs) to the specified Hub repository.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert AudioFlamingo3 to Hugging Face format.")
    ap.add_argument("--src_dir", required=True, help="Source model root directory")
    ap.add_argument("--dst_dir", required=True, help="Destination directory for converted model")
    ap.add_argument(
        "--push_to_hub",
        default=None,
        type=str,
        help=(
            "Optional repository ID to push the converted assets to the Hugging Face Hub, "
            "e.g. 'username/audio-flamingo-3'."
        ),
    )
    args = ap.parse_args()

    src_root = Path(args.src_dir).resolve()
    dst_root = Path(args.dst_dir).resolve()

    processor = write_processor(src_root, dst_root)
    model = merge_and_shard_weights(src_root, dst_root)
    gen_config = write_generation_config(dst_root)
    # Ensure the same generation config is shipped when pushing to Hub
    model.generation_config = gen_config

    # Optionally push converted assets using native push_to_hub only
    if args.push_to_hub:
        logger.info("Pushing processor to the Hub ...")
        processor.push_to_hub(args.push_to_hub)
        logger.info("Pushing model to the Hub ...")
        model.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
