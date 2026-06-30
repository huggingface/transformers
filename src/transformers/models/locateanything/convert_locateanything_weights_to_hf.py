# Copyright 2026 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
r"""
Convert the original `nvidia/LocateAnything-3B` checkpoint (which ships custom remote code) into a transformers-native
checkpoint that can be loaded with `from_pretrained` (no `trust_remote_code`).

Example:

```bash
python src/transformers/models/locateanything/convert_locateanything_weights_to_hf.py \
    --input_repo nvidia/LocateAnything-3B \
    --output_dir ./LocateAnything-3B-hf
```

Then it can be loaded natively:

```python
from transformers import AutoProcessor, LocateAnythingForConditionalGeneration

model = LocateAnythingForConditionalGeneration.from_pretrained("./LocateAnything-3B-hf")
processor = AutoProcessor.from_pretrained("./LocateAnything-3B-hf")
```
"""

import argparse
import glob
import json
import os

import torch
from safetensors.torch import load_file as safe_load_file

from transformers import (
    AutoTokenizer,
    LocateAnythingConfig,
    LocateAnythingForConditionalGeneration,
    LocateAnythingImageProcessor,
    LocateAnythingProcessor,
    LocateAnythingVisionConfig,
    Qwen2Config,
)


# Prefix-based renaming from the original checkpoint layout to the transformers-native layout. Order matters: the more
# specific prefixes must come before the broader ones.
KEY_RENAMES = [
    ("language_model.lm_head.", "lm_head."),
    ("language_model.model.", "model.language_model."),
    ("mlp1.0.", "model.multi_modal_projector.layer_norm."),
    ("mlp1.1.", "model.multi_modal_projector.linear_1."),
    ("mlp1.3.", "model.multi_modal_projector.linear_2."),
    ("vision_model.encoder.", "model.vision_tower."),
    ("vision_model.patch_embed.", "model.vision_tower.patch_embed."),
]


def convert_key(key: str) -> str:
    for old, new in KEY_RENAMES:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def build_config(src_dir: str) -> LocateAnythingConfig:
    with open(os.path.join(src_dir, "config.json"), encoding="utf-8") as f:
        orig = json.load(f)

    v = orig["vision_config"]
    t = orig["text_config"]

    vision_config = LocateAnythingVisionConfig(
        patch_size=v["patch_size"],
        init_pos_emb_height=v["init_pos_emb_height"],
        init_pos_emb_width=v["init_pos_emb_width"],
        num_attention_heads=v["num_attention_heads"],
        num_hidden_layers=v["num_hidden_layers"],
        hidden_size=v["hidden_size"],
        intermediate_size=v["intermediate_size"],
        merge_kernel_size=tuple(v["merge_kernel_size"]),
    )

    text_config = Qwen2Config(
        vocab_size=t["vocab_size"],
        hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"],
        num_hidden_layers=t["num_hidden_layers"],
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t["num_key_value_heads"],
        hidden_act=t.get("hidden_act", "silu"),
        max_position_embeddings=t["max_position_embeddings"],
        rms_norm_eps=t.get("rms_norm_eps", 1e-6),
        rope_theta=t.get("rope_theta", 1000000.0),
        bos_token_id=t.get("bos_token_id"),
        eos_token_id=t.get("eos_token_id"),
        attention_dropout=t.get("attention_dropout", 0.0),
        tie_word_embeddings=t.get("tie_word_embeddings", True),
    )

    config = LocateAnythingConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=orig.get("image_token_index", 151665),
        block_size=t.get("block_size", 6),
        causal_attn=t.get("causal_attn", False),
        box_start_token_id=orig["box_start_token_id"],
        box_end_token_id=orig["box_end_token_id"],
        coord_start_token_id=orig["coord_start_token_id"],
        coord_end_token_id=orig["coord_end_token_id"],
        ref_start_token_id=orig["ref_start_token_id"],
        ref_end_token_id=orig["ref_end_token_id"],
        none_token_id=orig["none_token_id"],
        null_token_id=t["null_token_id"],
        switch_token_id=t["switch_token_id"],
        text_mask_token_id=t["text_mask_token_id"],
        tie_word_embeddings=t.get("tie_word_embeddings", True),
    )
    # `magi` attention is a GPU-only custom backend in the original code; default the native checkpoint to sdpa.
    config._attn_implementation = "sdpa"
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_repo", default="nvidia/LocateAnything-3B", help="HF repo id or local directory.")
    parser.add_argument("--output_dir", required=True, help="Where to write the converted transformers checkpoint.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--push_to_hub", default=None, help="Optional repo id to push the converted model to.")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    if os.path.isdir(args.input_repo):
        src_dir = args.input_repo
    else:
        from huggingface_hub import snapshot_download

        src_dir = snapshot_download(
            args.input_repo,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*", "vocab*", "merges*"],
        )

    print(f"Loading source checkpoint from: {src_dir}")
    config = build_config(src_dir)

    print("Instantiating native LocateAnything model...")
    model = LocateAnythingForConditionalGeneration(config).to(dtype)
    expected = set(model.state_dict().keys())

    shards = sorted(glob.glob(os.path.join(src_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {src_dir}")

    # Load and copy weights shard-by-shard to keep peak memory low (only the model plus a single shard reside in RAM
    # at any time, instead of two full copies of the weights).
    loaded = set()
    for shard in shards:
        shard_sd = {convert_key(k): v.to(dtype) for k, v in safe_load_file(shard).items()}
        model.load_state_dict(shard_sd, strict=False, assign=False)
        loaded.update(shard_sd.keys())
        del shard_sd

    # `lm_head.weight` is tied to the input embeddings, so it is allowed to be absent from the source weights.
    missing = sorted(expected - loaded - {"lm_head.weight"})
    unexpected = sorted(loaded - expected)
    if missing:
        raise ValueError(f"Missing keys after conversion: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected keys after conversion: {unexpected}")
    print(f"All {len(loaded)} weights mapped cleanly (lm_head is tied).")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(src_dir)
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_path = os.path.join(src_dir, "chat_template.json")
    if chat_template is None and os.path.isfile(chat_template_path):
        with open(chat_template_path, encoding="utf-8") as f:
            chat_template = json.load(f)["chat_template"]
    image_processor = LocateAnythingImageProcessor()
    processor = LocateAnythingProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    processor.save_pretrained(args.output_dir)
    print(f"Saved processor/tokenizer to {args.output_dir}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
