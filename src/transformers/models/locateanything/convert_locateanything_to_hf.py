# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Convert an original (remote-code) LocateAnything checkpoint to the Transformers format.

python convert_locateanything_to_hf.py --input_dir <orig> --output_dir <hf>
"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from transformers import AutoTokenizer, LocateAnythingConfig, LocateAnythingForConditionalGeneration
from transformers.models.locateanything.image_processing_locateanything import LocateAnythingImageProcessor
from transformers.models.locateanything.processing_locateanything import LocateAnythingProcessor


def rope_perm(head_dim: int) -> torch.Tensor:
    """Maps MoonViT's interleaved-complex rope layout to the rotate_half layout of apply_rotary_pos_emb_vision."""
    quarter = head_dim // 4
    ar = torch.arange(quarter)
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[ar] = 4 * ar + 2
    perm[quarter + ar] = 4 * ar
    perm[2 * quarter + ar] = 4 * ar + 3
    perm[3 * quarter + ar] = 4 * ar + 1
    return perm


def permute_qk(tensor: torch.Tensor, num_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    shape = tensor.shape
    return tensor.view(num_heads, head_dim, *shape[1:])[:, perm].reshape(shape)


def remap_key(key: str) -> str:
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model.") :]
    if key == "language_model.lm_head.weight":
        return "lm_head.weight"
    if key.startswith("vision_model.encoder.blocks."):
        rest = key[len("vision_model.encoder.blocks.") :]
        rest = re.sub(r"^(\d+)\.norm0\.", r"\1.layer_norm1.", rest)
        rest = re.sub(r"^(\d+)\.norm1\.", r"\1.layer_norm2.", rest)
        rest = re.sub(r"^(\d+)\.wo\.", r"\1.self_attn.out_proj.", rest)
        rest = re.sub(r"^(\d+)\.mlp\.fc1\.", r"\1.mlp.fc2.", rest)
        rest = re.sub(r"^(\d+)\.mlp\.fc0\.", r"\1.mlp.fc1.", rest)
        return "model.vision_tower.encoder.layers." + rest
    if key.startswith("vision_model."):
        return "model.vision_tower." + key[len("vision_model.") :]
    if key.startswith("mlp1."):
        idx_map = {"0": "pre_norm", "1": "linear_1", "3": "linear_2"}
        match = re.match(r"mlp1\.(\d+)\.(weight|bias)", key)
        return f"model.multi_modal_projector.{idx_map[match.group(1)]}.{match.group(2)}"
    raise KeyError(f"unmapped key: {key}")


def build_config(input_dir: Path) -> LocateAnythingConfig:
    cfg = json.load(open(input_dir / "config.json"))
    vc = cfg["vision_config"]
    vision_config = {
        "hidden_size": vc["hidden_size"],
        "intermediate_size": vc["intermediate_size"],
        "num_hidden_layers": vc["num_hidden_layers"],
        "num_attention_heads": vc["num_attention_heads"],
        "patch_size": vc["patch_size"],
        "init_pos_emb_height": vc["init_pos_emb_height"],
        "init_pos_emb_width": vc["init_pos_emb_width"],
        "spatial_merge_size": vc["merge_kernel_size"][0],
    }
    return LocateAnythingConfig(
        vision_config=vision_config,
        text_config=dict(cfg["text_config"]),
        image_token_id=cfg["image_token_index"],
        box_start_token_id=cfg["box_start_token_id"],
        box_end_token_id=cfg["box_end_token_id"],
        coord_start_token_id=cfg["coord_start_token_id"],
        coord_end_token_id=cfg["coord_end_token_id"],
        ref_start_token_id=cfg["ref_start_token_id"],
        ref_end_token_id=cfg["ref_end_token_id"],
        none_token_id=cfg["none_token_id"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)

    config = build_config(input_dir)
    model = LocateAnythingForConditionalGeneration(config)
    num_heads = config.vision_config.num_attention_heads
    head_dim = config.vision_config.hidden_size // num_heads
    perm = rope_perm(head_dim)

    original = {}
    for shard in sorted(input_dir.glob("model-*.safetensors")):
        original.update(load_file(str(shard)))

    state_dict = {}
    for key, tensor in original.items():
        match = re.match(r"vision_model\.encoder\.blocks\.(\d+)\.wqkv\.(weight|bias)", key)
        if match:
            layer, suffix = match.group(1), match.group(2)
            q, k, v = tensor.chunk(3, dim=0)
            base = f"model.vision_tower.encoder.layers.{layer}.self_attn"
            state_dict[f"{base}.q_proj.{suffix}"] = permute_qk(q, num_heads, head_dim, perm)
            state_dict[f"{base}.k_proj.{suffix}"] = permute_qk(k, num_heads, head_dim, perm)
            state_dict[f"{base}.v_proj.{suffix}"] = v
        else:
            state_dict[remap_key(key)] = tensor

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight map incomplete: missing={missing}, unexpected={unexpected}")

    model.to(torch.bfloat16).save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    image_processor = LocateAnythingImageProcessor()
    chat_template = json.load(open(input_dir / "chat_template.json"))["chat_template"]
    processor = LocateAnythingProcessor(
        image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template
    )
    processor.save_pretrained(output_dir)
    print(f"saved Transformers checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
