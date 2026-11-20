# Copyright 2026 Deepseek AI and The HuggingFace Team. All rights reserved.
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
"""
Converts an original DeepSeek-VL2 checkpoint (deepseek-ai/deepseek-vl2-tiny, -small or deepseek-vl2) to the
Transformers format.

    python src/transformers/models/deepseek_vl_v2/convert_deepseek_vl_v2_weights_to_hf.py \
        --input_dir /path/to/deepseek-vl2-tiny --output_dir /path/to/deepseek-vl2-tiny-hf
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch
from safetensors import safe_open

from transformers import (
    AutoTokenizer,
    DeepseekVLV2Config,
    DeepseekVLV2ForConditionalGeneration,
    DeepseekVLV2ImageProcessor,
    DeepseekVLV2Processor,
)


# Keys of the timm attention-pooling head; DeepSeek-VL2 reads patch tokens and never runs it.
DROPPED_PREFIXES = ("vision.attn_pool.",)

RENAMES = [
    (r"^vision\.patch_embed\.proj\.", "model.vision_tower.embeddings.patch_embedding."),
    (r"^vision\.blocks\.(\d+)\.norm1\.", r"model.vision_tower.encoder.layers.\1.layer_norm1."),
    (r"^vision\.blocks\.(\d+)\.norm2\.", r"model.vision_tower.encoder.layers.\1.layer_norm2."),
    (r"^vision\.blocks\.(\d+)\.attn\.proj\.", r"model.vision_tower.encoder.layers.\1.self_attn.out_proj."),
    (r"^vision\.blocks\.(\d+)\.mlp\.", r"model.vision_tower.encoder.layers.\1.mlp."),
    (r"^vision\.norm\.", "model.vision_tower.post_layernorm."),
    (r"^projector\.layers\.0\.", "model.multi_modal_projector.linear_1."),
    (r"^projector\.layers\.2\.", "model.multi_modal_projector.linear_2."),
    (r"^image_newline$", "model.image_newline"),
    (r"^view_seperator$", "model.view_separator"),
    (r"^language\.lm_head\.", "lm_head."),
    (r"^language\.model\.", "model.language_model."),
]
QKV = re.compile(r"^vision\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)$")
EXPERT = re.compile(r"^language\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def convert_config(original: dict) -> DeepseekVLV2Config:
    language = original["language_config"]
    vision = original["vision_config"]
    if vision.get("model_name") != "siglip_so400m_patch14_384":
        raise ValueError(f"Only the SigLIP so400m/384 vision tower is supported, got {vision.get('model_name')}")

    text_keys = [
        "vocab_size", "hidden_size", "intermediate_size", "moe_intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "n_group", "topk_group", "topk_method", "max_position_embeddings", "bos_token_id", "eos_token_id",
    ]  # fmt: skip
    text_config = {key: language[key] for key in text_keys if language.get(key) is not None}
    text_config["routed_scaling_factor"] = language.get("routed_scaling_factor") or 1.0
    text_config["rms_norm_eps"] = language.get("rms_norm_eps", 1e-6)
    text_config["rope_parameters"] = {"rope_type": "default", "rope_theta": language.get("rope_theta", 10000.0)}
    num_dense = language.get("first_k_dense_replace", 0)

    if language.get("use_mla") is False:
        # deepseek-vl2-tiny: plain multi-head attention over a DeepSeek MoE, the text model DeepSeek-OCR-2 uses.
        text_config["model_type"] = "deepseek_ocr2_text"
        text_config["mlp_layer_types"] = ["dense"] * num_dense + ["sparse"] * (
            language["num_hidden_layers"] - num_dense
        )
    else:
        # deepseek-vl2-small and deepseek-vl2: MLA without query compression.
        text_config["model_type"] = "deepseek_v3" if language.get("topk_method") == "noaux_tc" else "deepseek_v2"
        text_config["first_k_dense_replace"] = num_dense
        for key in ["kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"]:
            text_config[key] = language.get(key)
        if text_config["model_type"] == "deepseek_v3":
            text_config["norm_topk_prob"] = language.get("norm_topk_prob", True)

    # Spelled out: sub-config defaults only apply when no vision_config is passed at all.
    vision_config = {
        "model_type": "siglip_vision_model",
        "hidden_size": vision["width"],
        "intermediate_size": 4304,
        "num_hidden_layers": vision["layers"],
        "num_attention_heads": 16,
        "image_size": 384,
        "patch_size": vision["patch_size"],
        "vision_use_head": False,
    }
    return DeepseekVLV2Config(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=128815,
    )


def convert_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    with open(os.path.join(input_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))

    state_dict = {}
    experts = defaultdict(dict)  # layer -> {(expert, proj): tensor}
    for shard in shards:
        with safe_open(os.path.join(input_dir, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith(DROPPED_PREFIXES):
                    continue
                tensor = f.get_tensor(key)
                if match := QKV.match(key):
                    layer, kind = match.groups()
                    prefix = f"model.vision_tower.encoder.layers.{layer}.self_attn"
                    for name, part in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
                        state_dict[f"{prefix}.{name}.{kind}"] = part.contiguous()
                elif match := EXPERT.match(key):
                    layer, expert, proj = match.groups()
                    experts[int(layer)][(int(expert), proj)] = tensor
                elif key == "vision.pos_embed":
                    state_dict["model.vision_tower.embeddings.position_embedding.weight"] = tensor.squeeze(0)
                else:
                    new_key = key
                    for pattern, replacement in RENAMES:
                        new_key, count = re.subn(pattern, replacement, new_key)
                        if count:
                            break
                    else:
                        raise ValueError(f"No conversion rule for checkpoint key {key}")
                    state_dict[new_key] = tensor

    for layer, tensors in experts.items():
        num_experts = 1 + max(expert for expert, _ in tensors)
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        state_dict[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([tensors[(e, "gate_proj")], tensors[(e, "up_proj")]], dim=0) for e in range(num_experts)]
        )
        state_dict[f"{prefix}.down_proj"] = torch.stack([tensors[(e, "down_proj")] for e in range(num_experts)])
    return state_dict


def convert(input_dir: str, output_dir: str, push_to_hub: str | None = None):
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        raise ValueError("input_dir and output_dir must differ")
    with open(os.path.join(input_dir, "config.json"), encoding="utf-8") as f:
        config = convert_config(json.load(f))

    with torch.device("meta"):
        model = DeepseekVLV2ForConditionalGeneration(config)
    state_dict = convert_state_dict(input_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"Conversion mismatch. Missing: {missing[:10]} Unexpected: {unexpected[:10]}")
    model.to(torch.bfloat16)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    with open(os.path.join(input_dir, "processor_config.json"), encoding="utf-8") as f:
        candidate_resolutions = json.load(f)["candidate_resolutions"]
    processor = DeepseekVLV2Processor(
        image_processor=DeepseekVLV2ImageProcessor(candidate_resolutions=candidate_resolutions),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(output_dir)
    print(f"Saved {len(state_dict)} tensors, config and processor to {output_dir}")

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with the original DeepSeek-VL2 checkpoint")
    parser.add_argument("--output_dir", required=True, help="Where to write the Transformers checkpoint")
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to upload the result to")
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.push_to_hub)
