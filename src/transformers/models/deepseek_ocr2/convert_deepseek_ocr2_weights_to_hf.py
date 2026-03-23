# Copyright 2026 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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

import argparse
import copy
import gc
import glob
import json
import os
import re

import torch
from safetensors import safe_open

from transformers import DeepseekOcr2Config, DeepseekOcr2ForConditionalGeneration, LlamaTokenizerFast


# fmt: off
# Mapping from HF Hub (original) key patterns to transformers key patterns.
# Order matters: more specific patterns must come before more general ones.
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # SAM vision encoder: blocks -> layers, rename norm/proj
    r"model\.sam_model\.blocks\.(\d+)\.norm1\.":            r"model.vision_tower.sam_encoder.layers.\1.layer_norm1.",
    r"model\.sam_model\.blocks\.(\d+)\.norm2\.":            r"model.vision_tower.sam_encoder.layers.\1.layer_norm2.",
    r"model\.sam_model\.blocks\.":                          r"model.vision_tower.sam_encoder.layers.",
    r"model\.sam_model\.patch_embed\.proj\.":               r"model.vision_tower.sam_encoder.patch_embed.projection.",
    r"model\.sam_model\.pos_embed":                         r"model.vision_tower.sam_encoder.pos_embed",

    # SAM neck: Sequential indices -> named layers
    r"model\.sam_model\.neck\.0\.":                         r"model.vision_tower.sam_encoder.neck.conv1.",
    r"model\.sam_model\.neck\.1\.":                         r"model.vision_tower.sam_encoder.neck.layer_norm1.",
    r"model\.sam_model\.neck\.2\.":                         r"model.vision_tower.sam_encoder.neck.conv2.",
    r"model\.sam_model\.neck\.3\.":                         r"model.vision_tower.sam_encoder.neck.layer_norm2.",
    # Vision proj: net_2/net_3 -> proj.conv1/conv2
    r"model\.sam_model\.net_2\.":                           r"model.vision_tower.sam_encoder.proj.conv1.",
    r"model\.sam_model\.net_3\.":                           r"model.vision_tower.sam_encoder.proj.conv2.",

    # Qwen2 vision encoder (remove extra .model nesting from original)
    r"model\.qwen2_model\.model\.model\.layers\.":          r"model.vision_tower.vision_encoder.layers.",
    r"model\.qwen2_model\.model\.model\.norm\.":            r"model.vision_tower.vision_encoder.norm.",
    r"model\.qwen2_model\.query_768\.":                     r"model.vision_tower.query_768.",
    r"model\.qwen2_model\.query_1024\.":                    r"model.vision_tower.query_1024.",

    # Projector: model.projector.layers -> model.multi_modal_projector.proj
    r"model\.projector\.layers\.":                          r"model.multi_modal_projector.proj.",

    # View separator (typo fix: "seperator" -> "separator")
    r"model\.view_seperator":                               r"model.view_separator",

    # Language model — bare decoder layers that live under model.*
    # These must come after all more specific model.* patterns above.
    r"model\.embed_tokens\.":                               r"model.language_model.embed_tokens.",
    r"model\.layers\.":                                     r"model.language_model.layers.",
    r"model\.norm\.":                                       r"model.language_model.norm.",

    # LM head (1:1 mapping)
    r"lm_head\.":                                           r"lm_head.",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    """
    Build a mapping from original keys to converted keys by applying regex
    replacements in order. Each key is transformed by the first matching
    pattern only.
    """
    output_dict = {}
    for old_key in state_dict_keys:
        new_key = old_key
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            new_key_candidate = re.sub(pattern, replacement, old_key)
            if new_key_candidate != old_key:
                new_key = new_key_candidate
                break
        output_dict[old_key] = new_key
    return output_dict


def convert_config(config_dict: dict) -> dict:
    """
    Convert a config.json from the HF Hub custom-code format to the native
    transformers format.
    """
    config_dict = copy.deepcopy(config_dict)

    # language_config -> text_config
    if "language_config" in config_dict:
        text_config = config_dict.pop("language_config")
        # This model uses MHA (use_mla=False), so MLA-specific fields are null.
        # DeepseekOcr2TextConfig defaults these to 0/None, so remove them.
        for mla_field in ("kv_lora_rank", "q_lora_rank"):
            if mla_field in text_config and text_config[mla_field] is None:
                del text_config[mla_field]
        config_dict["text_config"] = text_config

    # vision_config: restructure from original flat format
    vision_config = {}
    if "vision_config" in config_dict:
        orig_vision = config_dict.pop("vision_config")

        sam_info = orig_vision["width"]["sam_vit_b"]
        vision_config["sam_config"] = {
            "hidden_size": sam_info["width"],
            "num_hidden_layers": sam_info["layers"],
            "num_attention_heads": sam_info["heads"],
            "global_attn_indexes": sam_info["global_attn_indexes"],
            # Original config says [512, 1024] but actual weights are [512, 896].
            # See deepencoderv2.py: net_3 = nn.Conv2d(512, 896, ...)
            "downsample_channels": [512, 896],
        }

        # Qwen2 vision encoder: values from deepencoderv2.py build_qwen2_decoder_as_encoder()
        vision_config["hidden_size"] = orig_vision["width"]["qwen2-0-5b"]["dim"]
        vision_config["num_hidden_layers"] = 24
        vision_config["num_attention_heads"] = 14
        vision_config["num_key_value_heads"] = 2
        vision_config["intermediate_size"] = 4864
        vision_config["max_query"] = 400
        vision_config["rms_norm_eps"] = 1e-6
        vision_config["rope_theta"] = 1000000.0
        vision_config["vocab_size"] = 1

    # projector_config -> flat fields
    proj = config_dict.pop("projector_config")
    config_dict["projector_input_dim"] = proj["input_dim"]
    config_dict["projector_n_embed"] = proj["n_embed"]
    config_dict["projector_type"] = proj["projector_type"]

    config_dict["vision_config"] = vision_config
    config_dict["model_type"] = "deepseek_ocr2"

    return config_dict


def load_original_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    """Load all safetensors shards from *input_dir* into a single state dict."""
    safetensor_files = sorted(glob.glob(os.path.join(input_dir, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {input_dir}")

    state_dict = {}
    for path in safetensor_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def fuse_moe_experts(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Fuse individual MoE expert weights into 3D tensors.
    """
    expert_pattern = re.compile(
        r"(model\.language_model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    )

    # Collect expert weights grouped by layer prefix
    expert_groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    fused_keys = set()

    for key, tensor in state_dict.items():
        m = expert_pattern.match(key)
        if m:
            prefix, expert_idx, proj_type = m.group(1), int(m.group(2)), m.group(3)
            expert_groups.setdefault(prefix, {}).setdefault(expert_idx, {})[proj_type] = tensor
            fused_keys.add(key)

    # Build fused tensors
    fused = {}
    for prefix, experts in expert_groups.items():
        num_experts = len(experts)
        gate_up_list, down_list = [], []
        for idx in range(num_experts):
            gate_up_list.append(torch.cat([experts[idx]["gate_proj"], experts[idx]["up_proj"]], dim=0))
            down_list.append(experts[idx]["down_proj"])
        fused[f"{prefix}.gate_up_proj"] = torch.stack(gate_up_list, dim=0)
        fused[f"{prefix}.down_proj"] = torch.stack(down_list, dim=0)

    # Replace individual keys with fused
    for key in fused_keys:
        del state_dict[key]
    state_dict.update(fused)

    print(f"  Fused {len(fused_keys)} individual expert keys into {len(fused)} fused tensors")
    return state_dict


def convert_weights(input_dir: str, output_dir: str, push_to_hub: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Config ----
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)
    converted_config = convert_config(raw_config)

    config = DeepseekOcr2Config.from_dict(converted_config)
    config.save_pretrained(output_dir)
    print("Config saved to", output_dir)

    # ---- Weights ----
    print(f"Loading original weights from {input_dir} ...")
    original_state_dict = load_original_state_dict(input_dir)
    print(f"  Loaded {len(original_state_dict)} tensors.")

    # Remap keys
    all_keys = list(original_state_dict.keys())
    key_mapping = convert_old_keys_to_new_keys(all_keys)

    new_state_dict: dict[str, torch.Tensor] = {}
    for old_key in all_keys:
        new_state_dict[key_mapping[old_key]] = original_state_dict[old_key]

    del original_state_dict
    gc.collect()

    # Log renamed keys
    renamed = {k: v for k, v in key_mapping.items() if k != v}
    if renamed:
        print(f"  Renamed {len(renamed)} keys:")
        for old_k, new_k in list(renamed.items())[:20]:
            print(f"    {old_k}  ->  {new_k}")
        if len(renamed) > 20:
            print(f"    ... and {len(renamed) - 20} more")

    # Fuse MoE experts
    print("  Fusing MoE expert weights ...")
    new_state_dict = fuse_moe_experts(new_state_dict)

    # ---- Instantiate model and load ----
    print("Loading state dict into DeepseekOcr2ForConditionalGeneration ...")
    model = DeepseekOcr2ForConditionalGeneration(config)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    if missing_keys:
        print(f"  Missing keys ({len(missing_keys)}):")
        for k in missing_keys[:20]:
            print(f"    {k}")
    if unexpected_keys:
        print(f"  Unexpected keys ({len(unexpected_keys)}):")
        for k in unexpected_keys[:20]:
            print(f"    {k}")

    model = model.to(torch.bfloat16)
    print("  Model dtype:", model.dtype)

    # ---- Save ----
    print(f"Saving model to {output_dir} ...")
    model.save_pretrained(output_dir)

    del new_state_dict, model
    gc.collect()

    # ---- Tokenizer ----
    print("Copying tokenizer ...")
    tokenizer = LlamaTokenizerFast.from_pretrained(input_dir)
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer saved.")

    if push_to_hub:
        print("Pushing to hub ...")
        model = DeepseekOcr2ForConditionalGeneration.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
        model.push_to_hub("deepseek-ai/DeepSeek-OCR-2-hf")
        tokenizer.push_to_hub("deepseek-ai/DeepSeek-OCR-2-hf")

    print("Done.")


def main():
    """
    Download the original model and convert to transformers format:
        huggingface-cli download deepseek-ai/DeepSeek-OCR-2 --local-dir /path/to/DeepSeek-OCR-2

        python convert_deepseek_ocr2_weights_to_hf.py \
            --input_dir /path/to/DeepSeek-OCR-2 \
            --output_dir /path/to/output
    """
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek-OCR-2 weights from HF Hub custom-code format to transformers format.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the downloaded DeepSeek-OCR-2 checkpoint directory (with config.json and *.safetensors).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to write the converted transformers-compatible model.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model and tokenizer to the Hugging Face Hub.",
    )
    args = parser.parse_args()
    convert_weights(args.input_dir, args.output_dir, push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()
