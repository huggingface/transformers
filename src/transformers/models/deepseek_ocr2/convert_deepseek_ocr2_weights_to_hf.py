# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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

"""Convert DeepSeek-OCR-2 weights from HF Hub custom-code format to native transformers format."""

import argparse
import copy
import gc
import glob
import json
import os
import re

import torch
from safetensors import safe_open

from transformers import (
    DeepseekOcr2Config,
    DeepseekOcr2ForConditionalGeneration,
    DeepseekOcr2ImageProcessor,
    DeepseekOcr2Processor,
    PreTrainedTokenizerFast,
)


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # SAM vision encoder
    r"model\.sam_model\.blocks\.(\d+)\.norm1\.":            r"model.vision_tower.sam_encoder.layers.\1.layer_norm1.",
    r"model\.sam_model\.blocks\.(\d+)\.norm2\.":            r"model.vision_tower.sam_encoder.layers.\1.layer_norm2.",
    r"model\.sam_model\.blocks\.":                          r"model.vision_tower.sam_encoder.layers.",
    r"model\.sam_model\.patch_embed\.proj\.":               r"model.vision_tower.sam_encoder.patch_embed.projection.",
    r"model\.sam_model\.pos_embed":                         r"model.vision_tower.sam_encoder.pos_embed",
    # SAM neck
    r"model\.sam_model\.neck\.0\.":                         r"model.vision_tower.sam_encoder.neck.conv1.",
    r"model\.sam_model\.neck\.1\.":                         r"model.vision_tower.sam_encoder.neck.layer_norm1.",
    r"model\.sam_model\.neck\.2\.":                         r"model.vision_tower.sam_encoder.neck.conv2.",
    r"model\.sam_model\.neck\.3\.":                         r"model.vision_tower.sam_encoder.neck.layer_norm2.",
    # Vision proj
    r"model\.sam_model\.net_2\.":                           r"model.vision_tower.sam_encoder.proj.conv1.",
    r"model\.sam_model\.net_3\.":                           r"model.vision_tower.sam_encoder.proj.conv2.",
    # Qwen2 vision encoder
    r"model\.qwen2_model\.model\.model\.layers\.":          r"model.vision_tower.vision_encoder.layers.",
    r"model\.qwen2_model\.model\.model\.norm\.":            r"model.vision_tower.vision_encoder.norm.",
    r"model\.qwen2_model\.query_768\.":                     r"model.vision_tower.query_768.",
    r"model\.qwen2_model\.query_1024\.":                    r"model.vision_tower.query_1024.",
    # Projector
    r"model\.projector\.layers\.":                          r"model.multi_modal_projector.",
    # View separator (typo fix: "seperator" -> "separator")
    r"model\.view_seperator":                               r"model.view_separator",
    # Language model (must come after all more specific model.* patterns)
    r"model\.embed_tokens\.":                               r"model.language_model.embed_tokens.",
    r"model\.layers\.":                                     r"model.language_model.layers.",
    r"model\.norm\.":                                       r"model.language_model.norm.",
    # LM head
    r"lm_head\.":                                           r"lm_head.",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
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
    config_dict = copy.deepcopy(config_dict)

    if "language_config" in config_dict:
        text_config = config_dict.pop("language_config")
        for mla_field in ("kv_lora_rank", "q_lora_rank"):
            if mla_field in text_config and text_config[mla_field] is None:
                del text_config[mla_field]
        first_k = text_config.pop("first_k_dense_replace", 0)
        n_layers = text_config.get("num_hidden_layers", 28)
        text_config["mlp_layer_types"] = ["dense"] * first_k + ["sparse"] * (n_layers - first_k)
        config_dict["text_config"] = text_config

    vision_config = {}
    if "vision_config" in config_dict:
        orig_vision = config_dict.pop("vision_config")

        sam_info = orig_vision["width"]["sam_vit_b"]
        vision_config["sam_config"] = {
            "hidden_size": sam_info["width"],
            "num_hidden_layers": sam_info["layers"],
            "num_attention_heads": sam_info["heads"],
            "global_attn_indexes": sam_info["global_attn_indexes"],
            "downsample_channels": [512, 896],
        }

        vision_config["encoder_config"] = {
            "hidden_size": orig_vision["width"]["qwen2-0-5b"]["dim"],
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "vocab_size": 1,
        }

    proj = config_dict.pop("projector_config")
    config_dict["projector_input_dim"] = proj["input_dim"]
    config_dict["projector_n_embed"] = proj["n_embed"]

    config_dict["vision_config"] = vision_config
    config_dict["model_type"] = "deepseek_ocr2"

    return config_dict


def load_original_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
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
    expert_pattern = re.compile(
        r"(model\.language_model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    )

    expert_groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    fused_keys = set()

    for key, tensor in state_dict.items():
        m = expert_pattern.match(key)
        if m:
            prefix, expert_idx, proj_type = m.group(1), int(m.group(2)), m.group(3)
            expert_groups.setdefault(prefix, {}).setdefault(expert_idx, {})[proj_type] = tensor
            fused_keys.add(key)

    fused = {}
    for prefix, experts in expert_groups.items():
        gate_up_list, down_list = [], []
        for idx in range(len(experts)):
            gate_up_list.append(torch.cat([experts[idx]["gate_proj"], experts[idx]["up_proj"]], dim=0))
            down_list.append(experts[idx]["down_proj"])
        fused[f"{prefix}.gate_up_proj"] = torch.stack(gate_up_list, dim=0)
        fused[f"{prefix}.down_proj"] = torch.stack(down_list, dim=0)

    for key in fused_keys:
        del state_dict[key]
    state_dict.update(fused)

    print(f"  Fused {len(fused_keys)} individual expert keys into {len(fused)} fused tensors")
    return state_dict


def convert_weights(input_dir: str, output_dir: str, hub_repo_id: str | None = None):
    os.makedirs(output_dir, exist_ok=True)

    # Config
    with open(os.path.join(input_dir, "config.json")) as f:
        raw_config = json.load(f)

    config = DeepseekOcr2Config.from_dict(convert_config(raw_config))
    config.save_pretrained(output_dir)
    print("Config saved to", output_dir)

    # Weights
    print(f"Loading original weights from {input_dir} ...")
    original_state_dict = load_original_state_dict(input_dir)
    print(f"  Loaded {len(original_state_dict)} tensors.")

    all_keys = list(original_state_dict.keys())
    key_mapping = convert_old_keys_to_new_keys(all_keys)

    new_state_dict = {key_mapping[k]: original_state_dict[k] for k in all_keys}
    del original_state_dict
    gc.collect()

    renamed = {k: v for k, v in key_mapping.items() if k != v}
    if renamed:
        print(f"  Renamed {len(renamed)} keys:")
        for old_k, new_k in list(renamed.items())[:20]:
            print(f"    {old_k}  ->  {new_k}")
        if len(renamed) > 20:
            print(f"    ... and {len(renamed) - 20} more")

    print("  Fusing MoE expert weights ...")
    new_state_dict = fuse_moe_experts(new_state_dict)

    # Load into model
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

    # Save
    print(f"Saving model to {output_dir} ...")
    model.save_pretrained(output_dir)

    del new_state_dict, model
    gc.collect()

    print("Copying tokenizer ...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(input_dir)
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer saved.")

    print("Saving processor ...")
    image_processor = DeepseekOcr2ImageProcessor()
    processor = DeepseekOcr2Processor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(output_dir)
    print("Processor saved.")

    if hub_repo_id:
        print(f"Pushing to hub ({hub_repo_id}) ...")
        model = DeepseekOcr2ForConditionalGeneration.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
        model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)
        processor.push_to_hub(hub_repo_id)

    print("Done.")


def test(output_dir: str):
    """Run a quick inference test on the converted model."""
    import requests
    from PIL import Image

    image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"

    print(f"\n{'=' * 60}")
    print("Running inference test...")
    print(f"Image: {image_url}")

    model = DeepseekOcr2ForConditionalGeneration.from_pretrained(
        output_dir, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
    processor = DeepseekOcr2Processor(image_processor=DeepseekOcr2ImageProcessor(), tokenizer=tokenizer)

    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    print(f"Image size: {image.size[0]}x{image.size[1]}")

    inputs = processor(images=image, text="<image>\nFree OCR.", return_tensors="pt").to(
        model.device, dtype=torch.bfloat16
    )
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=4096,
            do_sample=False,
            no_repeat_ngram_size=35,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f"Generated {len(generated)} tokens")
    print(f"Output:\n{output_text[:500]}")
    print(f"{'=' * 60}")


def main():
    """
    Convert DeepSeek-OCR-2 weights from HF Hub custom-code format to native transformers format.

    Usage:
        # Step 1: Download the original checkpoint
        huggingface-cli download deepseek-ai/DeepSeek-OCR-2 --local-dir /path/to/DeepSeek-OCR-2

        # Step 2: Convert to native transformers format
        python convert_deepseek_ocr2_weights_to_hf.py \\
            --input_dir /path/to/DeepSeek-OCR-2 \\
            --output_dir /path/to/DeepSeek-OCR-2-hf

        # Step 3 (optional): Verify with a quick inference test
        python convert_deepseek_ocr2_weights_to_hf.py \\
            --input_dir /path/to/DeepSeek-OCR-2 \\
            --output_dir /path/to/DeepSeek-OCR-2-hf \\
            --test
    """
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to the downloaded DeepSeek-OCR-2 checkpoint."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Path to write the converted model.")
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Push converted model to this HF Hub repo (e.g. 'my-org/DeepSeek-OCR-2-hf').",
    )
    parser.add_argument("--test", action="store_true", help="Run inference test after conversion.")
    args = parser.parse_args()

    convert_weights(args.input_dir, args.output_dir, hub_repo_id=args.hub_repo_id)

    if args.test:
        test(args.output_dir)


if __name__ == "__main__":
    main()
