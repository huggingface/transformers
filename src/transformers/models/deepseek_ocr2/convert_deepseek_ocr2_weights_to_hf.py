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
import json
import os

import torch

from transformers import (
    DeepseekOcr2Config,
    DeepseekOcr2ForConditionalGeneration,
    DeepseekOcr2ImageProcessor,
    DeepseekOcr2Processor,
    PreTrainedTokenizerFast,
)


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

    config_dict.pop("projector_config", None)

    config_dict["vision_config"] = vision_config
    config_dict["model_type"] = "deepseek_ocr2"

    return config_dict


def convert_weights(input_dir: str, output_dir: str, hub_repo_id: str | None = None):
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        raise ValueError("`input_dir` and `output_dir` must be different directories.")

    os.makedirs(output_dir, exist_ok=True)

    # Config
    with open(os.path.join(input_dir, "config.json")) as f:
        raw_config = json.load(f)

    config = DeepseekOcr2Config.from_dict(convert_config(raw_config))
    config.save_pretrained(output_dir)
    print("Config saved to", output_dir)

    # Load with conversion_mapping.py (key remapping + MoE expert fusing) and save in HF format
    print(f"Loading model from {input_dir} with automatic weight conversion ...")
    model = DeepseekOcr2ForConditionalGeneration.from_pretrained(input_dir, config=config)

    print(f"Saving model to {output_dir} ...")
    model.save_pretrained(output_dir)
    del model

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
