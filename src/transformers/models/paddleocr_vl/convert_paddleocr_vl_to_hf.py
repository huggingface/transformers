# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import glob
import re

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AutoProcessor,
    PaddleOCRTextConfig,
    PaddleOCRVisionConfig,
    PaddleOCRVLConfig,
    PaddleOCRVLForConditionalGeneration,
)


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^visual\.": r"model.visual.",
    r"^mlp_AR\.": r"model.projector.",
    r"^model\.(?!visual\.|projector\.|language_model\.)": r"model.language_model.",
}

# Keys present in the original checkpoint that are not needed
KEYS_TO_IGNORE = [
    "packing_position_embedding",
    "vision_model.head",
]


def convert_old_keys_to_new_keys(state_dict_keys):
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)
                continue
            new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in sorted(glob.glob(f"{directory_path}/*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def get_paddleocr_vl_config():
    vision_config = PaddleOCRVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        spatial_merge_size=2,
    )

    text_config = PaddleOCRTextConfig(
        vocab_size=103424,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=18,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        use_bias=False,
        head_dim=128,
        rope_theta=500000.0,
        rope_scaling={
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default",
        },
    )

    config = PaddleOCRVLConfig(
        vision_config=vision_config.to_dict(),
        text_config=text_config.to_dict(),
        image_token_id=100295,
        video_token_id=101307,
        vision_start_token_id=101305,
        vision_end_token_id=101306,
        tie_word_embeddings=True,
    )

    return config


@torch.no_grad()
def convert_paddleocr_vl_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False, verify_logits=True):

    print(f"Loading original state dict from {model_name}...")
    original_state_dict = load_original_state_dict(model_name)
    print(f"Loaded {len(original_state_dict)} keys from original checkpoint.")

    # 2. Convert keys
    all_keys = list(original_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    state_dict = {}
    for old_key in all_keys:
        new_key = new_keys[old_key]

        if any(ignored in old_key for ignored in KEYS_TO_IGNORE):
            print(f"  Skipping: {old_key}")
            continue

        state_dict[new_key] = original_state_dict[old_key]

    embed_key = "model.language_model.embed_tokens.weight"
    lm_head_key = "lm_head.weight"
    if lm_head_key in state_dict and embed_key in state_dict:
        if torch.equal(state_dict[lm_head_key], state_dict[embed_key]):
            print("lm_head.weight is identical to embed_tokens.weight (will be tied after save).")
        else:
            print("WARNING: lm_head.weight differs from embed_tokens.weight.")

    print(f"Converted state dict has {len(state_dict)} keys.")

    config = get_paddleocr_vl_config()

    print("Loading weights into PaddleOCRVLForConditionalGeneration...")
    with torch.device("meta"):
        model = PaddleOCRVLForConditionalGeneration(config)

    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    print("Checkpoint loaded successfully.")

    print(f"Saving processor from {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving converted model to {pytorch_dump_folder_path}...")
    model.save_pretrained(pytorch_dump_folder_path)
    print("Model saved successfully.")

    if verify_logits:
        print("Verifying logits between original and converted model...")
        verify_model_outputs(model_name, pytorch_dump_folder_path, processor)

    if push_to_hub:
        print("Pushing model and processor to the hub...")
        model.push_to_hub(f"PaddlePaddle/PaddleOCR-VL-hf")
        processor.push_to_hub(f"PaddlePaddle/PaddleOCR-VL-hf")
        print("Pushed to hub successfully.")


def verify_model_outputs(original_model_name, converted_model_path, processor):
    print("  Loading original model via native PaddleOCRVLForConditionalGeneration...")
    original_model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        original_model_name,
        torch_dtype=torch.bfloat16,
    ).eval()

    # Load converted model
    print("  Loading converted model...")
    converted_model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        converted_model_path,
        torch_dtype=torch.bfloat16,
    ).eval()

    dummy_image = Image.new("RGB", (56, 56), color=(128, 100, 80))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    print("  Running forward pass on original model...")
    original_inputs = {k: v.to(original_model.device) for k, v in inputs.items()}
    original_outputs = original_model(**original_inputs)

    print("  Running forward pass on converted model...")
    converted_inputs = {k: v.to(converted_model.device) for k, v in inputs.items()}
    converted_outputs = converted_model(**converted_inputs)

    # Compare logits
    original_logits = original_outputs.logits
    converted_logits = converted_outputs.logits

    print(f"  Original logits shape: {original_logits.shape}")
    print(f"  Converted logits shape: {converted_logits.shape}")
    print(f"  Original logits sample: {original_logits[0, :3, :3]}")
    print(f"  Converted logits sample: {converted_logits[0, :3, :3]}")

    torch.testing.assert_close(original_logits, converted_logits, atol=1e-4, rtol=1e-4)
    print("  Logits match! Conversion verified successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="PaddlePaddle/PaddleOCR-VL",
        type=str,
        help="Hub ID of the original PaddleOCR-VL model.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=str,
        help="Path to the output directory where the converted model will be saved.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )
    parser.add_argument(
        "--no_verify_logits",
        action="store_true",
        help="Skip logits verification between original and converted model.",
    )

    args = parser.parse_args()
    convert_paddleocr_vl_checkpoint(
        model_name=args.model_name,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        push_to_hub=args.push_to_hub,
        verify_logits=not args.no_verify_logits,
    )


if __name__ == "__main__":
    main()
