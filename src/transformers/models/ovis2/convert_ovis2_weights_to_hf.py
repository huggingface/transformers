# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import os
import re

import requests
import torch
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.ovis2.configuration_ovis2 import Ovis2Config, Ovis2VisionConfig
from transformers.models.ovis2.image_processing_ovis2 import Ovis2ImageProcessor
from transformers.models.ovis2.modeling_ovis2 import Ovis2ForConditionalGeneration
from transformers.models.ovis2.processing_ovis2 import Ovis2Processor
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


# Constants
CONTEXT_LENGTH = 32768  # multimodal_max_length


# fmt: off

# Mapping from original model key patterns to HF key patterns
ORIGINAL_TO_HF_MAPPING = {
    r"trunk.blocks\.(\d+)\.norm_1":                 r"encoder.layers.\1.rms_norm1",
    r"trunk.blocks\.(\d+)\.norm_2":                 r"encoder.layers.\1.rms_norm2",
    r"trunk.blocks\.(\d+)\.attn.proj":              r"encoder.layers.\1.attention.out_proj",
    r"visual_tokenizer":                            r"model.vision_tower",
    r"backbone":                                    r"transformer",
    r"preprocessor":                                r"embeddings",
    r"patchifier.proj":                             r"patch_embedding",
    r"patchifier.norm":                             r"rms_norm",
    r"trunk.post_trunk_norm":                       r"rms_norm",
    r"trunk.blocks":                                r"encoder.layers",
    r"mlp.fc1":                                     r"ffn.gate_proj",
    r"mlp.fc2":                                     r"ffn.down_proj",
    r"mlp.fc3":                                     r"ffn.up_proj",
    r"head.0":                                      r"head_linear",
    r"head.1":                                      r"head_norm",
    r"vte.weight":                                  r"model.visual_embeddings_table.weight",
    r"llm.model":                                   r"model.language_model",
    r"llm.lm_head":                                 r"lm_head",
}
# fmt: on

# Special tokens for the tokenizer
SPECIAL_TOKENS = [
    "<IMG_ATOM>",
    "<IMG_START>",
    "<IMG_GRID>",
    "<IMG_COL>",
    "<IMG_ROW>",
    "<IMG_END>",
]

# Configuration keys to ignore when converting
UNNECESSARY_CONFIG_KEYS = [
    "_name_or_path",
    "_attn_implementation_autoset",
    "auto_map",
    "use_bfloat16",
    "use_flash_attn",
    "qk_normalization",
    "bias",
    "norm_type",
]

# Chat template for the tokenizer
CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}"
    "{{ '<image>\n' }}"
    "{% elif content['type'] == 'text' %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{'<|im_end|>\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def create_tokenizer(model_name_or_path, save_dir):
    """
    Create and configure a tokenizer for the Ovis2 model.

    Args:
        model_name_or_path: Path to the source model or tokenizer
        save_dir: Directory to save the tokenizer to

    Returns:
        The configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, return_token_type_ids=False)
    tokenizer.model_max_length = CONTEXT_LENGTH
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    tokenizer.chat_template = CHAT_TEMPLATE
    setattr(tokenizer, "image_token", "<IMG_ATOM>")  # 151665
    setattr(tokenizer, "image_token_id", tokenizer.convert_tokens_to_ids(tokenizer.image_token))

    return tokenizer


def create_image_processor(save_dir):
    """
    Create and save an image processor for the Ovis2 model.

    Args:
        save_dir: Directory to save the image processor to

    Returns:
        The configured image processor
    """
    image_processor = Ovis2ImageProcessor(
        crop_to_patches=True,
        size={"height": 448, "width": 448},
    )
    return image_processor


def extract_vision_config_from_original(orig_config):
    """
    Extract and format vision configuration from the original model config.

    Args:
        orig_config: Original model configuration

    Returns:
        dict: Cleaned vision configuration dictionary
    """
    visual_tokenizer_config = orig_config.visual_tokenizer_config.to_dict()
    # backbone_config = visual_tokenizer_config.pop("backbone_config")

    # Copy required fields from backbone config
    visual_tokenizer_config["hidden_size"] = orig_config.visual_tokenizer_config.backbone_config.hidden_size
    visual_tokenizer_config["intermediate_size"] = (
        orig_config.visual_tokenizer_config.backbone_config.intermediate_size
    )
    visual_tokenizer_config["num_attention_heads"] = (
        orig_config.visual_tokenizer_config.backbone_config.num_attention_heads
    )
    visual_tokenizer_config["num_hidden_layers"] = (
        orig_config.visual_tokenizer_config.backbone_config.num_hidden_layers
    )
    visual_tokenizer_config["rms_norm_eps"] = orig_config.visual_tokenizer_config.backbone_config.rms_norm_eps
    visual_tokenizer_config["image_size"] = orig_config.visual_tokenizer_config.backbone_config.image_size
    visual_tokenizer_config["num_channels"] = orig_config.visual_tokenizer_config.backbone_config.num_channels
    visual_tokenizer_config["patch_size"] = orig_config.visual_tokenizer_config.backbone_config.patch_size
    visual_tokenizer_config["qkv_bias"] = orig_config.visual_tokenizer_config.backbone_config.qkv_bias

    # Remove unnecessary keys
    return {k: v for k, v in visual_tokenizer_config.items() if k not in UNNECESSARY_CONFIG_KEYS}


def get_ovis2_config(model_name_or_path):
    """
    Create an Ovis2 configuration from the original model.

    Args:
        model_name_or_path: Path to the original model

    Returns:
        Ovis2Config: Configuration for the HF implementation
    """
    orig_config = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    ).config

    # Extract and clean LLM config
    llm_config = orig_config.llm_config.to_dict()
    llm_config = {k: v for k, v in llm_config.items() if k not in UNNECESSARY_CONFIG_KEYS}

    # Extract and clean vision config
    visual_tokenizer_config = extract_vision_config_from_original(orig_config)

    return Ovis2Config(
        text_config=Qwen2Config(**llm_config),
        vision_config=Ovis2VisionConfig(**visual_tokenizer_config),
        hidden_size=llm_config["hidden_size"],
        vocab_size=llm_config["vocab_size"],
        initializer_range=llm_config["initializer_range"],
    )


def load_orig_state_dict(model_name_or_path):
    """
    Load the state dictionary from the original model.

    Args:
        model_name_or_path: Path to the original model

    Returns:
        dict: Original model state dictionary
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    return model.state_dict()


def convert_orig2hf(state_dict, dim):
    """
    Convert original state dictionary keys to HF format.

    Args:
        state_dict: Original state dictionary
        dim: Hidden dimension for splitting QKV weights

    Returns:
        dict: Converted state dictionary for HF model
    """
    new_state_dict = {}

    for key, val in state_dict.items():
        orig_key = key

        # Apply regex pattern replacements
        for pattern, replacement in ORIGINAL_TO_HF_MAPPING.items():
            key = re.sub(pattern, replacement, key)

        # Handle special cases
        if "attn.qkv" in key:
            # Split QKV into separate Q, K, V matrices
            new_key_query = key.replace("attn.qkv", "attention.q_proj")
            new_state_dict[new_key_query] = state_dict[orig_key][:dim]

            new_key_key = key.replace("attn.qkv", "attention.k_proj")
            new_state_dict[new_key_key] = state_dict[orig_key][dim : 2 * dim]

            new_key_value = key.replace("attn.qkv", "attention.v_proj")
            new_state_dict[new_key_value] = state_dict[orig_key][-dim:]

        elif "pos_embed" in key:
            new_key = key.replace("pos_embed", "position_embedding.weight")
            new_state_dict[new_key] = state_dict[orig_key][0]

        else:
            new_state_dict[key] = val

    return new_state_dict


def convert_model(model_name_or_path):
    """
    Convert and save the model in HF format.

    Args:
        model_name_or_path: Path to the original model
        save_dir: Directory to save the converted model

    Returns:
        The converted model
    """

    config = get_ovis2_config(model_name_or_path)
    config.architectures = ["Ovis2ForConditionalGeneration"]

    # Load and convert weights
    orig_state_dict = load_orig_state_dict(model_name_or_path)
    new_state_dict = convert_orig2hf(orig_state_dict, config.vision_config.hidden_size)

    # Create model and load converted weights
    model = Ovis2ForConditionalGeneration(config)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # Report any issues with weight loading
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model


def main():
    """Process command line arguments and execute the conversion pipeline."""
    parser = argparse.ArgumentParser(description="Convert Ovis2 model to HF format")
    parser.add_argument(
        "--model_name_or_path",
        default="AIDC-AI/Ovis2-2B",
        choices=[
            "AIDC-AI/Ovis2-1B",
            "AIDC-AI/Ovis2-2B",
            "AIDC-AI/Ovis2-4B",
            "AIDC-AI/Ovis2-8B",
            "AIDC-AI/Ovis2-16B",
            "AIDC-AI/Ovis2-34B",
        ],
        help="Location of original Ovis2 model",
    )
    parser.add_argument("--save_dir", default="Ovis2-2B-hf", help="Location to write HF model and processors")
    parser.add_argument("--hub_dir", default="thisisiron/Ovis2-2B-hf", help="Hub repository name if pushing to hub")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether to push the converted model to the Hugging Face hub"
    )

    args = parser.parse_args()

    # Execute conversion pipeline
    print(f"Converting model from {args.model_name_or_path} to {args.save_dir}")

    # If already included in the transformers library, remove to avoid duplication.
    if "aimv2" in CONFIG_MAPPING_NAMES:
        CONFIG_MAPPING_NAMES.pop("aimv2")

    tokenizer = create_tokenizer(
        model_name_or_path=args.model_name_or_path,
        save_dir=args.save_dir,
    )

    image_processor = create_image_processor(
        save_dir=args.save_dir,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Convert and save the model
    model = convert_model(model_name_or_path=args.model_name_or_path)
    model.save_pretrained(args.save_dir)

    # Save the processor
    processor = Ovis2Processor(tokenizer=tokenizer, image_processor=image_processor, chat_template=CHAT_TEMPLATE)
    processor.save_pretrained(args.save_dir)

    # Push to hub if requested
    if args.push_to_hub:
        processor.push_to_hub(args.hub_dir, use_temp_dir=True)
        model.push_to_hub(args.hub_dir, use_temp_dir=True)

    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.save_dir,
            dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda:0")
    )
    processor = AutoProcessor.from_pretrained(args.save_dir)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ],
        },
    ]
    url = "http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    messages = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(messages)

    inputs = processor(
        images=[image],
        text=messages,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(output_text)


if __name__ == "__main__":
    main()
