# coding=utf-8
# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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
import gc
import json
import os
import re
from typing import Optional

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download

from transformers import (
    AutoTokenizer,
    BagelConfig,
    BagelForConditionalGeneration,
    BagelVQVAEConfig,
    Qwen2Config,
    SiglipVisionConfig,
)
from transformers.models.bagel.image_processing_bagel import BagelImageProcessor
from transformers.models.bagel.processing_bagel import BagelProcessor


MAPPINGS = {
    # Vision model
    r"vit_model.vision_model": r"model.vision_tower",
    # Connector and time embedder
    r"connector.fc1": r"model.vision_connecter.fc1",
    r"connector.fc2": r"model.vision_connecter.fc2",
    r"time_embedder.mlp.0": r"model.timestep_embedder.mlp.linear1",
    r"time_embedder.mlp.2": r"model.timestep_embedder.mlp.linear2",
    r"vae2llm": r"model.vae2llm_connector",
    r"llm2vae": r"model.llm2vae_connector",
    r"latent_pos_embed.pos_embed": r"model.latent_pos_embed",
    r"vit_pos_embed.pos_embed": r"model.vit_pos_embed",
    # Language model
    r"language_model.lm_head": r"lm_head",
    r"language_model.model.embed_tokens": r"model.language_model.embed_tokens",
    r"language_model.model.layers.(\d+).input_layernorm_moe_gen": r"model.language_model.layers.\1.input_layernorm_generation",
    r"language_model.model.layers.(\d+).input_layernorm": r"model.language_model.layers.\1.input_layernorm",
    r"language_model.model.layers.(\d+).mlp_moe_gen": r"model.language_model.layers.\1.mlp_generation",
    r"language_model.model.layers.(\d+).mlp": r"model.language_model.layers.\1.mlp",
    r"language_model.model.layers.(\d+).self_attn.k_norm_moe_gen": r"model.language_model.layers.\1.self_attn.k_norm_generation",
    r"language_model.model.layers.(\d+).self_attn.k_norm": r"model.language_model.layers.\1.self_attn.k_norm",
    r"language_model.model.layers.(\d+).self_attn.q_norm_moe_gen": r"model.language_model.layers.\1.self_attn.q_norm_generation",
    r"language_model.model.layers.(\d+).self_attn.q_norm": r"model.language_model.layers.\1.self_attn.q_norm",
    r"language_model.model.layers.(\d+).self_attn.q_proj_moe_gen": r"model.language_model.layers.\1.self_attn.q_proj_generation",
    r"language_model.model.layers.(\d+).self_attn.q_proj": r"model.language_model.layers.\1.self_attn.q_proj",
    r"language_model.model.layers.(\d+).self_attn.k_proj_moe_gen": r"model.language_model.layers.\1.self_attn.k_proj_generation",
    r"language_model.model.layers.(\d+).self_attn.k_proj": r"model.language_model.layers.\1.self_attn.k_proj",
    r"language_model.model.layers.(\d+).self_attn.v_proj_moe_gen": r"model.language_model.layers.\1.self_attn.v_proj_generation",
    r"language_model.model.layers.(\d+).self_attn.v_proj": r"model.language_model.layers.\1.self_attn.v_proj",
    r"language_model.model.layers.(\d+).self_attn.o_proj_moe_gen": r"model.language_model.layers.\1.self_attn.o_proj_generation",
    r"language_model.model.layers.(\d+).self_attn.o_proj": r"model.language_model.layers.\1.self_attn.o_proj",
    r"language_model.model.layers.(\d+).post_attention_layernorm_moe_gen": r"model.language_model.layers.\1.post_attention_layernorm_generation",
    r"language_model.model.layers.(\d+).post_attention_layernorm": r"model.language_model.layers.\1.post_attention_layernorm",
    r"language_model.model.norm_moe_gen": r"model.language_model.norm_generation",
    r"language_model.model.norm.weight": r"model.language_model.norm.weight",
}

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' or message['role'] == 'assistant' %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}<|vision_pad|>"
    "{% elif content['type'] == 'text' %}<|im_start|>{{ content['text'].strip() }}<|im_end|>"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>{% endif %}"
)


def convert_old_keys_to_new_keys(state_dict):
    keys_as_text = "\n".join(state_dict.keys())
    new_keys_as_text = keys_as_text
    for old, repl in MAPPINGS.items():
        if repl is None:
            new_keys_as_text = re.sub(old, "", new_keys_as_text)
        else:
            new_keys_as_text = re.sub(old, repl, new_keys_as_text)
    output_dict = dict(zip(keys_as_text.split("\n"), new_keys_as_text.split("\n")))
    return output_dict


def convert_state_dict_to_hf(state_dict):
    """Convert state dict keys to HF format."""
    conversion_dict = convert_old_keys_to_new_keys(state_dict)
    converted_state_dict = {}

    for old_key, new_key in conversion_dict.items():
        converted_state_dict[new_key] = state_dict[old_key]

    # Embeddings will not have initial dimension
    pos_embed_key = "model.vision_model.embeddings.position_embedding.weight"
    converted_state_dict[pos_embed_key] = converted_state_dict[pos_embed_key].squeeze(0)

    return converted_state_dict


def ensure_model_downloaded(
    repo_id: Optional[str] = None, revision: Optional[str] = None, local_dir: Optional[str] = None
) -> str:
    """
    Ensures model files are downloaded locally, downloads them if not.
    Returns path to local files.

    Args:
        repo_id: The Hugging Face model repo ID (required if local_dir not provided)
        revision: Optional git revision to use
        local_dir: Optional local directory path where model files should be stored/found
    """

    if local_dir is not None:
        if os.path.exists(local_dir):
            print(f"Using provided local directory: {local_dir}")
        else:
            # Create the local directory if it doesn't exist
            os.makedirs(local_dir, exist_ok=True)
            print(f"Created local directory: {local_dir}")

    print(f"Ensuring {repo_id} (revision: {revision or 'latest'}) is downloaded...")

    try:
        # First try to find files locally
        download_dir = snapshot_download(repo_id, revision=revision, local_files_only=True, local_dir=local_dir)
        print(f"Found model files locally at {download_dir}")
        return download_dir
    except Exception:
        # If files not found locally, download them
        print(f"Downloading model files for {repo_id}...")
        download_dir = snapshot_download(repo_id, revision=revision, local_files_only=False, local_dir=local_dir)
        print(f"Downloaded model files to {download_dir}")
        return download_dir


def load_model_state_dict(input_path: str) -> dict:
    """Load model state dict, handling both single and sharded files."""

    index_path = os.path.join(input_path, "pytorch_model.bin.index.json")
    single_file_path = os.path.join(input_path, "pytorch_model.bin")

    # Check if we have a sharded model
    if os.path.exists(index_path):
        print("Loading sharded model...")
        state_dict = {}
        with open(index_path, "r") as f:
            index = json.load(f)

        # Get unique shard files and load each one only once
        unique_shard_files = sorted(set(index["weight_map"].values()))
        for shard_file in unique_shard_files:
            print(f"Loading shard {shard_file}...")
            shard_path = os.path.join(input_path, shard_file)
            shard_dict = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard_dict)

        return state_dict

    # Single file model
    elif os.path.exists(single_file_path):
        print("Loading single file model...")
        return torch.load(single_file_path, map_location="cpu")

    else:
        raise ValueError(f"No model files found in {input_path}")


def convert_model(
    repo_id=None,
    local_dir=None,
    output_dir=None,
    output_hub_path=None,
    revision=None,
):
    """Convert and save the model weights, processor, and configuration."""
    if output_dir is None and output_hub_path is None:
        raise ValueError("At least one of output_dir or output_hub_path must be specified")

    if repo_id is None and local_dir is None:
        raise ValueError("Either repo_id or local_dir must be specified")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created/verified output directory: {output_dir}")

    torch.set_default_dtype(torch.float16)

    # Download or locate model files
    if local_dir is not None:
        input_path = local_dir
        if not os.path.exists(input_path):
            raise ValueError(f"Local directory {input_path} does not exist.")
    else:
        input_path = ensure_model_downloaded(repo_id=repo_id, revision=revision, local_dir=local_dir)

    # Load configuration files
    required_files = [
        "config.json",
        "vit_config.json",
        "llm_config.json",
        "generation_config.json",
        "tokenizer_config.json",
    ]

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(input_path, f))]
    if missing_files:
        raise ValueError(
            f"The following required configuration files are missing from {input_path}: {', '.join(missing_files)}. "
            "Please ensure you have downloaded all necessary model files."
        )

    with open(os.path.join(input_path, "llm_config.json"), "r") as f:
        llm_config = json.load(f)
    with open(os.path.join(input_path, "vit_config.json"), "r") as f:
        vit_config = json.load(f)
    with open(os.path.join(input_path, "tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)

    _ = llm_config.pop("architectures", None)
    _ = vit_config.pop("architectures", None)

    # Create tokenizer directly from tokenizer.json if it exists
    special_image_tokens = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "boi_token": "<|vision_start|>",
        "eoi_token": "<|vision_end|>",
    }

    tokenizer = AutoTokenizer.from_pretrained(
        input_path,  # This will load tokenizer.json directly
        model_max_length=tokenizer_config["model_max_length"],
        extra_special_tokens=special_image_tokens,
    )

    # ToDo - how to have both vit and vae default in preprocessor_config.json?
    image_processor = BagelImageProcessor()

    # Create processor with chat template
    processor = BagelProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    if output_dir:
        print(f"Saving processor to {output_dir}...")
        processor.save_pretrained(output_dir)
    if output_hub_path:
        print(f"Pushing processor to hub at {output_hub_path}...")
        processor.push_to_hub(output_hub_path)

    # Add token IDs from tokenizer
    llm_config.update(
        {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "boi_token_id": tokenizer.boi_token_id,
            "eoi_token_id": tokenizer.eoi_token_id,
        }
    )

    text_config = Qwen2Config(**llm_config)

    vision_config = SiglipVisionConfig(**vit_config)

    vq_config = BagelVQVAEConfig()

    # Create the main config
    config = BagelConfig(
        text_config=text_config,
        vision_config=vision_config,
        vq_config=vq_config,
    )

    # Save the config
    if output_dir:
        config.save_pretrained(output_dir)
    if output_hub_path:
        config.push_to_hub(output_hub_path)

    # Initialize model with empty weights
    print("Creating empty model...")
    # with init_empty_weights():
    #     model = BagelForConditionalGeneration(config)

    # model.generation_config.temperature = 1
    # model.generation_config.guidance_scale = 5
    # model.generation_config.pad_token_id = tokenizer.vocab.get("<\uff5c\u2581pad\u2581\uff5c>")
    # model.generation_config.generation_kwargs["boi_token_id"] = tokenizer.vocab.get("<begin_of_image>")

    # Load and convert state dict
    # print("Loading state dict...")
    # state_dict = load_model_state_dict(input_path)
    # state_dict = convert_state_dict_to_hf(state_dict)

    # # Load converted state dict
    # print("Loading converted weights into model...")
    # model.load_state_dict(state_dict, strict=True, assign=True)

    # # Tie weights before any device mapping
    # print("Tying weights...")
    # model.tie_weights()

    # # Save the model
    # if output_dir:
    #     print(f"Saving model to {output_dir}...")
    #     model.save_pretrained(output_dir, safe_serialization=True)
    # if output_hub_path:
    #     print(f"Pushing model to hub at {output_hub_path}...")
    #     model.push_to_hub(output_hub_path, safe_serialization=True)

    # del state_dict, model
    # gc.collect()

    # Validate the saved model if saved locally
    # if output_dir:
    #     print("Reloading the local model to check if it's saved correctly...")
    #     BagelForConditionalGeneration.from_pretrained(output_dir, device_map="auto")
    #     print("Local model reloaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        help="HuggingFace Hub repo ID for the model",
        default=None,
    )
    parser.add_argument(
        "--local_dir",
        help="Local directory containing the model files",
        default=None,
    )
    parser.add_argument(
        "--revision",
        help="Specific revision to download from the Hub",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model locally",
        default=None,
    )
    parser.add_argument(
        "--output_hub_path",
        help="Repository ID to push model to hub (e.g. 'username/model-name')",
        default=None,
    )
    args = parser.parse_args()

    if args.output_dir is None and args.output_hub_path is None:
        raise ValueError("At least one of --output_dir or --output_hub_path must be specified")

    if args.repo_id is None and args.local_dir is None:
        raise ValueError("Either --repo_id or --local_dir must be specified")

    convert_model(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        output_dir=args.output_dir,
        output_hub_path=args.output_hub_path,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
