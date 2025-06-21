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

"""
Example of run command (run from root):

python src/transformers/models/janus/convert_janus_weights_to_hf.py --repo_id deepseek-ai/Janus-Pro-1B --local_dir tmp/hub_code_in --output_dir tmp/hub_code_out --safe_serialization
Using provided local directory: tmp/hub_code_in
"""

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
    JanusForConditionalGeneration,
    JanusVisionConfig,
    JanusVQVAEConfig,
    LlamaConfig,
)
from transformers.models.janus.image_processing_janus import JanusImageProcessor
from transformers.models.janus.processing_janus import JanusProcessor


# Mappings
MAPPINGS = {}


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


def split_tensor(tensor, key):
    """Splits a merged tensor (qkv or kv) into separate tensors and creates keys for each part."""

    if "qkv" in key:
        prefix_to_replace = "qkv"
        num_splits = 3
        new_keys = ["q_proj", "k_proj", "v_proj"]
    else:
        raise ValueError(f"Unrecognized tensor type in key: {key}")

    split_size = tensor.shape[0] // num_splits
    tensors = torch.split(tensor, split_size, dim=0)
    return {key.replace(prefix_to_replace, new_keys[i]): tensors[i] for i in range(num_splits)}


def convert_state_dict_to_hf(state_dict):
    """Convert state dict keys to HF format."""
    conversion_dict = convert_old_keys_to_new_keys(state_dict)
    converted_state_dict = {}

    for old_key, new_key in conversion_dict.items():
        if new_key:
            if "qkv" in new_key or "kv" in new_key:  # Detect merged attention keys and split them.
                qkv_split_dict = split_tensor(state_dict[old_key], new_key)
                converted_state_dict.update(qkv_split_dict)
            else:
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

    if repo_id is None:
        raise ValueError("Either repo_id or local_dir must be provided")

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
    """
    Load model state dict, handling both single and sharded files.
    """
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
    text_model_id=None,
    output_dir=None,
    output_hub_path=None,
    safe_serialization=True,
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
    input_path = ensure_model_downloaded(repo_id=repo_id, revision=revision, local_dir=local_dir)

    # Load configuration files
    required_files = ["config.json", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json"]

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(input_path, f))]
    if missing_files:
        raise ValueError(
            f"The following required configuration files are missing from {input_path}: {', '.join(missing_files)}. "
            "Please ensure you have downloaded all necessary model files."
        )

    with open(os.path.join(input_path, "config.json"), "r") as f:
        config_data = json.load(f)
    with open(os.path.join(input_path, "preprocessor_config.json"), "r") as f:
        preprocessor_config = json.load(f)
    with open(os.path.join(input_path, "special_tokens_map.json"), "r") as f:
        special_tokens_map = json.load(f)
    with open(os.path.join(input_path, "tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)

    # Create tokenizer directly from tokenizer.json if it exists
    tokenizer_json_path = os.path.join(input_path, "tokenizer.json")
    special_image_tokens = {
        "image_token": "<image_placeholder>",
        "boi_token": "<begin_of_image>",
        "eoi_token": "<end_of_image>",
    }

    if os.path.exists(tokenizer_json_path) and not text_model_id:
        tokenizer = AutoTokenizer.from_pretrained(
            input_path,  # This will load tokenizer.json directly
            model_max_length=tokenizer_config["model_max_length"],
            extra_special_tokens=special_image_tokens,
        )
    else:
        # Fallback to creating from text_model_id with special tokens
        tokenizer = AutoTokenizer.from_pretrained(
            text_model_id,
            bos_token=special_tokens_map["bos_token"],
            eos_token=special_tokens_map["eos_token"],
            pad_token=special_tokens_map["pad_token"],
            additional_special_tokens=special_tokens_map["additional_special_tokens"],
            model_max_length=tokenizer_config["model_max_length"],
            extra_special_tokens=special_image_tokens,
        )

    # Create image processor from config
    image_processor_kwargs = {}
    for key in ["do_normalize", "image_mean", "image_std", "min_size", "rescale_factor"]:
        if key in preprocessor_config:
            image_processor_kwargs[key] = preprocessor_config[key]

    if "image_size" in preprocessor_config:
        image_processor_kwargs["size"] = {
            "height": preprocessor_config["image_size"],
            "width": preprocessor_config["image_size"],
        }

    image_processor = JanusImageProcessor(**image_processor_kwargs)

    # Create processor with chat template
    processor = JanusProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template="",
        use_default_system_prompt=True,
    )

    if output_dir:
        print(f"Saving processor to {output_dir}...")
        processor.save_pretrained(output_dir)
    if output_hub_path:
        print(f"Pushing processor to hub at {output_hub_path}...")
        processor.push_to_hub(output_hub_path)

    # Create model configurations
    text_config_kwargs = {}
    for key in [
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "torch_dtype",
    ]:
        if key in config_data["language_config"]:
            text_config_kwargs[key] = config_data["language_config"][key]

    # Add token IDs from tokenizer
    text_config_kwargs.update(
        {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    text_config = LlamaConfig(**text_config_kwargs)

    # Create vision config
    vision_config_kwargs = {}
    if "image_size" in config_data["vision_config"]["params"]:
        vision_config_kwargs["image_size"] = config_data["vision_config"]["params"]["image_size"]

    # Add aligner params if present
    if "aligner_config" in config_data and "params" in config_data["aligner_config"]:
        if "n_embed" in config_data["aligner_config"]["params"]:
            vision_config_kwargs["projection_dim"] = config_data["aligner_config"]["params"]["n_embed"]
        if "depth" in config_data["aligner_config"]["params"]:
            vision_config_kwargs["depth"] = config_data["aligner_config"]["params"]["depth"]

    vision_config = JanusVisionConfig(**vision_config_kwargs)

    vq_config = JanusVQVAEConfig(
        embed_dim=config_data["gen_vision_config"]["params"]["n_embed"],
        num_embeddings=config_data["gen_vision_config"]["params"]["image_token_size"],
        projection_dim=config_data["gen_aligner_config"]["params"]["n_embed"],
        depth=config_data["gen_aligner_config"]["params"]["depth"],
        image_token_embed_dim=config_data["gen_head_config"]["params"]["image_token_embed"],
    )

    # Create the main config
    config = BagelConfig(
        text_config=text_config,
        vision_config=vision_config,
        vq_config=vq_config,
        image_token_id=tokenizer.vocab.get("<image_placeholder>"),
    )

    # Save the config
    if output_dir:
        config.save_pretrained(output_dir)
    if output_hub_path:
        config.push_to_hub(output_hub_path)

    # Initialize model with empty weights
    print("Creating empty model...")
    with init_empty_weights():
        model = JanusForConditionalGeneration(config)

    model.generation_config.temperature = 1
    model.generation_config.guidance_scale = 5
    model.generation_config.pad_token_id = tokenizer.vocab.get("<\uff5c\u2581pad\u2581\uff5c>")
    model.generation_config.generation_kwargs["boi_token_id"] = tokenizer.vocab.get("<begin_of_image>")

    # Load and convert state dict
    print("Loading state dict...")
    state_dict = load_model_state_dict(input_path)
    state_dict = convert_state_dict_to_hf(state_dict)

    # Load converted state dict
    print("Loading converted weights into model...")
    model.load_state_dict(state_dict, strict=True, assign=True)

    # Tie weights before any device mapping
    print("Tying weights...")
    model.tie_weights()

    # Save the model
    if output_dir:
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    if output_hub_path:
        print(f"Pushing model to hub at {output_hub_path}...")
        model.push_to_hub(output_hub_path, safe_serialization=safe_serialization)

    del state_dict, model
    gc.collect()

    # Validate the saved model if saved locally
    if output_dir:
        print("Reloading the local model to check if it's saved correctly...")
        # TODO: warning about weights not being tied is raised here regardless of model.tie_weights() above
        JanusForConditionalGeneration.from_pretrained(output_dir, device_map="auto")
        print("Local model reloaded successfully.")


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
    parser.add_argument(
        "--text_model_id",
        help="Hub ID of the text model to get tokenizer from. Optional if tokenizer.json exists in the model directory.",
        required=False,
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Whether to save using safetensors",
    )
    args = parser.parse_args()

    if args.output_dir is None and args.output_hub_path is None:
        raise ValueError("At least one of --output_dir or --output_hub_path must be specified")

    if args.repo_id is None and args.local_dir is None:
        raise ValueError("Either --repo_id or --local_dir must be specified")

    convert_model(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        text_model_id=args.text_model_id,
        output_dir=args.output_dir,
        output_hub_path=args.output_hub_path,
        safe_serialization=args.safe_serialization,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
