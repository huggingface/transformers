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
import gc
import os
from pathlib import Path
import re
import json

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    JanusConfig,
    JanusForConditionalGeneration,
    JanusImageProcessor,
    JanusProcessor,
    SiglipVisionConfig,
    JanusTextConfig,
    JanusVisionConfig,
    JanusVQVAEConfig,
)
from safetensors.torch import load_file

# Mapping dictionaries for converting keys
VISION_MAPPINGS = {
    "vision_model.vision_tower.blocks": "vision_model.vision_tower.layers",
    "vision_model.vision_tower.pos_embed": "vision_model.embeddings.position_embeddings",
    "vision_model.vision_tower.patch_embed.proj": "vision_model.embeddings.patch_embeddings.projection",
    "vision_model.vision_tower.norm": "vision_model.post_layernorm",
    "vision_model.vision_tower.attn_pool": "vision_model.head",
    "proj": "projection_layer",
    "norm": "layer_norm",
    "norm1": "layer_norm1",
    "norm2": "layer_norm2",
}

VQ_MAPPINGS = {
    # VQ Model prefix conversion
    "res": "block",
    "mid.0": "mid.block_1",
    "mid.1": "mid.attn_1",
    "mid.2": "mid.block_2",
    # Aligner module changes
    "layers.0": "fc1",
    "layers.2": "hidden_layers.0",
    "gen_head.output_mlp_projector": "gen_head.proj_out",
}

CHAT_TEMPLATE = (
    "{%set seps=['\n\n','']%}"
    "{%set i=0%}"
    "{%for message in messages%}"
        "{%if message['role']!='system'%}"
            "{%if message['role']=='user'%}"
                "<|User|>: "
            "{%else%}"
                "<|Assistant|>: "
            "{%endif%}"
        "{%endif%}"
        "{%for content in message['content']%}"
            "{%if content['type']=='image'%}"
                "{%if not loop.first%}{{'\n'}}{%endif%}"
                "<image_placeholder>"
                "{%if not loop.last%}{{'\n'}}{%endif%}"
            "{%elif content['type']=='text'%}"
                "{%set text=content['text']%}"
                "{%if loop.first%}{%set text=text.lstrip()%}{%endif%}"
                "{%if loop.last%}{%set text=text.rstrip()%}{%endif%}"
                "{%if not loop.first and message['content'][loop.index0-1]['type']=='text'%}"
                    "{{' '+text}}"
                "{%else%}"
                    "{{text}}"
                "{%endif%}"
            "{%endif%}"
        "{%endfor%}"
        "{%if not loop.last or add_generation_prompt%}"
            "{%if message['role']=='system'%}"
                "{{seps[0]}}"
            "{%elif message['role']=='user'%}"
                "{{seps[0]}}"
            "{%else%}"
                "{{seps[1]}}"
            "{%endif%}"
        "{%endif%}"
    "{%endfor%}"
    "{%if add_generation_prompt%}<|Assistant|>:{%endif%}"
)

def convert_state_dict_to_hf(state_dict):
    """Convert state dict keys to HF format."""
    converted_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # No conversion as language model is same
        if "language_model" in key:
            converted_state_dict[key] = value
            continue

        new_key = new_key.replace("gen_vision_model", "vqmodel")

        if "vision_model" in new_key:
            for old, new in VISION_MAPPINGS.items():
                if re.search(rf'\b{re.escape(old)}\b', new_key):
                    new_key = new_key.replace(old, new)
            new_key = new_key.replace("vision_tower", "encoder")
        elif "vqmodel" in new_key:
            for old, new in VQ_MAPPINGS.items():
                new_key = new_key.replace(old, new)

            if "encoder" in new_key:
                new_key = new_key.replace("conv_blocks", "down")
            elif "decoder" in new_key:
                new_key = new_key.replace("conv_blocks", "up")

        converted_state_dict[new_key] = value

    if "vqmodel.quantize.codebook_used" in converted_state_dict:
        del converted_state_dict["vqmodel.quantize.codebook_used"]

    return converted_state_dict

def ensure_model_downloaded(repo_id: str = None, revision: str = None, local_dir: str = None) -> str:
    """
    Ensures model files are downloaded locally, downloads them if not.
    Returns path to local files.
    
    Args:
        repo_id: The Hugging Face model repo ID (required if local_dir not provided)
        revision: Optional git revision to use
        local_dir: Optional local directory path where model files should be stored/found
    """
    if repo_id is None:
        raise ValueError("repo_id must be provided")
        
    print(f"Ensuring {repo_id} (revision: {revision or 'latest'}) is downloaded...")
    
    if local_dir is not None:
        print(f"Using provided local directory: {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
    
    try:
        # First try to find files locally
        download_dir = snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=True,
            local_dir=local_dir
        )
        print(f"Found model files locally at {download_dir}")
        return download_dir
    except Exception:
        # If files not found locally, download them
        print(f"Downloading model files for {repo_id}...")
        download_dir = snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=False,
            local_dir=local_dir
        )
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
    repo_id,
    text_model_id,
    output_dir=None,
    output_hub_path=None,
    safe_serialization=True,
    revision=None,
):
    """Convert and save the model weights, processor, and configuration."""
    if output_dir is None and output_hub_path is None:
        raise ValueError("At least one of output_dir or output_hub_path must be specified")

    torch.set_default_dtype(torch.float16)

    # Download or locate model files
    input_path = ensure_model_downloaded(repo_id, revision)
    
    # Load configuration files
    with open(os.path.join(input_path, "config.json"), "r") as f:
        config_data = json.load(f)
    with open(os.path.join(input_path, "preprocessor_config.json"), "r") as f:
        preprocessor_config = json.load(f)
    with open(os.path.join(input_path, "processor_config.json"), "r") as f:
        processor_config = json.load(f)
    with open(os.path.join(input_path, "special_tokens_map.json"), "r") as f:
        special_tokens_map = json.load(f)
    with open(os.path.join(input_path, "tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)

    # Create tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        text_model_id,
        bos_token=special_tokens_map["bos_token"],
        eos_token=special_tokens_map["eos_token"],
        pad_token=special_tokens_map["pad_token"],
        additional_special_tokens=special_tokens_map["additional_special_tokens"],
        model_max_length=tokenizer_config["model_max_length"],
    )

    # Create image processor from config
    image_processor = JanusImageProcessor(
        do_resize=preprocessor_config.get("do_resize", True),
        size={"height": preprocessor_config["image_size"], "width": preprocessor_config["image_size"]},
        do_normalize=preprocessor_config.get("do_normalize", True),
        image_mean=preprocessor_config.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=preprocessor_config.get("image_std", [0.5, 0.5, 0.5]),
        min_size=preprocessor_config.get("min_size", 14),
    )
    
    # Create processor with chat template
    processor = JanusProcessor(
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

    # Create model configurations
    text_config = JanusTextConfig(
        vocab_size=config_data["language_config"]["vocab_size"],
        hidden_size=config_data["language_config"]["hidden_size"],
        intermediate_size=config_data["language_config"]["intermediate_size"],
        num_hidden_layers=config_data["language_config"]["num_hidden_layers"],
        num_attention_heads=config_data["language_config"]["num_attention_heads"],
        num_key_value_heads=config_data["language_config"].get("num_key_value_heads", None),
        hidden_act=config_data["language_config"].get("hidden_act", "silu"),
        max_position_embeddings=config_data["language_config"]["max_position_embeddings"],
        torch_dtype=config_data["language_config"].get("torch_dtype", "bfloat16"),
    )

    vision_config = JanusVisionConfig(
        hidden_size=config_data["vision_config"]["params"]["model_name"].endswith("large") and 1024 or 768,
        image_size=config_data["vision_config"]["params"]["image_size"],
        select_feature=config_data["vision_config"]["params"]["select_feature"],
        select_layer=config_data["vision_config"]["params"]["select_layer"],
        aligner_projection_size=config_data["aligner_config"]["params"]["n_embed"],
        depth=config_data["aligner_config"]["params"]["depth"],
    )

    vq_config = JanusVQVAEConfig(
        embed_dim=config_data["gen_vision_config"]["params"]["n_embed"],
        num_embeddings=config_data["gen_vision_config"]["params"]["image_token_size"],
        aligner_projection_size=config_data["gen_aligner_config"]["params"]["n_embed"],
        depth=config_data["gen_aligner_config"]["params"]["depth"],
        image_token_embed_size=config_data["gen_head_config"]["params"]["image_token_embed"],
    )

    # Create the main config
    config = JanusConfig(
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
    with init_empty_weights():
        model = JanusForConditionalGeneration(config)

    # Load and convert state dict
    print("Loading state dict...")
    state_dict = load_model_state_dict(input_path)
    state_dict = convert_state_dict_to_hf(state_dict)

    # Load converted state dict
    print("Loading converted weights into model...")
    model.load_state_dict(state_dict, strict=True)

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
        JanusForConditionalGeneration.from_pretrained(output_dir, device_map="auto")
        print("Local model reloaded successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        help="HuggingFace Hub repo ID for the model",
        required=True,
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
        help="Hub ID of the text model to get tokenizer from",
        required=True,
    )
    parser.add_argument(
        "--safe_serialization",
        type=bool,
        default=True,
        help="Whether to save using safetensors",
    )
    args = parser.parse_args()

    if args.output_dir is None and args.output_hub_path is None:
        raise ValueError("At least one of --output_dir or --output_hub_path must be specified")

    convert_model(
        repo_id=args.repo_id,
        text_model_id=args.text_model_id,
        output_dir=args.output_dir,
        output_hub_path=args.output_hub_path,
        safe_serialization=args.safe_serialization,
        revision=args.revision,
    )

if __name__ == "__main__":
    # main()
    ensure_model_downloaded("deepseek-ai/Janus-Pro-1B", local_dir="tmp/hub_code")
