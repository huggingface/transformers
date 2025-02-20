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
    JanusProcessor,
    SiglipVisionConfig,
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

def ensure_model_downloaded(repo_id: str, revision: str = None) -> str:
    """
    Ensures model files are downloaded locally, downloads them if not.
    Returns path to local files.
    """
    print(f"Ensuring {repo_id} (revision: {revision}) is downloaded...")
    local_files_only = False
    
    try:
        # First try to find files locally
        local_files_only = True
        local_dir = snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=local_files_only
        )
        print(f"Found model files locally at {local_dir}")
        return local_dir
    except Exception:
        # If files not found locally, download them
        print(f"Downloading model files for {repo_id}...")
        local_files_only = False
        local_dir = snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=local_files_only
        )
        print(f"Downloaded model files to {local_dir}")
        return local_dir

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
    vision_model_id,
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
    
    # First create and save the processor
    print("Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    
    print("Creating and saving processor...")
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

    # Create and save the config
    print("Creating model configuration...")
    text_config = AutoConfig.from_pretrained(text_model_id)
    vision_config = SiglipVisionConfig(
        hidden_size=1152,
        image_size=384,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=26,
        patch_size=14,
        vision_use_head=False,
    ).to_dict()

    # TODO: change this. The text and processor configs are probably right, but this is likely obtained from HUB
    config = JanusConfig(text_config=text_config, 
                         vision_config=vision_config,
                         vq_config=None)
    
    if output_dir:
        print(f"Saving config to {output_dir}...")
        config.save_pretrained(output_dir)
    if output_hub_path:
        print(f"Pushing config to hub at {output_hub_path}...")
        config.push_to_hub(output_hub_path)

    # Load and convert the model
    print(f"Loading weights from {input_path}")
    state_dict = load_model_state_dict(input_path)
    state_dict = convert_state_dict_to_hf(state_dict)
    
    print("Creating Janus model...")
    with init_empty_weights():
        model = JanusForConditionalGeneration(config)
    
    print("Loading state dict...")
    model.load_state_dict(state_dict, strict=True)
    
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
        "--vision_model_id", 
        help="Hub ID of the vision model to get image processor from",
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
        vision_model_id=args.vision_model_id,
        output_dir=args.output_dir,
        output_hub_path=args.output_hub_path,
        safe_serialization=args.safe_serialization,
        revision=args.revision,
    )

if __name__ == "__main__":
    main()
