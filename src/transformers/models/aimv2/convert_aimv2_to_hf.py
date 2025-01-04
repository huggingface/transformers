# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
import requests
from safetensors.torch import load_file as safe_load

from transformers import AIMv2Config, AIMv2Model, BitImageProcessor, AutoImageProcessor, AIMv2ForImageClassification
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def get_aimv2_config(model_name: str, image_classifier: bool = False) -> AIMv2Config:
    if model_name == "aimv2-large-patch14-224":
        config = AIMv2Config(
            image_size=224,
            patch_size=14,
            num_channels=3,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=8,
            intermediate_size=2816,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            qkv_bias=False,
            use_bias=False,
            use_cls_token=False,
            pos_embed_type="absolute",
            post_trunk_norm=False,
            probe_layers=[6],
            reduce=False,
            ffn_target_type="swiglu",
            is_causal=False,
            norm_layer=nn.RMSNorm,
        )
    elif model_name == "aimv2-huge-patch14-224":
        config = AIMv2Config(
            image_size=224,
            patch_size=14,
            num_channels=3,
            hidden_size=1536,
            num_hidden_layers=24,
            num_attention_heads=12,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            qkv_bias=False,
            use_bias=False,
            use_cls_token=False,
            pos_embed_type="absolute",
            post_trunk_norm=False,
            probe_layers=[6],
            reduce=False,
            ffn_target_type="swiglu",
            is_causal=False,
            norm_layer=nn.RMSNorm,
        )
    elif model_name == "aimv2-1B-patch14-224":
        config = AIMv2Config(
            image_size=224,
            patch_size=14,
            num_channels=3,
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=5632,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            qkv_bias=False,
            use_bias=False,
            use_cls_token=False,
            pos_embed_type="absolute",
            post_trunk_norm=False,
            probe_layers=[6],
            reduce=False,
            ffn_target_type="swiglu",
            is_causal=False,
            norm_layer=nn.RMSNorm,
        )
    elif model_name == "aimv2-3B-patch14-224":
        config = AIMv2Config(
            image_size=224,
            patch_size=14,
            num_channels=3,
            hidden_size=3072,
            num_hidden_layers=24,
            num_attention_heads=24,
            intermediate_size=8192,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            qkv_bias=False,
            use_bias=False,
            use_cls_token=False,
            pos_embed_type="absolute",
            post_trunk_norm=False,
            probe_layers=[6],
            reduce=False,
            ffn_target_type="swiglu",
            is_causal=False,
            norm_layer=nn.RMSNorm,
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")

    if image_classifier:
        config.num_labels = 1000
        config.image_size = 224
        config.patch_size = 14
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        config.id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        config.id2label = {int(k): v for k, v in config.id2label.items()}
        config.label2id = {v: k for k, v in config.id2label.items()}

    return config

def create_rename_keys(config: AIMv2Config) -> List[Tuple[str, str]]:
    """
    Generate a list of tuples, where each tuple contains the old key and the new key for renaming weights in the
    original checkpoint.

    Args:
        config (AIMv2Config): The configuration of the AIMv2 model.

    Returns:
        List[Tuple[str, str]]: A list of tuples representing the old and new keys for renaming weights.
    """
    rename_keys = []

    # Rename embedding layers
    rename_keys.append(("preprocessor.pos_embed", "embeddings.pos_embed"))
    rename_keys.append(("preprocessor.patchifier.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("preprocessor.patchifier.proj.bias", "embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("preprocessor.patchifier.norm.weight", "embeddings.patch_embeddings.norm.weight"))

    # Rename encoder layers
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"trunk.blocks.{i}.attn.qkv.weight", f"encoder.layer.{i}.attention.qkv.weight"))
        rename_keys.append((f"trunk.blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.out_proj.weight"))
        rename_keys.append((f"trunk.blocks.{i}.norm_1.weight", f"encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"trunk.blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.ffn.fc1.weight"))
        rename_keys.append((f"trunk.blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.ffn.fc2.weight"))
        rename_keys.append((f"trunk.blocks.{i}.mlp.fc3.weight", f"encoder.layer.{i}.ffn.fc3.weight"))
        rename_keys.append((f"trunk.blocks.{i}.norm_2.weight", f"encoder.layer.{i}.norm2.weight"))

    # Rename final norm layer
    rename_keys.append(("trunk.post_trunk_norm.weight", "encoder.norm.weight"))

    return rename_keys

def rename_state_dict_keys(
    state_dict: Dict[str, torch.Tensor], rename_keys: List[Tuple[str, str]]
) -> Dict[str, torch.Tensor]:
    """
    Rename the keys in the state dictionary based on the provided rename_keys mapping.

    Args:
        state_dict (Dict[str, torch.Tensor]): The state dictionary to be renamed.
        rename_keys (List[Tuple[str, str]]): A list of tuples representing the old and new keys for renaming weights.

    Returns:
        Dict[str, torch.Tensor]: The state dictionary with renamed keys.
    """
    new_state_dict = {}
    for old_key, new_key in rename_keys:
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict.pop(old_key)
    return new_state_dict

@torch.no_grad()
def convert_aimv2_checkpoint(
    model_name: str,
    pytorch_dump_folder_path: Union[str, Path],
    push_to_hub: bool = False,
    image_classifier: bool = False,
) -> None:
    """
    Convert the AIMv2 checkpoint from the original repository to the Hugging Face Transformers format.

    Args:
        model_name (str): The name of the AIMv2 model to convert.
        pytorch_dump_folder_path (Union[str, Path]): The path to the folder to save the converted model.
        push_to_hub (bool, optional): Whether to push the converted model to the Hugging Face Model Hub. Defaults to False.
    """
    # Load the configuration for the Hugging Face model
    config = get_aimv2_config(model_name, image_classifier=image_classifier)

    # Create an instance of the Hugging Face model
    if image_classifier:
        model = AIMv2ForImageClassification(config).eval()
    else:
        model = AIMv2Model(config).eval()

    # Load the state dict from the original checkpoint using hf_hub_download
    repo_id = f"apple/{model_name}"
    filename = "model.safetensors"  # Assuming the model is saved as safetensors
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except:
        filename = "pytorch_model.bin" # if not then try to find pytorch_model
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    state_dict = safe_load(local_path, device="cpu")

    # Generate the mapping for renaming keys
    rename_keys = create_rename_keys(config)

    # Rename the keys in the state_dict
    state_dict = rename_state_dict_keys(state_dict, rename_keys)

    # Load the renamed state dict into the Hugging Face model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    # Optionally, load image classification head weights
    if image_classifier:
        url = "https://ml-aim-public.s3.us-west-2.amazonaws.com/aim/torch/aimv2_classifier.pth"
        classifier_state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.classifier.load_state_dict(classifier_state_dict, strict=False)

    # Verify the conversion by comparing outputs
    image_processor = AutoImageProcessor.from_pretrained(f"apple/{model_name}")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    # Forward pass through the converted model
    hf_outputs = model(**inputs)

    # assert values
    if image_classifier:
        print("Predicted class:")
        class_idx = hf_outputs.logits.argmax(-1).item()
        print(model.config.id2label[class_idx])
    else:
        assert hf_outputs.last_hidden_state.shape[1:] == (config.image_size // config.patch_size, config.image_size // config.patch_size, config.hidden_size)

    print("Looks ok!")

    # Save the converted model
    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # Push the converted model to the Hugging Face Model Hub
    if push_to_hub:
        model.push_to_hub(f"apple/{model_name}")
        image_processor.push_to_hub(f"apple/{model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="aimv2-large-patch14-224",
        type=str,
        choices=["aimv2-large-patch14-224", "aimv2-huge-patch14-224", "aimv2-1B-patch14-224", "aimv2-3B-patch14-224"],
        help="Name of the AIMv2 model to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether to push the converted model to the Hugging Face Hub."
    )
    parser.add_argument(
        "--image_classifier", action="store_true", help="Whether to load the image classification model."
    )
    args = parser.parse_args()
    convert_aimv2_checkpoint(
        args.model_name,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
        args.image_classifier,
    )