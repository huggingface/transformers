# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Utility to convert EoMT-DINOv3 checkpoints to the 🤗 Transformers format."""

import argparse
import gc
import json
import os
import re
from typing import Dict, Optional

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download

from transformers import (
    DINOv3ViTModel,
    EomtDinov3Config,
    EomtDinov3ForUniversalSegmentation,
    EomtDinov3ImageProcessorFast,
)


# fmt: off
KEY_MAPPINGS = {
    # Backbone embeddings
    r"^encoder\.backbone\.patch_embed\.cls_token":            r"embeddings.cls_token",
    r"^encoder\.backbone\.patch_embed\.register_tokens":     r"embeddings.register_tokens",
    r"^encoder\.backbone\.patch_embed\.patch_embeddings":   r"embeddings.patch_embeddings",

    # Backbone blocks
    r"^encoder\.backbone\.blocks\.(\d+)\.":            r"layers.\1.",
    r"^encoder\.backbone\.norm":                              r"layernorm",

    # Query / head
    r"^q\.":                                                     r"query.",
    r"^class_head":                                                r"class_predictor",
    r"^mask_head\.0":                                            r"mask_head.fc1",
    r"^mask_head\.2":                                            r"mask_head.fc2",
    r"^mask_head\.4":                                            r"mask_head.fc3",
    r"^upscale\.(\d+)\.conv1":                              r"upscale_block.block.\1.conv1",
    r"^upscale\.(\d+)\.conv2":                              r"upscale_block.block.\1.conv2",
    r"^upscale\.(\d+)\.norm":                               r"upscale_block.block.\1.layernorm2d",

    # Misc
    r"^encoder\.attn_mask_probs":                               r"attn_mask_probs",
    r"^attn_mask_probs":                                           r"attn_mask_probs",
}
# fmt: on


def normalize_original_key(key: str) -> Optional[str]:
    """Normalizes a key from the original checkpoint before applying regex replacements."""

    if key.startswith("network."):
        key = key[len("network.") :]

    if key.startswith("encoder.pixel_mean") or key.startswith("encoder.pixel_std"):
        return None

    return key


def convert_old_keys_to_new_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Applies regex replacements to map the original weights to the Transformers naming scheme."""

    converted_state_dict: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        normalized_key = normalize_original_key(key)
        if normalized_key is None:
            continue

        new_key = normalized_key
        for pattern, replacement in KEY_MAPPINGS.items():
            new_key = re.sub(pattern, replacement, new_key)

        # Remove any remaining leading "encoder." prefix
        if new_key.startswith("encoder."):
            new_key = new_key[len("encoder.") :]

        converted_state_dict[new_key] = value

    return converted_state_dict


def convert_base_state_dict(base_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Renames the backbone weights from the base DINOv3 model to match the EoMT naming scheme."""

    converted: Dict[str, torch.Tensor] = {}
    for key, value in base_state_dict.items():
        if key.startswith("embeddings.mask_token"):
            # The EoMT embeddings do not expose the masked token.
            continue

        new_key = key
        new_key = re.sub(r"^layer\\.", "layers.", new_key)
        new_key = re.sub(r"^norm", "layernorm", new_key)
        converted[new_key] = value

    return converted


def merge_base_and_delta(base_state: Dict[str, torch.Tensor], delta_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Adds the delta weights on top of the base backbone."""

    final_state: Dict[str, torch.Tensor] = {k: v.clone() for k, v in base_state.items()}

    for key, tensor in delta_state.items():
        if key in final_state:
            final_state[key] = final_state[key].to(tensor.dtype) + tensor
        else:
            final_state[key] = tensor

    return final_state


def infer_processor_kwargs(repo_id: Optional[str], image_size: int) -> Dict[str, object]:
    repo_parts = repo_id.split("_") if repo_id else []
    if "semantic" in repo_parts:
        return {"size": {"shortest_edge": image_size, "longest_edge": None}, "do_split_image": True, "do_pad": False}
    else:
        return {
            "size": {"shortest_edge": image_size, "longest_edge": image_size},
            "do_split_image": False,
            "do_pad": True,
        }


def ensure_checkpoint_available(
    repo_id: Optional[str], local_dir: Optional[str], revision: Optional[str]
) -> str:
    if local_dir is not None:
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Local directory {local_dir} does not exist")
        return local_dir

    if repo_id is None:
        raise ValueError("Either repo_id or local_dir must be provided")

    return snapshot_download(repo_id, revision=revision)


def load_model_state_dict(input_path: str) -> Dict[str, torch.Tensor]:
    index_path = os.path.join(input_path, "pytorch_model.bin.index.json")
    model_path = os.path.join(input_path, "pytorch_model.bin")

    if os.path.isfile(index_path):
        with open(index_path, "r") as fp:
            index = json.load(fp)

        state_dict: Dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(input_path, shard_name)
            state_dict.update(torch.load(shard_path, map_location="cpu"))
        return state_dict

    if os.path.isfile(model_path):
        return torch.load(model_path, map_location="cpu")

    raise FileNotFoundError(f"Could not locate model weights under {input_path}")


def convert_model(
    repo_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_hub_path: Optional[str] = None,
    backbone_repo_id: Optional[str] = None,
    backbone_local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    backbone_revision: Optional[str] = None,
    safe_serialization: bool = True,
):
    if output_dir is None and output_hub_path is None:
        raise ValueError("At least one of output_dir or output_hub_path must be specified")

    if repo_id is None and local_dir is None:
        raise ValueError("Either repo_id or local_dir must be specified")

    if backbone_repo_id is None and backbone_local_dir is None:
        raise ValueError("Either backbone_repo_id or backbone_local_dir must be specified")

    torch.set_default_dtype(torch.float16)

    ckpt_path = ensure_checkpoint_available(repo_id, local_dir, revision)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The checkpoint at {ckpt_path} is missing config.json")

    with open(config_path, "r") as fp:
        config_data = json.load(fp)

    config_data.pop("backbone", None)

    if "num_labels" not in config_data and "num_classes" in config_data:
        config_data["num_labels"] = config_data.pop("num_classes")

    backbone_model = DINOv3ViTModel.from_pretrained(
        backbone_local_dir or backbone_repo_id,
        revision=backbone_revision,
        local_files_only=backbone_local_dir is not None,
    )
    base_state = convert_base_state_dict(backbone_model.state_dict())

    if "image_size" not in config_data:
        config_data["image_size"] = backbone_model.config.image_size
    if "patch_size" not in config_data:
        config_data["patch_size"] = backbone_model.config.patch_size
    if "num_channels" not in config_data:
        config_data["num_channels"] = backbone_model.config.num_channels

    config = EomtDinov3Config(
        **{
            **config_data,
            "hidden_size": backbone_model.config.hidden_size,
            "num_hidden_layers": backbone_model.config.num_hidden_layers,
            "num_attention_heads": backbone_model.config.num_attention_heads,
            "intermediate_size": backbone_model.config.intermediate_size,
            "num_register_tokens": backbone_model.config.num_register_tokens,
            "rope_theta": getattr(backbone_model.config, "rope_theta", 100.0),
            "query_bias": getattr(backbone_model.config, "query_bias", True),
            "key_bias": getattr(backbone_model.config, "key_bias", False),
            "value_bias": getattr(backbone_model.config, "value_bias", True),
            "proj_bias": getattr(backbone_model.config, "proj_bias", True),
            "mlp_bias": getattr(backbone_model.config, "mlp_bias", True),
            "use_gated_mlp": getattr(backbone_model.config, "use_gated_mlp", False),
            "hidden_act": backbone_model.config.hidden_act,
        }
    )

    del backbone_model

    processor_kwargs = infer_processor_kwargs(repo_id, config.image_size)
    processor = EomtDinov3ImageProcessorFast(**processor_kwargs)

    if output_dir:
        config.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    if output_hub_path:
        config.push_to_hub(output_hub_path)
        processor.push_to_hub(output_hub_path)

    with init_empty_weights():
        model = EomtDinov3ForUniversalSegmentation(config)

    delta_state = convert_old_keys_to_new_keys(load_model_state_dict(ckpt_path))

    final_state = merge_base_and_delta(base_state, delta_state)

    model.load_state_dict(final_state, strict=True, assign=True)

    if output_dir:
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)

    if output_hub_path:
        model.push_to_hub(output_hub_path, safe_serialization=safe_serialization)

    del model, final_state, delta_state, base_state
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dir", help="Path to the original EoMT-DINOv3 checkpoint", default=None)
    parser.add_argument("--hf_repo_id", help="Hub repo id for the original checkpoint", default=None)
    parser.add_argument("--revision", help="Revision for the original checkpoint", default=None)
    parser.add_argument("--output_dir", help="Where to store the converted model", default=None)
    parser.add_argument("--output_hub_path", help="Optional Hub repo to push the converted model", default=None)
    parser.add_argument("--backbone_local_dir", help="Local path to the DINOv3 base weights", default=None)
    parser.add_argument("--backbone_repo_id", help="Hub repo id for the DINOv3 base weights", default=None)
    parser.add_argument("--backbone_revision", help="Revision for the base DINOv3 model", default=None)
    parser.add_argument("--safe_serialization", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None and args.output_hub_path is None:
        raise ValueError("Specify at least --output_dir or --output_hub_path")

    if args.local_dir is None and args.hf_repo_id is None:
        raise ValueError("Specify either --local_dir or --hf_repo_id")

    convert_model(
        repo_id=args.hf_repo_id,
        local_dir=args.local_dir,
        revision=args.revision,
        output_dir=args.output_dir,
        output_hub_path=args.output_hub_path,
        backbone_repo_id=args.backbone_repo_id,
        backbone_local_dir=args.backbone_local_dir,
        backbone_revision=args.backbone_revision,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    main()
