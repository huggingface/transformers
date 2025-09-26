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
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download

from transformers import (
    DINOv3ViTConfig,
    DINOv3ViTModel,
    EomtDinov3Config,
    EomtDinov3ForUniversalSegmentation,
    EomtDinov3ImageProcessorFast,
)


try:  # pragma: no cover
    from safetensors.torch import load_file as load_safetensors

    SAFETENSORS_AVAILABLE = True
except ImportError:  # pragma: no cover
    SAFETENSORS_AVAILABLE = False


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class CheckpointMetadata:
    image_size: int
    num_labels: int
    num_queries: int
    num_blocks: int
    timm_arch: str
    default_backbone_repo: str
    processor_kwargs: dict[str, object]
    config_overrides: dict[str, object] = field(default_factory=dict)


TIMM_TO_CONFIG_KWARGS: dict[str, dict[str, object]] = {
    "vit_small_patch16_dinov3": {
        "patch_size": 16,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "proj_bias": True,
        "num_register_tokens": 4,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
    },
    "vit_small_plus_patch16_dinov3": {
        "patch_size": 16,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_register_tokens": 4,
        "use_gated_mlp": True,
        "hidden_act": "silu",
    },
    "vit_base_patch16_dinov3": {
        "patch_size": 16,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "proj_bias": True,
        "num_register_tokens": 4,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
    },
    "vit_large_patch16_dinov3": {
        "patch_size": 16,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_register_tokens": 4,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
    },
    "vit_huge_plus_patch16_dinov3": {
        "patch_size": 16,
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "num_register_tokens": 4,
        "use_gated_mlp": True,
        "hidden_act": "silu",
    },
    "vit_7b_patch16_dinov3": {
        "patch_size": 16,
        "hidden_size": 4096,
        "intermediate_size": 8192,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "query_bias": False,
        "value_bias": False,
        "num_register_tokens": 4,
        "use_gated_mlp": True,
        "hidden_act": "silu",
    },
}


CHECKPOINT_METADATA: dict[str, CheckpointMetadata] = {
    "coco_panoptic_eomt_small_640_dinov3": CheckpointMetadata(
        image_size=640,
        num_labels=133,
        num_queries=200,
        num_blocks=3,
        timm_arch="vit_small_patch16_dinov3",
        default_backbone_repo="timm/vit_small_patch16_dinov3.lvd1689m",
        processor_kwargs={
            "size": {"shortest_edge": 640, "longest_edge": 640},
            "do_split_image": False,
            "do_pad": True,
        },
    ),
    "coco_panoptic_eomt_base_640_dinov3": CheckpointMetadata(
        image_size=640,
        num_labels=133,
        num_queries=200,
        num_blocks=3,
        timm_arch="vit_base_patch16_dinov3",
        default_backbone_repo="timm/vit_base_patch16_dinov3.lvd1689m",
        processor_kwargs={
            "size": {"shortest_edge": 640, "longest_edge": 640},
            "do_split_image": False,
            "do_pad": True,
        },
    ),
    "coco_panoptic_eomt_large_640_dinov3": CheckpointMetadata(
        image_size=640,
        num_labels=133,
        num_queries=200,
        num_blocks=4,
        timm_arch="vit_large_patch16_dinov3",
        default_backbone_repo="timm/vit_large_patch16_dinov3.lvd1689m",
        processor_kwargs={
            "size": {"shortest_edge": 640, "longest_edge": 640},
            "do_split_image": False,
            "do_pad": True,
        },
    ),
    "coco_panoptic_eomt_large_1280_dinov3": CheckpointMetadata(
        image_size=1280,
        num_labels=133,
        num_queries=200,
        num_blocks=4,
        timm_arch="vit_large_patch16_dinov3",
        default_backbone_repo="timm/vit_large_patch16_dinov3.lvd1689m",
        processor_kwargs={
            "size": {"shortest_edge": 1280, "longest_edge": 1280},
            "do_split_image": False,
            "do_pad": True,
        },
    ),
    "coco_instance_eomt_large_640_dinov3": CheckpointMetadata(
        image_size=640,
        num_labels=80,
        num_queries=200,
        num_blocks=4,
        timm_arch="vit_large_patch16_dinov3",
        default_backbone_repo="timm/vit_large_patch16_dinov3.lvd1689m",
        processor_kwargs={
            "size": {"shortest_edge": 640, "longest_edge": 640},
            "do_split_image": False,
            "do_pad": True,
        },
    ),
    "coco_instance_eomt_large_1280_dinov3": CheckpointMetadata(
        image_size=1280,
        num_labels=80,
        num_queries=200,
        num_blocks=4,
        timm_arch="vit_large_patch16_dinov3",
        default_backbone_repo="timm/vit_large_patch16_dinov3.lvd1689m",
        processor_kwargs={
            "size": {"shortest_edge": 1280, "longest_edge": 1280},
            "do_split_image": False,
            "do_pad": True,
        },
    ),
    "ade20k_semantic_eomt_large_512_dinov3": CheckpointMetadata(
        image_size=512,
        num_labels=150,
        num_queries=100,
        num_blocks=4,
        timm_arch="vit_large_patch16_dinov3",
        default_backbone_repo="timm/vit_large_patch16_dinov3.lvd1689m",
        processor_kwargs={
            "size": {"shortest_edge": 512, "longest_edge": None},
            "do_split_image": True,
            "do_pad": False,
        },
    ),
}


# fmt: off
DELTA_KEY_MAPPINGS = {
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

TIMM_KEY_MAPPINGS = {
    r"cls_token":                   r"embeddings.cls_token",
    r"mask_token":                  r"embeddings.mask_token",
    r"reg_token":                   r"embeddings.register_tokens",
    r"storage_tokens":              r"embeddings.register_tokens",
    r"patch_embed.proj":            r"embeddings.patch_embeddings",
    r"periods":                     r"inv_freq",
    r"rope_embed":                  r"rope_embeddings",
    r"blocks.(\d+).attn.proj":      r"layer.\1.attention.o_proj",
    r"blocks.(\d+).attn.":          r"layer.\1.attention.",
    r"blocks.(\d+).ls(\d+).gamma":  r"layer.\1.layer_scale\2.lambda1",
    r"blocks.(\d+).gamma_1":        r"layer.\1.layer_scale1.lambda1",
    r"blocks.(\d+).gamma_2":        r"layer.\1.layer_scale2.lambda1",
    r"blocks.(\d+).mlp.fc1":        r"layer.\1.mlp.up_proj",
    r"blocks.(\d+).mlp.fc2":        r"layer.\1.mlp.down_proj",
    r"blocks.(\d+).mlp":            r"layer.\1.mlp",
    r"blocks.(\d+).norm":           r"layer.\1.norm",
    r"w1":                          r"gate_proj",
    r"w2":                          r"up_proj",
    r"w3":                          r"down_proj",
}
# fmt: on


def normalize_original_key(key: str) -> Optional[str]:
    """Normalizes a key from the original checkpoint before applying regex replacements."""

    if key.startswith("network."):
        key = key[len("network.") :]

    if key.startswith("encoder.pixel_mean") or key.startswith("encoder.pixel_std"):
        return None

    return key


def convert_delta_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Applies regex replacements to map the original EoMT deltas to the Transformers naming scheme."""

    converted_state_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        normalized_key = normalize_original_key(key)
        if normalized_key is None:
            continue

        new_key = normalized_key
        for pattern, replacement in DELTA_KEY_MAPPINGS.items():
            new_key = re.sub(pattern, replacement, new_key)

        # Remove any remaining leading "encoder." prefix
        if new_key.startswith("encoder."):
            new_key = new_key[len("encoder.") :]

        converted_state_dict[new_key] = value

    return converted_state_dict


def convert_base_state_dict(base_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Renames the backbone weights from the base DINOv3 model to match the EoMT naming scheme."""

    converted: dict[str, torch.Tensor] = {}
    for key, value in base_state_dict.items():
        if key.startswith("embeddings.mask_token"):
            # The EoMT embeddings do not expose the masked token.
            continue

        new_key = key
        new_key = re.sub(r"^layer\.", "layers.", new_key)
        new_key = re.sub(r"^norm", "layernorm", new_key)
        converted[new_key] = value

    return converted


def merge_base_and_delta(
    base_state: dict[str, torch.Tensor], delta_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Adds the delta weights on top of the base backbone."""

    final_state: dict[str, torch.Tensor] = {k: v.clone() for k, v in base_state.items()}

    for key, tensor in delta_state.items():
        if key in final_state:
            final_state[key] = final_state[key].to(tensor.dtype) + tensor
        else:
            final_state[key] = tensor

    return final_state


def determine_checkpoint_key(
    repo_id: Optional[str], local_dir: Optional[str], checkpoint_type: Optional[str]
) -> Optional[str]:
    if checkpoint_type is not None:
        return checkpoint_type

    if repo_id is not None:
        return repo_id.split("/")[-1]

    if local_dir is not None:
        return os.path.basename(os.path.abspath(local_dir))

    return None


def ensure_checkpoint_available(repo_id: Optional[str], local_dir: Optional[str], revision: Optional[str]) -> str:
    if local_dir is not None:
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Local directory {local_dir} does not exist")
        return local_dir

    if repo_id is None:
        raise ValueError("Either repo_id or local_dir must be provided")

    return snapshot_download(repo_id, revision=revision)


def load_model_state_dict(input_path: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(input_path, "pytorch_model.bin.index.json")
    model_path = os.path.join(input_path, "pytorch_model.bin")
    safetensor_path = os.path.join(input_path, "model.safetensors")

    if os.path.isfile(index_path):
        with open(index_path, "r") as fp:
            index = json.load(fp)

        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(input_path, shard_name)
            state_dict.update(torch.load(shard_path, map_location="cpu"))
        return state_dict

    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location="cpu")
        return state if isinstance(state, dict) else state.state_dict()

    if os.path.isfile(safetensor_path):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("Install safetensors to load .safetensors checkpoints")
        return load_safetensors(safetensor_path)

    raise FileNotFoundError(f"Could not locate model weights under {input_path}")


def split_qkv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = state_dict.copy()
    keys = [x for x in state_dict.keys() if "qkv" in x]
    for key in keys:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace("qkv", "q_proj")] = q
        state_dict[key.replace("qkv", "k_proj")] = k
        state_dict[key.replace("qkv", "v_proj")] = v
    return state_dict


def convert_timm_state_to_transformers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    original_state = split_qkv(state_dict)
    keys = list(original_state.keys())
    mapping = {}

    for key in keys:
        new_key = key
        for pattern, replacement in TIMM_KEY_MAPPINGS.items():
            new_key = re.sub(pattern, replacement, new_key)
        mapping[key] = new_key

    converted_state: dict[str, torch.Tensor] = {}
    for old_key, new_key in mapping.items():
        weight_tensor = original_state[old_key]

        if "bias_mask" in old_key or "attn.k_proj.bias" in old_key or "local_cls_norm" in old_key:
            continue
        if "embeddings.mask_token" in new_key:
            weight_tensor = weight_tensor.unsqueeze(1)
        if "inv_freq" in new_key:
            continue

        converted_state[new_key] = weight_tensor

    return converted_state


def load_timm_backbone(
    backbone_path: str, metadata: CheckpointMetadata
) -> tuple[dict[str, torch.Tensor], DINOv3ViTConfig, OrderedDict]:
    config_path = os.path.join(backbone_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json in {backbone_path}")

    with open(config_path, "r") as fp:
        config_data = json.load(fp)

    architecture = config_data.get("architecture")
    if architecture is None:
        raise ValueError("Unable to infer architecture from timm config.json")

    if architecture not in TIMM_TO_CONFIG_KWARGS:
        raise ValueError(f"Unsupported timm architecture: {architecture}")

    backbone_state = load_model_state_dict(backbone_path)
    if not isinstance(backbone_state, OrderedDict):
        backbone_state = OrderedDict(backbone_state)

    converted_state = convert_timm_state_to_transformers(backbone_state)
    base_config = DINOv3ViTConfig(**TIMM_TO_CONFIG_KWARGS[architecture])

    return converted_state, base_config, backbone_state


def load_backbone(
    metadata: CheckpointMetadata,
    backbone_repo_id: Optional[str],
    backbone_local_dir: Optional[str],
    backbone_revision: Optional[str],
) -> tuple[dict[str, torch.Tensor], DINOv3ViTConfig, Optional[OrderedDict]]:
    repo_to_use = backbone_repo_id or metadata.default_backbone_repo
    backbone_path = ensure_checkpoint_available(repo_to_use, backbone_local_dir, backbone_revision)

    config_path = os.path.join(backbone_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as fp:
            config_json = json.load(fp)
    else:
        config_json = {}

    if config_json.get("model_type") == "dinov3_vit":
        model = DINOv3ViTModel.from_pretrained(
            backbone_local_dir or repo_to_use,
            revision=backbone_revision,
            local_files_only=backbone_local_dir is not None,
        )
        base_state = convert_base_state_dict(model.state_dict())
        base_config = model.config
        return base_state, base_config, None

    converted_state, base_config, raw_state = load_timm_backbone(backbone_path, metadata)
    base_state = convert_base_state_dict(converted_state)
    return base_state, base_config, raw_state


def infer_metadata(
    repo_id: Optional[str], local_dir: Optional[str], checkpoint_type: Optional[str]
) -> CheckpointMetadata:
    checkpoint_key = determine_checkpoint_key(repo_id, local_dir, checkpoint_type)
    if checkpoint_key is None:
        raise ValueError("Unable to determine checkpoint type. Please provide --checkpoint_type explicitly.")

    if checkpoint_key not in CHECKPOINT_METADATA:
        available = ", ".join(sorted(CHECKPOINT_METADATA))
        raise ValueError(f"Unknown checkpoint '{checkpoint_key}'. Supported values are: {available}")

    return CHECKPOINT_METADATA[checkpoint_key]


def build_config(metadata: CheckpointMetadata, base_config: DINOv3ViTConfig) -> EomtDinov3Config:
    default_overrides = {
        "image_size": metadata.image_size,
        "num_labels": metadata.num_labels,
        "num_queries": metadata.num_queries,
        "num_blocks": metadata.num_blocks,
        "num_channels": base_config.num_channels,
        "patch_size": base_config.patch_size,
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "class_weight": 2.0,
        "mask_weight": 5.0,
        "dice_weight": 5.0,
        "no_object_weight": 0.1,
        "train_num_points": 12544,
        "oversample_ratio": 3.0,
        "importance_sample_ratio": 0.75,
    }

    overrides = {**default_overrides, **metadata.config_overrides}

    return EomtDinov3Config(
        **overrides,
        hidden_size=base_config.hidden_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        intermediate_size=base_config.intermediate_size,
        num_register_tokens=base_config.num_register_tokens,
        rope_theta=getattr(base_config, "rope_theta", 100.0),
        query_bias=getattr(base_config, "query_bias", True),
        key_bias=getattr(base_config, "key_bias", False),
        value_bias=getattr(base_config, "value_bias", True),
        proj_bias=getattr(base_config, "proj_bias", True),
        mlp_bias=getattr(base_config, "mlp_bias", True),
        use_gated_mlp=getattr(base_config, "use_gated_mlp", False),
        hidden_act=base_config.hidden_act,
    )


def _format_slice(tensor: torch.Tensor) -> torch.Tensor:
    """Returns a `(3, 3)` slice of the provided tensor for logging purposes."""

    if tensor.dim() == 4:
        tensor = tensor.flatten(2).transpose(1, 2)

    height = min(3, tensor.shape[1])
    width = min(3, tensor.shape[2])

    return tensor[0, :height, :width].detach().cpu()


def _log_stage_differences(stage: str, hf_tensor: torch.Tensor, original_tensor: torch.Tensor) -> None:
    if hf_tensor.shape != original_tensor.shape:
        LOGGER.warning(
            "Stage %s shape mismatch: hf=%s original=%s",
            stage,
            tuple(hf_tensor.shape),
            tuple(original_tensor.shape),
        )
        return

    aligned_original = original_tensor.to(device=hf_tensor.device, dtype=hf_tensor.dtype)
    difference = (hf_tensor - aligned_original).abs().max().item()
    hf_slice = _format_slice(hf_tensor)
    original_slice = _format_slice(original_tensor)

    LOGGER.info("%s HF[0,:3,:3]=%s", stage, hf_slice)
    LOGGER.info("%s ORIG[0,:3,:3]=%s", stage, original_slice)
    LOGGER.info("%s max |Δ| = %.6e", stage, difference)


def debug_backbone_layers(
    hf_model: EomtDinov3ForUniversalSegmentation,
    original_model: Any,
    pixel_values: torch.Tensor,
    original_inputs: torch.Tensor,
) -> None:
    backbone = original_model.encoder.backbone
    device = pixel_values.device

    with torch.no_grad():
        target_dtype = hf_model.embeddings.patch_embeddings.weight.dtype
        hf_patch = hf_model.embeddings.patch_embeddings(pixel_values.to(dtype=target_dtype))
        hf_patch = hf_patch.flatten(2).transpose(1, 2)

        normalized_inputs = (original_inputs - original_model.encoder.pixel_mean) / original_model.encoder.pixel_std
        normalized_inputs = normalized_inputs.to(device=device, dtype=hf_patch.dtype)

        original_patch = backbone.patch_embed(normalized_inputs)
        _log_stage_differences("patch_embed", hf_patch, original_patch)

        hf_hidden = hf_model.embeddings(pixel_values)

        original_hidden = original_patch
        if hasattr(backbone, "_pos_embed"):
            original_hidden = backbone._pos_embed(original_hidden)
        if hasattr(backbone, "patch_drop"):
            original_hidden = backbone.patch_drop(original_hidden)
        if hasattr(backbone, "norm_pre"):
            original_hidden = backbone.norm_pre(original_hidden)

        _log_stage_differences("pos_embed", hf_hidden, original_hidden)

        hf_position_embeddings = hf_model.rope_embeddings(pixel_values)
        original_rope = None
        if hasattr(backbone, "rope_embeddings"):
            original_rope = backbone.rope_embeddings(normalized_inputs)

        query_insert_idx = hf_model.config.num_hidden_layers - hf_model.config.num_blocks

        for idx, layer_module in enumerate(hf_model.layers):
            stage = f"layer_{idx:02d}"

            if idx == query_insert_idx:
                hf_query = hf_model.query.weight[None, :, :].expand(hf_hidden.shape[0], -1, -1)
                hf_hidden = torch.cat((hf_query, hf_hidden), dim=1)

                original_query = original_model.q.weight[None, :, :].expand(original_hidden.shape[0], -1, -1)
                original_hidden = torch.cat((original_query, original_hidden), dim=1)

            attention_mask = None
            original_attention_mask = None

            if idx >= query_insert_idx:
                hf_norm_hidden = hf_model.layernorm(hf_hidden)
                original_norm_hidden = backbone.norm(original_hidden)

                hf_masks, hf_classes = hf_model.predict(hf_norm_hidden)
                original_masks, original_classes = original_model._predict(original_norm_hidden)

                mask_diff = (hf_masks - original_masks.to(dtype=hf_masks.dtype)).abs().max().item()
                class_diff = (hf_classes - original_classes.to(dtype=hf_classes.dtype)).abs().max().item()
                LOGGER.info("%s mask logits max |Δ| = %.6e", stage, mask_diff)
                LOGGER.info("%s class logits max |Δ| = %.6e", stage, class_diff)

                original_attention_mask = original_model._attn_mask(original_hidden, original_masks, idx)
                attention_mask_bool = original_attention_mask[:, None, ...].expand(
                    -1, hf_model.config.num_attention_heads, -1, -1
                )
                attention_mask = attention_mask_bool.float().masked_fill(~attention_mask_bool, -1e9)

            hf_hidden = layer_module(
                hf_hidden,
                attention_mask=attention_mask,
                position_embeddings=hf_position_embeddings,
            )

            block = backbone.blocks[idx]
            attn_module = getattr(block, "attn", getattr(block, "attention", None))
            if attn_module is None:
                raise AttributeError(f"Block {idx} is missing an attention module")

            attn_output = original_model._attn(
                attn_module,
                block.norm1(original_hidden),
                original_attention_mask,
                rope=original_rope,
            )

            if hasattr(block, "ls1"):
                original_hidden = original_hidden + block.ls1(attn_output)
            elif hasattr(block, "layer_scale1"):
                original_hidden = original_hidden + block.layer_scale1(attn_output)
            else:
                original_hidden = original_hidden + attn_output

            mlp_output = block.mlp(block.norm2(original_hidden))
            if hasattr(block, "ls2"):
                original_hidden = original_hidden + block.ls2(mlp_output)
            elif hasattr(block, "layer_scale2"):
                original_hidden = original_hidden + block.layer_scale2(mlp_output)
            else:
                original_hidden = original_hidden + mlp_output

            _log_stage_differences(stage, hf_hidden, original_hidden)


def verify_conversion(
    config: EomtDinov3Config,
    processor: EomtDinov3ImageProcessorFast,
    hf_model: EomtDinov3ForUniversalSegmentation,
    metadata: CheckpointMetadata,
    raw_delta_state: dict[str, torch.Tensor],
    raw_backbone_state: Optional[OrderedDict],
    original_repo_path: Optional[str],
    image_url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    debug_layers: bool = False,
) -> None:
    if original_repo_path is None:
        raise ValueError("Verification requested but --original_repo_path was not provided")

    if raw_backbone_state is None:
        raise ValueError("Verification requires timm backbone weights. Provide a timm backbone repo.")

    import sys
    from io import BytesIO

    import requests
    from PIL import Image

    sys.path.insert(0, original_repo_path)
    try:
        from models.eomt import EoMT  # type: ignore
        from models.vit import ViT  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import the original EoMT implementation. Ensure --original_repo_path "
            "points to a clone of https://github.com/tue-mps/eomt."
        ) from exc

    device = torch.device("cpu")

    encoder = ViT(
        img_size=(metadata.image_size, metadata.image_size),
        patch_size=config.patch_size,
        backbone_name=metadata.timm_arch,
    )
    encoder.backbone.load_state_dict(raw_backbone_state, strict=True)

    original_model = EoMT(
        encoder=encoder,
        num_classes=metadata.num_labels,
        num_q=metadata.num_queries,
        num_blocks=metadata.num_blocks,
    ).to(device)
    original_model.eval()

    # Zero non-encoder parameters to mimic delta addition
    with torch.no_grad():
        for name, param in original_model.named_parameters():
            if not name.startswith("encoder."):
                param.zero_()

    current_state = original_model.state_dict()
    delta_state = {
        key[len("network.") :]: value for key, value in raw_delta_state.items() if key.startswith("network.")
    }

    missing_keys = [key for key in current_state.keys() if key not in delta_state]
    if missing_keys:
        raise KeyError(f"Delta checkpoint is missing keys required for verification: {missing_keys[:5]}")

    summed_state = {key: current_state[key] + delta_state[key] for key in current_state.keys()}
    original_model.load_state_dict(summed_state, strict=True)

    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    mean = torch.tensor(processor.image_mean, dtype=pixel_values.dtype).view(1, -1, 1, 1)
    std = torch.tensor(processor.image_std, dtype=pixel_values.dtype).view(1, -1, 1, 1)
    original_inputs = pixel_values * std + mean

    if debug_layers:
        debug_backbone_layers(hf_model, original_model, pixel_values, original_inputs)

    with torch.no_grad():
        hf_outputs = hf_model(pixel_values=pixel_values)
        orig_masks, orig_classes = original_model(original_inputs)

    orig_mask_logits = orig_masks[-1]
    orig_class_logits = orig_classes[-1]

    torch.testing.assert_close(hf_outputs.masks_queries_logits, orig_mask_logits, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(hf_outputs.class_queries_logits, orig_class_logits, atol=5e-4, rtol=5e-4)
    LOGGER.info("Verification against the original implementation succeeded.")


def convert_model(
    repo_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_hub_path: Optional[str] = None,
    backbone_repo_id: Optional[str] = None,
    backbone_local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    backbone_revision: Optional[str] = None,
    checkpoint_type: Optional[str] = None,
    safe_serialization: bool = True,
    verify: bool = False,
    original_repo_path: Optional[str] = None,
    debug_layers: bool = False,
):
    if output_dir is None and output_hub_path is None:
        raise ValueError("At least one of output_dir or output_hub_path must be specified")

    if repo_id is None and local_dir is None:
        raise ValueError("Either repo_id or local_dir must be specified")

    metadata = infer_metadata(repo_id, local_dir, checkpoint_type)

    torch.set_default_dtype(torch.float16)

    ckpt_path = ensure_checkpoint_available(repo_id, local_dir, revision)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_state, base_config, raw_backbone_state = load_backbone(
        metadata, backbone_repo_id, backbone_local_dir, backbone_revision
    )

    config = build_config(metadata, base_config)
    processor = EomtDinov3ImageProcessorFast(num_labels=metadata.num_labels, **metadata.processor_kwargs)

    if output_dir:
        config.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    if output_hub_path:
        config.push_to_hub(output_hub_path)
        processor.push_to_hub(output_hub_path)

    with init_empty_weights():
        model = EomtDinov3ForUniversalSegmentation(config)

    raw_delta_state = load_model_state_dict(ckpt_path)
    delta_state = convert_delta_keys(raw_delta_state)

    final_state = merge_base_and_delta(base_state, delta_state)

    model.load_state_dict(final_state, strict=True, assign=True)
    model = model.to(torch.float32)
    model.eval()

    if verify:
        verify_conversion(
            config,
            processor,
            model,
            metadata,
            raw_delta_state,
            raw_backbone_state,
            original_repo_path,
            debug_layers=debug_layers,
        )

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
    parser.add_argument("--checkpoint_type", help="Explicit checkpoint type to convert", default=None)
    parser.add_argument(
        "--original_repo_path", help="Path to the original EoMT repository for verification", default=None
    )
    parser.add_argument("--safe_serialization", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify conversion against the original implementation")
    parser.add_argument(
        "--debug-layers",
        action="store_true",
        help="Log intermediate activations for both implementations during verification",
    )
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
        checkpoint_type=args.checkpoint_type,
        safe_serialization=args.safe_serialization,
        verify=args.verify,
        original_repo_path=args.original_repo_path,
        debug_layers=args.debug_layers,
    )


if __name__ == "__main__":
    main()
