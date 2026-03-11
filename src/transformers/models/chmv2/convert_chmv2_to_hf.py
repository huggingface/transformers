# Copyright 2026 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Convert CHMv2 checkpoints from the original repository.

Usage:
    python -m transformers.models.chmv2.convert_chmv2_to_hf \
        --head_checkpoint_path /path/to/checkpoint.pth \
        --pytorch_dump_folder_path /path/to/output \
        --model_name chmv2

Or with a DINOv3 backbone from HuggingFace:
    python -m transformers.models.chmv2.convert_chmv2_to_hf \
        --head_checkpoint_path /path/to/head_checkpoint.pth \
        --backbone_repo_id facebook/dinov3-vitl16-pretrain-lvd1689m \
        --pytorch_dump_folder_path /path/to/output \
        --model_name chmv2
"""

import argparse
import os

import regex as re
import torch
from PIL import Image

from transformers.models.dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.convert_dinov3_vit_to_hf import (
    convert_old_keys_to_new_keys,
    get_dinov3_config,
    split_qkv,
)
from transformers.utils import logging

from .configuration_chmv2 import CHMv2Config
from .image_processing_chmv2_fast import CHMv2ImageProcessorFast
from .modeling_chmv2 import CHMv2ForDepthEstimation


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# Model configurations for CHMv2 head
MODEL_CONFIGS = {
    "chmv2": {
        "backbone_model": "vitl16_sat493m",
        "out_indices": [6, 12, 18, 24],
        "post_process_channels": [128, 256, 512, 1024],
        "fusion_hidden_size": 256,
    },
}

# Head key mapping: (head\.)? prefix for full-model checkpoints, no prefix for head-only
# fmt: off
HEAD_ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^(head\.)?reassemble_blocks\.projects\.(\d+)\.conv\.(weight|bias)$":                    r"head.reassemble_stage.layers.\2.projection.\3",
    r"^(head\.)?reassemble_blocks\.resize_layers\.(\d+)\.(weight|bias)$":                    r"head.reassemble_stage.layers.\2.resize.\3",
    r"^(head\.)?reassemble_blocks\.batchnorm_layers\.(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$": r"head.reassemble_stage.layers.\2.batchnorm.\3",
    r"^(head\.)?reassemble_blocks\.readout_projects\.(\d+)\.0\.(weight|bias)$":             r"head.reassemble_stage.readout_projects.\2.0.\3",
    r"^(head\.)?convs\.(\d+)\.conv\.weight$":                                               r"head.convs.\2.weight",
    r"^(head\.)?fusion_blocks\.(\d+)\.res_conv_unit1\.conv1\.conv\.(weight|bias)$":         r"head.fusion_layers.\2.residual_layer1.convolution1.\3",
    r"^(head\.)?fusion_blocks\.(\d+)\.res_conv_unit1\.conv2\.conv\.(weight|bias)$":         r"head.fusion_layers.\2.residual_layer1.convolution2.\3",
    r"^(head\.)?fusion_blocks\.(\d+)\.res_conv_unit2\.conv1\.conv\.(weight|bias)$":         r"head.fusion_layers.\2.residual_layer2.convolution1.\3",
    r"^(head\.)?fusion_blocks\.(\d+)\.res_conv_unit2\.conv2\.conv\.(weight|bias)$":         r"head.fusion_layers.\2.residual_layer2.convolution2.\3",
    r"^(head\.)?fusion_blocks\.(\d+)\.project\.conv\.(weight|bias)$":                       r"head.fusion_layers.\2.projection.\3",
    r"^(head\.)?conv_depth\.head\.(0|2|4)\.(weight|bias)$":                                 r"head.conv_depth.head.\2.\3",
}
# fmt: on


def convert_head_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    """Convert original CHMv2 head keys to HuggingFace format using regex patterns."""
    if not state_dict_keys:
        return {}
    old_text = "\n".join(state_dict_keys)
    new_text = old_text
    for pattern, replacement in HEAD_ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)
    return dict(zip(old_text.split("\n"), new_text.split("\n")))


def load_original_state_dict(checkpoint_path: str) -> dict:
    """Load checkpoint and return state dict (handles 'model' / 'state_dict' wrappers)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def get_chmv2_config(model_name: str, backbone_repo_id: str | None = None) -> CHMv2Config:
    """Create CHMv2 config based on model name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    model_cfg = MODEL_CONFIGS[model_name]

    if backbone_repo_id:
        backbone_config = DINOv3ViTConfig.from_pretrained(
            backbone_repo_id,
            out_indices=model_cfg["out_indices"],
            apply_layernorm=True,
            reshape_hidden_states=True,
            return_class_token=True,
        )
    else:
        backbone_config = get_dinov3_config(model_cfg["backbone_model"])
        backbone_config.out_indices = model_cfg["out_indices"]
        backbone_config.apply_layernorm = True
        backbone_config.reshape_hidden_states = True
        backbone_config.return_class_token = True

    return CHMv2Config(
        backbone_config=backbone_config,
        backbone_type="dinov3_vit",
        patch_size=backbone_config.patch_size,
        post_process_channels=model_cfg["post_process_channels"],
        fusion_hidden_size=model_cfg["fusion_hidden_size"],
        min_depth=0.001,
        max_depth=96.0,
        bins_strategy="chmv2_mixlog",
        norm_strategy="chmv2_mixlog",
    )


def convert_backbone_keys(state_dict: dict) -> dict:
    """Convert backbone keys using dinov3_vit conversion functions."""
    backbone_keys = [k for k in state_dict.keys() if k.startswith("backbone.")]
    if not backbone_keys:
        return state_dict

    stripped = {k[len("backbone.") :]: state_dict.pop(k) for k in backbone_keys}
    stripped = split_qkv(stripped)
    key_mapping = convert_old_keys_to_new_keys(list(stripped.keys()))

    result = {}
    for old_key, value in stripped.items():
        new_key = key_mapping.get(old_key, old_key)
        if "bias_mask" in new_key or "k_proj.bias" in new_key or "local_cls_norm" in new_key:
            continue
        if "inv_freq" in new_key:
            continue
        if "mask_token" in new_key and value.dim() == 2:
            value = value.unsqueeze(1)
        result[f"backbone.{new_key}"] = value

    return result


def convert_head_keys(state_dict: dict) -> dict:
    """Apply head key renames via regex mapping and return only head.* keys."""
    head_like = re.compile(r"^(head\.)?(reassemble_blocks|convs|fusion_blocks|conv_depth)\.")
    head_keys = [k for k in state_dict.keys() if head_like.match(k)]
    key_mapping = convert_head_keys_to_new_keys(head_keys)
    result = {}
    for old_key in head_keys:
        new_key = key_mapping.get(old_key, old_key)
        if new_key.startswith("head."):
            result[new_key] = state_dict[old_key]
    return result


@torch.no_grad()
def convert_chmv2_checkpoint(
    head_checkpoint_path: str,
    pytorch_dump_folder_path: str,
    backbone_checkpoint_path: str | None = None,
    model_name: str = "chmv2",
    backbone_repo_id: str | None = None,
    push_to_hub: bool = False,
    verify_image_path: str | None = None,
) -> None:
    """
    Convert CHMv2 checkpoints to HuggingFace format.

    Accepts head-only or combined (backbone + head) checkpoint. Optionally a separate
    backbone checkpoint or backbone_repo_id for a pre-converted DINOv3.
    """
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    config = get_chmv2_config(model_name=model_name, backbone_repo_id=backbone_repo_id)

    # Load checkpoint(s)
    logger.info(f"Loading checkpoint from {head_checkpoint_path}")
    head_ckpt = load_original_state_dict(head_checkpoint_path)

    has_backbone_keys = any(k.startswith("backbone.") for k in head_ckpt.keys())
    head_only = (not has_backbone_keys) and any(k.startswith("reassemble_blocks.") for k in head_ckpt.keys())

    state_dict = {}

    if backbone_checkpoint_path is not None:
        logger.info(f"Loading backbone from {backbone_checkpoint_path}")
        backbone_raw = load_original_state_dict(backbone_checkpoint_path)
        backbone_raw = {f"backbone.{k}": v for k, v in backbone_raw.items()}
        state_dict.update(convert_backbone_keys(backbone_raw))
    elif has_backbone_keys:
        logger.info("Converting backbone keys from checkpoint")
        state_dict.update(convert_backbone_keys(head_ckpt))

    logger.info(f"Converting head weights (head_only={head_only})")
    state_dict.update(convert_head_keys(head_ckpt))

    # Load into model
    model = CHMv2ForDepthEstimation(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    missing_non_inv = [k for k in missing if "inv_freq" not in k]
    if missing_non_inv:
        logger.warning(f"Missing keys (non-inv_freq): {missing_non_inv}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.eval()

    # Optional verification
    if verify_image_path is not None:
        logger.info(f"Verifying with image: {verify_image_path}")
        image = Image.open(verify_image_path)
        processor = CHMv2ImageProcessorFast()
        inputs = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        depth = results[0]["predicted_depth"]
        print(
            f"Predicted depth — shape: {depth.shape}  mean: {depth.mean():.4f}  range: [{depth.min():.4f}, {depth.max():.4f}]"
        )

    # Save
    logger.info(f"Saving to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    processor = CHMv2ImageProcessorFast()
    processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        repo_id = f"facebook/{model_name}-hf"
        logger.info(f"Pushing to hub: {repo_id}")
        model.push_to_hub(repo_id=repo_id)
        processor.push_to_hub(repo_id=repo_id)

    print("Conversion complete!")
    print(f"Model saved to: {pytorch_dump_folder_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CHMv2 checkpoints to HuggingFace format")
    parser.add_argument("--head_checkpoint_path", type=str, required=True, help="Path to CHMv2 head checkpoint (.pth)")
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Output directory for converted model"
    )
    parser.add_argument(
        "--backbone_checkpoint_path",
        type=str,
        default=None,
        help="Path to separate DINOv3 backbone checkpoint, if not in head checkpoint",
    )
    parser.add_argument("--model_name", type=str, default="chmv2", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument(
        "--backbone_repo_id", type=str, default=None, help="HuggingFace repo ID for pre-converted DINOv3 backbone"
    )
    parser.add_argument("--verify_image_path", type=str, default=None, help="Image path to verify conversion")
    parser.add_argument("--push_to_hub", action="store_true")

    args = parser.parse_args()

    convert_chmv2_checkpoint(
        head_checkpoint_path=args.head_checkpoint_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        backbone_checkpoint_path=args.backbone_checkpoint_path,
        model_name=args.model_name,
        backbone_repo_id=args.backbone_repo_id,
        verify_image_path=args.verify_image_path,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
