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
        --checkpoint_path /path/to/checkpoint.pth \
        --pytorch_dump_folder_path /path/to/output \
        --model_name chmv2

Or with a DINOv3 backbone from HuggingFace:
    python -m transformers.models.chmv2.convert_chmv2_to_hf \
        --checkpoint_path /path/to/head_checkpoint.pth \
        --backbone_repo_id facebook/dinov3-vitl16-pretrain-lvd1689m \
        --pytorch_dump_folder_path /path/to/output \
        --model_name chmv2
"""

import argparse
from pathlib import Path

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
        "backbone_model": "vitl16_sat493m",  # DINOv3 backbone variant
        "out_indices": [6, 12, 18, 24],  # BackboneMixin stage indices (stageN = after layer N-1)
        "post_process_channels": [128, 256, 512, 1024],
        "fusion_hidden_size": 256,
    },
}


def get_chmv2_config(
    model_name: str,
    backbone_repo_id: str | None = None,
    min_depth: float = 0.001,
    max_depth: float = 96.0,
    bins_strategy: str = "chmv2_mixlog",
    norm_strategy: str = "chmv2_mixlog",
) -> CHMv2Config:
    """Create CHMv2 config based on model name.

    Reuses get_dinov3_config from dinov3_vit conversion script for backbone config.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    model_cfg = MODEL_CONFIGS[model_name]

    # Create backbone config - reuse from dinov3_vit conversion script
    if backbone_repo_id:
        backbone_config = DINOv3ViTConfig.from_pretrained(
            backbone_repo_id,
            out_indices=model_cfg["out_indices"],
            apply_layernorm=True,
            reshape_hidden_states=True,
            return_class_token=True,
        )
    else:
        # Use get_dinov3_config from dinov3_vit conversion script
        backbone_config = get_dinov3_config(model_cfg["backbone_model"])
        backbone_config.out_indices = model_cfg["out_indices"]
        backbone_config.apply_layernorm = True
        backbone_config.reshape_hidden_states = True
        backbone_config.return_class_token = True

    config = CHMv2Config(
        backbone_config=backbone_config,
        backbone_type="dinov3_vit",
        patch_size=backbone_config.patch_size,
        post_process_channels=model_cfg["post_process_channels"],
        fusion_hidden_size=model_cfg["fusion_hidden_size"],
        min_depth=min_depth,
        max_depth=max_depth,
        bins_strategy=bins_strategy,
        norm_strategy=norm_strategy,
    )

    return config


def convert_backbone_keys(state_dict: dict) -> dict:
    """Convert backbone keys using dinov3_vit conversion functions.

    This replaces the old create_rename_keys_backbone function by reusing
    the regex-based conversion from convert_dinov3_vit_to_hf.py.
    """
    # Extract backbone keys
    backbone_keys = [k for k in state_dict.keys() if k.startswith("backbone.")]
    if not backbone_keys:
        return state_dict

    # Strip "backbone." prefix for conversion
    stripped_state_dict = {}
    for key in backbone_keys:
        stripped_key = key[len("backbone.") :]
        stripped_state_dict[stripped_key] = state_dict.pop(key)

    # Split QKV weights using dinov3_vit function
    stripped_state_dict = split_qkv(stripped_state_dict)

    # Get key mapping using dinov3_vit function
    key_mapping = convert_old_keys_to_new_keys(list(stripped_state_dict.keys()))

    # Apply key mapping and add "backbone." prefix back
    for old_key in list(stripped_state_dict.keys()):
        new_key = key_mapping.get(old_key, old_key)
        value = stripped_state_dict[old_key]

        # Skip keys that should be excluded (matching convert_dinov3_vit_to_hf.py)
        if "bias_mask" in new_key or "k_proj.bias" in new_key or "local_cls_norm" in new_key:
            continue

        # Handle mask_token shape: [1, hidden_size] -> [1, 1, hidden_size]
        if "mask_token" in new_key and value.dim() == 2:
            value = value.unsqueeze(1)

        # Skip inv_freq (computed from config)
        if "inv_freq" in new_key:
            continue

        state_dict[f"backbone.{new_key}"] = value

    return state_dict


def create_rename_keys_head(head_only: bool = False) -> list[tuple[str, str]]:
    """
    Create rename keys for CHMv2 head weights.

    Args:
        config: CHMv2 configuration
        head_only: If True, source keys have no prefix (decoder-only checkpoint).
                   If False, source keys have 'head.' prefix (full model checkpoint).

    Original structure (DINOv3 OSS code):
        [prefix.]reassemble_blocks.projects.{i}.conv
        [prefix.]reassemble_blocks.resize_layers.{i}
        [prefix.]reassemble_blocks.batchnorm_layers.{i}
        [prefix.]reassemble_blocks.readout_projects.{i}
        [prefix.]convs.{i}.conv
        [prefix.]fusion_blocks.{i}.res_conv_unit1.conv1.conv / conv2.conv
        [prefix.]fusion_blocks.{i}.res_conv_unit2.conv1.conv / conv2.conv
        [prefix.]conv_depth.head.{0,2,4}

    HF structure:
        head.reassemble_stage.layers.{i}.projection  (plain nn.Conv2d, no .conv)
        head.reassemble_stage.layers.{i}.resize
        head.reassemble_stage.layers.{i}.batchnorm
        head.reassemble_stage.readout_projects.{i}
        head.convs.{i} (plain Conv2d, no .conv)
        head.fusion_layers.{j}.projection  (renamed from .project, plain nn.Conv2d)
        head.fusion_layers.{j}.residual_layer1.convolution1 / convolution2
        head.fusion_layers.{j}.residual_layer2.convolution1 / convolution2
        head.conv_depth.head.{0,2,4}
    """
    rename_keys = []

    # Source prefix: empty for head-only checkpoint, "head." for full model
    src_prefix = "" if head_only else "head."

    # Reassemble stage
    for i in range(4):
        # Projection conv (plain nn.Conv2d — no .conv. wrapper)
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.projects.{i}.conv.weight",
                f"head.reassemble_stage.layers.{i}.projection.weight",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.projects.{i}.conv.bias",
                f"head.reassemble_stage.layers.{i}.projection.bias",
            )
        )

        # Resize layers (layer 2 is Identity, so it has no weights)
        if i != 2:
            rename_keys.append(
                (
                    f"{src_prefix}reassemble_blocks.resize_layers.{i}.weight",
                    f"head.reassemble_stage.layers.{i}.resize.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}reassemble_blocks.resize_layers.{i}.bias",
                    f"head.reassemble_stage.layers.{i}.resize.bias",
                )
            )

        # Batchnorm (if present - may be Identity)
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.batchnorm_layers.{i}.weight",
                f"head.reassemble_stage.layers.{i}.batchnorm.weight",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.batchnorm_layers.{i}.bias",
                f"head.reassemble_stage.layers.{i}.batchnorm.bias",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.batchnorm_layers.{i}.running_mean",
                f"head.reassemble_stage.layers.{i}.batchnorm.running_mean",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.batchnorm_layers.{i}.running_var",
                f"head.reassemble_stage.layers.{i}.batchnorm.running_var",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.batchnorm_layers.{i}.num_batches_tracked",
                f"head.reassemble_stage.layers.{i}.batchnorm.num_batches_tracked",
            )
        )

        # Readout projects (if present)
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.readout_projects.{i}.0.weight",
                f"head.reassemble_stage.readout_projects.{i}.0.weight",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}reassemble_blocks.readout_projects.{i}.0.bias",
                f"head.reassemble_stage.readout_projects.{i}.0.bias",
            )
        )
    for i in range(4):
        rename_keys.append((f"{src_prefix}convs.{i}.conv.weight", f"head.convs.{i}.weight"))

    for i in range(4):
        # res_conv_unit1 -> residual_layer1 (only for i > 0, since fusion_blocks[0].res_conv_unit1 = None)
        if i > 0:
            rename_keys.append(
                (
                    f"{src_prefix}fusion_blocks.{i}.res_conv_unit1.conv1.conv.weight",
                    f"head.fusion_layers.{i}.residual_layer1.convolution1.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}fusion_blocks.{i}.res_conv_unit1.conv1.conv.bias",
                    f"head.fusion_layers.{i}.residual_layer1.convolution1.bias",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}fusion_blocks.{i}.res_conv_unit1.conv2.conv.weight",
                    f"head.fusion_layers.{i}.residual_layer1.convolution2.weight",
                )
            )
            rename_keys.append(
                (
                    f"{src_prefix}fusion_blocks.{i}.res_conv_unit1.conv2.conv.bias",
                    f"head.fusion_layers.{i}.residual_layer1.convolution2.bias",
                )
            )

        # res_conv_unit2 -> residual_layer2 (all layers have this)
        rename_keys.append(
            (
                f"{src_prefix}fusion_blocks.{i}.res_conv_unit2.conv1.conv.weight",
                f"head.fusion_layers.{i}.residual_layer2.convolution1.weight",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}fusion_blocks.{i}.res_conv_unit2.conv1.conv.bias",
                f"head.fusion_layers.{i}.residual_layer2.convolution1.bias",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}fusion_blocks.{i}.res_conv_unit2.conv2.conv.weight",
                f"head.fusion_layers.{i}.residual_layer2.convolution2.weight",
            )
        )
        rename_keys.append(
            (
                f"{src_prefix}fusion_blocks.{i}.res_conv_unit2.conv2.conv.bias",
                f"head.fusion_layers.{i}.residual_layer2.convolution2.bias",
            )
        )
        # Projection layer in fusion blocks (renamed .project → .projection, plain nn.Conv2d)
        rename_keys.append(
            (f"{src_prefix}fusion_blocks.{i}.project.conv.weight", f"head.fusion_layers.{i}.projection.weight")
        )
        rename_keys.append(
            (f"{src_prefix}fusion_blocks.{i}.project.conv.bias", f"head.fusion_layers.{i}.projection.bias")
        )

    # UpConvHead - both use nn.Sequential with same indices
    # head[0]: Conv2d(features, features // 2, kernel_size=3)
    # head[1]: Interpolate (no weights)
    # head[2]: Conv2d(features // 2, n_hidden_channels, kernel_size=3)
    # head[3]: ReLU (no weights)
    # head[4]: Conv2d(n_hidden_channels, n_output_channels, kernel_size=1)
    rename_keys.append((f"{src_prefix}conv_depth.head.0.weight", "head.conv_depth.head.0.weight"))
    rename_keys.append((f"{src_prefix}conv_depth.head.0.bias", "head.conv_depth.head.0.bias"))
    rename_keys.append((f"{src_prefix}conv_depth.head.2.weight", "head.conv_depth.head.2.weight"))
    rename_keys.append((f"{src_prefix}conv_depth.head.2.bias", "head.conv_depth.head.2.bias"))
    rename_keys.append((f"{src_prefix}conv_depth.head.4.weight", "head.conv_depth.head.4.weight"))
    rename_keys.append((f"{src_prefix}conv_depth.head.4.bias", "head.conv_depth.head.4.bias"))

    return rename_keys


def rename_key(state_dict: dict, old: str, new: str) -> None:
    """Rename a key in state dict if it exists."""
    if old in state_dict:
        val = state_dict.pop(old)
        state_dict[new] = val


@torch.no_grad()
def convert_chmv2_checkpoint(
    head_checkpoint_path: str,
    pytorch_dump_folder_path: str,
    backbone_checkpoint_path: str | None = None,
    model_name: str = "chmv2",
    backbone_repo_id: str | None = None,
    min_depth: float = 0.001,
    max_depth: float = 96.0,
    bins_strategy: str = "chmv2_mixlog",
    norm_strategy: str = "chmv2_mixlog",
    push_to_hub: bool = False,
    verify_image_path: str | None = None,
) -> None:
    """
    Convert CHMv2 checkpoints to HuggingFace format and save the model + image processor.

    Accepts two separate checkpoint files (the original release format):
      - ``backbone_checkpoint_path``: DINOv3 backbone weights (.pth)
      - ``head_checkpoint_path``: CHMv2 DPT head weights (.pth), keys like ``reassemble_blocks.*``

    Also accepts a single combined checkpoint via ``head_checkpoint_path`` if it already
    contains both ``backbone.*`` and ``head.*`` keys.

    Args:
        head_checkpoint_path: Path to the head-only (or combined) checkpoint.
        pytorch_dump_folder_path: Directory to write the converted model + image processor.
        backbone_checkpoint_path: Path to the separate backbone checkpoint (optional).
        model_name: Key in ``MODEL_CONFIGS`` (default ``"chmv2"``).
        backbone_repo_id: HuggingFace repo ID for a pre-converted DINOv3 backbone.
        verify_image_path: If set, run a forward pass on this image (path to a .tif or any PIL-readable
            file) and print the predicted depth stats to verify the conversion.
        push_to_hub: Whether to push the converted model to the HuggingFace Hub.
    """
    config = get_chmv2_config(
        model_name=model_name,
        backbone_repo_id=backbone_repo_id,
        min_depth=min_depth,
        max_depth=max_depth,
        bins_strategy=bins_strategy,
        norm_strategy=norm_strategy,
    )

    # --- Build state dict -------------------------------------------------
    head_ckpt = torch.load(head_checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in head_ckpt:
        head_ckpt = head_ckpt["model"]
    elif "state_dict" in head_ckpt:
        head_ckpt = head_ckpt["state_dict"]

    # Determine if the head checkpoint also contains backbone keys
    has_backbone_keys = any(k.startswith("backbone.") for k in head_ckpt.keys())
    head_only = (not has_backbone_keys) and any(k.startswith("reassemble_blocks.") for k in head_ckpt.keys())

    state_dict: dict = {}

    if backbone_checkpoint_path is not None:
        logger.info(f"Loading backbone from {backbone_checkpoint_path} ...")
        backbone_raw = torch.load(backbone_checkpoint_path, map_location="cpu", weights_only=True)
        backbone_raw = {f"backbone.{k}": v for k, v in backbone_raw.items()}
        state_dict.update(convert_backbone_keys(backbone_raw))
    elif has_backbone_keys:
        logger.info("Converting backbone keys embedded in checkpoint ...")
        state_dict.update(convert_backbone_keys(head_ckpt))

    logger.info(f"Converting head weights (head_only={head_only}) ...")
    rename_keys = create_rename_keys_head(head_only=head_only)
    head_sd = head_ckpt.copy()
    for src, dest in rename_keys:
        rename_key(head_sd, src, dest)
    # Only keep keys that were successfully renamed (have head. prefix) or are already correct
    state_dict.update({k: v for k, v in head_sd.items() if k.startswith("head.")})

    # --- Load model -------------------------------------------------------
    model = CHMv2ForDepthEstimation(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    missing_non_inv = [k for k in missing if "inv_freq" not in k]
    if missing_non_inv:
        logger.warning(f"Missing keys (non-inv_freq): {missing_non_inv}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.eval()

    # --- Image processor --------------------------------------------------
    processor = CHMv2ImageProcessorFast()

    # --- Optional verification --------------------------------------------
    if verify_image_path is not None:
        logger.info(f"Verifying with image: {verify_image_path}")
        image = Image.open(verify_image_path)
        inputs = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        original_size = (image.height, image.width)
        results = processor.post_process_depth_estimation(outputs, target_sizes=[original_size])
        depth = results[0]["predicted_depth"]
        print(
            f"Predicted depth — shape: {depth.shape}  mean: {depth.mean():.4f}  range: [{depth.min():.4f}, {depth.max():.4f}]"
        )

    # --- Save -------------------------------------------------------------
    if pytorch_dump_folder_path:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
        logger.info("Done.")

    if push_to_hub:
        logger.info("Pushing model and processor to hub ...")
        model.push_to_hub(repo_id=f"facebook/{model_name}-hf")
        processor.push_to_hub(repo_id=f"facebook/{model_name}-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CHMv2 checkpoints to HuggingFace format")

    parser.add_argument(
        "--head_checkpoint_path",
        type=str,
        required=True,
        help="Path to the CHMv2 head checkpoint (.pth). May also contain backbone keys.",
    )
    parser.add_argument(
        "--backbone_checkpoint_path",
        type=str,
        default=None,
        help="Path to the separate DINOv3 backbone checkpoint (.pth), if not embedded in the head checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        required=True,
        help="Path to the output directory for the converted HuggingFace model + image processor.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="chmv2",
        choices=list(MODEL_CONFIGS.keys()),
        help="Name of the model variant",
    )
    parser.add_argument(
        "--backbone_repo_id",
        type=str,
        default=None,
        help="HuggingFace repo ID for a pre-converted DINOv3 backbone.",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=96.0,
    )
    parser.add_argument(
        "--bins_strategy",
        type=str,
        default="chmv2_mixlog",
        choices=["linear", "log", "chmv2_mixlog"],
    )
    parser.add_argument(
        "--norm_strategy",
        type=str,
        default="chmv2_mixlog",
        choices=["linear", "softmax", "sigmoid", "chmv2_mixlog"],
    )
    parser.add_argument(
        "--verify_image_path",
        type=str,
        default=None,
        help="Path to an image (tif or any PIL-readable format) to verify the conversion.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )

    args = parser.parse_args()

    convert_chmv2_checkpoint(
        head_checkpoint_path=args.head_checkpoint_path,
        backbone_checkpoint_path=args.backbone_checkpoint_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        model_name=args.model_name,
        backbone_repo_id=args.backbone_repo_id,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        bins_strategy=args.bins_strategy,
        norm_strategy=args.norm_strategy,
        verify_image_path=args.verify_image_path,
        push_to_hub=args.push_to_hub,
    )
