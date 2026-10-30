# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Conversion script for BEVFormer (tiny checkpoint) to Hugging Face format."""

import argparse
import os
import re
import urllib.request

import torch

from transformers import (
    BevformerConfig,
    BevformerFor3DObjectDetection,
    BevformerImageProcessor,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

DEFAULT_CHECKPOINT_URL = "https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth"


def map_upstream_key_to_hf(key: str) -> str | None:
    """Map an upstream mmdet3d / BEVFormer state_dict key to the HF BevformerFor3DObjectDetection parameter name.

    Returns None if the key is not a model parameter (e.g. training artifact).
    """
    if key == "pts_bbox_head.code_weights":
        return None

    # 1. ResNet-50 backbone
    if key.startswith("img_backbone."):
        rem = key[len("img_backbone.") :]
        if rem == "conv1.weight":
            return "bevformer.conv_encoder.model.embedder.embedder.convolution.weight"
        if rem.startswith("bn1."):
            norm_param = rem[len("bn1.") :]
            return f"bevformer.conv_encoder.model.embedder.embedder.normalization.{norm_param}"

        m = re.match(r"^layer(\d+)\.(\d+)\.(.*)", rem)
        if m:
            stage_idx = int(m.group(1)) - 1
            layer_idx = int(m.group(2))
            rest = m.group(3)

            m_conv = re.match(r"^conv(\d+)\.weight$", rest)
            if m_conv:
                conv_idx = int(m_conv.group(1)) - 1
                return f"bevformer.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{conv_idx}.convolution.weight"

            m_bn = re.match(r"^bn(\d+)\.(.*)", rest)
            if m_bn:
                conv_idx = int(m_bn.group(1)) - 1
                norm_param = m_bn.group(2)
                return f"bevformer.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{conv_idx}.normalization.{norm_param}"

            if rest == "downsample.0.weight":
                return f"bevformer.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.convolution.weight"
            if rest.startswith("downsample.1."):
                norm_param = rest[len("downsample.1.") :]
                return f"bevformer.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.{norm_param}"

    # 2. Neck / FPN
    if key.startswith("img_neck."):
        rem = key[len("img_neck.") :]
        if rem.startswith("lateral_convs.0.conv."):
            param = rem[len("lateral_convs.0.conv.") :]
            return f"bevformer.fpn.lateral_convs.0.{param}"
        if rem.startswith("fpn_convs.0.conv."):
            param = rem[len("fpn_convs.0.conv.") :]
            return f"bevformer.fpn.fpn_convs.0.{param}"

    # 3. Embeddings, Positional Encodings & Can-bus
    if key == "pts_bbox_head.bev_embedding.weight":
        return "bevformer.query_embedding.bev_embedding.weight"
    if key == "pts_bbox_head.query_embedding.weight":
        return "bevformer.query_embedding.query_embedding.weight"
    if key == "pts_bbox_head.positional_encoding.row_embed.weight":
        return "bevformer.positional_encoding.row_embed.weight"
    if key == "pts_bbox_head.positional_encoding.col_embed.weight":
        return "bevformer.positional_encoding.col_embed.weight"
    if key == "pts_bbox_head.transformer.level_embeds":
        return "bevformer.encoder.level_embeds"
    if key == "pts_bbox_head.transformer.cams_embeds":
        return "bevformer.encoder.cams_embeds"
    if key.startswith("pts_bbox_head.transformer.reference_points."):
        param = key[len("pts_bbox_head.transformer.reference_points.") :]
        return f"bevformer.query_embedding.reference_points.{param}"

    if key.startswith("pts_bbox_head.transformer.can_bus_mlp."):
        rem = key[len("pts_bbox_head.transformer.can_bus_mlp.") :]
        if rem.startswith("norm."):
            norm_param = rem[len("norm.") :]
            return f"bevformer.encoder.can_bus_mlp.can_bus_norm.{norm_param}"
        return f"bevformer.encoder.can_bus_mlp.can_bus_mlp.{rem}"

    # 4. Encoder Layers
    if key.startswith("pts_bbox_head.transformer.encoder.layers."):
        rem = key[len("pts_bbox_head.transformer.encoder.layers.") :]
        m = re.match(r"^(\d+)\.(.*)", rem)
        if m:
            layer_idx = m.group(1)
            rest = m.group(2)

            if rest.startswith("attentions.0."):
                sub = rest[len("attentions.0.") :]
                return f"bevformer.encoder.layers.{layer_idx}.self_attn.{sub}"
            if rest.startswith("norms.0."):
                sub = rest[len("norms.0.") :]
                return f"bevformer.encoder.layers.{layer_idx}.norm1.{sub}"
            if rest.startswith("attentions.1."):
                sub = rest[len("attentions.1.") :]
                return f"bevformer.encoder.layers.{layer_idx}.cross_attn.{sub}"
            if rest.startswith("norms.1."):
                sub = rest[len("norms.1.") :]
                return f"bevformer.encoder.layers.{layer_idx}.norm2.{sub}"
            if rest.startswith("ffns.0.layers.0.0."):
                sub = rest[len("ffns.0.layers.0.0.") :]
                return f"bevformer.encoder.layers.{layer_idx}.ffn.0.{sub}"
            if rest.startswith("ffns.0.layers.1."):
                sub = rest[len("ffns.0.layers.1.") :]
                return f"bevformer.encoder.layers.{layer_idx}.ffn.3.{sub}"
            if rest.startswith("norms.2."):
                sub = rest[len("norms.2.") :]
                return f"bevformer.encoder.layers.{layer_idx}.norm3.{sub}"

    # 5. Decoder Layers
    if key.startswith("pts_bbox_head.transformer.decoder.layers."):
        rem = key[len("pts_bbox_head.transformer.decoder.layers.") :]
        m = re.match(r"^(\d+)\.(.*)", rem)
        if m:
            layer_idx = m.group(1)
            rest = m.group(2)

            if rest.startswith("attentions.0.attn."):
                sub = rest[len("attentions.0.attn.") :]
                return f"decoder.layers.{layer_idx}.self_attn.{sub}"
            if rest.startswith("norms.0."):
                sub = rest[len("norms.0.") :]
                return f"decoder.layers.{layer_idx}.norm1.{sub}"
            if rest.startswith("attentions.1."):
                sub = rest[len("attentions.1.") :]
                return f"decoder.layers.{layer_idx}.cross_attn.{sub}"
            if rest.startswith("norms.1."):
                sub = rest[len("norms.1.") :]
                return f"decoder.layers.{layer_idx}.norm2.{sub}"
            if rest.startswith("ffns.0.layers.0.0."):
                sub = rest[len("ffns.0.layers.0.0.") :]
                return f"decoder.layers.{layer_idx}.ffn.0.{sub}"
            if rest.startswith("ffns.0.layers.1."):
                sub = rest[len("ffns.0.layers.1.") :]
                return f"decoder.layers.{layer_idx}.ffn.3.{sub}"
            if rest.startswith("norms.2."):
                sub = rest[len("norms.2.") :]
                return f"decoder.layers.{layer_idx}.norm3.{sub}"

    # 6. Prediction Heads: cls_branches, reg_branches
    if key.startswith("pts_bbox_head.cls_branches."):
        rem = key[len("pts_bbox_head.cls_branches.") :]
        m = re.match(r"^(\d+)\.(\d+)\.(.*)", rem)
        if m:
            return f"cls_branches.{m.group(1)}.layers.{m.group(2)}.{m.group(3)}"

    if key.startswith("pts_bbox_head.reg_branches."):
        rem = key[len("pts_bbox_head.reg_branches.") :]
        m = re.match(r"^(\d+)\.(\d+)\.(.*)", rem)
        if m:
            return f"reg_branches.{m.group(1)}.layers.{m.group(2)}.{m.group(3)}"

    return None


def convert_bevformer_checkpoint(checkpoint_path: str, config: BevformerConfig) -> dict:
    """Load an upstream checkpoint and convert state_dict to Hugging Face naming."""
    if not os.path.exists(checkpoint_path):
        if checkpoint_path.startswith("http://") or checkpoint_path.startswith("https://"):
            local_target = os.path.join("/tmp/opencode", os.path.basename(checkpoint_path))
            logger.info(f"Downloading checkpoint from {checkpoint_path} to {local_target}...")
            urllib.request.urlretrieve(checkpoint_path, local_target)
            checkpoint_path = local_target
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    orig_state_dict = checkpoint.get("state_dict", checkpoint)

    converted_state_dict = {}
    for orig_k, orig_v in orig_state_dict.items():
        hf_k = map_upstream_key_to_hf(orig_k)
        if hf_k is None:
            continue

        # Slicing for single level in tiny config
        if orig_k == "pts_bbox_head.transformer.level_embeds":
            orig_v = orig_v[: config.num_feature_levels]

        converted_state_dict[hf_k] = orig_v

    logger.info(f"Successfully converted {len(converted_state_dict)} tensors.")
    return converted_state_dict


def convert_bevformer(
    checkpoint_path: str,
    pytorch_dump_out: str,
    push_to_hub: bool = False,
    repo_id: str | None = None,
):
    """Convert official BEVFormer checkpoint and write model, preprocessor, and config to disk."""
    config = BevformerConfig()
    processor = BevformerImageProcessor()

    converted_state_dict = convert_bevformer_checkpoint(checkpoint_path, config)

    logger.info("Instantiating BevformerFor3DObjectDetection model...")
    model = BevformerFor3DObjectDetection(config)
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=True)
    logger.info(f"State dict loaded with strict=True: missing={missing}, unexpected={unexpected}")

    os.makedirs(pytorch_dump_out, exist_ok=True)
    logger.info(f"Saving converted model and configs to {pytorch_dump_out}...")
    model.save_pretrained(pytorch_dump_out, safe_serialization=True)
    processor.save_pretrained(pytorch_dump_out)

    if push_to_hub:
        if repo_id is None:
            raise ValueError("--repo_id must be provided when --push_to_hub is True.")
        logger.info(f"Pushing model and image processor to Hugging Face Hub repository: {repo_id}...")
        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)


def main():
    parser = argparse.ArgumentParser(description="Convert official BEVFormer checkpoint to Hugging Face format.")
    parser.add_argument(
        "--checkpoint_path",
        default=DEFAULT_CHECKPOINT_URL,
        type=str,
        help="Path or URL to the official BEVFormer .pth checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_out",
        required=True,
        type=str,
        help="Path to the output directory where HF model and processor will be written.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hub repository ID (e.g. username/bevformer-tiny). Required if --push_to_hub is set.",
    )
    args = parser.parse_args()

    convert_bevformer(
        checkpoint_path=args.checkpoint_path,
        pytorch_dump_out=args.pytorch_dump_out,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
