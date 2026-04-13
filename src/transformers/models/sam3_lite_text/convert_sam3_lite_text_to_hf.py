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

"""
Convert SAM3-LiteText (MobileCLIP-S0) checkpoints from the original implementation to HuggingFace format.

Original repository: https://github.com/SimonZeng7108/efficientsam3 (sam3_litetext branch)
"""

import argparse
import gc
import os

import regex as re
import torch

from transformers import CLIPTokenizerFast, Sam3LiteTextConfig, Sam3LiteTextModel
from transformers.models.sam3_lite_text.image_processing_sam3_lite_text import Sam3LiteTextImageProcessor
from transformers.models.sam3_lite_text.processing_sam3_lite_text import Sam3LiteTextProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Strip detector/student_trunk prefixes
    r"^detector\.":                                                              r"",
    r"^student_trunk\.":                                                         r"",
    r"^sam3_model\.":                                                            r"",

    # ============================================================================
    # Text Encoder (MobileCLIP-S0)
    # ============================================================================
    r"^backbone\.language_backbone\.encoder\.":                                  r"text_encoder.",
    r"^text_encoder\.positional_embedding\.pos_embed\.pos_embed":               r"text_encoder.positional_embedding.pos_embed",
    r"^text_encoder\.transformer\.":                                             r"text_encoder.layers.",

    # TransformerEncoder layers: pre_norm_mha → attn_norm + attention
    r"^(text_encoder\.layers\.\d+\.)pre_norm_mha\.0\.":                         r"\1attn_norm.",
    r"^(text_encoder\.layers\.\d+\.)pre_norm_mha\.1\.":                         r"\1attention.",
    r"^(text_encoder\.layers\.\d+\.)pre_norm_mha\.2\.":                         r"\1attn_dropout.",

    # TransformerEncoder layers: pre_norm_ffn → ffn_norm + fc1 + fc2
    r"^(text_encoder\.layers\.\d+\.)pre_norm_ffn\.0\.":                         r"\1ffn_norm.",
    r"^(text_encoder\.layers\.\d+\.)pre_norm_ffn\.1\.":                         r"\1fc1.",
    r"^(text_encoder\.layers\.\d+\.)pre_norm_ffn\.4\.":                         r"\1fc2.",

    # Text projector (MobileCLIP dim → SAM3 d_model=256)
    r"^backbone\.language_backbone\.projector\.":                                r"text_projection.",

    # ============================================================================
    # Vision Encoder - ViT Backbone (identical to SAM3)
    # ============================================================================
    r"^backbone\.vision_backbone\.trunk\.":                                      r"vision_encoder.backbone.",
    r"^vision_encoder\.backbone\.pos_embed":                                     r"vision_encoder.backbone.embeddings.position_embeddings",
    r"^vision_encoder\.backbone\.patch_embed\.proj\.":                           r"vision_encoder.backbone.embeddings.patch_embeddings.projection.",
    r"^vision_encoder\.backbone\.ln_pre\.":                                      r"vision_encoder.backbone.layer_norm.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.norm1\.":                        r"vision_encoder.backbone.layers.\1.layer_norm1.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.norm2\.":                        r"vision_encoder.backbone.layers.\1.layer_norm2.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.qkv\.":                   r"vision_encoder.backbone.layers.\1.attention.qkv.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.proj\.":                   r"vision_encoder.backbone.layers.\1.attention.o_proj.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.freqs_cis":               r"vision_encoder.backbone.layers.\1.rotary_emb.rope_embeddings",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc1\.":                     r"vision_encoder.backbone.layers.\1.mlp.fc1.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc2\.":                     r"vision_encoder.backbone.layers.\1.mlp.fc2.",

    # Vision Encoder - FPN Neck
    r"^backbone\.vision_backbone\.neck\.fpn\.(\d+)\.":                           r"vision_encoder.neck.fpn_layers.\1.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_0\.":                  r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_1\.":                  r"vision_encoder.neck.fpn_layers.\1.scale_layers.2.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2\.":                    r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.maxpool_2x2\.":                  r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.conv_1x1\.":                     r"vision_encoder.neck.fpn_layers.\1.proj1.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.conv_3x3\.":                     r"vision_encoder.neck.fpn_layers.\1.proj2.",

    # ============================================================================
    # Geometry Encoder (identical to SAM3)
    # ============================================================================
    r"^geometry_encoder\.encode\.(\d+)\.cross_attn_image\.out_proj\.":            r"geometry_encoder.layers.\1.cross_attn.o_proj.",
    r"^geometry_encoder\.encode\.(\d+)\.cross_attn_image\.":                      r"geometry_encoder.layers.\1.cross_attn.",
    r"^geometry_encoder\.encode\.(\d+)\.self_attn\.out_proj\.":                   r"geometry_encoder.layers.\1.self_attn.o_proj.",
    r"^geometry_encoder\.encode\.(\d+)\.self_attn\.":                             r"geometry_encoder.layers.\1.self_attn.",
    r"^geometry_encoder\.encode\.(\d+)\.linear1\.":                               r"geometry_encoder.layers.\1.mlp.fc1.",
    r"^geometry_encoder\.encode\.(\d+)\.linear2\.":                               r"geometry_encoder.layers.\1.mlp.fc2.",
    r"^geometry_encoder\.encode\.(\d+)\.norm1\.":                                 r"geometry_encoder.layers.\1.layer_norm1.",
    r"^geometry_encoder\.encode\.(\d+)\.norm2\.":                                 r"geometry_encoder.layers.\1.layer_norm2.",
    r"^geometry_encoder\.encode\.(\d+)\.norm3\.":                                 r"geometry_encoder.layers.\1.layer_norm3.",
    r"^geometry_encoder\.img_pre_norm\.":                                          r"geometry_encoder.vision_layer_norm.",
    r"^geometry_encoder\.norm\.":                                                  r"geometry_encoder.prompt_layer_norm.",
    r"^geometry_encoder\.encode_norm\.":                                           r"geometry_encoder.output_layer_norm.",

    # ============================================================================
    # DETR Encoder (identical to SAM3)
    # ============================================================================
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.out_proj\.":        r"detr_encoder.layers.\1.cross_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.":                  r"detr_encoder.layers.\1.cross_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.":               r"detr_encoder.layers.\1.self_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.self_attn\.":                         r"detr_encoder.layers.\1.self_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn\.out_proj\.":              r"detr_encoder.layers.\1.cross_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn\.":                        r"detr_encoder.layers.\1.cross_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.linear1\.":                           r"detr_encoder.layers.\1.mlp.fc1.",
    r"^transformer\.encoder\.layers\.(\d+)\.linear2\.":                           r"detr_encoder.layers.\1.mlp.fc2.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm1\.":                             r"detr_encoder.layers.\1.layer_norm1.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm2\.":                             r"detr_encoder.layers.\1.layer_norm2.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm3\.":                             r"detr_encoder.layers.\1.layer_norm3.",

    # ============================================================================
    # DETR Decoder (identical to SAM3)
    # ============================================================================
    r"^transformer\.decoder\.query_embed\.":                                       r"detr_decoder.query_embed.",
    r"^transformer\.decoder\.reference_points\.":                                  r"detr_decoder.reference_points.",
    r"^transformer\.decoder\.instance_query_embed\.":                              r"detr_decoder.instance_query_embed.",
    r"^transformer\.decoder\.instance_reference_points\.":                         r"detr_decoder.instance_reference_points.",
    r"^transformer\.decoder\.presence_token\.":                                    r"detr_decoder.presence_token.",
    r"^transformer\.decoder\.presence_token_head\.layers\.0\.":                    r"detr_decoder.presence_head.layer1.",
    r"^transformer\.decoder\.presence_token_head\.layers\.1\.":                    r"detr_decoder.presence_head.layer2.",
    r"^transformer\.decoder\.presence_token_head\.layers\.2\.":                    r"detr_decoder.presence_head.layer3.",
    r"^transformer\.decoder\.presence_token_out_norm\.":                           r"detr_decoder.presence_layer_norm.",
    r"^transformer\.decoder\.norm\.":                                              r"detr_decoder.output_layer_norm.",
    r"^transformer\.decoder\.bbox_embed\.layers\.0\.":                             r"detr_decoder.box_head.layer1.",
    r"^transformer\.decoder\.bbox_embed\.layers\.1\.":                             r"detr_decoder.box_head.layer2.",
    r"^transformer\.decoder\.bbox_embed\.layers\.2\.":                             r"detr_decoder.box_head.layer3.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.0\.":                    r"detr_decoder.instance_box_head.layer1.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.1\.":                    r"detr_decoder.instance_box_head.layer2.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.2\.":                    r"detr_decoder.instance_box_head.layer3.",
    r"^transformer\.decoder\.ref_point_head\.layers\.0\.":                         r"detr_decoder.ref_point_head.layer1.",
    r"^transformer\.decoder\.ref_point_head\.layers\.1\.":                         r"detr_decoder.ref_point_head.layer2.",
    r"^transformer\.decoder\.boxRPB_embed_x\.layers\.0\.":                         r"detr_decoder.box_rpb_embed_x.layer1.",
    r"^transformer\.decoder\.boxRPB_embed_x\.layers\.1\.":                         r"detr_decoder.box_rpb_embed_x.layer2.",
    r"^transformer\.decoder\.boxRPB_embed_y\.layers\.0\.":                         r"detr_decoder.box_rpb_embed_y.layer1.",
    r"^transformer\.decoder\.boxRPB_embed_y\.layers\.1\.":                         r"detr_decoder.box_rpb_embed_y.layer2.",
    r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.":                r"detr_decoder.layers.\1.self_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.":                          r"detr_decoder.layers.\1.self_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.ca_text\.out_proj\.":                  r"detr_decoder.layers.\1.text_cross_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.ca_text\.":                            r"detr_decoder.layers.\1.text_cross_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.out_proj\.":               r"detr_decoder.layers.\1.vision_cross_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.":                         r"detr_decoder.layers.\1.vision_cross_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.linear1\.":                            r"detr_decoder.layers.\1.mlp.fc1.",
    r"^transformer\.decoder\.layers\.(\d+)\.linear2\.":                            r"detr_decoder.layers.\1.mlp.fc2.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm1\.":                              r"detr_decoder.layers.\1.vision_cross_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.catext_norm\.":                        r"detr_decoder.layers.\1.text_cross_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm2\.":                              r"detr_decoder.layers.\1.self_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm3\.":                              r"detr_decoder.layers.\1.mlp_layer_norm.",

    # ============================================================================
    # Dot Product Scoring (identical to SAM3)
    # ============================================================================
    r"^dot_prod_scoring\.prompt_mlp\.layers\.0\.":                                 r"dot_product_scoring.text_mlp.layer1.",
    r"^dot_prod_scoring\.prompt_mlp\.layers\.1\.":                                 r"dot_product_scoring.text_mlp.layer2.",
    r"^dot_prod_scoring\.prompt_mlp\.out_norm\.":                                  r"dot_product_scoring.text_mlp_out_norm.",
    r"^dot_prod_scoring\.prompt_proj\.":                                            r"dot_product_scoring.text_proj.",
    r"^dot_prod_scoring\.hs_proj\.":                                                r"dot_product_scoring.query_proj.",

    # ============================================================================
    # Mask Decoder (identical to SAM3)
    # ============================================================================
    r"^segmentation_head\.pixel_decoder\.conv_layers\.(\d+)\.":                    r"mask_decoder.pixel_decoder.conv_layers.\1.",
    r"^segmentation_head\.pixel_decoder\.norms\.(\d+)\.":                           r"mask_decoder.pixel_decoder.norms.\1.",
    r"^segmentation_head\.mask_embed\.layers\.(\d+)\.":                             r"mask_decoder.mask_embedder.layers.\1.",
    r"^segmentation_head\.mask_predictor\.mask_embed\.layers\.(\d+)\.":             r"mask_decoder.mask_embedder.layers.\1.",
    r"^segmentation_head\.instance_seg_head\.":                                     r"mask_decoder.instance_projection.",
    r"^segmentation_head\.semantic_seg_head\.":                                     r"mask_decoder.semantic_projection.",
    r"^segmentation_head\.cross_attend_prompt\.out_proj\.":                         r"mask_decoder.prompt_cross_attn.o_proj.",
    r"^segmentation_head\.cross_attend_prompt\.":                                   r"mask_decoder.prompt_cross_attn.",
    r"^segmentation_head\.cross_attn_norm\.":                                       r"mask_decoder.prompt_cross_attn_norm.",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def split_qkv(state_dict: dict) -> dict:
    """Split combined QKV weights/biases in the vision backbone into separate Q, K, V projections."""
    keys_to_split = [key for key in state_dict.keys() if ".attention.qkv." in key]
    for key in keys_to_split:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace(".qkv.", ".q_proj.")] = q
        state_dict[key.replace(".qkv.", ".k_proj.")] = k
        state_dict[key.replace(".qkv.", ".v_proj.")] = v

    # Handle DETR decoder cross-attention in_proj_*
    in_proj_keys = [key for key in state_dict.keys() if ".in_proj_" in key]
    for key in in_proj_keys:
        in_proj = state_dict.pop(key)
        q, k, v = torch.chunk(in_proj, 3, dim=0)
        if key.endswith("in_proj_weight"):
            base = key.replace("in_proj_weight", "")
            state_dict[base + "q_proj.weight"] = q
            state_dict[base + "k_proj.weight"] = k
            state_dict[base + "v_proj.weight"] = v
        elif key.endswith("in_proj_bias"):
            base = key.replace("in_proj_bias", "")
            state_dict[base + "q_proj.bias"] = q
            state_dict[base + "k_proj.bias"] = k
            state_dict[base + "v_proj.bias"] = v

    return state_dict


def truncate_positional_embeddings(state_dict: dict, context_length: int = 16) -> dict:
    """Truncate MobileCLIP positional embeddings from pretrained length (77) to target context length."""
    pos_key = "text_encoder.positional_embedding.pos_embed"
    if pos_key in state_dict:
        pos_embed = state_dict[pos_key]
        # Shape: (1, 1, original_length, dim)
        original_length = pos_embed.shape[2]
        if original_length > context_length:
            print(f"Truncating positional embeddings from {original_length} to {context_length}")
            state_dict[pos_key] = pos_embed[:, :, :context_length, :]
    return state_dict


def convert_sam3_lite_text_checkpoint(
    checkpoint_path: str,
    output_path: str,
    push_to_hub: bool = False,
    repo_id: str | None = None,
):
    os.makedirs(output_path, exist_ok=True)

    config = Sam3LiteTextConfig()
    config.architectures = ["Sam3LiteTextModel"]
    config.save_pretrained(output_path)
    print("Config saved")

    # Load original checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in checkpoint:
        state_dict_old = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict_old = checkpoint["state_dict"]
    else:
        state_dict_old = checkpoint
    print(f"Loaded {len(state_dict_old)} keys")

    # Convert keys
    print("Converting keys...")
    all_keys = list(state_dict_old.keys())
    key_mapping = convert_old_keys_to_new_keys(all_keys)

    state_dict_new = {}
    for old_key in all_keys:
        new_key = key_mapping.get(old_key, old_key)

        # Strip cls token from vision backbone position embeddings
        if new_key == "vision_encoder.backbone.embeddings.position_embeddings":
            state_dict_new[new_key] = state_dict_old[old_key][:, 1:, :]
        # Skip the MobileCLIP projection_layer (we use text_projection linear instead)
        elif new_key == "text_encoder.projection_layer":
            print(f"Skipping {old_key} (projection handled by text_projection linear)")
            continue
        else:
            state_dict_new[new_key] = state_dict_old[old_key]

    del state_dict_old
    gc.collect()

    # Split QKV projections in vision backbone and DETR
    print("Splitting QKV projections...")
    state_dict_new = split_qkv(state_dict_new)

    # Truncate positional embeddings to context_length
    state_dict_new = truncate_positional_embeddings(state_dict_new, config.text_config.context_length)

    # Load into model
    print("Loading weights into Sam3LiteTextModel...")
    model = Sam3LiteTextModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}):")
        for key in missing_keys:
            logger.warning(f"  - {key}")

    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys:
            logger.warning(f"  - {key}")

    # Save model
    print(f"Saving to {output_path}")
    model.save_pretrained(output_path)

    # Save processor
    print("Saving processor...")
    image_processor = Sam3LiteTextImageProcessor()
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", max_length=16, model_max_length=16)
    processor = Sam3LiteTextProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(output_path)

    if push_to_hub:
        if repo_id is None:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        print(f"Pushing to Hub: {repo_id}")
        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)

    del state_dict_new, model
    gc.collect()

    # Verify
    print("Verifying...")
    try:
        model = Sam3LiteTextModel.from_pretrained(output_path)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Successfully loaded model with {param_count:,} parameters")
        del model
        gc.collect()
    except Exception as e:
        print(f"Failed to reload: {e}")

    print(f"\nConversion complete! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert SAM3-LiteText checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to original .pt checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save converted model")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", type=str, default=None)

    args = parser.parse_args()
    convert_sam3_lite_text_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
