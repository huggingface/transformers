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
"""Convert EfficientSAM3 LiteText checkpoints to Hugging Face format."""

import argparse
import gc
import os

import regex as re
import torch
from huggingface_hub import hf_hub_download

from transformers import CLIPTokenizerFast, Sam3ImageProcessor, Sam3Processor
from transformers.models.sam3_lite_text.configuration_sam3_lite_text import Sam3LiteTextConfig, Sam3LiteTextTextConfig
from transformers.models.sam3_lite_text.modeling_sam3_lite_text import Sam3LiteTextModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Strip the "detector." prefix that wraps all components
    r"^detector\.":                                                          r"",

    # ============================================================================
    # Vision Encoder - ViT Backbone
    # ============================================================================
    r"^backbone\.vision_backbone\.trunk\.":                                  r"vision_encoder.backbone.",
    r"^vision_encoder\.backbone\.pos_embed":                                 r"vision_encoder.backbone.embeddings.position_embeddings",
    r"^vision_encoder\.backbone\.patch_embed\.proj\.":                       r"vision_encoder.backbone.embeddings.patch_embeddings.projection.",
    r"^vision_encoder\.backbone\.ln_pre\.":                                  r"vision_encoder.backbone.layer_norm.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.norm1\.":                    r"vision_encoder.backbone.layers.\1.layer_norm1.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.norm2\.":                    r"vision_encoder.backbone.layers.\1.layer_norm2.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.qkv\.":               r"vision_encoder.backbone.layers.\1.attention.qkv.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.proj\.":              r"vision_encoder.backbone.layers.\1.attention.o_proj.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.freqs_cis":           r"vision_encoder.backbone.layers.\1.rotary_emb.rope_embeddings",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc1\.":                r"vision_encoder.backbone.layers.\1.mlp.fc1.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc2\.":                r"vision_encoder.backbone.layers.\1.mlp.fc2.",

    # Vision Encoder - FPN Neck
    r"^backbone\.vision_backbone\.neck\.fpn\.(\d+)\.":                      r"vision_encoder.neck.fpn_layers.\1.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_0\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_1\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.2.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2\.":               r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.maxpool_2x2\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.conv_1x1\.":                r"vision_encoder.neck.fpn_layers.\1.proj1.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.conv_3x3\.":                r"vision_encoder.neck.fpn_layers.\1.proj2.",

    # ============================================================================
    # Text Encoder - LiteText (MobileCLIP student)
    # ============================================================================
    # Embeddings
    r"^backbone\.language_backbone\.encoder\.embedding_layer\.":            r"text_encoder.embeddings.token_embedding.",
    r"^backbone\.language_backbone\.encoder\.positional_embedding\.pos_embed\.pos_embed$": r"text_encoder.embeddings.position_embedding.position_embedding",
    r"^backbone\.language_backbone\.encoder\.final_layer_norm\.":           r"text_encoder.final_layer_norm.",
    r"^backbone\.language_backbone\.encoder\.projection_layer$":            r"text_encoder.projection.weight",
    # text_projection: projects from text hidden-dim to DETR hidden-dim
    r"^backbone\.language_backbone\.projector\.":                           r"text_projection.",
    # RepMixer blocks (layer 0 and the last layer in the mct variant)
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.layer_scale$": r"text_encoder.layers.\1.layer_scale",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.token_mixer\.layer_scale$": r"text_encoder.layers.\1.token_mixer.layer_scale",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.token_mixer\.norm\.rbr_skip\.": r"text_encoder.layers.\1.token_mixer.reference_batchnorm.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.token_mixer\.mixer\.rbr_skip\.": r"text_encoder.layers.\1.token_mixer.mixer.batchnorm_skip.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.token_mixer\.mixer\.rbr_conv\.0\.conv\.": r"text_encoder.layers.\1.token_mixer.mixer.conv.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.token_mixer\.mixer\.rbr_conv\.0\.bn\.": r"text_encoder.layers.\1.token_mixer.mixer.batchnorm_conv.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.convffn\.conv\.conv\.": r"text_encoder.layers.\1.conv_feed_forward.depthwise_conv.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.convffn\.conv\.bn\.": r"text_encoder.layers.\1.conv_feed_forward.depthwise_batchnorm.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.convffn\.fc1\.": r"text_encoder.layers.\1.conv_feed_forward.mlp.fc1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.convffn\.fc2\.": r"text_encoder.layers.\1.conv_feed_forward.mlp.fc2.",
    # Standard transformer layers (pre-norm MHA + FFN)
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.pre_norm_mha\.0\.": r"text_encoder.layers.\1.layer_norm1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.pre_norm_mha\.1\.qkv_proj\.": r"text_encoder.layers.\1.self_attn.in_proj_",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.pre_norm_mha\.1\.out_proj\.": r"text_encoder.layers.\1.self_attn.out_proj.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.pre_norm_ffn\.0\.": r"text_encoder.layers.\1.layer_norm2.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.pre_norm_ffn\.1\.": r"text_encoder.layers.\1.mlp.fc1.",
    r"^backbone\.language_backbone\.encoder\.transformer\.(\d+)\.pre_norm_ffn\.4\.": r"text_encoder.layers.\1.mlp.fc2.",

    # ============================================================================
    # Geometry Encoder
    # ============================================================================
    r"^geometry_encoder\.points_direct_project\.":                         r"geometry_encoder.boxes_direct_project.",
    r"^geometry_encoder\.points_pool_project\.":                           r"geometry_encoder.boxes_pool_project.",
    r"^geometry_encoder\.points_pos_enc_project\.":                        r"geometry_encoder.boxes_pos_enc_project.",
    r"^geometry_encoder\.encode\.(\d+)\.cross_attn_image\.out_proj\.":     r"geometry_encoder.layers.\1.cross_attn.o_proj.",
    r"^geometry_encoder\.encode\.(\d+)\.cross_attn_image\.":               r"geometry_encoder.layers.\1.cross_attn.",
    r"^geometry_encoder\.encode\.(\d+)\.self_attn\.out_proj\.":            r"geometry_encoder.layers.\1.self_attn.o_proj.",
    r"^geometry_encoder\.encode\.(\d+)\.self_attn\.":                      r"geometry_encoder.layers.\1.self_attn.",
    r"^geometry_encoder\.encode\.(\d+)\.linear1\.":                        r"geometry_encoder.layers.\1.mlp.fc1.",
    r"^geometry_encoder\.encode\.(\d+)\.linear2\.":                        r"geometry_encoder.layers.\1.mlp.fc2.",
    r"^geometry_encoder\.encode\.(\d+)\.norm1\.":                          r"geometry_encoder.layers.\1.layer_norm1.",
    r"^geometry_encoder\.encode\.(\d+)\.norm2\.":                          r"geometry_encoder.layers.\1.layer_norm2.",
    r"^geometry_encoder\.encode\.(\d+)\.norm3\.":                          r"geometry_encoder.layers.\1.layer_norm3.",
    r"^geometry_encoder\.img_pre_norm\.":                                   r"geometry_encoder.vision_layer_norm.",
    r"^geometry_encoder\.norm\.":                                           r"geometry_encoder.prompt_layer_norm.",
    r"^geometry_encoder\.encode_norm\.":                                    r"geometry_encoder.output_layer_norm.",

    # ============================================================================
    # DETR Encoder
    # ============================================================================
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.out_proj\.":  r"detr_encoder.layers.\1.cross_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.":            r"detr_encoder.layers.\1.cross_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.":         r"detr_encoder.layers.\1.self_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.self_attn\.":                   r"detr_encoder.layers.\1.self_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn\.out_proj\.":        r"detr_encoder.layers.\1.cross_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn\.":                  r"detr_encoder.layers.\1.cross_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.linear1\.":                     r"detr_encoder.layers.\1.mlp.fc1.",
    r"^transformer\.encoder\.layers\.(\d+)\.linear2\.":                     r"detr_encoder.layers.\1.mlp.fc2.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm1\.":                       r"detr_encoder.layers.\1.layer_norm1.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm2\.":                       r"detr_encoder.layers.\1.layer_norm2.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm3\.":                       r"detr_encoder.layers.\1.layer_norm3.",

    # ============================================================================
    # DETR Decoder
    # ============================================================================
    r"^transformer\.decoder\.query_embed\.":                                r"detr_decoder.query_embed.",
    r"^transformer\.decoder\.reference_points\.":                           r"detr_decoder.reference_points.",
    r"^transformer\.decoder\.instance_query_embed\.":                       r"detr_decoder.instance_query_embed.",
    r"^transformer\.decoder\.instance_reference_points\.":                  r"detr_decoder.instance_reference_points.",
    r"^transformer\.decoder\.presence_token\.":                             r"detr_decoder.presence_token.",
    r"^transformer\.decoder\.presence_token_head\.layers\.0\.":             r"detr_decoder.presence_head.layer1.",
    r"^transformer\.decoder\.presence_token_head\.layers\.1\.":             r"detr_decoder.presence_head.layer2.",
    r"^transformer\.decoder\.presence_token_head\.layers\.2\.":             r"detr_decoder.presence_head.layer3.",
    r"^transformer\.decoder\.presence_token_out_norm\.":                    r"detr_decoder.presence_layer_norm.",
    r"^transformer\.decoder\.norm\.":                                       r"detr_decoder.output_layer_norm.",
    r"^transformer\.decoder\.bbox_embed\.layers\.0\.":                      r"detr_decoder.box_head.layer1.",
    r"^transformer\.decoder\.bbox_embed\.layers\.1\.":                      r"detr_decoder.box_head.layer2.",
    r"^transformer\.decoder\.bbox_embed\.layers\.2\.":                      r"detr_decoder.box_head.layer3.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.0\.":             r"detr_decoder.instance_box_head.layer1.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.1\.":             r"detr_decoder.instance_box_head.layer2.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.2\.":             r"detr_decoder.instance_box_head.layer3.",
    r"^transformer\.decoder\.ref_point_head\.layers\.0\.":                  r"detr_decoder.ref_point_head.layer1.",
    r"^transformer\.decoder\.ref_point_head\.layers\.1\.":                  r"detr_decoder.ref_point_head.layer2.",
    r"^transformer\.decoder\.boxRPB_embed_x\.layers\.0\.":                  r"detr_decoder.box_rpb_embed_x.layer1.",
    r"^transformer\.decoder\.boxRPB_embed_x\.layers\.1\.":                  r"detr_decoder.box_rpb_embed_x.layer2.",
    r"^transformer\.decoder\.boxRPB_embed_y\.layers\.0\.":                  r"detr_decoder.box_rpb_embed_y.layer1.",
    r"^transformer\.decoder\.boxRPB_embed_y\.layers\.1\.":                  r"detr_decoder.box_rpb_embed_y.layer2.",
    r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.":         r"detr_decoder.layers.\1.self_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.":                   r"detr_decoder.layers.\1.self_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.ca_text\.out_proj\.":           r"detr_decoder.layers.\1.text_cross_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.ca_text\.":                     r"detr_decoder.layers.\1.text_cross_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.out_proj\.":        r"detr_decoder.layers.\1.vision_cross_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.":                  r"detr_decoder.layers.\1.vision_cross_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.linear1\.":                     r"detr_decoder.layers.\1.mlp.fc1.",
    r"^transformer\.decoder\.layers\.(\d+)\.linear2\.":                     r"detr_decoder.layers.\1.mlp.fc2.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm1\.":                       r"detr_decoder.layers.\1.vision_cross_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.catext_norm\.":                 r"detr_decoder.layers.\1.text_cross_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm2\.":                       r"detr_decoder.layers.\1.self_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm3\.":                       r"detr_decoder.layers.\1.mlp_layer_norm.",

    # ============================================================================
    # Dot Product Scoring
    # ============================================================================
    r"^dot_prod_scoring\.prompt_mlp\.layers\.0\.":                          r"dot_product_scoring.text_mlp.layer1.",
    r"^dot_prod_scoring\.prompt_mlp\.layers\.1\.":                          r"dot_product_scoring.text_mlp.layer2.",
    r"^dot_prod_scoring\.prompt_mlp\.out_norm\.":                           r"dot_product_scoring.text_mlp_out_norm.",
    r"^dot_prod_scoring\.prompt_proj\.":                                    r"dot_product_scoring.text_proj.",
    r"^dot_prod_scoring\.hs_proj\.":                                        r"dot_product_scoring.query_proj.",

    # ============================================================================
    # Mask Decoder
    # ============================================================================
    r"^segmentation_head\.pixel_decoder\.conv_layers\.(\d+)\.":             r"mask_decoder.pixel_decoder.conv_layers.\1.",
    r"^segmentation_head\.pixel_decoder\.norms\.(\d+)\.":                   r"mask_decoder.pixel_decoder.norms.\1.",
    r"^segmentation_head\.mask_embed\.layers\.(\d+)\.":                     r"mask_decoder.mask_embedder.layers.\1.",
    r"^segmentation_head\.mask_predictor\.mask_embed\.layers\.(\d+)\.":     r"mask_decoder.mask_embedder.layers.\1.",
    r"^segmentation_head\.instance_seg_head\.":                             r"mask_decoder.instance_projection.",
    r"^segmentation_head\.semantic_seg_head\.":                             r"mask_decoder.semantic_projection.",
    r"^segmentation_head\.cross_attend_prompt\.out_proj\.":                 r"mask_decoder.prompt_cross_attn.o_proj.",
    r"^segmentation_head\.cross_attend_prompt\.":                           r"mask_decoder.prompt_cross_attn.",
    r"^segmentation_head\.cross_attn_norm\.":                               r"mask_decoder.prompt_cross_attn_norm.",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    """
    Convert original SAM3 LiteText checkpoint keys to HuggingFace format.

    Applies all regex patterns in `ORIGINAL_TO_CONVERTED_KEY_MAPPING` at once
    using a multiline bulk-substitution.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def split_qkv(state_dict: dict) -> dict:
    """Split combined QKV projections into separate Q, K, V projections."""
    # Vision backbone: .attention.qkv.* → .attention.{q,k,v}_proj.*
    for key in [k for k in state_dict if ".attention.qkv." in k]:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace(".qkv.", ".q_proj.")] = q
        state_dict[key.replace(".qkv.", ".k_proj.")] = k
        state_dict[key.replace(".qkv.", ".v_proj.")] = v

    # Text encoder & attention layers: .in_proj_weight/bias → .{q,k,v}_proj.*
    for key in [k for k in state_dict if ".in_proj_" in k]:
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


def load_original_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load the original EfficientSAM3 LiteText checkpoint."""
    print(f"Loading original checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    print(f"Loaded {len(state_dict)} keys from checkpoint")
    return state_dict


def _infer_text_config(state_dict: dict[str, torch.Tensor]) -> Sam3LiteTextTextConfig:
    """Infer LiteText encoder hyper-parameters from the raw checkpoint."""
    prefix = "detector.backbone.language_backbone.encoder."
    hidden_size = state_dict[f"{prefix}embedding_layer.weight"].shape[1]
    context_length = state_dict[f"{prefix}positional_embedding.pos_embed.pos_embed"].shape[2]
    use_repmixer_blocks = any(f"{prefix}transformer.0.token_mixer" in k for k in state_dict)
    if use_repmixer_blocks:
        num_hidden_layers = 6
    else:
        layer_ids = {
            int(k.split("transformer.")[1].split(".")[0])
            for k in state_dict
            if f"{prefix}transformer." in k and ".pre_norm_mha." in k
        }
        num_hidden_layers = max(layer_ids) + 1

    return Sam3LiteTextTextConfig(
        vocab_size=49408,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=hidden_size // 64,
        max_position_embeddings=context_length,
        projection_dim=hidden_size,
        use_repmixer_blocks=use_repmixer_blocks,
    )


def get_sam3_lite_text_config(state_dict: dict[str, torch.Tensor]) -> Sam3LiteTextConfig:
    """Build a Sam3LiteTextConfig inferred from the raw checkpoint."""
    text_config = _infer_text_config(state_dict)
    config = Sam3LiteTextConfig()
    config.text_config = text_config
    return config


def convert_sam3_lite_text_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config: Sam3LiteTextConfig | None = None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
):
    """
    Convert an EfficientSAM3 LiteText checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to the original `.pt` checkpoint file.
        output_path: Directory where the converted model will be saved.
        config: Optional pre-built `Sam3LiteTextConfig` (defaults to auto-inferred).
        push_to_hub: Whether to push the model to the Hugging Face Hub.
        repo_id: Hub repository ID (required when ``push_to_hub=True``).
    """
    os.makedirs(output_path, exist_ok=True)

    # Load original checkpoint
    state_dict_old = load_original_state_dict(checkpoint_path)

    # Build config from checkpoint
    if config is None:
        config = get_sam3_lite_text_config(state_dict_old)

    config.architectures = ["Sam3LiteTextModel"]
    config.save_pretrained(output_path)
    print("Model config saved successfully")

    # Convert keys
    print("Converting checkpoint keys...")
    all_keys = list(state_dict_old.keys())
    key_mapping = convert_old_keys_to_new_keys(all_keys)

    state_dict_new = {}
    for old_key in all_keys:
        new_key = key_mapping.get(old_key, old_key)
        # num_batches_tracked from BatchNorm is not needed
        if "num_batches_tracked" in new_key:
            continue
        # Parallel SAM2 neck branch in the original checkpoint; HF vision uses `convs` only.
        if "vision_backbone.sam2_convs" in new_key:
            continue
        # Drop keys whose names were not transformed (unrecognised / legacy keys)
        if new_key == old_key:
            continue
        # Strip the first position (cls token) from ViT position embeddings
        if new_key == "vision_encoder.backbone.embeddings.position_embeddings":
            state_dict_new[new_key] = state_dict_old[old_key][:, 1:, :]
        else:
            state_dict_new[new_key] = state_dict_old[old_key]

    del state_dict_old
    gc.collect()

    # Split combined QKV projections
    print("Splitting QKV projections...")
    state_dict_new = split_qkv(state_dict_new)

    # HF models compute the RoPE table on the fly
    for k in list(state_dict_new.keys()):
        if k.endswith("rotary_emb.rope_embeddings"):
            state_dict_new.pop(k)

    print(
        "Converted key counts:",
        {
            prefix: sum(1 for k in state_dict_new if k.startswith(prefix))
            for prefix in (
                "vision_encoder.",
                "text_encoder.",
                "geometry_encoder.",
                "detr_encoder.",
                "detr_decoder.",
                "mask_decoder.",
            )
        },
    )

    # Load weights into HF model
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
    print(f"Saving converted model to {output_path}")
    model.save_pretrained(output_path)

    # Save processor
    print("Creating and saving processor...")
    image_processor = Sam3ImageProcessor()
    tokenizer = CLIPTokenizerFast.from_pretrained(
        "openai/clip-vit-base-patch32",
        max_length=config.text_config.max_position_embeddings,
        model_max_length=config.text_config.max_position_embeddings,
    )
    processor = Sam3Processor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(output_path)

    if push_to_hub:
        if repo_id is None:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        print(f"Pushing model to Hub: {repo_id}")
        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)

    print("Conversion complete!")

    # Cleanup and verify
    del state_dict_new, model
    gc.collect()

    print("\nVerifying converted checkpoint can be loaded...")
    try:
        model = Sam3LiteTextModel.from_pretrained(output_path)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Successfully loaded model with {param_count:,} parameters")
        del model
        gc.collect()
    except Exception as e:
        print(f"Failed to reload model: {e}")

    print("\n" + "=" * 80)
    print("Conversion finished!")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print("\nTo use the model:")
    print(">>> from transformers import Sam3LiteTextModel, Sam3Processor")
    print(f">>> model = Sam3LiteTextModel.from_pretrained('{output_path}')")
    print("=" * 80)


MODEL_VARIANTS = {
    "s0": "sam3_litetext/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt",
    "s1": "sam3_litetext/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt",
    "l": "sam3_litetext/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt",
}


def main():
    parser = argparse.ArgumentParser(description="Convert EfficientSAM3 LiteText checkpoint to HuggingFace format")
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=list(MODEL_VARIANTS),
        default=None,
        help="Model variant to download and convert: 's0' (MobileCLIP-S0, 42M), 's1' (MobileCLIP-S1, 63M), "
        "or 'l' (MobileCLIP2-L, 124M). Takes precedence over --filename when set.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the original .pt checkpoint file. If omitted, the checkpoint is downloaded from the Hub.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where the converted checkpoint will be saved.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Simon7108528/EfficientSAM3",
        help="Hub repository ID to download the checkpoint from (used when --checkpoint_path is not provided).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="sam3_litetext/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt",
        help="Filename within the Hub repository to download (ignored when --model_variant is set).",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hub repository ID to push to (e.g. 'my-org/sam3-litetext-s0').",
    )
    args = parser.parse_args()

    filename = MODEL_VARIANTS[args.model_variant] if args.model_variant else args.filename

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        print(f"Downloading checkpoint {filename} from {args.repo_id}...")
        checkpoint_path = hf_hub_download(args.repo_id, filename)

    convert_sam3_lite_text_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()
