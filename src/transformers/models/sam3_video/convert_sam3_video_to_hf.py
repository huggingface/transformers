# Copyright 2025 The Meta AI Authors and The HuggingFace Inc. team. All rights reserved.
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
Convert SAM3 checkpoints from the original implementation to HuggingFace format.

"""

import argparse
import gc
import os
from typing import Optional

import regex as re
import torch

from transformers import CLIPTokenizerFast
from transformers.models.sam2_video.video_processing_sam2_video import Sam2VideoVideoProcessor
from transformers.models.sam3.image_processing_sam3_fast import Sam3ImageProcessorFast
from transformers.models.sam3.modeling_sam3 import Sam3Model
from transformers.models.sam3_tracker.modeling_sam3_tracker import Sam3TrackerModel
from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import Sam3TrackerVideoModel
from transformers.models.sam3_video.configuration_sam3_video import Sam3VideoConfig
from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoModel
from transformers.models.sam3_video.processing_sam3_video import Sam3VideoProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^sam3_model\.": r"detector_model.",
    r"^sam2_predictor\.model\.": r"tracker_model.",

    # ============================================================================
    # Vision Encoder - ViT Backbone
    # ============================================================================
    r"backbone\.vision_backbone\.trunk\.":                                 r"vision_encoder.backbone.",
    r"vision_encoder\.backbone\.pos_embed":                                r"vision_encoder.backbone.embeddings.position_embeddings",
    r"vision_encoder\.backbone\.patch_embed\.proj\.":                      r"vision_encoder.backbone.embeddings.patch_embeddings.projection.",
    r"vision_encoder\.backbone\.ln_pre\.":                                 r"vision_encoder.backbone.layer_norm.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.norm1\.":                   r"vision_encoder.backbone.layers.\1.layer_norm1.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.norm2\.":                   r"vision_encoder.backbone.layers.\1.layer_norm2.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.qkv\.":               r"vision_encoder.backbone.layers.\1.attention.qkv.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.proj\.":              r"vision_encoder.backbone.layers.\1.attention.o_proj.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.attn\.freqs_cis":           r"vision_encoder.backbone.layers.\1.rotary_emb.rope_embeddings",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc1\.":                r"vision_encoder.backbone.layers.\1.mlp.fc1.",
    r"vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc2\.":                r"vision_encoder.backbone.layers.\1.mlp.fc2.",

    # ision Encoder - FPN Neck
    r"backbone\.vision_backbone\.neck\.fpn\.(\d+)\.":                      r"vision_encoder.neck.fpn_layers.\1.",
    r"backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_0\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_1\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.2.",
    r"backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2\.":               r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"backbone\.vision_backbone\.convs\.(\d+)\.maxpool_2x2\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"backbone\.vision_backbone\.convs\.(\d+)\.conv_1x1\.":                r"vision_encoder.neck.fpn_layers.\1.proj1.",
    r"backbone\.vision_backbone\.convs\.(\d+)\.conv_3x3\.":                r"vision_encoder.neck.fpn_layers.\1.proj2.",

    # ision Encoder - Tracker FPN Neck
    r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.dconv_2x2_0\.":             r"vision_encoder.tracker_neck.fpn_layers.\1.scale_layers.0.",
    r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.dconv_2x2_1\.":             r"vision_encoder.tracker_neck.fpn_layers.\1.scale_layers.2.",
    r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.dconv_2x2\.":               r"vision_encoder.tracker_neck.fpn_layers.\1.scale_layers.0.",
    r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.maxpool_2x2\.":             r"vision_encoder.tracker_neck.fpn_layers.\1.scale_layers.0.",
    r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.conv_1x1\.":                r"vision_encoder.tracker_neck.fpn_layers.\1.proj1.",
    r"backbone\.vision_backbone\.sam2_convs\.(\d+)\.conv_3x3\.":                r"vision_encoder.tracker_neck.fpn_layers.\1.proj2.",

    # ===========================================================================
    # ext Encoder (CLIP)
    # ===========================================================================
    r"backbone\.language_backbone\.encoder\.":                             r"text_encoder.",
    r"text_encoder\.token_embedding\.":                                    r"text_encoder.text_model.embeddings.token_embedding.",
    r"text_encoder\.positional_embedding":                                 r"text_encoder.text_model.embeddings.position_embedding.weight",
    r"text_encoder\.ln_final\.":                                           r"text_encoder.text_model.final_layer_norm.",
    r"text_encoder\.text_projection":                                      r"text_encoder.text_projection.weight",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.attn\.in_proj_":       r"text_encoder.text_model.encoder.layers.\1.self_attn.in_proj_",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.attn\.out_proj\.":     r"text_encoder.text_model.encoder.layers.\1.self_attn.out_proj.",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.ln_1\.":               r"text_encoder.text_model.encoder.layers.\1.layer_norm1.",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.ln_2\.":               r"text_encoder.text_model.encoder.layers.\1.layer_norm2.",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_fc\.":          r"text_encoder.text_model.encoder.layers.\1.mlp.fc1.",
    r"text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_proj\.":        r"text_encoder.text_model.encoder.layers.\1.mlp.fc2.",
    r"backbone\.language_backbone\.resizer\.":                             r"text_projection.",

    # ===========================================================================
    # eometry Encoder
    # ===========================================================================
    r"geometry_encoder\.encode\.(\d+)\.cross_attn_image\.out_proj\.":     r"geometry_encoder.layers.\1.cross_attn.o_proj.",
    r"geometry_encoder\.encode\.(\d+)\.cross_attn_image\.":               r"geometry_encoder.layers.\1.cross_attn.",
    r"geometry_encoder\.encode\.(\d+)\.self_attn\.out_proj\.":            r"geometry_encoder.layers.\1.self_attn.o_proj.",
    r"geometry_encoder\.encode\.(\d+)\.self_attn\.":                      r"geometry_encoder.layers.\1.self_attn.",
    r"geometry_encoder\.encode\.(\d+)\.linear1\.":                        r"geometry_encoder.layers.\1.mlp.fc1.",
    r"geometry_encoder\.encode\.(\d+)\.linear2\.":                        r"geometry_encoder.layers.\1.mlp.fc2.",
    r"geometry_encoder\.encode\.(\d+)\.norm1\.":                          r"geometry_encoder.layers.\1.layer_norm1.",
    r"geometry_encoder\.encode\.(\d+)\.norm2\.":                          r"geometry_encoder.layers.\1.layer_norm2.",
    r"geometry_encoder\.encode\.(\d+)\.norm3\.":                          r"geometry_encoder.layers.\1.layer_norm3.",
    r"geometry_encoder\.img_pre_norm\.":                                   r"geometry_encoder.vision_layer_norm.",
    r"geometry_encoder\.norm\.":                                           r"geometry_encoder.prompt_layer_norm.",
    r"geometry_encoder\.encode_norm\.":                                    r"geometry_encoder.output_layer_norm.",

    # ===========================================================================
    # ETR Encoder
    # ===========================================================================
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.out_proj\.":  r"detector_model.detr_encoder.layers.\1.cross_attn.o_proj.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.":            r"detector_model.detr_encoder.layers.\1.cross_attn.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.":         r"detector_model.detr_encoder.layers.\1.self_attn.o_proj.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.self_attn\.":                   r"detector_model.detr_encoder.layers.\1.self_attn.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.cross_attn\.out_proj\.":        r"detector_model.detr_encoder.layers.\1.cross_attn.o_proj.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.cross_attn\.":                  r"detector_model.detr_encoder.layers.\1.cross_attn.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.linear1\.":                     r"detector_model.detr_encoder.layers.\1.mlp.fc1.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.linear2\.":                     r"detector_model.detr_encoder.layers.\1.mlp.fc2.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.norm1\.":                       r"detector_model.detr_encoder.layers.\1.layer_norm1.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.norm2\.":                       r"detector_model.detr_encoder.layers.\1.layer_norm2.",
    r"detector_model.transformer\.encoder\.layers\.(\d+)\.norm3\.":                       r"detector_model.detr_encoder.layers.\1.layer_norm3.",

    # ===========================================================================
    # ETR Decoder
    # ===========================================================================
    r"transformer\.decoder\.query_embed\.":                                r"detr_decoder.query_embed.",
    r"transformer\.decoder\.reference_points\.":                           r"detr_decoder.reference_points.",
    r"transformer\.decoder\.instance_query_embed\.":                       r"detr_decoder.instance_query_embed.",
    r"transformer\.decoder\.instance_reference_points\.":                  r"detr_decoder.instance_reference_points.",
    r"transformer\.decoder\.presence_token\.":                             r"detr_decoder.presence_token.",
    r"transformer\.decoder\.presence_token_head\.layers\.0\.":             r"detr_decoder.presence_head.layer1.",
    r"transformer\.decoder\.presence_token_head\.layers\.1\.":             r"detr_decoder.presence_head.layer2.",
    r"transformer\.decoder\.presence_token_head\.layers\.2\.":             r"detr_decoder.presence_head.layer3.",
    r"transformer\.decoder\.presence_token_out_norm\.":                    r"detr_decoder.presence_layer_norm.",
    r"transformer\.decoder\.norm\.":                                       r"detr_decoder.output_layer_norm.",
    r"transformer\.decoder\.bbox_embed\.layers\.0\.":                      r"detr_decoder.box_head.layer1.",
    r"transformer\.decoder\.bbox_embed\.layers\.1\.":                      r"detr_decoder.box_head.layer2.",
    r"transformer\.decoder\.bbox_embed\.layers\.2\.":                      r"detr_decoder.box_head.layer3.",
    r"transformer\.decoder\.instance_bbox_embed\.layers\.0\.":             r"detr_decoder.instance_box_head.layer1.",
    r"transformer\.decoder\.instance_bbox_embed\.layers\.1\.":             r"detr_decoder.instance_box_head.layer2.",
    r"transformer\.decoder\.instance_bbox_embed\.layers\.2\.":             r"detr_decoder.instance_box_head.layer3.",
    r"transformer\.decoder\.ref_point_head\.layers\.0\.":                  r"detr_decoder.ref_point_head.layer1.",
    r"transformer\.decoder\.ref_point_head\.layers\.1\.":                  r"detr_decoder.ref_point_head.layer2.",
    r"transformer\.decoder\.boxRPB_embed_x\.layers\.0\.":                  r"detr_decoder.box_rpb_embed_x.layer1.",
    r"transformer\.decoder\.boxRPB_embed_x\.layers\.1\.":                  r"detr_decoder.box_rpb_embed_x.layer2.",
    r"transformer\.decoder\.boxRPB_embed_y\.layers\.0\.":                  r"detr_decoder.box_rpb_embed_y.layer1.",
    r"transformer\.decoder\.boxRPB_embed_y\.layers\.1\.":                  r"detr_decoder.box_rpb_embed_y.layer2.",
    r"transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.":         r"detr_decoder.layers.\1.self_attn.o_proj.",
    r"transformer\.decoder\.layers\.(\d+)\.self_attn\.":                   r"detr_decoder.layers.\1.self_attn.",
    r"transformer\.decoder\.layers\.(\d+)\.ca_text\.out_proj\.":           r"detr_decoder.layers.\1.text_cross_attn.o_proj.",
    r"transformer\.decoder\.layers\.(\d+)\.ca_text\.":                     r"detr_decoder.layers.\1.text_cross_attn.",
    r"transformer\.decoder\.layers\.(\d+)\.cross_attn\.out_proj\.":        r"detr_decoder.layers.\1.vision_cross_attn.o_proj.",
    r"transformer\.decoder\.layers\.(\d+)\.cross_attn\.":                  r"detr_decoder.layers.\1.vision_cross_attn.",
    r"transformer\.decoder\.layers\.(\d+)\.linear1\.":                     r"detr_decoder.layers.\1.mlp.fc1.",
    r"transformer\.decoder\.layers\.(\d+)\.linear2\.":                     r"detr_decoder.layers.\1.mlp.fc2.",
    r"transformer\.decoder\.layers\.(\d+)\.norm1\.":                       r"detr_decoder.layers.\1.vision_cross_attn_layer_norm.",
    r"transformer\.decoder\.layers\.(\d+)\.catext_norm\.":                 r"detr_decoder.layers.\1.text_cross_attn_layer_norm.",
    r"transformer\.decoder\.layers\.(\d+)\.norm2\.":                       r"detr_decoder.layers.\1.self_attn_layer_norm.",
    r"transformer\.decoder\.layers\.(\d+)\.norm3\.":                       r"detr_decoder.layers.\1.mlp_layer_norm.",

    # ===========================================================================
    # ot Product Scoring
    # ===========================================================================
    r"dot_prod_scoring\.prompt_mlp\.layers\.0\.":                          r"dot_product_scoring.text_mlp.layer1.",
    r"dot_prod_scoring\.prompt_mlp\.layers\.1\.":                          r"dot_product_scoring.text_mlp.layer2.",
    r"dot_prod_scoring\.prompt_mlp\.out_norm\.":                           r"dot_product_scoring.text_mlp_out_norm.",
    r"dot_prod_scoring\.prompt_proj\.":                                    r"dot_product_scoring.text_proj.",
    r"dot_prod_scoring\.hs_proj\.":                                        r"dot_product_scoring.query_proj.",

    # ===========================================================================
    # ask Decoder
    # ===========================================================================
    r"segmentation_head\.pixel_decoder\.conv_layers\.(\d+)\.":             r"mask_decoder.pixel_decoder.conv_layers.\1.",
    r"segmentation_head\.pixel_decoder\.norms\.(\d+)\.":                   r"mask_decoder.pixel_decoder.norms.\1.",
    r"segmentation_head\.mask_embed\.layers\.(\d+)\.":                     r"mask_decoder.mask_embedder.layers.\1.",
    r"segmentation_head\.mask_predictor\.mask_embed\.layers\.(\d+)\.":     r"mask_decoder.mask_embedder.layers.\1.",
    r"segmentation_head\.instance_seg_head\.":                             r"mask_decoder.instance_projection.",
    r"segmentation_head\.semantic_seg_head\.":                             r"mask_decoder.semantic_projection.",
    r"segmentation_head\.cross_attend_prompt\.out_proj\.":                 r"mask_decoder.prompt_cross_attn.o_proj.",
    r"segmentation_head\.cross_attend_prompt\.":                           r"mask_decoder.prompt_cross_attn.",
    r"segmentation_head\.cross_attn_norm\.":                               r"mask_decoder.prompt_cross_attn_norm.",

    r"^detector_model\.vision_encoder\.tracker_neck\.": r"tracker_neck.",

}
# fmt: on

KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "dwconv": "depthwise_conv",
    "pwconv": "pointwise_conv",
    "fuser": "memory_fuser",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "obj_ptr_tpos_proj": "temporal_positional_encoding_projection_layer",
    "no_obj_embed_spatial": "occlusion_spatial_embedding_parameter",
    "sam_prompt_encoder": "prompt_encoder",
    "sam_mask_decoder": "mask_decoder",
    "maskmem_tpos_enc": "memory_temporal_positional_encoding",
    "gamma": "scale",
    "image_encoder.neck": "vision_encoder.neck",
    "image_encoder": "vision_encoder.backbone",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "pix_feat_proj": "feature_projection",
    "patch_embed.proj": "patch_embed.projection",
    "no_mem_embed": "no_memory_embedding",
    "no_mem_pos_enc": "no_memory_positional_encoding",
    "obj_ptr": "object_pointer",
    ".norm": ".layer_norm",
    "trunk.": "",
    "out_proj": "o_proj",
}


def adapt_internal_ckpt(ov_sd):
    # Replace values instead of keys, and remove any isinstance checks
    sam2_sd = {k: v.replace("backbone.vision_backbone.trunk", "image_encoder.trunk") for k, v in ov_sd.items()}
    sam2_sd = {k: v.replace("backbone.vision_backbone.convs", "image_encoder.neck.convs") for k, v in sam2_sd.items()}
    # rename components to be consitent with paper and public release
    sam2_sd = {k: v.replace("transformer.encoder", "memory_attention") for k, v in sam2_sd.items()}
    sam2_sd = {k: v.replace("maskmem_backbone", "memory_encoder") for k, v in sam2_sd.items()}
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.0.",
            "memory_encoder.mask_downsampler.layers.0.conv.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.1.",
            "memory_encoder.mask_downsampler.layers.0.layer_norm.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.3.",
            "memory_encoder.mask_downsampler.layers.1.conv.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.4.",
            "memory_encoder.mask_downsampler.layers.1.layer_norm.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.6.",
            "memory_encoder.mask_downsampler.layers.2.conv.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.7.",
            "memory_encoder.mask_downsampler.layers.2.layer_norm.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.9.",
            "memory_encoder.mask_downsampler.layers.3.conv.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.10.",
            "memory_encoder.mask_downsampler.layers.3.layer_norm.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.mask_downsampler.encoder.12.",
            "memory_encoder.mask_downsampler.final_conv.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "memory_encoder.o_proj.",
            "memory_encoder.projection.",
        )
        for k, v in sam2_sd.items()
    }
    # MLPBLock to MLP
    sam2_sd = {
        k: v.replace("mask_decoder.transformer.layers.0.mlp.lin1", "mask_decoder.transformer.layers.0.mlp.layers.0")
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace("mask_decoder.transformer.layers.0.mlp.lin2", "mask_decoder.transformer.layers.0.mlp.layers.1")
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace("mask_decoder.transformer.layers.1.mlp.lin1", "mask_decoder.transformer.layers.1.mlp.layers.0")
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace("mask_decoder.transformer.layers.1.mlp.lin2", "mask_decoder.transformer.layers.1.mlp.layers.1")
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "mask_decoder.transformer.layers.0.mlp.layers.0.",
            "mask_decoder.transformer.layers.0.mlp.proj_in.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "mask_decoder.transformer.layers.0.mlp.layers.1.",
            "mask_decoder.transformer.layers.0.mlp.proj_out.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "mask_decoder.transformer.layers.1.mlp.layers.0.",
            "mask_decoder.transformer.layers.1.mlp.proj_in.",
        )
        for k, v in sam2_sd.items()
    }
    sam2_sd = {
        k: v.replace(
            "mask_decoder.transformer.layers.1.mlp.layers.1.",
            "mask_decoder.transformer.layers.1.mlp.proj_out.",
        )
        for k, v in sam2_sd.items()
    }
    # FFN to MLP
    # sam2_sd = {k: v.replace(".fc1", ".layers.0") for k, v in sam2_sd.items()}
    # sam2_sd = {k: v.replace(".fc2", ".layers.1") for k, v in sam2_sd.items()}
    return sam2_sd


def replace_keys(key_mapping: dict):
    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"
    output_mask_decoder_mlps_pattern = r"tracker_model.mask_decoder.transformer.layers.(\d+).mlp.layers.(\d+).*"
    output_mask_decoder_score_head_pattern = r"tracker_model.mask_decoder.pred_obj_score_head.layers.(\d+).*"
    output_vision_encoder_mlps_pattern = r"vision_encoder.backbone.blocks.(\d+).mlp.layers.(\d+).*"
    output_vision_encoder_neck_pattern = r"tracker_model.vision_encoder.neck.convs.(\d+).conv"
    output_memory_encoder_projection_pattern = r"tracker_model.memory_encoder.o_proj.*"
    output_object_pointer_proj_pattern = r"tracker_model.object_pointer_proj.layers.(\d+).*"
    output_memory_encoder_mask_downsampler_pattern = r"tracker_model.memory_encoder.mask_downsampler.encoder.(\d+).*"
    key_mapping_copy = key_mapping.copy()
    for value, key in key_mapping_copy.items():
        if not value.startswith("sam2_predictor."):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        # vision_encoder.blocks.0.mlp.layers.1.weight -> vision_encoder.blocks.0.mlp.proj_out.weight
        if re.match(output_vision_encoder_mlps_pattern, key):
            layer_nb = int(re.match(output_vision_encoder_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "proj_out")

        # mask_decoder.transformer.layers.0.mlp.layers.1.weight -> mask_decoder.transformer.layers.1.mlp.proj_out.weight
        if re.match(output_mask_decoder_mlps_pattern, key):
            layer_nb = int(re.match(output_mask_decoder_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("mlp.layers.0", "mlp.proj_in")
            elif layer_nb == 1:
                key = key.replace("mlp.layers.1", "mlp.proj_out")

        # mask_decoder.pred_obj_score_head.layers.1.weight -> mask_decoder.pred_obj_score_head.proj_in.weight
        if re.match(output_mask_decoder_score_head_pattern, key):
            layer_nb = int(re.match(output_mask_decoder_score_head_pattern, key).group(1))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        # vision_encoder.neck.convs.1.conv.bias -> vision_encoder.neck.convs.1.bias
        if re.match(output_vision_encoder_neck_pattern, key):
            key = key.replace(".conv.", ".")

        # memory_encoder.o_proj.weight -> memory_encoder.projection.weight
        if re.match(output_memory_encoder_projection_pattern, key):
            key = key.replace(".o_proj.", ".projection.")

        if re.match(output_object_pointer_proj_pattern, key):
            layer_nb = int(re.match(output_object_pointer_proj_pattern, key).group(1))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        if re.match(output_memory_encoder_mask_downsampler_pattern, key):
            layer_nb = int(re.match(output_memory_encoder_mask_downsampler_pattern, key).group(1))
            if layer_nb == 12:
                key = key.replace(f"encoder.{layer_nb}", "final_conv")
            elif layer_nb % 3 == 0:
                key = key.replace(f"encoder.{layer_nb}", f"layers.{layer_nb // 3}.conv")
            elif layer_nb % 3 == 1:
                key = key.replace(f"encoder.{layer_nb}", f"layers.{layer_nb // 3}.layer_norm")
        key_mapping[value] = key

    return key_mapping


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    """
    Convert original SAM3 checkpoint keys to HuggingFace format.

    This function applies regex patterns to efficiently rename keys in bulk.

    Args:
        state_dict_keys: List of original checkpoint keys

    Returns:
        Dictionary mapping original keys to new keys
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text

        # Apply all regex patterns
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)

        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))

    output_dict = replace_keys(output_dict)
    output_dict = adapt_internal_ckpt(output_dict)

    return output_dict


def split_qkv(state_dict: dict) -> dict:
    """
    Split combined QKV weights/biases into separate Q, K, V projections.

    Both the vision backbone and text encoder in the original SAM3 use combined QKV projections,
    but the refactored model uses separate Q, K, V projections.

    Args:
        state_dict: State dictionary with combined QKV weights

    Returns:
        State dictionary with split Q, K, V weights
    """
    # Handle vision backbone: .attention.qkv.* → .attention.{q,k,v}_proj.*
    vision_keys_to_split = [key for key in state_dict.keys() if ".attention.qkv." in key]

    for key in vision_keys_to_split:
        qkv = state_dict.pop(key)
        # Split into 3 equal chunks along dimension 0 (output dimension)
        q, k, v = torch.chunk(qkv, 3, dim=0)

        # Create new keys for q_proj, k_proj, v_proj
        state_dict[key.replace(".qkv.", ".q_proj.")] = q
        state_dict[key.replace(".qkv.", ".k_proj.")] = k
        state_dict[key.replace(".qkv.", ".v_proj.")] = v

    # Handle all attention layers with in_proj_* (text encoder, DETR decoder cross-attention, mask decoder)
    # These use: .{attn_type}.in_proj_* → .{attn_type}.{q,k,v}_proj.*
    in_proj_keys_to_split = [key for key in state_dict.keys() if ".in_proj_" in key]

    for key in in_proj_keys_to_split:
        in_proj = state_dict.pop(key)
        # Split into 3 equal chunks along dimension 0 (output dimension)
        q, k, v = torch.chunk(in_proj, 3, dim=0)

        # Create new keys for q_proj, k_proj, v_proj
        # Replace "in_proj_weight" with "q_proj.weight" (or "in_proj_bias" with "q_proj.bias")
        if key.endswith("in_proj_weight"):
            base_key = key.replace("in_proj_weight", "")
            state_dict[base_key + "q_proj.weight"] = q
            state_dict[base_key + "k_proj.weight"] = k
            state_dict[base_key + "v_proj.weight"] = v
        elif key.endswith("in_proj_bias"):
            base_key = key.replace("in_proj_bias", "")
            state_dict[base_key + "q_proj.bias"] = q
            state_dict[base_key + "k_proj.bias"] = k
            state_dict[base_key + "v_proj.bias"] = v

    return state_dict


def load_original_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load the original SAM3 checkpoint."""
    print(f"Loading original checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"Loaded {len(state_dict)} keys from checkpoint")
    return state_dict


def get_sam3_video_config(
    vision_config: Optional[dict] = None,
    text_config: Optional[dict] = None,
) -> Sam3VideoConfig:
    """
    Create SAM3 configuration.

    Args:
        vision_config: Optional vision encoder configuration overrides
        text_config: Optional text encoder configuration overrides

    Returns:
        Sam3Config instance
    """
    config = Sam3VideoConfig()

    # Update with any provided overrides
    if vision_config is not None:
        for key, value in vision_config.items():
            setattr(config.vision_config, key, value)

    if text_config is not None:
        # Text config is a CLIPTextConfig
        for key, value in text_config.items():
            setattr(config.text_config, key, value)

    return config


def convert_sam3_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config: Optional[Sam3VideoConfig] = None,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    safe_serialization: bool = True,
):
    """
    Convert SAM3 checkpoint from original format to HuggingFace format.

    Args:
        checkpoint_path: Path to the original checkpoint file
        output_path: Path to save the converted checkpoint
        config: Optional Sam3VideoConfig to use (otherwise creates default)
        push_to_hub: Whether to push the model to the Hub
        repo_id: Repository ID for pushing to Hub
        safe_serialization: Whether to save using safetensors
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load configuration
    if config is None:
        config = get_sam3_video_config()

    config.architectures = ["Sam3VideoModel"]
    config.save_pretrained(output_path)
    print("Model config saved successfully")

    # Load and convert weights
    print("Loading original checkpoint...")
    state_dict_old = load_original_state_dict(checkpoint_path)

    print("Converting checkpoint keys...")
    all_keys = list(state_dict_old.keys())
    key_mapping = convert_old_keys_to_new_keys(all_keys)

    # Create new state dict with converted keys
    state_dict_new = {}

    for old_key in all_keys:
        new_key = key_mapping.get(old_key, old_key)
        # Special handling: Strip cls token from vision backbone position embeddings
        if new_key == "detector_model.vision_encoder.backbone.embeddings.position_embeddings":
            # Original has [1, 577, 1024] with cls token, but refactored expects [1, 576, 1024] without cls token
            # Strip the first position (cls token position)
            state_dict_new[new_key] = state_dict_old[old_key][:, 1:, :]
        else:
            state_dict_new[new_key] = state_dict_old[old_key]

    state_dict_new["tracker_model.shared_image_embedding.positional_embedding"] = state_dict_new[
        "tracker_model.prompt_encoder.shared_embedding.positional_embedding"
    ]
    state_dict_new["tracker_model.prompt_encoder.point_embed.weight"] = torch.cat(
        [state_dict_new.pop(f"tracker_model.prompt_encoder.point_embed.{i}.weight") for i in range(4)],
        dim=0,
    )
    del state_dict_old
    gc.collect()

    # Split combined QKV projections into separate Q, K, V projections
    print("Splitting QKV projections...")
    state_dict_new = split_qkv(state_dict_new)

    # Transpose CLIP text projection (stored transposed in original)
    if "detector_model.text_encoder.text_projection.weight" in state_dict_new:
        print("Transposing CLIP text_projection...")
        state_dict_new["detector_model.text_encoder.text_projection.weight"] = state_dict_new[
            "detector_model.text_encoder.text_projection.weight"
        ].T

    # Load into HF models
    print("Loading weights into Sam3VideoModel...")
    model = Sam3VideoModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}):")
        for key in missing_keys:  # Show more keys for debugging
            logger.warning(f"  - {key}")

    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys:  # Show more keys for debugging
            logger.warning(f"  - {key}")

    # Note: Some missing/unexpected keys are expected:
    # - vision_encoder.backbone.embeddings.patch_embeddings.projection.bias: patch projection has bias=False
    # - geometry_encoder.mask_encoder.projection.*: this is nn.Identity() in original (no weights)
    # - rotary_emb.rope_embeddings: pre-computed in original, computed on-the-fly in refactored
    # - text_encoder.text_projection.bias: projection layer might not have bias

    # Save model
    print(f"Saving converted model to {output_path}")
    model.save_pretrained(
        output_path,
        safe_serialization=safe_serialization,
    )

    # Save processor
    print("Creating and saving processor...")
    image_processor = Sam3ImageProcessorFast()
    video_processor = Sam2VideoVideoProcessor(
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], size={"height": 1008, "width": 1008}
    )
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", max_length=32, model_max_length=32)
    processor = Sam3VideoProcessor(
        image_processor=image_processor, video_processor=video_processor, tokenizer=tokenizer
    )
    processor.save_pretrained(output_path)

    # Push to hub if requested
    if push_to_hub:
        if repo_id is None:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        print(f"Pushing model to Hub: {repo_id}")
        model.push_to_hub(repo_id, use_temp_dir=True, private=True)
        processor.push_to_hub(repo_id, use_temp_dir=True, private=True)

    print("Conversion complete!")
    print(f"Model saved successfully to: {output_path}")

    # Cleanup
    del state_dict_new, model
    gc.collect()

    # Verify the conversion by reloading
    print("Loading saved weights into Sam3TrackerVideoModel...")
    try:
        model = Sam3TrackerVideoModel.from_pretrained(output_path)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Successfully loaded model with {param_count:,} parameters")
        del model
        gc.collect()
    except Exception as e:
        print(f"✗ Failed to reload model: {e}")
        raise e
    print("Loading saved weights into Sam3TrackerModel...")
    try:
        model = Sam3TrackerModel.from_pretrained(output_path)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Successfully loaded model with {param_count:,} parameters")
        del model
        gc.collect()
    except Exception as e:
        print(f"✗ Failed to reload model: {e}")
        raise e
    print("Loading saved weights into Sam3Model...")
    try:
        model = Sam3Model.from_pretrained(output_path)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Successfully loaded model with {param_count:,} parameters")
        del model
        gc.collect()
    except Exception as e:
        print(f"✗ Failed to reload model: {e}")
        raise e

    print("\nVerifying converted checkpoint can be loaded...")
    try:
        model = Sam3VideoModel.from_pretrained(output_path)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Successfully loaded model with {param_count:,} parameters")
        del model
        gc.collect()
    except Exception as e:
        print(f"✗ Failed to reload model: {e}")

    print("\n" + "=" * 80)
    print("Conversion finished!")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print("\nTo test the model, you can run:")
    print(">>> from transformers import Sam3Model")
    print(f">>> model = Sam3Model.from_pretrained('{output_path}')")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Convert SAM3 checkpoint to HuggingFace format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the original SAM3 checkpoint file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted checkpoint",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model to the Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Repository ID for pushing to Hub (e.g., 'facebook/sam3-large')",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Whether to save using safetensors format",
    )

    args = parser.parse_args()

    convert_sam3_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        safe_serialization=args.safe_serialization,
    )


if __name__ == "__main__":
    main()
