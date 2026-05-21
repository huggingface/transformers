# Copyright 2026 The Meta AI Authors and The HuggingFace Inc. team. All rights reserved.
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
Convert full SAM3.1 video PCS checkpoints from the original Meta implementation to HuggingFace format.

Composes the SAM3.1 tracker remaps (TriNeck + multiplex tracker, in `convert_sam3_1_tracker_video_to_hf.py`)
with the SAM3 detector head remaps (DETR / CLIP text / geometry / dot-product / mask decoder, in
`convert_sam3_video_to_hf.py`), routing them under the SAM3.1 video composition layout:

  * `tracker_model.vision_encoder.*` — shared TriNeck (lives on the tracker; the detector reuses it via
    the `Sam31VideoModel._tri_neck_to_sam3_view` adapter at runtime).
  * `tracker_model.*`                — multiplex SAM3.1 video tracker body.
  * `detector_model.*`               — heads-only `Sam3Model` (no vision encoder).

Example:
    python convert_sam3_1_video_to_hf.py \\
        --checkpoint_path /path/to/sam3_1_full.pt \\
        --output_path /path/to/output_dir
"""

import argparse
import gc
import os

import regex as re
import torch

from transformers import CLIPTokenizerFast
from transformers.models.sam2_video.video_processing_sam2_video import Sam2VideoVideoProcessor
from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor
from transformers.models.sam3_1_video.configuration_sam3_1_video import Sam31VideoConfig
from transformers.models.sam3_1_video.modeling_sam3_1_video import Sam31VideoModel

# Reuse the tracker-side textual rewrites verbatim; we just re-prefix their destinations to
# `tracker_model.*` after the fact.
from transformers.models.sam3_1_tracker_video.convert_sam3_1_tracker_video_to_hf import (
    TRACKER_REGEX_RENAMES as _TRK_REGEX,
)
from transformers.models.sam3_1_tracker_video.convert_sam3_1_tracker_video_to_hf import (
    TRACKER_SUBMODULE_RENAMES as _TRK_SUBMODULE,
)
from transformers.models.sam3_1_tracker_video.convert_sam3_1_tracker_video_to_hf import (
    add_identity_memory_projection,
    concat_point_embeddings,
    reduce_occlusion_spatial_embedding,
    split_qkv as split_qkv_tracker,
)
from transformers.models.sam3_video.processing_sam3_video import Sam3VideoProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# ============================================================================================
# Phase 1: top-level prefix regexes (original key → HF composition-prefixed key)
# ============================================================================================
# Conventions (matches SAM3.1 full PCS .pt layout from `facebook_sam3/sam3/model/`):
#   * `detector.backbone.vision_backbone.*`                → `tracker_model.vision_encoder.*`
#     (shared TriNeck — lives on the tracker side under the composition).
#   * `detector.text_encoder.*`                            → `detector_model.text_encoder.*`
#   * `detector.backbone.language_backbone.*`              → `detector_model.text_encoder.*`
#   * `detector.transformer.{encoder|decoder}.*`           → `detector_model.detr_{encoder|decoder}.*`
#   * `detector.geometry_encoder.*`                        → `detector_model.geometry_encoder.*`
#   * `detector.dot_prod_scoring.*`                        → `detector_model.dot_product_scoring.*`
#   * `detector.segmentation_head.*`                       → `detector_model.mask_decoder.*`
#   * `tracker.model.*`                                    → `tracker_model.*`
# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING: dict[str, str] = {
    # =====================================================================================
    # Shared Vision Encoder: ViT trunk → tracker_model.vision_encoder.backbone
    # =====================================================================================
    r"^detector\.backbone\.vision_backbone\.trunk\.pos_embed$":                       r"tracker_model.vision_encoder.backbone.embeddings.position_embeddings",
    r"^detector\.backbone\.vision_backbone\.trunk\.patch_embed\.proj\.":              r"tracker_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.",
    r"^detector\.backbone\.vision_backbone\.trunk\.ln_pre\.":                         r"tracker_model.vision_encoder.backbone.layer_norm.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.norm1\.":           r"tracker_model.vision_encoder.backbone.layers.\1.layer_norm1.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.norm2\.":           r"tracker_model.vision_encoder.backbone.layers.\1.layer_norm2.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.qkv\.":       r"tracker_model.vision_encoder.backbone.layers.\1.attention.qkv.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.proj\.":      r"tracker_model.vision_encoder.backbone.layers.\1.attention.o_proj.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.mlp\.fc1\.":        r"tracker_model.vision_encoder.backbone.layers.\1.mlp.fc1.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.mlp\.fc2\.":        r"tracker_model.vision_encoder.backbone.layers.\1.mlp.fc2.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.freqs_cis$":  r"__DROP__.tracker_model.vision_encoder.backbone.layers.\1.attention.freqs_cis",

    # =====================================================================================
    # Shared Vision Encoder: TriNeck FPN heads (sam3 / interactive / propagation)
    # =====================================================================================
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_0\.":             r"tracker_model.vision_encoder.neck.sam3_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_1\.":             r"tracker_model.vision_encoder.neck.sam3_fpn_layers.\1.scale_layers.2.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2\.":               r"tracker_model.vision_encoder.neck.sam3_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.maxpool_2x2\.":             r"tracker_model.vision_encoder.neck.sam3_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.conv_1x1\.":                r"tracker_model.vision_encoder.neck.sam3_fpn_layers.\1.proj1.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.conv_3x3\.":                r"tracker_model.vision_encoder.neck.sam3_fpn_layers.\1.proj2.",

    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.dconv_2x2_0\.": r"tracker_model.vision_encoder.neck.interactive_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.dconv_2x2_1\.": r"tracker_model.vision_encoder.neck.interactive_fpn_layers.\1.scale_layers.2.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.dconv_2x2\.":   r"tracker_model.vision_encoder.neck.interactive_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.maxpool_2x2\.": r"tracker_model.vision_encoder.neck.interactive_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.conv_1x1\.":    r"tracker_model.vision_encoder.neck.interactive_fpn_layers.\1.proj1.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.conv_3x3\.":    r"tracker_model.vision_encoder.neck.interactive_fpn_layers.\1.proj2.",

    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.dconv_2x2_0\.": r"tracker_model.vision_encoder.neck.propagation_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.dconv_2x2_1\.": r"tracker_model.vision_encoder.neck.propagation_fpn_layers.\1.scale_layers.2.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.dconv_2x2\.":   r"tracker_model.vision_encoder.neck.propagation_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.maxpool_2x2\.": r"tracker_model.vision_encoder.neck.propagation_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.conv_1x1\.":    r"tracker_model.vision_encoder.neck.propagation_fpn_layers.\1.proj1.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.conv_3x3\.":    r"tracker_model.vision_encoder.neck.propagation_fpn_layers.\1.proj2.",

    # =====================================================================================
    # Detector heads — Text Encoder (CLIP) under `detector.backbone.language_backbone` or `detector.text_encoder`
    # =====================================================================================
    r"^detector\.backbone\.language_backbone\.encoder\.":                             r"detector_model.text_encoder.",
    r"^detector\.backbone\.language_backbone\.resizer\.":                             r"detector_model.text_projection.",
    r"^detector_model\.text_encoder\.token_embedding\.":                              r"detector_model.text_encoder.text_model.embeddings.token_embedding.",
    r"^detector_model\.text_encoder\.positional_embedding":                           r"detector_model.text_encoder.text_model.embeddings.position_embedding.weight",
    r"^detector_model\.text_encoder\.ln_final\.":                                     r"detector_model.text_encoder.text_model.final_layer_norm.",
    r"^detector_model\.text_encoder\.text_projection":                                r"detector_model.text_encoder.text_projection.weight",
    r"^detector_model\.text_encoder\.transformer\.resblocks\.(\d+)\.attn\.in_proj_":  r"detector_model.text_encoder.text_model.encoder.layers.\1.self_attn.in_proj_",
    r"^detector_model\.text_encoder\.transformer\.resblocks\.(\d+)\.attn\.out_proj\.": r"detector_model.text_encoder.text_model.encoder.layers.\1.self_attn.out_proj.",
    r"^detector_model\.text_encoder\.transformer\.resblocks\.(\d+)\.ln_1\.":          r"detector_model.text_encoder.text_model.encoder.layers.\1.layer_norm1.",
    r"^detector_model\.text_encoder\.transformer\.resblocks\.(\d+)\.ln_2\.":          r"detector_model.text_encoder.text_model.encoder.layers.\1.layer_norm2.",
    r"^detector_model\.text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_fc\.":     r"detector_model.text_encoder.text_model.encoder.layers.\1.mlp.fc1.",
    r"^detector_model\.text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_proj\.":   r"detector_model.text_encoder.text_model.encoder.layers.\1.mlp.fc2.",

    # =====================================================================================
    # Detector heads — Geometry Encoder
    # =====================================================================================
    r"^detector\.geometry_encoder\.encode\.(\d+)\.cross_attn_image\.out_proj\.":      r"detector_model.geometry_encoder.layers.\1.cross_attn.o_proj.",
    r"^detector\.geometry_encoder\.encode\.(\d+)\.cross_attn_image\.":                r"detector_model.geometry_encoder.layers.\1.cross_attn.",
    r"^detector\.geometry_encoder\.encode\.(\d+)\.self_attn\.out_proj\.":             r"detector_model.geometry_encoder.layers.\1.self_attn.o_proj.",
    r"^detector\.geometry_encoder\.encode\.(\d+)\.self_attn\.":                       r"detector_model.geometry_encoder.layers.\1.self_attn.",
    r"^detector\.geometry_encoder\.encode\.(\d+)\.linear1\.":                         r"detector_model.geometry_encoder.layers.\1.mlp.fc1.",
    r"^detector\.geometry_encoder\.encode\.(\d+)\.linear2\.":                         r"detector_model.geometry_encoder.layers.\1.mlp.fc2.",
    r"^detector\.geometry_encoder\.encode\.(\d+)\.norm1\.":                           r"detector_model.geometry_encoder.layers.\1.layer_norm1.",
    r"^detector\.geometry_encoder\.encode\.(\d+)\.norm2\.":                           r"detector_model.geometry_encoder.layers.\1.layer_norm2.",
    r"^detector\.geometry_encoder\.encode\.(\d+)\.norm3\.":                           r"detector_model.geometry_encoder.layers.\1.layer_norm3.",
    r"^detector\.geometry_encoder\.img_pre_norm\.":                                   r"detector_model.geometry_encoder.vision_layer_norm.",
    r"^detector\.geometry_encoder\.norm\.":                                           r"detector_model.geometry_encoder.prompt_layer_norm.",
    r"^detector\.geometry_encoder\.encode_norm\.":                                    r"detector_model.geometry_encoder.output_layer_norm.",

    # =====================================================================================
    # Detector heads — DETR Encoder
    # =====================================================================================
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.out_proj\.":  r"detector_model.detr_encoder.layers.\1.cross_attn.o_proj.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.":            r"detector_model.detr_encoder.layers.\1.cross_attn.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.":         r"detector_model.detr_encoder.layers.\1.self_attn.o_proj.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.self_attn\.":                   r"detector_model.detr_encoder.layers.\1.self_attn.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.cross_attn\.out_proj\.":        r"detector_model.detr_encoder.layers.\1.cross_attn.o_proj.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.cross_attn\.":                  r"detector_model.detr_encoder.layers.\1.cross_attn.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.linear1\.":                     r"detector_model.detr_encoder.layers.\1.mlp.fc1.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.linear2\.":                     r"detector_model.detr_encoder.layers.\1.mlp.fc2.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.norm1\.":                       r"detector_model.detr_encoder.layers.\1.layer_norm1.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.norm2\.":                       r"detector_model.detr_encoder.layers.\1.layer_norm2.",
    r"^detector\.transformer\.encoder\.layers\.(\d+)\.norm3\.":                       r"detector_model.detr_encoder.layers.\1.layer_norm3.",

    # =====================================================================================
    # Detector heads — DETR Decoder
    # =====================================================================================
    r"^detector\.transformer\.decoder\.query_embed\.":                                r"detector_model.detr_decoder.query_embed.",
    r"^detector\.transformer\.decoder\.reference_points\.":                           r"detector_model.detr_decoder.reference_points.",
    r"^detector\.transformer\.decoder\.instance_query_embed\.":                       r"detector_model.detr_decoder.instance_query_embed.",
    r"^detector\.transformer\.decoder\.instance_reference_points\.":                  r"detector_model.detr_decoder.instance_reference_points.",
    r"^detector\.transformer\.decoder\.presence_token\.":                             r"detector_model.detr_decoder.presence_token.",
    r"^detector\.transformer\.decoder\.presence_token_head\.layers\.0\.":             r"detector_model.detr_decoder.presence_head.layer1.",
    r"^detector\.transformer\.decoder\.presence_token_head\.layers\.1\.":             r"detector_model.detr_decoder.presence_head.layer2.",
    r"^detector\.transformer\.decoder\.presence_token_head\.layers\.2\.":             r"detector_model.detr_decoder.presence_head.layer3.",
    r"^detector\.transformer\.decoder\.presence_token_out_norm\.":                    r"detector_model.detr_decoder.presence_layer_norm.",
    r"^detector\.transformer\.decoder\.norm\.":                                       r"detector_model.detr_decoder.output_layer_norm.",
    r"^detector\.transformer\.decoder\.bbox_embed\.layers\.0\.":                      r"detector_model.detr_decoder.box_head.layer1.",
    r"^detector\.transformer\.decoder\.bbox_embed\.layers\.1\.":                      r"detector_model.detr_decoder.box_head.layer2.",
    r"^detector\.transformer\.decoder\.bbox_embed\.layers\.2\.":                      r"detector_model.detr_decoder.box_head.layer3.",
    r"^detector\.transformer\.decoder\.instance_bbox_embed\.layers\.0\.":             r"detector_model.detr_decoder.instance_box_head.layer1.",
    r"^detector\.transformer\.decoder\.instance_bbox_embed\.layers\.1\.":             r"detector_model.detr_decoder.instance_box_head.layer2.",
    r"^detector\.transformer\.decoder\.instance_bbox_embed\.layers\.2\.":             r"detector_model.detr_decoder.instance_box_head.layer3.",
    r"^detector\.transformer\.decoder\.ref_point_head\.layers\.0\.":                  r"detector_model.detr_decoder.ref_point_head.layer1.",
    r"^detector\.transformer\.decoder\.ref_point_head\.layers\.1\.":                  r"detector_model.detr_decoder.ref_point_head.layer2.",
    r"^detector\.transformer\.decoder\.boxRPB_embed_x\.layers\.0\.":                  r"detector_model.detr_decoder.box_rpb_embed_x.layer1.",
    r"^detector\.transformer\.decoder\.boxRPB_embed_x\.layers\.1\.":                  r"detector_model.detr_decoder.box_rpb_embed_x.layer2.",
    r"^detector\.transformer\.decoder\.boxRPB_embed_y\.layers\.0\.":                  r"detector_model.detr_decoder.box_rpb_embed_y.layer1.",
    r"^detector\.transformer\.decoder\.boxRPB_embed_y\.layers\.1\.":                  r"detector_model.detr_decoder.box_rpb_embed_y.layer2.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.":         r"detector_model.detr_decoder.layers.\1.self_attn.o_proj.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.self_attn\.":                   r"detector_model.detr_decoder.layers.\1.self_attn.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.ca_text\.out_proj\.":           r"detector_model.detr_decoder.layers.\1.text_cross_attn.o_proj.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.ca_text\.":                     r"detector_model.detr_decoder.layers.\1.text_cross_attn.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.cross_attn\.out_proj\.":        r"detector_model.detr_decoder.layers.\1.vision_cross_attn.o_proj.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.cross_attn\.":                  r"detector_model.detr_decoder.layers.\1.vision_cross_attn.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.linear1\.":                     r"detector_model.detr_decoder.layers.\1.mlp.fc1.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.linear2\.":                     r"detector_model.detr_decoder.layers.\1.mlp.fc2.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.norm1\.":                       r"detector_model.detr_decoder.layers.\1.vision_cross_attn_layer_norm.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.catext_norm\.":                 r"detector_model.detr_decoder.layers.\1.text_cross_attn_layer_norm.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.norm2\.":                       r"detector_model.detr_decoder.layers.\1.self_attn_layer_norm.",
    r"^detector\.transformer\.decoder\.layers\.(\d+)\.norm3\.":                       r"detector_model.detr_decoder.layers.\1.mlp_layer_norm.",

    # =====================================================================================
    # Detector heads — Dot-product scoring
    # =====================================================================================
    r"^detector\.dot_prod_scoring\.prompt_mlp\.layers\.0\.":                          r"detector_model.dot_product_scoring.text_mlp.layer1.",
    r"^detector\.dot_prod_scoring\.prompt_mlp\.layers\.1\.":                          r"detector_model.dot_product_scoring.text_mlp.layer2.",
    r"^detector\.dot_prod_scoring\.prompt_mlp\.out_norm\.":                           r"detector_model.dot_product_scoring.text_mlp_out_norm.",
    r"^detector\.dot_prod_scoring\.prompt_proj\.":                                    r"detector_model.dot_product_scoring.text_proj.",
    r"^detector\.dot_prod_scoring\.hs_proj\.":                                        r"detector_model.dot_product_scoring.query_proj.",

    # =====================================================================================
    # Detector heads — Mask Decoder (segmentation head)
    # =====================================================================================
    r"^detector\.segmentation_head\.pixel_decoder\.conv_layers\.(\d+)\.":             r"detector_model.mask_decoder.pixel_decoder.conv_layers.\1.",
    r"^detector\.segmentation_head\.pixel_decoder\.norms\.(\d+)\.":                   r"detector_model.mask_decoder.pixel_decoder.norms.\1.",
    r"^detector\.segmentation_head\.mask_embed\.layers\.(\d+)\.":                     r"detector_model.mask_decoder.mask_embedder.layers.\1.",
    r"^detector\.segmentation_head\.mask_predictor\.mask_embed\.layers\.(\d+)\.":     r"detector_model.mask_decoder.mask_embedder.layers.\1.",
    r"^detector\.segmentation_head\.instance_seg_head\.":                             r"detector_model.mask_decoder.instance_projection.",
    r"^detector\.segmentation_head\.semantic_seg_head\.":                             r"detector_model.mask_decoder.semantic_projection.",
    r"^detector\.segmentation_head\.cross_attend_prompt\.out_proj\.":                 r"detector_model.mask_decoder.prompt_cross_attn.o_proj.",
    r"^detector\.segmentation_head\.cross_attend_prompt\.":                           r"detector_model.mask_decoder.prompt_cross_attn.",
    r"^detector\.segmentation_head\.cross_attn_norm\.":                               r"detector_model.mask_decoder.prompt_cross_attn_norm.",

    # =====================================================================================
    # Tracker body: peel `tracker.model.` → `tracker_model.`
    # =====================================================================================
    r"^tracker\.model\.":                                                             r"tracker_model.",

    # Any remaining `detector.*` keys are unrecognised heads — drop them so the load is strict.
    # The tracker converter used a `__DROP__.detector.` sentinel because the standalone PVS tracker
    # legitimately discards the detector. Here we expect every detector key to be claimed by one of
    # the rules above; if not, surface it via the sentinel and warn in the loader.
    r"^detector\.":                                                                   r"__DROP__.detector.",
}
# fmt: on


def _reprefix(rule: tuple[str, str], prefix: str) -> tuple[str, str]:
    """Rewrite a `(needle, replacement)` rule so its replacement is prefixed with `prefix`.

    The needles are anchored at the start of the (re-prefixed) target string, so we add `prefix`
    to the replacement and to the needle's leading anchor if any. Submodule renames have no anchor
    so we only need to prefix the replacement.
    """
    needle, replacement = rule
    if needle.startswith("^"):
        # Regex pattern with explicit start anchor: insert the prefix after the anchor.
        return ("^" + re.escape(prefix) + needle[1:], prefix + replacement)
    return (needle, prefix + replacement) if needle.startswith(prefix) else (needle, replacement)


# Phase 2: apply the tracker-side textual rules, but only on keys that already live under
# `tracker_model.*` (everything else has been routed to detector_model.* / tracker_model.vision_encoder.*).
# We do plain substring substitutions, scoped to keys starting with the right prefix.
TRACKER_SUBMODULE_RENAMES: list[tuple[str, str]] = list(_TRK_SUBMODULE)


def _re_prefix_tracker_regex(needle: str, repl: str) -> tuple[str, str]:
    """Re-anchor a `(needle, replacement)` rule from the standalone-tracker namespace into the
    composed `tracker_model.*` namespace.

    Anchored rules (those starting with `^`) target the top of a key like `^memory_attention.`; we
    rewrite them to `^tracker_model.memory_attention.` so they still match after composition.
    Unanchored rules (e.g. the `__FF_proj_in__ → proj_in` placeholder resolution, which fires anywhere
    in the path) must be left untouched — they're already namespace-agnostic.
    """
    if needle.startswith("^"):
        return ("^tracker_model\\." + needle[1:], "tracker_model." + repl)
    return (needle, repl)


TRACKER_REGEX_RENAMES: list[tuple[str, str]] = [_re_prefix_tracker_regex(n, r) for n, r in _TRK_REGEX]


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    """Map original SAM3.1 full-PCS state-dict keys to `Sam31VideoModel` keys.

    Same three-pass pipeline as the tracker converter:
      1. Top-level prefix regexes (`ORIGINAL_TO_CONVERTED_KEY_MAPPING`) split keys between the
         detector heads, the shared vision encoder, and the tracker.
      2. Literal submodule renames (`TRACKER_SUBMODULE_RENAMES`, scoped to `tracker_model.*`).
      3. Deeper regex rewrites (`TRACKER_REGEX_RENAMES`, also scoped to `tracker_model.*`).
    """
    if not state_dict_keys:
        return {}

    old_text = "\n".join(state_dict_keys)
    new_text = old_text

    for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)

    # Submodule renames are applied as plain string replacements but only on `tracker_model.*` lines
    # (to avoid touching `detector_model.*` which happens to share some component names).
    out_lines = []
    for line in new_text.split("\n"):
        if line.startswith("tracker_model.") and not line.startswith("tracker_model.vision_encoder."):
            for needle, replacement in TRACKER_SUBMODULE_RENAMES:
                line = line.replace(needle, replacement)
        out_lines.append(line)
    new_text = "\n".join(out_lines)

    for pattern, replacement in TRACKER_REGEX_RENAMES:
        new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)

    return dict(zip(old_text.split("\n"), new_text.split("\n")))


def split_qkv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split the vision-backbone combined QKV projections into separate Q / K / V tensors.

    The TriNeck shares a single ViT trunk between detector and tracker. The shared backbone
    stores QKV combined under `tracker_model.vision_encoder.backbone.layers.{i}.attention.qkv.*`;
    we re-use the tracker converter's `split_qkv` for that branch. For the detector-side CLIP
    text encoder, splits are handled below via the `in_proj_*` rewrite (it uses a different
    convention than the ViT QKV).
    """
    state_dict = split_qkv_tracker(state_dict)

    # Detector CLIP text encoder uses `in_proj_{weight,bias}` for combined QKV; split into q/k/v.
    in_proj_keys = [k for k in list(state_dict) if ".in_proj_" in k]
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


def load_original_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load the original SAM3.1 full-PCS checkpoint, unwrapping the optional `model` / `state_dict` keys."""
    print(f"Loading original checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"Loaded {len(state_dict)} keys from checkpoint")
    return state_dict


def get_sam3_1_video_config() -> Sam31VideoConfig:
    """Return the default `Sam31VideoConfig` matching the published full SAM3.1 PCS checkpoint."""
    return Sam31VideoConfig()


def convert_sam3_1_video_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config: Sam31VideoConfig | None = None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
) -> None:
    """Convert a SAM3.1 full PCS checkpoint to a HuggingFace `Sam31VideoModel` directory."""
    os.makedirs(output_path, exist_ok=True)

    if config is None:
        config = get_sam3_1_video_config()

    config.architectures = ["Sam31VideoModel"]
    config.save_pretrained(output_path)
    print("Model config saved successfully")

    print("Loading original checkpoint...")
    state_dict_old = load_original_state_dict(checkpoint_path)

    print("Converting checkpoint keys...")
    all_keys = list(state_dict_old.keys())
    key_mapping = convert_old_keys_to_new_keys(all_keys)

    state_dict_new: dict[str, torch.Tensor] = {}
    dropped_keys: list[str] = []
    for old_key in all_keys:
        new_key = key_mapping.get(old_key, old_key)
        if new_key.startswith("__DROP__."):
            dropped_keys.append(old_key)
            continue
        tensor = state_dict_old[old_key]
        if new_key == "tracker_model.vision_encoder.backbone.embeddings.position_embeddings":
            # ViT pos_embed comes with a leading CLS token slot (1, num_patches + 1, hidden); the HF
            # backbone allocates only `num_patches` slots, so strip the first one.
            tensor = tensor[:, 1:, :]
        state_dict_new[new_key] = tensor

    print(f"Dropped {len(dropped_keys)} rope-buffer / unrouted detector keys (expected for the HF format)")
    if dropped_keys:
        for k in dropped_keys[:5]:
            print(f"  - {k}")
        if len(dropped_keys) > 5:
            print(f"  ... and {len(dropped_keys) - 5} more")

    # Tracker-side post-processing (mirrors `convert_sam3_1_tracker_video_to_hf.convert_sam3_1_checkpoint`).
    state_dict_new = concat_point_embeddings_video(state_dict_new)
    state_dict_new = split_qkv(state_dict_new)
    state_dict_new = reduce_occlusion_spatial_embedding_video(state_dict_new)
    state_dict_new = add_identity_memory_projection_video(state_dict_new, hidden_dim=config.tracker_config.vision_config.fpn_hidden_size)

    # Detector-side post-processing: transpose CLIP text projection (stored transposed in original).
    proj_key = "detector_model.text_encoder.text_projection.weight"
    if proj_key in state_dict_new:
        print("Transposing CLIP text_projection...")
        state_dict_new[proj_key] = state_dict_new[proj_key].T

    del state_dict_old
    gc.collect()

    print("Loading weights into Sam31VideoModel...")
    model = Sam31VideoModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}):")
        for key in missing_keys:
            logger.warning(f"  - {key}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys:
            logger.warning(f"  - {key}")

    print(f"Saving converted model to {output_path}")
    model.save_pretrained(output_path)

    print("Creating and saving processor...")
    from transformers.image_utils import PILImageResampling

    image_processor = Sam3ImageProcessor()
    video_processor = Sam2VideoVideoProcessor(
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], size={"height": 1008, "width": 1008}
    )
    video_processor.resample = PILImageResampling.BICUBIC
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", max_length=32, model_max_length=32)
    processor = Sam3VideoProcessor(
        image_processor=image_processor, video_processor=video_processor, tokenizer=tokenizer
    )
    processor.save_pretrained(output_path)

    if push_to_hub:
        if repo_id is None:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        print(f"Pushing model to Hub: {repo_id}")
        model.push_to_hub(repo_id, private=True)
        processor.push_to_hub(repo_id, private=True)

    del state_dict_new, model
    gc.collect()

    print("\nVerifying converted checkpoint can be reloaded...")
    reloaded = Sam31VideoModel.from_pretrained(output_path)
    param_count = sum(p.numel() for p in reloaded.parameters())
    print(f"\u2713 Sam31VideoModel.from_pretrained succeeded with {param_count:,} parameters")
    del reloaded
    gc.collect()

    print("\n" + "=" * 80)
    print("Conversion finished!")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print("\nTo test the model, you can run:")
    print(">>> from transformers import Sam31VideoModel")
    print(f">>> model = Sam31VideoModel.from_pretrained('{output_path}')")
    print("=" * 80)


def concat_point_embeddings_video(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concat the four `tracker_model.prompt_encoder.point_embeddings.{i}.weight` tensors."""
    parts = [state_dict.pop(f"tracker_model.prompt_encoder.point_embeddings.{i}.weight", None) for i in range(4)]
    if all(p is not None for p in parts):
        state_dict["tracker_model.prompt_encoder.point_embed.weight"] = torch.cat(parts, dim=0)
    elif any(p is not None for p in parts):
        for i, p in enumerate(parts):
            if p is not None:
                state_dict[f"tracker_model.prompt_encoder.point_embeddings.{i}.weight"] = p
    return state_dict


def reduce_occlusion_spatial_embedding_video(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Forward the tracker-namespaced occlusion embedding through the standalone-tracker helper."""
    key = "tracker_model.occlusion_spatial_embedding_parameter"
    if key not in state_dict:
        return state_dict
    # Repackage with the tracker's local key name, call the helper, then re-prefix.
    bare = {"occlusion_spatial_embedding_parameter": state_dict.pop(key)}
    bare = reduce_occlusion_spatial_embedding(bare)
    state_dict[key] = bare["occlusion_spatial_embedding_parameter"]
    return state_dict


def add_identity_memory_projection_video(
    state_dict: dict[str, torch.Tensor], hidden_dim: int
) -> dict[str, torch.Tensor]:
    """Fill in `tracker_model.memory_encoder.projection.{weight,bias}` with identity init."""
    weight_key = "tracker_model.memory_encoder.projection.weight"
    bias_key = "tracker_model.memory_encoder.projection.bias"
    if weight_key in state_dict:
        return state_dict
    # Reuse the standalone-tracker helper by repackaging with bare keys.
    bare: dict[str, torch.Tensor] = {}
    bare = add_identity_memory_projection(bare, hidden_dim)
    state_dict[weight_key] = bare["memory_encoder.projection.weight"]
    state_dict[bias_key] = bare["memory_encoder.projection.bias"]
    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SAM3.1 full PCS checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original SAM3.1 full PCS .pt file")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the converted model")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the Hub")
    parser.add_argument("--repo_id", type=str, default=None, help="Hub repo id (e.g. `facebook/sam3.1`)")
    args = parser.parse_args()

    convert_sam3_1_video_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
