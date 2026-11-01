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
Convert SAM3.1 multiplex checkpoints from the original implementation to HuggingFace format.

Example:
    python convert_sam3_1_tracker_video_to_hf.py \
        --checkpoint_path /path/to/sam3.1_multiplex.pt \
        --output_path /path/to/output_dir
"""

import argparse
import gc
import os

import regex as re
import torch

from transformers.models.sam2_video.video_processing_sam2_video import Sam2VideoVideoProcessor
from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor
from transformers.models.sam3_1_tracker_video.configuration_sam3_1_tracker_video import Sam31TrackerVideoConfig
from transformers.models.sam3_1_tracker_video.modeling_sam3_1_tracker_video import Sam31TrackerVideoModel
from transformers.models.sam3_1_tracker_video.processing_sam3_1_tracker_video import Sam31TrackerVideoProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# Regex-based mappings from original SAM3.1 checkpoint keys to HuggingFace `Sam31TrackerVideoModel` keys.
# Patterns are applied in declaration order; the right-hand side may reference capture groups via \1, \2, ...
# Conventions (matches SAM3.1 multiplex.pt layout):
#   * `tracker.model.*`                                  → tracker submodules (no prefix in standalone model).
#   * `detector.backbone.vision_backbone.{trunk|*_convs}` → shared vision encoder (we adopt them in the tracker
#                                                          since `Sam31TrackerVideoModel` owns its `vision_encoder`).
#   * Any other `detector.*` key                          → silently dropped (detector head is not part of PVS).
# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # ===========================================================================
    # Vision Encoder: ViT trunk (shared with the detector backbone)
    # ===========================================================================
    r"^detector\.backbone\.vision_backbone\.trunk\.pos_embed$":                       r"vision_encoder.backbone.embeddings.position_embeddings",
    r"^detector\.backbone\.vision_backbone\.trunk\.patch_embed\.proj\.":              r"vision_encoder.backbone.embeddings.patch_embeddings.projection.",
    r"^detector\.backbone\.vision_backbone\.trunk\.ln_pre\.":                         r"vision_encoder.backbone.layer_norm.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.norm1\.":           r"vision_encoder.backbone.layers.\1.layer_norm1.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.norm2\.":           r"vision_encoder.backbone.layers.\1.layer_norm2.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.qkv\.":       r"vision_encoder.backbone.layers.\1.attention.qkv.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.proj\.":      r"vision_encoder.backbone.layers.\1.attention.o_proj.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.mlp\.fc1\.":        r"vision_encoder.backbone.layers.\1.mlp.fc1.",
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.mlp\.fc2\.":        r"vision_encoder.backbone.layers.\1.mlp.fc2.",
    # `attn.freqs_cis` is the original pre-computed RoPE buffer; HF recomputes it at init and does not
    # need it from the checkpoint. We send it to a sentinel name that we filter out before load.
    r"^detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.freqs_cis$":  r"__DROP__.vision_encoder.backbone.layers.\1.attention.freqs_cis",

    # ===========================================================================
    # Vision Encoder: TriNeck FPN heads (sam3 / interactive / propagation)
    # ===========================================================================
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_0\.":             r"vision_encoder.neck.sam3_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_1\.":             r"vision_encoder.neck.sam3_fpn_layers.\1.scale_layers.2.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2\.":               r"vision_encoder.neck.sam3_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.maxpool_2x2\.":             r"vision_encoder.neck.sam3_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.conv_1x1\.":                r"vision_encoder.neck.sam3_fpn_layers.\1.proj1.",
    r"^detector\.backbone\.vision_backbone\.convs\.(\d+)\.conv_3x3\.":                r"vision_encoder.neck.sam3_fpn_layers.\1.proj2.",

    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.dconv_2x2_0\.": r"vision_encoder.neck.interactive_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.dconv_2x2_1\.": r"vision_encoder.neck.interactive_fpn_layers.\1.scale_layers.2.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.dconv_2x2\.":   r"vision_encoder.neck.interactive_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.maxpool_2x2\.": r"vision_encoder.neck.interactive_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.conv_1x1\.":    r"vision_encoder.neck.interactive_fpn_layers.\1.proj1.",
    r"^detector\.backbone\.vision_backbone\.interactive_convs\.(\d+)\.conv_3x3\.":    r"vision_encoder.neck.interactive_fpn_layers.\1.proj2.",

    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.dconv_2x2_0\.": r"vision_encoder.neck.propagation_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.dconv_2x2_1\.": r"vision_encoder.neck.propagation_fpn_layers.\1.scale_layers.2.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.dconv_2x2\.":   r"vision_encoder.neck.propagation_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.maxpool_2x2\.": r"vision_encoder.neck.propagation_fpn_layers.\1.scale_layers.0.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.conv_1x1\.":    r"vision_encoder.neck.propagation_fpn_layers.\1.proj1.",
    r"^detector\.backbone\.vision_backbone\.propagation_convs\.(\d+)\.conv_3x3\.":    r"vision_encoder.neck.propagation_fpn_layers.\1.proj2.",

    # ===========================================================================
    # Tracker prefix stripping
    # ===========================================================================
    r"^tracker\.model\.":                                                             r"",

    # Every remaining `detector.*` belongs to the SAM3.1 detector head, which is not implemented in
    # `Sam31TrackerVideoModel`. Send them all to a sentinel prefix so we filter them out before load.
    r"^detector\.":                                                                   r"__DROP__.detector.",
}
# fmt: on


# Sub-module rename map applied AFTER the prefix rewrites above (so we always operate inside the standalone
# tracker namespace). We use plain string replacements here because most of these are non-overlapping
# substitutions that need to fire anywhere in the path.
TRACKER_SUBMODULE_RENAMES: list[tuple[str, str]] = [
    # Top-level submodule rewrites (order matters: interactive ones first to avoid the sam_mask_decoder
    # prefix also matching inside interactive_sam_mask_decoder).
    ("interactive_sam_prompt_encoder.", "prompt_encoder."),
    ("interactive_sam_mask_decoder.", "mask_decoder."),
    ("sam_mask_decoder.", "propagation_mask_decoder."),
    ("interactive_mask_downsample.", "mask_downsample."),
    ("maskmem_backbone.", "memory_encoder."),
    ("transformer.encoder.", "memory_attention."),
    ("image_pe_layer.positional_encoding_gaussian_matrix", "shared_image_embedding.positional_embedding"),
    # SAM3.1 keeps two distinct 3-layer FFs for object-pointer projection:
    # `interactive_obj_ptr_proj` is used on initial click / box / mask conditioning frames
    # (HF `_forward_sam_heads`), and `obj_ptr_proj` is used during multiplex propagation
    # (HF `_run_multiplex_propagation`). The conversion now loads both — earlier HF versions
    # dropped the interactive head and reused the propagation projection on both paths,
    # which silently fed the wrong cond-frame pointer into every subsequent propagation
    # step's memory attention. Match `interactive_obj_ptr_proj` first so the next rule
    # doesn't also rewrite it.
    ("interactive_obj_ptr_proj.", "interactive_object_pointer_proj."),
    ("obj_ptr_proj.", "object_pointer_proj."),
    ("obj_ptr_tpos_proj.", "temporal_positional_encoding_projection_layer."),
    # SAM3.1 swapped the static (`no_obj_ptr`, `(1, hidden_dim)`) no-object pointer for a learned linear
    # projection (`no_obj_ptr_linear`, `(hidden_dim, hidden_dim)`) applied to the predicted pointer when
    # a slot is "not appearing" (Meta `use_linear_no_obj_ptr=True`). We map the linear directly to the
    # HF `no_obj_ptr_linear` module that `_blend_no_object_pointer` consumes; the legacy zero-init
    # `no_object_pointer` parameter remains for backward compatibility but is no longer used at runtime.
    ("no_obj_ptr_linear.", "no_obj_ptr_linear."),
    # SAM3.1 per-slot mask-decoder suppression embeddings (multiplex padding vs valid slots).
    ("output_valid_embed", "output_valid_embed"),
    ("output_invalid_embed", "output_invalid_embed"),
    ("maskmem_tpos_enc", "memory_temporal_positional_encoding"),
    ("interactivity_no_mem_embed", "no_memory_embedding"),
    ("no_obj_embed_spatial", "occlusion_spatial_embedding_parameter"),
    # Memory attention: the original encoder uses `norm{1,2,3}`; HF uses `layer_norm{1,2,3}`. Disambiguate
    # against the final encoder norm (handled separately just below).
    ("memory_attention.norm.", "memory_attention.layer_norm."),
    # Prompt encoder mask_downscaling sequential indices.
    (
        "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
        "prompt_encoder.shared_embedding.positional_embedding",
    ),
    ("prompt_encoder.mask_downscaling.0.", "prompt_encoder.mask_embed.conv1."),
    ("prompt_encoder.mask_downscaling.1.", "prompt_encoder.mask_embed.layer_norm1."),
    ("prompt_encoder.mask_downscaling.3.", "prompt_encoder.mask_embed.conv2."),
    ("prompt_encoder.mask_downscaling.4.", "prompt_encoder.mask_embed.layer_norm2."),
    ("prompt_encoder.mask_downscaling.6.", "prompt_encoder.mask_embed.conv3."),
    # Memory encoder (former maskmem_backbone) sub-renames.
    ("memory_encoder.pix_feat_proj.", "memory_encoder.feature_projection."),
    ("memory_encoder.fuser.", "memory_encoder.memory_fuser."),
]


# Regex rewrites applied AFTER the prefix and submodule renames. These handle the deeper structural
# differences between the original (sequential / lin1-lin2 / norm1-norm4 / etc.) and HF naming.
# fmt: off
TRACKER_REGEX_RENAMES: list[tuple[str, str]] = [
    # ---------------- Memory attention layers ----------------
    (r"^memory_attention\.layers\.(\d+)\.norm1\.",  r"memory_attention.layers.\1.layer_norm1."),
    (r"^memory_attention\.layers\.(\d+)\.norm2\.",  r"memory_attention.layers.\1.layer_norm2."),
    (r"^memory_attention\.layers\.(\d+)\.norm3\.",  r"memory_attention.layers.\1.layer_norm3."),

    # ---------------- Mask decoder two-way transformer ----------------
    # (matches both the interactive `mask_decoder` and the multiplex `propagation_mask_decoder`)
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.layers\.(\d+)\.norm1\.",  r"\1.transformer.layers.\2.layer_norm1."),
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.layers\.(\d+)\.norm2\.",  r"\1.transformer.layers.\2.layer_norm2."),
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.layers\.(\d+)\.norm3\.",  r"\1.transformer.layers.\2.layer_norm3."),
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.layers\.(\d+)\.norm4\.",  r"\1.transformer.layers.\2.layer_norm4."),
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.layers\.(\d+)\.mlp\.lin1\.",  r"\1.transformer.layers.\2.mlp.proj_in."),
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.layers\.(\d+)\.mlp\.lin2\.",  r"\1.transformer.layers.\2.mlp.proj_out."),
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.layers\.(\d+)\.(self_attn|cross_attn_token_to_image|cross_attn_image_to_token)\.out_proj\.",
        r"\1.transformer.layers.\2.\3.o_proj."),
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.final_attn_token_to_image\.out_proj\.",
        r"\1.transformer.final_attn_token_to_image.o_proj."),
    (r"^(mask_decoder|propagation_mask_decoder)\.transformer\.norm_final_attn\.",
        r"\1.transformer.layer_norm_final_attn."),
    # output_upscaling Sequential[0..3] → upscale_conv1 / upscale_layer_norm / upscale_conv2
    (r"^(mask_decoder|propagation_mask_decoder)\.output_upscaling\.0\.",  r"\1.upscale_conv1."),
    (r"^(mask_decoder|propagation_mask_decoder)\.output_upscaling\.1\.",  r"\1.upscale_layer_norm."),
    (r"^(mask_decoder|propagation_mask_decoder)\.output_upscaling\.3\.",  r"\1.upscale_conv2."),

    # ---------------- FeedForward (3-layer MLP) splits ----------------
    # Used by: output_hypernetworks_mlps[j], iou_prediction_head, pred_obj_score_head, object_pointer_proj.
    # The original stores three Linear layers as `layers.{0,1,2}`; HF stores them as
    # `proj_in / layers.0 / proj_out`. We can't write that with a single regex, so we emit three
    # ordered rewrites and rely on the second one running *after* the first/third have already
    # consumed their target indices (we add a temporary suffix to guarantee that).
    (r"^(mask_decoder|propagation_mask_decoder)\.output_hypernetworks_mlps\.(\d+)\.layers\.0\.",
        r"\1.output_hypernetworks_mlps.\2.__FF_proj_in__."),
    (r"^(mask_decoder|propagation_mask_decoder)\.output_hypernetworks_mlps\.(\d+)\.layers\.1\.",
        r"\1.output_hypernetworks_mlps.\2.__FF_mid_0__."),
    (r"^(mask_decoder|propagation_mask_decoder)\.output_hypernetworks_mlps\.(\d+)\.layers\.2\.",
        r"\1.output_hypernetworks_mlps.\2.__FF_proj_out__."),

    (r"^(mask_decoder|propagation_mask_decoder)\.iou_prediction_head\.layers\.0\.",  r"\1.iou_prediction_head.__FF_proj_in__."),
    (r"^(mask_decoder|propagation_mask_decoder)\.iou_prediction_head\.layers\.1\.",  r"\1.iou_prediction_head.__FF_mid_0__."),
    (r"^(mask_decoder|propagation_mask_decoder)\.iou_prediction_head\.layers\.2\.",  r"\1.iou_prediction_head.__FF_proj_out__."),

    (r"^(mask_decoder|propagation_mask_decoder)\.pred_obj_score_head\.layers\.0\.",  r"\1.pred_obj_score_head.__FF_proj_in__."),
    (r"^(mask_decoder|propagation_mask_decoder)\.pred_obj_score_head\.layers\.1\.",  r"\1.pred_obj_score_head.__FF_mid_0__."),
    (r"^(mask_decoder|propagation_mask_decoder)\.pred_obj_score_head\.layers\.2\.",  r"\1.pred_obj_score_head.__FF_proj_out__."),

    (r"^object_pointer_proj\.layers\.0\.",  r"object_pointer_proj.__FF_proj_in__."),
    (r"^object_pointer_proj\.layers\.1\.",  r"object_pointer_proj.__FF_mid_0__."),
    (r"^object_pointer_proj\.layers\.2\.",  r"object_pointer_proj.__FF_proj_out__."),

    (r"^interactive_object_pointer_proj\.layers\.0\.",  r"interactive_object_pointer_proj.__FF_proj_in__."),
    (r"^interactive_object_pointer_proj\.layers\.1\.",  r"interactive_object_pointer_proj.__FF_mid_0__."),
    (r"^interactive_object_pointer_proj\.layers\.2\.",  r"interactive_object_pointer_proj.__FF_proj_out__."),

    # ---------------- Memory encoder mask downsampler ----------------
    # Original sequential [conv(0), ln(1), gelu(2), conv(3), ln(4), gelu(5), conv(6), ln(7), gelu(8),
    #                     conv(9), ln(10), gelu(11), conv(12)]
    # HF: layers.{0..3}.{conv, layer_norm} + final_conv.
    (r"^memory_encoder\.mask_downsampler\.encoder\.0\.",   r"memory_encoder.mask_downsampler.layers.0.conv."),
    (r"^memory_encoder\.mask_downsampler\.encoder\.1\.",   r"memory_encoder.mask_downsampler.layers.0.layer_norm."),
    (r"^memory_encoder\.mask_downsampler\.encoder\.3\.",   r"memory_encoder.mask_downsampler.layers.1.conv."),
    (r"^memory_encoder\.mask_downsampler\.encoder\.4\.",   r"memory_encoder.mask_downsampler.layers.1.layer_norm."),
    (r"^memory_encoder\.mask_downsampler\.encoder\.6\.",   r"memory_encoder.mask_downsampler.layers.2.conv."),
    (r"^memory_encoder\.mask_downsampler\.encoder\.7\.",   r"memory_encoder.mask_downsampler.layers.2.layer_norm."),
    (r"^memory_encoder\.mask_downsampler\.encoder\.9\.",   r"memory_encoder.mask_downsampler.layers.3.conv."),
    (r"^memory_encoder\.mask_downsampler\.encoder\.10\.",  r"memory_encoder.mask_downsampler.layers.3.layer_norm."),
    (r"^memory_encoder\.mask_downsampler\.encoder\.12\.",  r"memory_encoder.mask_downsampler.final_conv."),

    # ---------------- Memory encoder fuser ----------------
    (r"^memory_encoder\.memory_fuser\.layers\.(\d+)\.gamma$",   r"memory_encoder.memory_fuser.layers.\1.scale"),
    (r"^memory_encoder\.memory_fuser\.layers\.(\d+)\.dwconv\.", r"memory_encoder.memory_fuser.layers.\1.depthwise_conv."),
    (r"^memory_encoder\.memory_fuser\.layers\.(\d+)\.norm\.",   r"memory_encoder.memory_fuser.layers.\1.layer_norm."),
    (r"^memory_encoder\.memory_fuser\.layers\.(\d+)\.pwconv1\.", r"memory_encoder.memory_fuser.layers.\1.pointwise_conv1."),
    (r"^memory_encoder\.memory_fuser\.layers\.(\d+)\.pwconv2\.", r"memory_encoder.memory_fuser.layers.\1.pointwise_conv2."),

    # Resolve the temporary FF placeholders (these always run last because they only match the sentinels).
    (r"__FF_proj_in__",   r"proj_in"),
    (r"__FF_proj_out__",  r"proj_out"),
    (r"__FF_mid_0__",     r"layers.0"),
]
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: list[str]) -> dict[str, str]:
    """Map original SAM3.1 multiplex state-dict keys to `Sam31TrackerVideoModel` keys.

    The mapping is purely textual:
      1. Apply the top-level prefix regexes in `ORIGINAL_TO_CONVERTED_KEY_MAPPING`.
      2. Apply the literal submodule renames in `TRACKER_SUBMODULE_RENAMES`.
      3. Apply the deeper regex rewrites in `TRACKER_REGEX_RENAMES`.

    Keys that should be dropped (RoPE buffers, detector head) are prefixed with `__DROP__.`; the caller
    filters them out before loading into the model.
    """
    output_dict = {}
    if not state_dict_keys:
        return output_dict

    old_text = "\n".join(state_dict_keys)
    new_text = old_text

    for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)
    for needle, replacement in TRACKER_SUBMODULE_RENAMES:
        new_text = new_text.replace(needle, replacement)
    for pattern, replacement in TRACKER_REGEX_RENAMES:
        new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)

    output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def split_qkv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split the vision-backbone combined QKV projections into separate Q / K / V tensors.

    The SAM3.1 multiplex checkpoint already stores the rest of the model with split Q/K/V (memory
    attention, mask decoders), so this only handles the ViT trunk where the keys come in as
    `vision_encoder.backbone.layers.{i}.attention.qkv.{weight,bias}`.
    """
    keys_to_split = [k for k in list(state_dict) if ".attention.qkv." in k]
    for key in keys_to_split:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace(".qkv.", ".q_proj.")] = q
        state_dict[key.replace(".qkv.", ".k_proj.")] = k
        state_dict[key.replace(".qkv.", ".v_proj.")] = v
    return state_dict


def concat_point_embeddings(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concat the four `prompt_encoder.point_embeddings.{i}.weight` tensors into one `point_embed.weight`."""
    parts = [state_dict.pop(f"prompt_encoder.point_embeddings.{i}.weight", None) for i in range(4)]
    if all(p is not None for p in parts):
        state_dict["prompt_encoder.point_embed.weight"] = torch.cat(parts, dim=0)
    elif any(p is not None for p in parts):
        # Restore any popped tensor we didn't end up using so the caller can flag the missing pieces.
        for i, p in enumerate(parts):
            if p is not None:
                state_dict[f"prompt_encoder.point_embeddings.{i}.weight"] = p
    return state_dict


def add_identity_memory_projection(state_dict: dict[str, torch.Tensor], hidden_dim: int) -> dict[str, torch.Tensor]:
    """Fill in `memory_encoder.projection` with identity weights / zero bias.

    The original `SimpleMaskEncoder.out_proj` is `nn.Identity()` whereas the HF `Sam2VideoMemoryEncoder`
    (parent of the SAM3.1 memory encoder) always allocates a `Conv2d(hidden_dim, hidden_dim, kernel_size=1)`
    `projection` layer. Initializing it to identity reproduces the original architecture exactly.
    """
    weight_key = "memory_encoder.projection.weight"
    bias_key = "memory_encoder.projection.bias"
    if weight_key in state_dict:
        return state_dict
    state_dict[weight_key] = torch.eye(hidden_dim).reshape(hidden_dim, hidden_dim, 1, 1)
    state_dict[bias_key] = torch.zeros(hidden_dim)
    return state_dict


def reduce_occlusion_spatial_embedding(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pass-through for the per-slot SAM3.1 occlusion embedding.

    `Sam31TrackerVideoModel.occlusion_spatial_embedding_parameter` now keeps Meta SAM3.1's
    full `(multiplex_count, hidden_dim)` per-slot layout (see the parameter's construction
    and the per-slot weighted-sum applied in `_encode_new_memory`). Earlier conversions
    collapsed the slot axis by averaging, which both rescaled the learned weights by
    `1 / multiplex_count` and dropped the padding-slot contribution that Meta always adds
    to every bucket. We keep the function as a no-op (and as the place to surface a
    one-time warning) so older callers continue to work without behavior change.
    """
    key = "occlusion_spatial_embedding_parameter"
    tensor = state_dict.get(key)
    if tensor is None:
        return state_dict
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        logger.warning(
            f"`{key}` is shape (1, {tensor.shape[-1]}) — looks like it came from the older "
            "conversion that averaged the per-slot rows. Re-run the converter from the original "
            "Meta checkpoint to recover the per-slot `(multiplex_count, mem_dim)` tensor; the "
            "current `_encode_new_memory` expects the full per-slot layout."
        )
    return state_dict


def load_original_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load the original SAM3.1 multiplex checkpoint, unwrapping the optional `model` / `state_dict` keys."""
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


def get_sam3_1_tracker_video_config() -> Sam31TrackerVideoConfig:
    """Return the default `Sam31TrackerVideoConfig` matching the published SAM3.1 multiplex checkpoint."""
    return Sam31TrackerVideoConfig()


def convert_sam3_1_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config: Sam31TrackerVideoConfig | None = None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
) -> None:
    """Convert a SAM3.1 multiplex checkpoint to a HuggingFace-format `Sam31TrackerVideoModel` directory."""
    os.makedirs(output_path, exist_ok=True)

    if config is None:
        config = get_sam3_1_tracker_video_config()

    config.architectures = ["Sam31TrackerVideoModel"]
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
        if new_key == "vision_encoder.backbone.embeddings.position_embeddings":
            # ViT pos_embed comes with a leading CLS token slot (1, num_patches + 1, hidden); the HF
            # backbone allocates only `num_patches` slots, so strip the first one.
            tensor = tensor[:, 1:, :]
        state_dict_new[new_key] = tensor

    print(f"Dropped {len(dropped_keys)} detector / rope buffer keys (expected for the standalone PVS tracker)")

    state_dict_new = concat_point_embeddings(state_dict_new)
    state_dict_new = split_qkv(state_dict_new)
    state_dict_new = reduce_occlusion_spatial_embedding(state_dict_new)
    state_dict_new = add_identity_memory_projection(state_dict_new, hidden_dim=config.vision_config.fpn_hidden_size)

    del state_dict_old
    gc.collect()

    print("Loading weights into Sam31TrackerVideoModel...")
    model = Sam31TrackerVideoModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys ({len(missing_keys)}):")
        for key in missing_keys:
            logger.warning(f"  - {key}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys:
            logger.warning(f"  - {key}")

    # Expected to be missing (no source in the original checkpoint):
    # * `no_memory_positional_encoding`  — SAM3.1 doesn't materialize this; left at zero init.
    # * `backbone.layers.{i}.rotary_emb.rope_embeddings_{cos,sin}`  — recomputed at init.
    # * `no_object_pointer`  — legacy SAM3 constant pointer; kept as a zero buffer for
    #   backward compat but no longer used at runtime (Meta SAM3.1 uses `no_obj_ptr_linear`
    #   instead — see `_blend_no_object_pointer`).
    # Loaded when `config.add_output_suppression_embeddings` is True (default):
    # * `output_valid_embed`, `output_invalid_embed` — from `tracker.model.output_{valid,invalid}_embed`.
    # Loaded from Meta SAM3.1 directly (previously dropped):
    # * `no_obj_ptr_linear.{weight,bias}`  — Meta `use_linear_no_obj_ptr=True` head.
    # * `occlusion_spatial_embedding_parameter` keeps the full `(multiplex_count, mem_dim)`
    #   per-slot layout (previously collapsed via averaging).

    print(f"Saving converted model to {output_path}")
    model.save_pretrained(output_path)

    print("Creating and saving processor...")
    from transformers.image_utils import PILImageResampling

    image_processor = Sam3ImageProcessor()
    video_processor = Sam2VideoVideoProcessor(
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], size={"height": 1008, "width": 1008}
    )
    # Match Meta `cv2.resize(..., cv2.INTER_CUBIC)` in `facebook_sam3/sam3/model/io_utils.py`.
    video_processor.resample = PILImageResampling.BICUBIC
    processor = Sam31TrackerVideoProcessor(image_processor=image_processor, video_processor=video_processor)
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
    reloaded = Sam31TrackerVideoModel.from_pretrained(output_path)
    param_count = sum(p.numel() for p in reloaded.parameters())
    print(f"\u2713 Sam31TrackerVideoModel.from_pretrained succeeded with {param_count:,} parameters")
    del reloaded
    gc.collect()

    print("\n" + "=" * 80)
    print("Conversion finished!")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print("\nTo test the model, you can run:")
    print(">>> from transformers import Sam31TrackerVideoModel")
    print(f">>> model = Sam31TrackerVideoModel.from_pretrained('{output_path}')")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SAM3.1 multiplex checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original SAM3.1 .pt file")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the converted model")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the Hub")
    parser.add_argument("--repo_id", type=str, default=None, help="Hub repo id (e.g. `facebook/sam3.1`)")
    args = parser.parse_args()

    convert_sam3_1_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
