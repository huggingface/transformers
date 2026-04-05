# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Convert the original SAM 3.1 multiplex checkpoint to the Hugging Face format."""

import argparse
import importlib
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from transformers.models.sam3_1_video.configuration_sam3_1_video import Sam3_1VideoConfig
from transformers.models.sam3_1_video.modeling_sam3_1_video import Sam3_1VideoModel


UNUSED_PREFIXES = (
    "detector.transformer.",
    "detector.geometry_encoder.",
    "detector.segmentation_head.",
    "detector.dot_prod_scoring.",
    "detector.backbone.language_backbone.",
)

STRING_REPLACEMENTS = {
    "interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix": (
        "interactive_sam_prompt_encoder.shared_embedding.positional_embedding"
    ),
    "interactive_sam_prompt_encoder.mask_downscaling.0": "interactive_sam_prompt_encoder.mask_embed.conv1",
    "interactive_sam_prompt_encoder.mask_downscaling.1": "interactive_sam_prompt_encoder.mask_embed.layer_norm1",
    "interactive_sam_prompt_encoder.mask_downscaling.3": "interactive_sam_prompt_encoder.mask_embed.conv2",
    "interactive_sam_prompt_encoder.mask_downscaling.4": "interactive_sam_prompt_encoder.mask_embed.layer_norm2",
    "interactive_sam_prompt_encoder.mask_downscaling.6": "interactive_sam_prompt_encoder.mask_embed.conv3",
    "interactive_sam_mask_decoder.output_upscaling.0": "interactive_sam_mask_decoder.upscale_conv1",
    "interactive_sam_mask_decoder.output_upscaling.1": "interactive_sam_mask_decoder.upscale_layer_norm",
    "interactive_sam_mask_decoder.output_upscaling.3": "interactive_sam_mask_decoder.upscale_conv2",
    "interactive_sam_mask_decoder.iou_prediction_head.layers.0": "interactive_sam_mask_decoder.iou_prediction_head.proj_in",
    "interactive_sam_mask_decoder.iou_prediction_head.layers.1": "interactive_sam_mask_decoder.iou_prediction_head.layers.0",
    "interactive_sam_mask_decoder.iou_prediction_head.layers.2": "interactive_sam_mask_decoder.iou_prediction_head.proj_out",
}

FPN_LAYER_REPLACEMENTS = {
    ".dconv_2x2_0.": ".scale_layers.0.",
    ".dconv_2x2_1.": ".scale_layers.2.",
    ".dconv_2x2.": ".scale_layers.0.",
    ".conv_1x1.": ".proj1.",
    ".conv_3x3.": ".proj2.",
}

MASK_DECODER_TRANSFORMER_REPLACEMENTS = {
    ".self_attn.out_proj.": ".self_attn.o_proj.",
    ".cross_attn_token_to_image.out_proj.": ".cross_attn_token_to_image.o_proj.",
    ".cross_attn_image_to_token.out_proj.": ".cross_attn_image_to_token.o_proj.",
    ".final_attn_token_to_image.out_proj.": ".final_attn_token_to_image.o_proj.",
    ".norm1.": ".layer_norm1.",
    ".norm2.": ".layer_norm2.",
    ".norm3.": ".layer_norm3.",
    ".norm4.": ".layer_norm4.",
    ".norm_final_attn.": ".layer_norm_final_attn.",
}


def maybe_download_checkpoint(checkpoint_path: str | None) -> str:
    if checkpoint_path is not None:
        return checkpoint_path
    if hf_hub_download is None:
        raise ImportError("huggingface_hub is required to auto-download `facebook/sam3.1`.")
    return hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt")


def _convert_interactive_point_embeddings(
    state_dict: dict[str, torch.Tensor], converted_state_dict: dict[str, torch.Tensor]
):
    point_embedding_keys = [
        f"interactive_sam_prompt_encoder.point_embeddings.{idx}.weight"
        for idx in range(4)
        if f"interactive_sam_prompt_encoder.point_embeddings.{idx}.weight" in state_dict
    ]
    if point_embedding_keys:
        converted_state_dict["interactive_sam_prompt_encoder.point_embed.weight"] = torch.cat(
            [state_dict[key] for key in point_embedding_keys], dim=0
        )


def _rename_backbone_fpn_key(key: str) -> str:
    for source, target in FPN_LAYER_REPLACEMENTS.items():
        key = key.replace(source, target)
    return key


def _rename_mask_decoder_transformer_key(key: str) -> str:
    for source, target in MASK_DECODER_TRANSFORMER_REPLACEMENTS.items():
        key = key.replace(source, target)
    return key


def _convert_backbone_trunk_key(key: str, value: torch.Tensor, converted_state_dict: dict[str, torch.Tensor]) -> bool:
    if key == "backbone.vision_backbone.trunk.pos_embed":
        if value.ndim == 3 and value.shape[1] > 1:
            num_tokens_without_cls = value.shape[1] - 1
            if int(num_tokens_without_cls**0.5) ** 2 == num_tokens_without_cls:
                value = value[:, 1:]
        converted_state_dict["backbone.vision_backbone.trunk.embeddings.position_embeddings"] = value
        return True

    if key.startswith("backbone.vision_backbone.trunk.patch_embed."):
        converted_state_dict[
            key.replace(
                "backbone.vision_backbone.trunk.patch_embed.proj",
                "backbone.vision_backbone.trunk.embeddings.patch_embeddings.projection",
            )
        ] = value
        return True

    if key.startswith("backbone.vision_backbone.trunk.norm."):
        converted_state_dict[key.replace(".trunk.norm.", ".trunk.layer_norm.")] = value
        return True

    if key.startswith("backbone.vision_backbone.trunk.ln_pre."):
        converted_state_dict[key.replace(".trunk.ln_pre.", ".trunk.layer_norm.")] = value
        return True

    if ".attn.freqs_cis" in key:
        return True

    if key.startswith("backbone.vision_backbone.trunk.blocks."):
        if ".attn.qkv." in key:
            prefix, suffix = key.split(".attn.qkv.")
            prefix = prefix.replace(".blocks.", ".layers.")
            q_tensor, k_tensor, v_tensor = value.chunk(3, dim=0)
            converted_state_dict[f"{prefix}.attention.q_proj.{suffix}"] = q_tensor
            converted_state_dict[f"{prefix}.attention.k_proj.{suffix}"] = k_tensor
            converted_state_dict[f"{prefix}.attention.v_proj.{suffix}"] = v_tensor
            return True

        new_key = key.replace(".blocks.", ".layers.")
        new_key = new_key.replace(".attn.proj.", ".attention.o_proj.")
        new_key = new_key.replace(".norm1.", ".layer_norm1.")
        new_key = new_key.replace(".norm2.", ".layer_norm2.")
        converted_state_dict[new_key] = value
        return True

    return False


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    normalized_state_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith(UNUSED_PREFIXES):
            continue

        if key.startswith("tracker.model."):
            new_key = key.removeprefix("tracker.model.")
        elif key.startswith("detector.backbone."):
            new_key = "backbone." + key.removeprefix("detector.backbone.")
        else:
            continue

        if new_key.startswith("backbone.vision_backbone.trunk."):
            handled = _convert_backbone_trunk_key(new_key, value, normalized_state_dict)
            if handled:
                continue

        if (
            ".vision_backbone.convs." in new_key
            or ".vision_backbone.interactive_convs." in new_key
            or ".vision_backbone.propagation_convs." in new_key
        ):
            new_key = _rename_backbone_fpn_key(new_key)

        if new_key.startswith("sam_mask_decoder.transformer."):
            new_key = new_key.replace("sam_mask_decoder.transformer.", "sam_mask_decoder.transformer.inner.")

        if new_key.startswith("interactive_sam_mask_decoder.transformer.") or new_key.startswith(
            "sam_mask_decoder.transformer.inner."
        ):
            new_key = _rename_mask_decoder_transformer_key(new_key)
            new_key = new_key.replace(".mlp.lin1.", ".mlp.proj_in.")
            new_key = new_key.replace(".mlp.lin2.", ".mlp.proj_out.")

        for source, target in STRING_REPLACEMENTS.items():
            new_key = new_key.replace(source, target)

        if (
            new_key.startswith("interactive_sam_mask_decoder.transformer.layers.")
            or new_key.startswith("sam_mask_decoder.transformer.inner.layers.")
        ) and ".mlp.layers." in new_key:
            match = re.search(r"\.mlp\.layers\.(\d+)\.", new_key)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx == 0:
                    new_key = new_key.replace(".mlp.layers.0.", ".mlp.proj_in.")
                elif layer_idx == 1:
                    new_key = new_key.replace(".mlp.layers.1.", ".mlp.proj_out.")

        if new_key.startswith("interactive_sam_mask_decoder.output_hypernetworks_mlps.") and ".layers." in new_key:
            match = re.search(r"\.output_hypernetworks_mlps\.(\d+)\.layers\.(\d+)\.", new_key)
            if match:
                layer_idx = int(match.group(2))
                if layer_idx == 0:
                    new_key = new_key.replace(".layers.0.", ".proj_in.")
                elif layer_idx == 1:
                    new_key = new_key.replace(".layers.1.", ".layers.0.")
                elif layer_idx == 2:
                    new_key = new_key.replace(".layers.2.", ".proj_out.")

        if new_key.startswith("interactive_sam_mask_decoder.pred_obj_score_head.layers."):
            match = re.search(r"\.pred_obj_score_head\.layers\.(\d+)\.", new_key)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx == 0:
                    new_key = new_key.replace(".layers.0.", ".proj_in.")
                elif layer_idx == 1:
                    new_key = new_key.replace(".layers.1.", ".layers.0.")
                elif layer_idx == 2:
                    new_key = new_key.replace(".layers.2.", ".proj_out.")

        normalized_state_dict[new_key] = value

    converted_state_dict: dict[str, torch.Tensor] = {}
    _convert_interactive_point_embeddings(normalized_state_dict, converted_state_dict)

    for key, value in normalized_state_dict.items():
        if key.startswith("interactive_sam_prompt_encoder.point_embeddings."):
            continue
        converted_state_dict[key] = value

    return converted_state_dict


def compare_tensors(
    name: str, reference: torch.Tensor, converted: torch.Tensor, atol: float = 1e-3, rtol: float = 1e-3
):
    if not torch.allclose(reference, converted, atol=atol, rtol=rtol):
        diff = (reference - converted).abs().max().item()
        raise AssertionError(f"{name} mismatch. max_abs_diff={diff}")


def _register_upstream_stubbed_modules(sam3_repo_path: str):
    sam3_root = Path(sam3_repo_path) / "sam3"
    if not sam3_root.exists():
        raise FileNotFoundError(f"Could not find upstream SAM 3 repo at {sam3_root}")

    sam3_pkg = types.ModuleType("sam3")
    sam3_pkg.__path__ = [str(sam3_root)]
    sam3_model_pkg = types.ModuleType("sam3.model")
    sam3_model_pkg.__path__ = [str(sam3_root / "model")]
    sam3_sam_pkg = types.ModuleType("sam3.sam")
    sam3_sam_pkg.__path__ = [str(sam3_root / "sam")]
    sam3_perflib_pkg = types.ModuleType("sam3.perflib")
    sam3_perflib_pkg.__path__ = [str(sam3_root / "perflib")]

    sys.modules["sam3"] = sam3_pkg
    sys.modules["sam3.model"] = sam3_model_pkg
    sys.modules["sam3.sam"] = sam3_sam_pkg
    sys.modules["sam3.perflib"] = sam3_perflib_pkg

    triton = types.ModuleType("triton")
    triton.__version__ = "0.0"
    triton.jit = lambda fn=None, **kwargs: (lambda value: value)(fn) if fn is not None else (lambda value: value)
    triton.autotune = lambda *args, **kwargs: (lambda fn: fn)

    class _Config:
        def __init__(self, *args, **kwargs):
            pass

    triton.Config = _Config
    triton.cdiv = lambda a, b: (a + b - 1) // b

    triton_language = types.ModuleType("triton.language")
    triton_language.constexpr = object()
    triton_language.dtype = type("dtype", (), {})
    triton.language = triton_language

    sys.modules["triton"] = triton
    sys.modules["triton.language"] = triton_language

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    def trunc_normal_(
        tensor: torch.Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = -2.0,
        b: float = 2.0,
    ) -> torch.Tensor:
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    timm = types.ModuleType("timm")
    timm_layers = types.ModuleType("timm.layers")
    timm_models = types.ModuleType("timm.models")
    timm_models_layers = types.ModuleType("timm.models.layers")
    for module in (timm_layers, timm_models_layers):
        module.DropPath = DropPath
        module.trunc_normal_ = trunc_normal_

    timm.layers = timm_layers
    timm.models = timm_models
    timm.models.layers = timm_models_layers

    sys.modules["timm"] = timm
    sys.modules["timm.layers"] = timm_layers
    sys.modules["timm.models"] = timm_models
    sys.modules["timm.models.layers"] = timm_models_layers

    def _unsupported_torchvision_op(*args, **kwargs):
        raise RuntimeError("A torchvision op required by the upstream SAM 3.1 parity helper was unexpectedly called.")

    class RoIAlign(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            return _unsupported_torchvision_op(*args, **kwargs)

    def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
        if masks.numel() == 0:
            return torch.zeros((0, 4), dtype=masks.dtype, device=masks.device)

        boxes = []
        for mask in masks:
            ys, xs = torch.where(mask > 0)
            if ys.numel() == 0:
                boxes.append(torch.zeros(4, dtype=masks.dtype, device=masks.device))
            else:
                boxes.append(
                    torch.stack([xs.min(), ys.min(), xs.max(), ys.max()]).to(device=masks.device, dtype=masks.dtype)
                )
        return torch.stack(boxes)

    torchvision = types.ModuleType("torchvision")
    torchvision.__version__ = "0.0"
    torchvision.__path__ = []
    torchvision_ops = types.ModuleType("torchvision.ops")
    torchvision_ops.__path__ = []
    torchvision_ops_roi_align = types.ModuleType("torchvision.ops.roi_align")
    torchvision_ops_roi_align.RoIAlign = RoIAlign
    torchvision_ops_roi_align.roi_align = _unsupported_torchvision_op
    torchvision_ops.roi_align = _unsupported_torchvision_op
    torchvision_ops.RoIAlign = RoIAlign
    torchvision_ops.masks_to_boxes = masks_to_boxes
    torchvision.ops = torchvision_ops

    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_transforms.__path__ = []
    torchvision_transforms_functional = types.ModuleType("torchvision.transforms.functional")
    torchvision_transforms_v2 = types.ModuleType("torchvision.transforms.v2")
    torchvision_transforms.functional = torchvision_transforms_functional
    torchvision_transforms.v2 = torchvision_transforms_v2
    torchvision.transforms = torchvision_transforms

    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.ops"] = torchvision_ops
    sys.modules["torchvision.ops.roi_align"] = torchvision_ops_roi_align
    sys.modules["torchvision.transforms"] = torchvision_transforms
    sys.modules["torchvision.transforms.functional"] = torchvision_transforms_functional
    sys.modules["torchvision.transforms.v2"] = torchvision_transforms_v2

    importlib.invalidate_caches()


def _build_original_sam3_1_model(
    checkpoint_path: str,
    sam3_repo_path: str,
    device: str,
):
    _register_upstream_stubbed_modules(sam3_repo_path)

    position_encoding_module = importlib.import_module("sam3.model.position_encoding")
    memory_module = importlib.import_module("sam3.model.memory")
    decoder_module = importlib.import_module("sam3.model.decoder")
    model_misc_module = importlib.import_module("sam3.model.model_misc")
    multiplex_utils_module = importlib.import_module("sam3.model.multiplex_utils")
    necks_module = importlib.import_module("sam3.model.necks")
    vl_combiner_module = importlib.import_module("sam3.model.vl_combiner")
    fused_module = importlib.import_module("sam3.perflib.fused")

    def addmm_act_cpu_fallback(activation, linear, mat1):
        bias = linear.bias.detach()
        weight = linear.weight.detach()
        hidden_states = torch.addmm(bias, mat1.reshape(-1, mat1.shape[-1]), weight.t())
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            hidden_states = torch.relu(hidden_states)
        elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            hidden_states = torch.nn.functional.gelu(hidden_states)
        else:
            raise ValueError(f"Unexpected activation {activation}")
        return hidden_states.view(mat1.shape[:-1] + (hidden_states.shape[-1],))

    fused_module.addmm_act = addmm_act_cpu_fallback

    vitdet_module = importlib.import_module("sam3.model.vitdet")
    video_demo_module = importlib.import_module("sam3.model.video_tracking_multiplex_demo")
    vitdet_module.addmm_act = addmm_act_cpu_fallback

    precompute_resolution = 1008 if torch.device(device).type == "cuda" else None
    position_encoding = position_encoding_module.PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )

    maskmem_backbone = memory_module.SimpleMaskEncoder(
        out_dim=256,
        position_encoding=position_encoding,
        mask_downsampler=memory_module.SimpleMaskDownSampler(
            kernel_size=3,
            stride=2,
            padding=1,
            interpol_size=[1152, 1152],
            multiplex_count=16,
            starting_out_chan=4,
            input_channel_multiplier=2,
        ),
        fuser=memory_module.SimpleFuser(
            layer=memory_module.CXBlock(
                dim=256,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1.0e-06,
                use_dwconv=True,
            ),
            num_layers=2,
        ),
    )

    transformer = model_misc_module.TransformerWrapper(
        encoder=decoder_module.TransformerEncoderDecoupledCrossAttention(
            d_model=256,
            frozen=False,
            pos_enc_at_input=True,
            use_image_in_output=False,
            layer=decoder_module.DecoupledTransformerDecoderLayerv2(
                activation="gelu",
                d_model=256,
                num_heads=8,
                dropout=0.1,
                dim_feedforward=2048,
                pos_enc_at_attn=False,
                pre_norm=True,
                pos_enc_at_cross_attn_keys=True,
                pos_enc_at_cross_attn_queries=False,
                self_attention_rope=decoder_module.SimpleRoPEAttention(
                    d_model=256,
                    num_heads=8,
                    dropout_p=0.1,
                    rope_theta=10000.0,
                    feat_sizes=[72, 72],
                    use_fa3=False,
                    use_rope_real=False,
                ),
                cross_attention_rope=decoder_module.SimpleRoPEAttention(
                    d_model=256,
                    num_heads=8,
                    dropout_p=0.1,
                    rope_theta=10000.0,
                    feat_sizes=[72, 72],
                    rope_k_repeat=True,
                    use_fa3=False,
                    use_rope_real=False,
                ),
            ),
            num_layers=4,
            use_act_checkpoint=False,
            batch_first=True,
        ),
        decoder=None,
        d_model=256,
    )

    vit_backbone = vitdet_module.ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
        use_fa3=False,
        use_rope_real=False,
    )

    tri_neck = necks_module.Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )

    backbone = vl_combiner_module.TriHeadVisionOnly(visual=tri_neck, n_features=256, scalp=0)
    multiplex_controller = multiplex_utils_module.MultiplexController(multiplex_count=16, eval_multiplex_count=16)

    original_model = video_demo_module.Sam3VideoTrackingMultiplexDemo(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=multiplex_controller,
        image_size=1008,
        backbone_stride=14,
        num_maskmem=7,
        use_high_res_features_in_sam=True,
        use_obj_ptrs_in_encoder=True,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_mlp_for_obj_ptr_proj=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        use_no_obj_ptr=True,
        use_linear_no_obj_ptr=True,
        no_obj_embed_spatial=True,
        sincos_tpos_enc=True,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_multimask_token_for_obj_ptr=True,
        num_multimask_outputs=3,
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        sigmoid_scale_for_mem_enc=2.0,
        sigmoid_bias_for_mem_enc=-1.0,
        non_overlap_masks_for_mem_enc=False,
        add_output_suppression_embeddings=True,
        add_object_conditional_embeddings=False,
        condition_as_mask_input=True,
        condition_as_mask_input_fg=1.0,
        condition_as_mask_input_bg=0.0,
        use_maskmem_tpos_v2=True,
        save_image_features=True,
        randomness_fix=True,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        forward_backbone_per_frame_for_eval=True,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        max_cond_frames_in_attn=4,
        is_dynamic_model=True,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        compile_all_components=False,
        use_memory_selection=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = checkpoint["model"] if isinstance(checkpoint.get("model"), dict) else checkpoint

    original_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("tracker.model."):
            original_state_dict[key.removeprefix("tracker.model.")] = value
        elif key.startswith("detector.backbone.language_backbone."):
            continue
        elif key.startswith("detector.backbone."):
            original_state_dict["backbone." + key.removeprefix("detector.backbone.")] = value

    missing_keys, unexpected_keys = original_model.load_state_dict(original_state_dict, strict=False)
    if missing_keys:
        raise RuntimeError(f"Upstream SAM 3.1 model still has missing keys after remapping: {missing_keys[:20]}")
    if unexpected_keys:
        raise RuntimeError(f"Upstream SAM 3.1 model still has unexpected keys after remapping: {unexpected_keys[:20]}")

    return original_model.to(device).eval()


def verify_conversion(
    hf_model: Sam3_1VideoModel,
    checkpoint_path: str,
    sam3_repo_path: str,
    device: str = "cpu",
):
    original_model = _build_original_sam3_1_model(
        checkpoint_path=checkpoint_path,
        sam3_repo_path=sam3_repo_path,
        device=device,
    )
    hf_model = hf_model.to(device).eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    pixel_values = torch.randn(
        1,
        3,
        hf_model.config.image_size,
        hf_model.config.image_size,
        generator=generator,
        device=device,
    )

    with torch.no_grad():
        original_backbone_outputs = original_model.backbone.forward_image(
            pixel_values,
            need_sam3_out=False,
            need_interactive_out=True,
            need_propagation_out=True,
        )
        hf_backbone_outputs = hf_model.get_image_features(
            pixel_values,
            need_sam3_out=False,
            need_interactive_out=True,
            need_propagation_out=True,
        )

        original_interactive_fpn = list(original_backbone_outputs["interactive"]["backbone_fpn"])
        original_interactive_fpn[0].tensors = original_model.interactive_sam_mask_decoder.conv_s0(
            original_interactive_fpn[0].tensors
        )
        original_interactive_fpn[1].tensors = original_model.interactive_sam_mask_decoder.conv_s1(
            original_interactive_fpn[1].tensors
        )
        original_propagation_fpn = list(original_backbone_outputs["sam2_backbone_out"]["backbone_fpn"])
        original_propagation_fpn[0].tensors = original_model.sam_mask_decoder.conv_s0(
            original_propagation_fpn[0].tensors
        )
        original_propagation_fpn[1].tensors = original_model.sam_mask_decoder.conv_s1(
            original_propagation_fpn[1].tensors
        )

        compare_tensors(
            "interactive_backbone_last",
            original_backbone_outputs["interactive"]["vision_features"],
            hf_backbone_outputs.interactive["vision_features"],
        )
        compare_tensors(
            "propagation_backbone_last",
            original_backbone_outputs["sam2_backbone_out"]["vision_features"],
            hf_backbone_outputs.sam2_backbone_out["vision_features"],
        )
        compare_tensors(
            "interactive_backbone_s0",
            original_interactive_fpn[0].tensors,
            hf_backbone_outputs.interactive["backbone_fpn"][0],
        )
        compare_tensors(
            "propagation_backbone_s0",
            original_propagation_fpn[0].tensors,
            hf_backbone_outputs.sam2_backbone_out["backbone_fpn"][0],
        )

        prompt_coords = torch.tensor([[[256.0, 256.0]]], device=device)
        prompt_labels = torch.tensor([[1]], dtype=torch.int64, device=device)

        original_sparse_embeddings, original_dense_embeddings = original_model.interactive_sam_prompt_encoder(
            points=(prompt_coords, prompt_labels), boxes=None, masks=None
        )
        original_low_res_masks, original_iou_scores, _, original_object_score_logits = (
            original_model.interactive_sam_mask_decoder(
                image_embeddings=original_backbone_outputs["interactive"]["vision_features"],
                image_pe=original_model.interactive_sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=original_sparse_embeddings,
                dense_prompt_embeddings=original_dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=[original_interactive_fpn[0].tensors, original_interactive_fpn[1].tensors],
            )
        )

        hf_sparse_embeddings, hf_dense_embeddings = hf_model.interactive_sam_prompt_encoder(
            input_points=prompt_coords.unsqueeze(1),
            input_labels=prompt_labels.unsqueeze(1),
            input_boxes=None,
            input_masks=None,
        )
        hf_low_res_masks, hf_iou_scores, _, hf_object_score_logits = hf_model.interactive_sam_mask_decoder(
            image_embeddings=hf_backbone_outputs.interactive["vision_features"],
            image_positional_embeddings=hf_model.get_image_wide_positional_embeddings(
                batch_size=1,
                device=device,
                dtype=hf_backbone_outputs.interactive["vision_features"].dtype,
            ),
            sparse_prompt_embeddings=hf_sparse_embeddings,
            dense_prompt_embeddings=hf_dense_embeddings,
            multimask_output=False,
            high_resolution_features=list(hf_backbone_outputs.interactive["backbone_fpn"][:-1]),
        )

        compare_tensors("interactive_low_res_masks", original_low_res_masks, hf_low_res_masks.squeeze(1))
        compare_tensors("interactive_iou_scores", original_iou_scores, hf_iou_scores.squeeze(1))
        compare_tensors(
            "interactive_object_score_logits",
            original_object_score_logits,
            hf_object_score_logits.squeeze(1),
        )


def convert_checkpoint(
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    verify: bool = True,
    sam3_repo_path: str = "/Users/nielsrogge/Documents/python_projecten/sam3",
    push_to_hub: bool = False,
    repo_id: str | None = None,
):
    if push_to_hub and repo_id is None:
        raise ValueError("repo_id must be provided when push_to_hub=True")

    checkpoint_path = maybe_download_checkpoint(checkpoint_path)
    print(f"Loading original SAM 3.1 checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    converted_state_dict = convert_state_dict(state_dict)
    print(f"Remapped {len(converted_state_dict)} checkpoint tensors to the Transformers SAM 3.1 format.")

    model = Sam3_1VideoModel(Sam3_1VideoConfig())
    missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:20]}")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:20]}")
    if not missing_keys and not unexpected_keys:
        print("Checkpoint loaded into Sam3_1VideoModel without missing or unexpected keys.")

    if verify:
        print(f"Running parity verification against the upstream SAM 3.1 implementation in {sam3_repo_path}")
        verify_conversion(model, checkpoint_path=checkpoint_path, sam3_repo_path=sam3_repo_path)
        print("Verification passed: the converted Transformers model matches the upstream SAM 3.1 implementation.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Saved converted Transformers checkpoint to {output_dir}")

    if push_to_hub:
        print(f"Pushing converted Transformers checkpoint to the Hub: {repo_id}")
        model.push_to_hub(repo_id)
        print(f"Successfully pushed the converted SAM 3.1 video checkpoint to {repo_id}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_verify", dest="verify", action="store_false")
    parser.set_defaults(verify=True)
    parser.add_argument("--sam3_repo_path", type=str, default="/Users/nielsrogge/Documents/python_projecten/sam3")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", type=str, default=None)
    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        verify=args.verify,
        sam3_repo_path=args.sam3_repo_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )
