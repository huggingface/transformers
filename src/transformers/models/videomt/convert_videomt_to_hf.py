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
"""Convert official VidEoMT checkpoints from https://huggingface.co/tue-mps/VidEoMT to HF format.

URL of the original Github implementation: https://github.com/tue-mps/VidEoMT. We have cloned it locally at /Users/nielsrogge/Documents/python_projecten/videomt.

The easiest way to verify conversion is by using print statements within both the original implementation and within the converted model.

To run:

```bash
# Single checkpoint (image-size and num-frames auto-derived from registry)
python src/transformers/models/videomt/convert_videomt_to_hf.py --checkpoint-filename yt_2019_vit_small_52.8.pth --verify

# All supported DINOv2 checkpoints
python src/transformers/models/videomt/convert_videomt_to_hf.py --all --push-to-hub
```
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from transformers import VideomtConfig, VideomtForUniversalSegmentation


MODEL_REPO_ID = "tue-mps/VidEoMT"

# fmt: off
CHECKPOINT_CONFIGS = {
    "yt_2019_vit_small_52.8.pth":   {"image_size": 640,  "num_frames": 2, "hub_name": "videomt-dinov2-small-ytvis2019", "dataset": "ytvis_2019"},
    "yt_2019_vit_base_58.2.pth":    {"image_size": 640,  "num_frames": 2, "hub_name": "videomt-dinov2-base-ytvis2019",  "dataset": "ytvis_2019"},
    "yt_2019_vit_large_68.6.pth":   {"image_size": 640,  "num_frames": 2, "hub_name": "videomt-dinov2-large-ytvis2019", "dataset": "ytvis_2019"},
    "yt_2021_vit_large_63.1.pth":   {"image_size": 640,  "num_frames": 2, "hub_name": "videomt-dinov2-large-ytvis2021", "dataset": "ytvis_2021"},
    "yt_2022_vit_large_42.6.pth":   {"image_size": 640,  "num_frames": 2, "hub_name": "videomt-dinov2-large-ytvis2022", "dataset": "ytvis_2021"},
    "ovis_vit_large_52.5.pth":      {"image_size": 640,  "num_frames": 2, "hub_name": "videomt-dinov2-large-ovis",      "dataset": "ovis"},
    "vipseg_vit_large_55.2.pth":    {"image_size": 640,  "num_frames": 2, "hub_name": "videomt-dinov2-large-vipseg",    "dataset": "vipseg"},
    "vspw_vit_large_95.0_64.9.pth": {"image_size": 1280, "num_frames": 2, "hub_name": "videomt-dinov2-large-vspw",      "dataset": "vipseg"},
}

YTVIS_2019_ID2LABEL = {
    0: "person", 1: "giant_panda", 2: "lizard", 3: "parrot", 4: "skateboard",
    5: "sedan", 6: "ape", 7: "dog", 8: "snake", 9: "monkey",
    10: "hand", 11: "rabbit", 12: "duck", 13: "cat", 14: "cow",
    15: "fish", 16: "train", 17: "horse", 18: "turtle", 19: "bear",
    20: "motorbike", 21: "giraffe", 22: "leopard", 23: "fox", 24: "deer",
    25: "owl", 26: "surfboard", 27: "airplane", 28: "truck", 29: "zebra",
    30: "tiger", 31: "elephant", 32: "snowboard", 33: "boat", 34: "shark",
    35: "mouse", 36: "frog", 37: "eagle", 38: "earless_seal", 39: "tennis_racket",
}

YTVIS_2021_ID2LABEL = {
    0: "airplane", 1: "bear", 2: "bird", 3: "boat", 4: "car",
    5: "cat", 6: "cow", 7: "deer", 8: "dog", 9: "duck",
    10: "earless_seal", 11: "elephant", 12: "fish", 13: "flying_disc", 14: "fox",
    15: "frog", 16: "giant_panda", 17: "giraffe", 18: "horse", 19: "leopard",
    20: "lizard", 21: "monkey", 22: "motorbike", 23: "mouse", 24: "parrot",
    25: "person", 26: "rabbit", 27: "shark", 28: "skateboard", 29: "snake",
    30: "snowboard", 31: "squirrel", 32: "surfboard", 33: "tennis_racket", 34: "tiger",
    35: "train", 36: "truck", 37: "turtle", 38: "whale", 39: "zebra",
}

OVIS_ID2LABEL = {
    0: "Person", 1: "Bird", 2: "Cat", 3: "Dog", 4: "Horse",
    5: "Sheep", 6: "Cow", 7: "Elephant", 8: "Bear", 9: "Zebra",
    10: "Giraffe", 11: "Poultry", 12: "Giant_panda", 13: "Lizard", 14: "Parrot",
    15: "Monkey", 16: "Rabbit", 17: "Tiger", 18: "Fish", 19: "Turtle",
    20: "Bicycle", 21: "Motorcycle", 22: "Airplane", 23: "Boat", 24: "Vehical",
}

VIPSEG_ID2LABEL = {
    0: "wall", 1: "ceiling", 2: "door", 3: "stair", 4: "ladder",
    5: "escalator", 6: "Playground_slide", 7: "handrail_or_fence", 8: "window", 9: "rail",
    10: "goal", 11: "pillar", 12: "pole", 13: "floor", 14: "ground",
    15: "grass", 16: "sand", 17: "athletic_field", 18: "road", 19: "path",
    20: "crosswalk", 21: "building", 22: "house", 23: "bridge", 24: "tower",
    25: "windmill", 26: "well_or_well_lid", 27: "other_construction", 28: "sky", 29: "mountain",
    30: "stone", 31: "wood", 32: "ice", 33: "snowfield", 34: "grandstand",
    35: "sea", 36: "river", 37: "lake", 38: "waterfall", 39: "water",
    40: "billboard_or_Bulletin_Board", 41: "sculpture", 42: "pipeline", 43: "flag", 44: "parasol_or_umbrella",
    45: "cushion_or_carpet", 46: "tent", 47: "roadblock", 48: "car", 49: "bus",
    50: "truck", 51: "bicycle", 52: "motorcycle", 53: "wheeled_machine", 54: "ship_or_boat",
    55: "raft", 56: "airplane", 57: "tyre", 58: "traffic_light", 59: "lamp",
    60: "person", 61: "cat", 62: "dog", 63: "horse", 64: "cattle",
    65: "other_animal", 66: "tree", 67: "flower", 68: "other_plant", 69: "toy",
    70: "ball_net", 71: "backboard", 72: "skateboard", 73: "bat", 74: "ball",
    75: "cupboard_or_showcase_or_storage_rack", 76: "box", 77: "traveling_case_or_trolley_case", 78: "basket", 79: "bag_or_package",
    80: "trash_can", 81: "cage", 82: "plate", 83: "tub_or_bowl_or_pot", 84: "bottle_or_cup",
    85: "barrel", 86: "fishbowl", 87: "bed", 88: "pillow", 89: "table_or_desk",
    90: "chair_or_seat", 91: "bench", 92: "sofa", 93: "shelf", 94: "bathtub",
    95: "gun", 96: "commode", 97: "roaster", 98: "other_machine", 99: "refrigerator",
    100: "washing_machine", 101: "Microwave_oven", 102: "fan", 103: "curtain", 104: "textiles",
    105: "clothes", 106: "painting_or_poster", 107: "mirror", 108: "flower_pot_or_vase", 109: "clock",
    110: "book", 111: "tool", 112: "blackboard", 113: "tissue", 114: "screen_or_television",
    115: "computer", 116: "printer", 117: "Mobile_phone", 118: "keyboard", 119: "other_electronic_product",
    120: "fruit", 121: "food", 122: "instrument", 123: "train",
}

DATASET_TO_ID2LABEL = {
    "ytvis_2019": YTVIS_2019_ID2LABEL,
    "ytvis_2021": YTVIS_2021_ID2LABEL,
    "ovis": OVIS_ID2LABEL,
    "vipseg": VIPSEG_ID2LABEL,
}
# fmt: on


def infer_num_attention_heads(checkpoint_filename: str, hidden_size: int) -> int:
    if "vit_small" in checkpoint_filename:
        return 6
    if "vit_base" in checkpoint_filename:
        return 12
    if "vit_large" in checkpoint_filename:
        return 16
    if hidden_size % 64 == 0:
        return hidden_size // 64
    raise ValueError(f"Could not infer num_attention_heads from checkpoint name '{checkpoint_filename}'.")


def infer_videomt_config(
    state_dict: dict[str, torch.Tensor], checkpoint_filename: str, image_size: int, num_frames: int
):
    hidden_size = state_dict["backbone.encoder.backbone.cls_token"].shape[-1]
    num_hidden_layers = (
        max(
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"backbone\.encoder\.backbone\.blocks\.(\d+)\.norm1\.weight", key))
        )
        + 1
    )

    return VideomtConfig(
        hidden_size=hidden_size,
        num_attention_heads=infer_num_attention_heads(checkpoint_filename, hidden_size),
        mlp_ratio=state_dict["backbone.encoder.backbone.blocks.0.mlp.fc1.weight"].shape[0] // hidden_size,
        image_size=image_size,
        patch_size=state_dict["backbone.encoder.backbone.patch_embed.proj.weight"].shape[-1],
        num_register_tokens=state_dict["backbone.encoder.backbone.reg_token"].shape[1],
        num_hidden_layers=num_hidden_layers,
        num_queries=state_dict["backbone.q.weight"].shape[0],
        num_blocks=state_dict["backbone.attn_mask_probs"].shape[0],
        num_labels=state_dict["backbone.class_head.weight"].shape[0] - 1,
        num_frames=num_frames,
    )


def infer_backbone_model_name(checkpoint_filename: str) -> str:
    # Official VidEoMT configs point to timm DINOv2 register-token backbones, e.g.
    # `vit_small_patch14_reg4_dinov2` in `configs/ytvis19/videomt/vit-small/videomt_online_ViTS.yaml`.
    if "vit_small" in checkpoint_filename:
        return "vit_small_patch14_reg4_dinov2"
    if "vit_base" in checkpoint_filename:
        return "vit_base_patch14_reg4_dinov2"
    if "vit_large" in checkpoint_filename:
        return "vit_large_patch14_reg4_dinov2"
    raise ValueError(f"Could not infer timm backbone model from checkpoint name '{checkpoint_filename}'.")


def _build_reference_load_dict(
    original_state_dict: dict[str, torch.Tensor], reference_state_dict: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    loadable_reference_state_dict = {}
    skipped_reference_keys = []

    for key, value in original_state_dict.items():
        if not key.startswith("backbone."):
            continue

        stripped_key = key[len("backbone.") :]
        candidate_key = stripped_key

        if candidate_key.endswith(".ls1.gamma"):
            gamma_key = candidate_key.replace(".ls1.gamma", ".gamma_1")
            if gamma_key in reference_state_dict:
                candidate_key = gamma_key
        elif candidate_key.endswith(".ls2.gamma"):
            gamma_key = candidate_key.replace(".ls2.gamma", ".gamma_2")
            if gamma_key in reference_state_dict:
                candidate_key = gamma_key
        elif (
            candidate_key.endswith(".reg_token")
            and candidate_key.replace(".reg_token", ".register_tokens") in reference_state_dict
        ):
            candidate_key = candidate_key.replace(".reg_token", ".register_tokens")

        if candidate_key.endswith(".attn.qkv.bias"):
            base_key = candidate_key[: -len(".qkv.bias")]
            hidden_size = value.shape[0] // 3
            q_bias, _k_bias, v_bias = value.split(hidden_size, dim=0)
            q_bias_key = f"{base_key}.q_bias"
            v_bias_key = f"{base_key}.v_bias"
            if q_bias_key in reference_state_dict and v_bias_key in reference_state_dict:
                loadable_reference_state_dict[q_bias_key] = q_bias
                loadable_reference_state_dict[v_bias_key] = v_bias
                continue

        if candidate_key in reference_state_dict and tuple(reference_state_dict[candidate_key].shape) == tuple(
            value.shape
        ):
            loadable_reference_state_dict[candidate_key] = value
        else:
            skipped_reference_keys.append(stripped_key)

    return loadable_reference_state_dict, skipped_reference_keys


class _ReferenceLayerScaleAdapter(nn.Module):
    def __init__(self, gamma: torch.Tensor):
        super().__init__()
        self.gamma = gamma

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.gamma


def _prepare_reference_model_for_verify(reference_model: nn.Module) -> None:
    # Keep verification deterministic and avoid timm patch-drop index path differences across backbones.
    reference_model.encoder.backbone.patch_drop = nn.Identity()

    original_pos_embed = reference_model.encoder.backbone._pos_embed

    def _safe_pos_embed(x: torch.Tensor):
        # timm EVA `_pos_embed` internally calls `self.patch_drop(x)` and expects `(x, keep_indices)`.
        # Upstream VidEoMT wrapper then calls `patch_drop` once more and expects a tensor.
        # We temporarily disable the internal patch_drop call to avoid API mismatch, while keeping
        # the outer wrapper path deterministic via `nn.Identity`.
        original_patch_drop = reference_model.encoder.backbone.patch_drop
        reference_model.encoder.backbone.patch_drop = None
        pos_embed_output = original_pos_embed(x)
        reference_model.encoder.backbone.patch_drop = original_patch_drop

        # Newer timm EVA backbones may return `(tokens, rope)` while the upstream VidEoMT wrapper
        # expects `_pos_embed` to return only tokens.
        if isinstance(pos_embed_output, tuple):
            return pos_embed_output[0]
        return pos_embed_output

    reference_model.encoder.backbone._pos_embed = _safe_pos_embed

    # timm EVA blocks expose gamma_1/gamma_2, while the VidEoMT wrapper calls ls1/ls2 modules.
    for block in reference_model.encoder.backbone.blocks:
        if not hasattr(block, "ls1") and hasattr(block, "gamma_1"):
            block.ls1 = _ReferenceLayerScaleAdapter(block.gamma_1)
        if not hasattr(block, "ls2") and hasattr(block, "gamma_2"):
            block.ls2 = _ReferenceLayerScaleAdapter(block.gamma_2)

        # Upstream wrapper `_attn` expects timm attention modules to expose `head_dim`.
        if hasattr(block, "attn") and not hasattr(block.attn, "head_dim") and hasattr(block.attn, "qkv"):
            block.attn.head_dim = block.attn.qkv.weight.shape[0] // (3 * block.attn.num_heads)


def load_reference_videomt_class(reference_repo_path: Path):
    base_path = reference_repo_path / "videomt" / "modeling" / "backbone"

    detectron2_modeling = types.ModuleType("detectron2.modeling")

    class _Backbone:
        pass

    class _Registry:
        def register(self):
            def _deco(cls):
                return cls

            return _deco

    detectron2_modeling.Backbone = _Backbone
    detectron2_modeling.BACKBONE_REGISTRY = _Registry()
    sys.modules["detectron2.modeling"] = detectron2_modeling

    for module_name in ["scale_block", "vit", "videomt"]:
        spec = importlib.util.spec_from_file_location(
            f"hf_videomt_reference.backbone.{module_name}", base_path / f"{module_name}.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module '{module_name}' from {base_path}.")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "hf_videomt_reference.backbone"
        sys.modules[f"hf_videomt_reference.backbone.{module_name}"] = module
        spec.loader.exec_module(module)

    return sys.modules["hf_videomt_reference.backbone.videomt"].VidEoMT_CLASS


# fmt: off
MAPPINGS = {
    r"backbone\.encoder\.backbone\.cls_token":                       r"embeddings.cls_token",
    r"backbone\.encoder\.backbone\.reg_token":                       r"embeddings.register_tokens",
    r"backbone\.encoder\.backbone\.pos_embed":                       r"embeddings.position_embeddings.weight",
    r"backbone\.encoder\.backbone\.patch_embed\.proj":               r"embeddings.patch_embeddings.projection",
    r"backbone\.encoder\.backbone\.blocks\.(\d+)\.norm1":            r"layers.\1.norm1",
    r"backbone\.encoder\.backbone\.blocks\.(\d+)\.attn\.proj":       r"layers.\1.attention.out_proj",
    r"backbone\.encoder\.backbone\.blocks\.(\d+)\.ls1\.gamma":       r"layers.\1.layer_scale1.lambda1",
    r"backbone\.encoder\.backbone\.blocks\.(\d+)\.norm2":            r"layers.\1.norm2",
    r"backbone\.encoder\.backbone\.blocks\.(\d+)\.ls2\.gamma":       r"layers.\1.layer_scale2.lambda1",
    r"backbone\.encoder\.backbone\.blocks\.(\d+)\.attn":             r"layers.\1.attention",
    r"backbone\.encoder\.backbone\.blocks\.(\d+)\.mlp":              r"layers.\1.mlp",
    r"backbone\.encoder\.backbone\.norm":                            r"layernorm",
    r"backbone\.q\.weight":                                          r"query.weight",
    r"backbone\.query_updater":                                      r"query_updater",
    r"backbone\.class_head":                                         r"class_predictor",
    r"backbone\.upscale\.(\d+)\.conv1":                              r"upscale_block.block.\1.conv1",
    r"backbone\.upscale\.(\d+)\.conv2":                              r"upscale_block.block.\1.conv2",
    r"backbone\.upscale\.(\d+)\.norm":                               r"upscale_block.block.\1.layernorm2d",
    r"backbone\.mask_head\.0":                                       r"mask_head.fc1",
    r"backbone\.mask_head\.2":                                       r"mask_head.fc2",
    r"backbone\.mask_head\.4":                                       r"mask_head.fc3",
    r"backbone\.attn_mask_probs":                                    r"attn_mask_probs",
}
# fmt: on


def _rename_key(key: str) -> str | None:
    for pattern, replacement in MAPPINGS.items():
        new_key = re.sub(pattern, replacement, key)
        if new_key != key:
            return new_key
    return None


def _split_qkv(key: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    split_tensors = torch.split(tensor, tensor.shape[0] // 3, dim=0)
    return {key.replace("qkv", proj): t for proj, t in zip(["q_proj", "k_proj", "v_proj"], split_tensors)}


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], set[str]]:
    converted = {}
    consumed_keys = set()

    for old_key, value in state_dict.items():
        new_key = _rename_key(old_key)
        if new_key is None:
            continue
        consumed_keys.add(old_key)
        if "qkv" in new_key:
            converted.update(_split_qkv(new_key, value))
        else:
            converted[new_key] = value

    pos_key = "embeddings.position_embeddings.weight"
    if pos_key in converted:
        converted[pos_key] = converted[pos_key].squeeze(0)

    return converted, consumed_keys


def convert_checkpoint(
    checkpoint_filename: str,
    image_size: int,
    num_frames: int,
    output_dir: str | None = None,
    verify: bool = False,
    reference_repo_path: str | None = None,
    push_to_hub: bool = False,
) -> None:
    checkpoint_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=checkpoint_filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    original_state_dict = checkpoint.get("model", checkpoint)

    config = infer_videomt_config(
        original_state_dict, checkpoint_filename, image_size=image_size, num_frames=num_frames
    )

    ckpt_cfg = CHECKPOINT_CONFIGS.get(checkpoint_filename)
    if ckpt_cfg is not None:
        id2label = DATASET_TO_ID2LABEL[ckpt_cfg["dataset"]]
        if len(id2label) != config.num_labels:
            raise ValueError(
                f"id2label length ({len(id2label)}) does not match num_labels ({config.num_labels}) "
                f"for checkpoint '{checkpoint_filename}'."
            )
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    model = VideomtForUniversalSegmentation(config)
    converted_state_dict, consumed_keys = convert_state_dict(original_state_dict)

    load_info = model.load_state_dict(converted_state_dict, strict=False)

    dummy_video = torch.randn(1, num_frames, 3, config.image_size, config.image_size)
    with torch.no_grad():
        outputs = model(pixel_values_videos=dummy_video)

    if (
        not torch.isfinite(outputs.class_queries_logits).all()
        or not torch.isfinite(outputs.masks_queries_logits).all()
    ):
        raise ValueError("Converted model produced non-finite outputs.")

    print(f"checkpoint={checkpoint_filename}")
    print(f"missing_keys={len(load_info.missing_keys)}")
    print(f"unexpected_keys={len(load_info.unexpected_keys)}")
    print(f"class_logits_shape={tuple(outputs.class_queries_logits.shape)}")
    print(f"mask_logits_shape={tuple(outputs.masks_queries_logits.shape)}")

    if load_info.missing_keys:
        print("missing_key_list=")
        for key in load_info.missing_keys:
            print(f"  - {key}")

    if load_info.unexpected_keys:
        print("unexpected_key_list=")
        for key in load_info.unexpected_keys:
            print(f"  - {key}")

    unconverted_source_keys = sorted(set(original_state_dict.keys()) - consumed_keys)
    print(f"unconverted_source_keys={len(unconverted_source_keys)}")
    if unconverted_source_keys:
        print("unconverted_source_key_list=")
        for key in unconverted_source_keys:
            print(f"  - {key}")

        query_updater_keys = [key for key in unconverted_source_keys if "query_updater" in key]
        if query_updater_keys:
            print("note=unconverted_query_updater_keys_detected; temporal-frame forward parity may differ")

    if output_dir is not None:
        model.save_pretrained(output_dir)
        config.save_pretrained(output_dir)
        print(f"saved_to={output_dir}")

    if push_to_hub:
        ckpt_cfg = CHECKPOINT_CONFIGS.get(checkpoint_filename)
        if ckpt_cfg is None:
            raise ValueError(
                f"Cannot push to Hub: checkpoint '{checkpoint_filename}' has no entry in CHECKPOINT_CONFIGS."
            )
        hub_name = ckpt_cfg["hub_name"]
        model.push_to_hub(hub_name)
        print(f"pushed_to_hub={hub_name}")

    if verify:
        verify_ok = verify_conversion_against_github_reference(
            hf_model=model,
            original_state_dict=original_state_dict,
            checkpoint_filename=checkpoint_filename,
            image_size=image_size,
            num_frames=num_frames,
            reference_repo_path=reference_repo_path,
        )
        print(f"verify_ok={verify_ok}")


def verify_conversion_against_github_reference(
    hf_model: VideomtForUniversalSegmentation,
    original_state_dict: dict[str, torch.Tensor],
    checkpoint_filename: str,
    image_size: int,
    num_frames: int,
    reference_repo_path: str | None = None,
    atol: float = 2e-4,
) -> bool:
    dummy_video = torch.randn(1, num_frames, 3, image_size, image_size, generator=torch.Generator().manual_seed(0))

    with tempfile.TemporaryDirectory(prefix="videomt_ref_") as tmp_dir:
        repo_path = Path(reference_repo_path) if reference_repo_path is not None else Path(tmp_dir) / "videomt"
        if reference_repo_path is None:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/tue-mps/videomt", str(repo_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        import timm
        from timm.layers import pos_embed_sincos
        from timm.models import eva as timm_eva

        original_create_model = timm.create_model
        original_apply_keep_indices_nlc = pos_embed_sincos.apply_keep_indices_nlc
        original_eva_apply_keep_indices_nlc = timm_eva.apply_keep_indices_nlc

        def _create_model_no_pretrained(*args, **kwargs):
            kwargs["pretrained"] = False
            return original_create_model(*args, **kwargs)

        def _safe_apply_keep_indices_nlc(x, pos_embed, keep_indices, pos_embed_has_batch: bool = False):
            if keep_indices.dtype not in (torch.int32, torch.int64):
                keep_indices = keep_indices.to(dtype=torch.int64)
            if torch.any(keep_indices < 0):
                keep_indices = keep_indices.clamp_min(0)
            return original_apply_keep_indices_nlc(x, pos_embed, keep_indices, pos_embed_has_batch=pos_embed_has_batch)

        timm.create_model = _create_model_no_pretrained
        pos_embed_sincos.apply_keep_indices_nlc = _safe_apply_keep_indices_nlc
        timm_eva.apply_keep_indices_nlc = _safe_apply_keep_indices_nlc
        reference_cls = load_reference_videomt_class(repo_path)

        backbone_model_name = infer_backbone_model_name(checkpoint_filename)
        try:
            reference_model = reference_cls(
                img_size=image_size,
                num_classes=hf_model.config.num_labels,
                name=backbone_model_name,
                num_frames=num_frames,
                num_q=hf_model.config.num_queries,
                segmenter_blocks=list(
                    range(
                        hf_model.config.num_hidden_layers - hf_model.config.num_blocks,
                        hf_model.config.num_hidden_layers,
                    )
                ),
            ).eval()
        finally:
            timm.create_model = original_create_model
            pos_embed_sincos.apply_keep_indices_nlc = original_apply_keep_indices_nlc
            timm_eva.apply_keep_indices_nlc = original_eva_apply_keep_indices_nlc

        loadable_state_dict, _ = _build_reference_load_dict(original_state_dict, reference_model.state_dict())
        reference_model.load_state_dict(loadable_state_dict, strict=False)
        _prepare_reference_model_for_verify(reference_model)

        with torch.no_grad():
            hf_outputs = hf_model(pixel_values_videos=dummy_video)
            reference_outputs = reference_model(dummy_video.reshape(-1, 3, image_size, image_size))

        reference_logits = reference_outputs["pred_logits"].reshape(
            -1, hf_model.config.num_queries, hf_model.config.num_labels + 1
        )
        reference_masks = (
            reference_outputs["pred_masks"]
            .permute(0, 2, 1, 3, 4)
            .reshape(
                -1,
                hf_model.config.num_queries,
                reference_outputs["pred_masks"].shape[-2],
                reference_outputs["pred_masks"].shape[-1],
            )
        )

        logits_diff = (hf_outputs.class_queries_logits - reference_logits).abs().max().item()
        masks_diff = (hf_outputs.masks_queries_logits - reference_masks).abs().max().item()
        print(f"verify_logits_max_abs_diff={logits_diff:.8f}")
        print(f"verify_masks_max_abs_diff={masks_diff:.8f}")

        outputs_match = logits_diff < atol and masks_diff < atol
        print(f"verify_forward_pass_ok={outputs_match}")
        return outputs_match


def _resolve_checkpoint_params(
    checkpoint_filename: str,
    image_size: int | None,
    num_frames: int | None,
) -> tuple[int, int]:
    ckpt_cfg = CHECKPOINT_CONFIGS.get(checkpoint_filename)
    if image_size is None:
        if ckpt_cfg is None:
            raise ValueError(
                f"--image-size is required for unknown checkpoint '{checkpoint_filename}'. "
                f"Known checkpoints: {list(CHECKPOINT_CONFIGS)}"
            )
        image_size = ckpt_cfg["image_size"]
    if num_frames is None:
        num_frames = ckpt_cfg["num_frames"] if ckpt_cfg is not None else 2
    return image_size, num_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official VidEoMT checkpoints to HF format.")
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default=None,
        help="Filename on tue-mps/VidEoMT (required unless --all is set)",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Auto-derived from registry when omitted")
    parser.add_argument("--num-frames", type=int, default=None, help="Auto-derived from registry when omitted")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--reference-repo-path", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Convert all supported DINOv2 checkpoints")
    parser.add_argument("--push-to-hub", action="store_true", help="Push converted models to the Hugging Face Hub")
    args = parser.parse_args()

    if not args.all and args.checkpoint_filename is None:
        parser.error("--checkpoint-filename is required unless --all is set")

    return args


def main() -> None:
    args = parse_args()

    if args.all:
        filenames = list(CHECKPOINT_CONFIGS)
    else:
        filenames = [args.checkpoint_filename]

    for checkpoint_filename in filenames:
        image_size, num_frames = _resolve_checkpoint_params(checkpoint_filename, args.image_size, args.num_frames)
        print(f"\n{'=' * 60}")
        print(f"Converting {checkpoint_filename} (image_size={image_size}, num_frames={num_frames})")
        print(f"{'=' * 60}")
        convert_checkpoint(
            checkpoint_filename=checkpoint_filename,
            image_size=image_size,
            num_frames=num_frames,
            output_dir=args.output_dir,
            verify=args.verify,
            reference_repo_path=args.reference_repo_path,
            push_to_hub=args.push_to_hub,
        )


if __name__ == "__main__":
    main()
