#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchvision",
#     "datasets>=3.0.0",
#     "numpy",
#     "timm",
#     "scipy",
#     "fairscale",
#     "huggingface-hub",
# ]
# ///

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection, LwDetrConfig
from transformers.loss.loss_lw_detr import LwDetrHungarianMatcher, LwDetrImageLoss
from transformers.models.lw_detr.convert_lw_detr_to_hf import (
    convert_original_checkpoint_state_dict,
    get_checkpoint_state_dict,
    get_model_config,
)


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LW-DETR HF vs original implementation on a tiny HF dataset subset."
    )
    parser.add_argument("--dataset-id", type=str, default="nielsr/tray-cart-detection")
    parser.add_argument("--model-id", type=str, default="AnnaZhang/lwdetr_tiny_60e_coco")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument(
        "--hf-use-pixel-mask",
        action="store_true",
        help="If set, pass pixel_mask to HF model (default: False for parity with original list-input path).",
    )
    parser.add_argument(
        "--hf-copy-original-two-stage-head-init",
        action="store_true",
        help="If set, copy main class/bbox heads into encoder two-stage heads after loading.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--original-repo-url", type=str, default="https://github.com/NielsRogge/LW-DETR.git")
    parser.add_argument("--original-repo-commit", type=str, default="d5e6e6c4add2d24dafb965ced8b50163c50b9788")
    parser.add_argument("--checkpoint-repo-id", type=str, default="xbsu/LW-DETR")
    parser.add_argument("--checkpoint-filename", type=str, default="pretrain_weights/LWDETR_tiny_60e_coco.pth")
    parser.add_argument(
        "--hf-load-from-original-checkpoint",
        action="store_true",
        help="If set, build HF model from config and load converted weights directly from original .pth checkpoint.",
    )
    parser.add_argument(
        "--hf-checkpoint-model-name",
        type=str,
        default="lwdetr_tiny_60e_coco",
        help="Model name used by conversion utilities when --hf-load-from-original-checkpoint is enabled.",
    )
    parser.add_argument(
        "--hf-attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa"],
        help="HF attention backend. 'auto' keeps default behavior from model/config.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="If set and CUDA is available, disable TensorFloat-32 matmul/cudnn paths for stricter numerical parity.",
    )
    parser.add_argument(
        "--compute-gradient-parity",
        action="store_true",
        help="If set, run a first-batch backward pass parity diagnostic and compare HF vs converted-original gradients.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_image_annotations_as_coco(
    image_id: int,
    categories: list[int],
    areas: list[float],
    bboxes: list[list[float]],
    iscrowd: list[bool] | None = None,
    category_offset: int = 1,
) -> dict[str, Any]:
    annotations = []
    for idx, (category, area, bbox) in enumerate(zip(categories, areas, bboxes)):
        crowd = 0
        if iscrowd is not None and idx < len(iscrowd):
            crowd = int(iscrowd[idx])
        annotations.append(
            {
                "image_id": int(image_id),
                "category_id": int(category) + category_offset,
                "iscrowd": crowd,
                "area": float(area),
                "bbox": [float(v) for v in bbox],
            }
        )
    return {"image_id": int(image_id), "annotations": annotations}


def validate_raw_bbox_format(examples: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_count = 0
    total_boxes = 0
    min_x = float("inf")
    min_y = float("inf")
    max_x2 = float("-inf")
    max_y2 = float("-inf")
    for sample in examples:
        width, height = sample["image"].size
        for bbox in sample["objects"]["bbox"]:
            total_boxes += 1
            if len(bbox) != 4:
                invalid_count += 1
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                invalid_count += 1
            if x < 0 or y < 0 or x + w > width + 1e-3 or y + h > height + 1e-3:
                invalid_count += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x2 = max(max_x2, x + w)
            max_y2 = max(max_y2, y + h)
    if total_boxes == 0:
        min_x = min_y = max_x2 = max_y2 = 0.0
    return {
        "total_boxes": total_boxes,
        "invalid_count": invalid_count,
        "bbox_min_x": float(min_x),
        "bbox_min_y": float(min_y),
        "bbox_max_x2": float(max_x2),
        "bbox_max_y2": float(max_y2),
    }


def build_processed_samples(
    examples: list[dict[str, Any]],
    image_processor: AutoImageProcessor,
    category_offset: int = 1,
) -> list[dict[str, Any]]:
    images = []
    annotations = []
    for sample in examples:
        images.append(np.array(sample["image"].convert("RGB")))
        annotations.append(
            format_image_annotations_as_coco(
                image_id=int(sample["image_id"]),
                categories=sample["objects"]["category"],
                areas=sample["objects"]["area"],
                bboxes=sample["objects"]["bbox"],
                iscrowd=sample["objects"].get("is_crowd"),
                category_offset=category_offset,
            )
        )
    outputs = image_processor(images=images, annotations=annotations, return_tensors="pt")
    samples = []
    pixel_mask = outputs.get("pixel_mask")
    for idx in range(len(examples)):
        sample = {
            "pixel_values": outputs["pixel_values"][idx],
            "labels": outputs["labels"][idx],
        }
        if pixel_mask is not None:
            sample["pixel_mask"] = pixel_mask[idx]
        samples.append(sample)
    return samples


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    collated["pixel_values"] = torch.stack([sample["pixel_values"] for sample in batch])
    collated["labels"] = [sample["labels"] for sample in batch]
    if "pixel_mask" in batch[0]:
        collated["pixel_mask"] = torch.stack([sample["pixel_mask"] for sample in batch])
    return collated


def summarize_loading_info(loading_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "missing_keys_count": len(loading_info.get("missing_keys", [])),
        "unexpected_keys_count": len(loading_info.get("unexpected_keys", [])),
        "mismatched_keys_count": len(loading_info.get("mismatched_keys", [])),
        "mismatched_keys": [
            {"name": key_name, "checkpoint_shape": list(checkpoint_shape), "model_shape": list(model_shape)}
            for key_name, checkpoint_shape, model_shape in loading_info.get("mismatched_keys", [])
        ],
    }


def inspect_custom_head_initialization(model: torch.nn.Module) -> dict[str, Any]:
    prior_prob = 0.01
    prior_bias_value = -math.log((1 - prior_prob) / prior_prob)

    class_embed_weight = model.class_embed.weight.detach().cpu()
    class_embed_bias = model.class_embed.bias.detach().cpu()
    enc_out_heads = model.model.enc_out_class_embed

    enc_bias_equal_to_prior_count = 0
    enc_bias_equal_main_count = 0
    enc_weight_equal_main_count = 0
    enc_weight_equal_first_count = 0
    enc_bias_means = []
    enc_bias_mins = []
    enc_bias_maxs = []

    first_enc_weight = enc_out_heads[0].weight.detach().cpu()
    for enc_head in enc_out_heads:
        enc_weight = enc_head.weight.detach().cpu()
        enc_bias = enc_head.bias.detach().cpu()
        enc_bias_means.append(float(enc_bias.mean().item()))
        enc_bias_mins.append(float(enc_bias.min().item()))
        enc_bias_maxs.append(float(enc_bias.max().item()))

        if torch.allclose(enc_bias, torch.full_like(enc_bias, prior_bias_value), atol=1e-6, rtol=0.0):
            enc_bias_equal_to_prior_count += 1
        if torch.allclose(enc_bias, class_embed_bias):
            enc_bias_equal_main_count += 1
        if torch.allclose(enc_weight, class_embed_weight):
            enc_weight_equal_main_count += 1
        if torch.allclose(enc_weight, first_enc_weight):
            enc_weight_equal_first_count += 1

    return {
        "prior_bias_value": float(prior_bias_value),
        "num_enc_out_heads": int(len(enc_out_heads)),
        "class_embed_bias_min": float(class_embed_bias.min().item()),
        "class_embed_bias_max": float(class_embed_bias.max().item()),
        "class_embed_bias_mean": float(class_embed_bias.mean().item()),
        "enc_out_bias_equal_to_prior_count": int(enc_bias_equal_to_prior_count),
        "enc_out_bias_equal_main_class_head_count": int(enc_bias_equal_main_count),
        "enc_out_weight_equal_main_class_head_count": int(enc_weight_equal_main_count),
        "enc_out_weight_equal_first_head_count": int(enc_weight_equal_first_count),
        "enc_out_bias_mean_min": float(min(enc_bias_means)) if enc_bias_means else None,
        "enc_out_bias_mean_max": float(max(enc_bias_means)) if enc_bias_means else None,
        "enc_out_bias_value_min": float(min(enc_bias_mins)) if enc_bias_mins else None,
        "enc_out_bias_value_max": float(max(enc_bias_maxs)) if enc_bias_maxs else None,
    }


@torch.no_grad()
def apply_original_two_stage_head_init(model: torch.nn.Module) -> dict[str, Any]:
    if not hasattr(model, "model") or not hasattr(model.model, "enc_out_class_embed"):
        return {"applied": False, "reason": "missing_two_stage_heads"}

    enc_out_class_heads = model.model.enc_out_class_embed
    enc_out_bbox_heads = model.model.enc_out_bbox_embed

    for enc_class_head in enc_out_class_heads:
        enc_class_head.load_state_dict(model.class_embed.state_dict())
    for enc_bbox_head in enc_out_bbox_heads:
        enc_bbox_head.load_state_dict(model.bbox_embed.state_dict())

    return {
        "applied": True,
        "num_enc_out_class_heads": int(len(enc_out_class_heads)),
        "num_enc_out_bbox_heads": int(len(enc_out_bbox_heads)),
    }


def get_hf_initialization_summary(model_id: str, label_names: list[str]) -> dict[str, Any]:
    base_config = AutoConfig.from_pretrained(model_id)
    base_config.dropout = 0.0
    base_model, base_info = AutoModelForObjectDetection.from_pretrained(
        model_id, config=base_config, output_loading_info=True
    )

    custom_id2label = dict(enumerate(label_names))
    custom_label2id = {name: idx for idx, name in custom_id2label.items()}
    custom_config = AutoConfig.from_pretrained(
        model_id,
        num_labels=len(custom_id2label),
        id2label=custom_id2label,
        label2id=custom_label2id,
    )
    custom_config.dropout = 0.0
    custom_model, custom_info = AutoModelForObjectDetection.from_pretrained(
        model_id,
        config=custom_config,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )

    custom_head_init = inspect_custom_head_initialization(custom_model)
    del base_model
    del custom_model
    return {
        "default_num_labels_load": summarize_loading_info(base_info),
        "custom_num_labels_load": summarize_loading_info(custom_info),
        "custom_head_initialization": custom_head_init,
    }


def load_hf_model_from_original_checkpoint(
    checkpoint_repo_id: str,
    checkpoint_filename: str,
    checkpoint_model_name: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = hf_hub_download(repo_id=checkpoint_repo_id, filename=checkpoint_filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config_dict = get_model_config(checkpoint_model_name)
    model_config = LwDetrConfig(**model_config_dict)
    model_config.dropout = 0.0
    hf_model = AutoModelForObjectDetection.from_config(model_config)

    original_state_dict = get_checkpoint_state_dict(checkpoint)
    converted_state_dict = convert_original_checkpoint_state_dict(original_state_dict, model_config)
    load_result = hf_model.load_state_dict(converted_state_dict, strict=False)

    return hf_model, {
        "model_source": "original_checkpoint_converted_into_auto_model",
        "checkpoint_repo_id": checkpoint_repo_id,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_model_name": checkpoint_model_name,
        "checkpoint_path": checkpoint_path,
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def clone_original_repo(repo_url: str, commit: str, root_dir: Path) -> Path:
    repo_dir = root_dir / "lw_detr_original"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run_cmd(["git", "clone", repo_url, str(repo_dir)])
    run_cmd(["git", "checkout", commit], cwd=repo_dir)
    return repo_dir


def run_cmd(command: list[str], cwd: Path | None = None) -> None:
    import subprocess

    log(f"Running command: {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, check=True)


def build_original_args(device: str, checkpoint_args: Any | None = None) -> argparse.Namespace:
    defaults = {
        "lr": 1e-4,
        "lr_encoder": 1.5e-4,
        "batch_size": 1,
        "weight_decay": 1e-4,
        "epochs": 60,
        "lr_drop": 60,
        "clip_max_norm": 0.1,
        "lr_vit_layer_decay": 0.8,
        "lr_component_decay": 0.7,
        "dropout": 0.0,
        "drop_path": 0.0,
        "drop_mode": "standard",
        "drop_schedule": "constant",
        "cutoff_epoch": 0,
        "pretrained_encoder": None,
        "pretrain_weights": None,
        "pretrain_exclude_keys": None,
        "pretrain_keys_modify_to_load": None,
        "encoder": "vit_tiny",
        "vit_encoder_num_layers": 10,
        "window_block_indexes": [0, 1, 3, 6, 7, 9],
        "position_embedding": "sine",
        "out_feature_indexes": [2, 4, 5, 9],
        "dec_layers": 3,
        "dim_feedforward": 2048,
        "hidden_dim": 256,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "num_queries": 300,
        "group_detr": 13,
        "two_stage": True,
        "projector_scale": ["P4"],
        "lite_refpoint_refine": True,
        "num_select": 300,
        "dec_n_points": 2,
        "decoder_norm": "LN",
        "bbox_reparam": True,
        "set_cost_class": 2.0,
        "set_cost_bbox": 5.0,
        "set_cost_giou": 2.0,
        "cls_loss_coef": 1.0,
        "bbox_loss_coef": 5.0,
        "giou_loss_coef": 2.0,
        "focal_alpha": 0.25,
        "aux_loss": True,
        "sum_group_losses": False,
        "use_varifocal_loss": False,
        "use_position_supervised_loss": False,
        "ia_bce_loss": True,
        "dataset_file": "coco",
        "num_feature_levels": 1,  # will be refreshed from projector_scale below
    }
    if checkpoint_args is not None:
        if isinstance(checkpoint_args, dict):
            defaults.update(checkpoint_args)
        else:
            defaults.update(vars(checkpoint_args))
    defaults["device"] = device
    if "projector_scale" in defaults and defaults["projector_scale"] is not None:
        defaults["num_feature_levels"] = len(defaults["projector_scale"])
    return argparse.Namespace(**defaults)


def init_original_model(
    repo_dir: Path,
    checkpoint_repo_id: str,
    checkpoint_filename: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    # Inject a fairscale checkpoint wrapper stub to avoid requiring fairscale in light-weight parity runs.
    if "fairscale" not in sys.modules:
        fairscale_module = types.ModuleType("fairscale")
        fairscale_nn_module = types.ModuleType("fairscale.nn")
        fairscale_checkpoint_module = types.ModuleType("fairscale.nn.checkpoint")

        def checkpoint_wrapper(module, *args, **kwargs):
            del args, kwargs
            return module

        fairscale_checkpoint_module.checkpoint_wrapper = checkpoint_wrapper
        fairscale_nn_module.checkpoint = fairscale_checkpoint_module
        fairscale_module.nn = fairscale_nn_module
        sys.modules["fairscale"] = fairscale_module
        sys.modules["fairscale.nn"] = fairscale_nn_module
        sys.modules["fairscale.nn.checkpoint"] = fairscale_checkpoint_module

    # Inject a stub extension module to allow importing original code without building custom CUDA ops.
    if "MultiScaleDeformableAttention" not in sys.modules:
        stub = SimpleNamespace(
            ms_deform_attn_forward=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("MultiScaleDeformableAttention extension path should not be used in this script.")
            ),
            ms_deform_attn_backward=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("MultiScaleDeformableAttention extension path should not be used in this script.")
            ),
        )
        sys.modules["MultiScaleDeformableAttention"] = stub

    sys.path.insert(0, str(repo_dir))
    try:
        from models import build_model as build_original_model
    finally:
        sys.path.pop(0)

    checkpoint_path = hf_hub_download(repo_id=checkpoint_repo_id, filename=checkpoint_filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    args = build_original_args(device=str(device), checkpoint_args=checkpoint_args)
    # Ensure the comparison run does not try to fetch extra pretrained weights from paths in the original train env.
    args.pretrained_encoder = None
    args.pretrain_weights = None
    args.resume = ""
    model, criterion, _ = build_original_model(args)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=False)

    # Force pure PyTorch deformable attention path to avoid custom extension usage.
    fallback_count = 0
    for module in model.modules():
        if module.__class__.__name__ == "MSDeformAttn":
            module._export = True
            fallback_count += 1

    model.to(device)
    criterion.to(device)

    loading_summary = {
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "ms_deform_attn_fallback_modules": fallback_count,
        "checkpoint_arch_args": {
            "encoder": getattr(args, "encoder", None),
            "vit_encoder_num_layers": getattr(args, "vit_encoder_num_layers", None),
            "window_block_indexes": getattr(args, "window_block_indexes", None),
            "out_feature_indexes": getattr(args, "out_feature_indexes", None),
            "num_queries": getattr(args, "num_queries", None),
            "num_select": getattr(args, "num_select", None),
            "dropout": getattr(args, "dropout", None),
        },
    }
    return model, criterion, loading_summary


def to_device_labels(labels: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in sample.items()} for sample in labels]


def to_original_targets(labels: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    targets = []
    for sample in labels:
        target = {
            "labels": sample["class_labels"],
            "boxes": sample["boxes"],
        }
        for key in ["area", "iscrowd", "orig_size", "size", "image_id"]:
            if key in sample:
                target[key] = sample[key]
        targets.append(target)
    return targets


def post_process_scores(
    image_processor: AutoImageProcessor,
    model_outputs: Any,
    target_sizes: torch.Tensor,
    thresholds: tuple[float, ...] = (0.0, 0.5),
) -> dict[str, dict[str, float]]:
    if isinstance(model_outputs, dict):
        logits = model_outputs.get("pred_logits", model_outputs.get("logits"))
        pred_boxes = model_outputs.get("pred_boxes", model_outputs.get("boxes"))
        if logits is None or pred_boxes is None:
            raise KeyError("Could not find logits/pred_boxes in model outputs dictionary.")
        wrapped_outputs = SimpleNamespace(logits=logits, pred_boxes=pred_boxes)
    else:
        logits = getattr(model_outputs, "pred_logits", None)
        if logits is None:
            logits = getattr(model_outputs, "logits")
        pred_boxes = getattr(model_outputs, "pred_boxes", None)
        if pred_boxes is None:
            pred_boxes = getattr(model_outputs, "boxes")
        wrapped_outputs = SimpleNamespace(logits=logits, pred_boxes=pred_boxes)

    scores_by_threshold: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        detections = image_processor.post_process_object_detection(
            wrapped_outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        scores = detections["scores"].detach().cpu().tolist()
        scores_by_threshold[str(threshold)] = {
            "num_predictions": float(len(scores)),
            "max_score": float(max(scores) if scores else 0.0),
            "mean_score": float(sum(scores) / len(scores) if scores else 0.0),
        }
    return scores_by_threshold


@torch.no_grad()
def compute_forward_parity(
    hf_model: torch.nn.Module,
    original_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    hf_use_pixel_mask: bool = False,
) -> dict[str, Any]:
    batch = next(iter(dataloader))
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)

    hf_model.eval()
    original_model.eval()

    if hf_use_pixel_mask and pixel_mask is not None:
        hf_outputs = hf_model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    else:
        hf_outputs = hf_model(pixel_values=pixel_values)
    original_outputs = original_model(list(pixel_values))

    hf_logits = hf_outputs.logits
    hf_boxes = hf_outputs.pred_boxes
    original_logits = original_outputs["pred_logits"]
    original_boxes = original_outputs["pred_boxes"]

    logit_diff = torch.max(torch.abs(hf_logits - original_logits)).item()
    box_diff = torch.max(torch.abs(hf_boxes - original_boxes)).item()
    return {
        "max_abs_diff_logits": float(logit_diff),
        "max_abs_diff_boxes": float(box_diff),
        "hf_logits_shape": list(hf_logits.shape),
        "original_logits_shape": list(original_logits.shape),
    }


@torch.no_grad()
def compute_train_forward_parity(
    hf_model: torch.nn.Module,
    original_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    hf_use_pixel_mask: bool = False,
) -> dict[str, Any]:
    batch = next(iter(dataloader))
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)
    labels = to_device_labels(batch["labels"], device)
    targets = to_original_targets(labels)

    hf_model.train()
    original_model.train()

    if hf_use_pixel_mask and pixel_mask is not None:
        hf_outputs = hf_model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    else:
        hf_outputs = hf_model(pixel_values=pixel_values)
    original_outputs = original_model(list(pixel_values), targets=targets)

    hf_logits = hf_outputs.logits
    hf_boxes = hf_outputs.pred_boxes
    original_logits = original_outputs["pred_logits"]
    original_boxes = original_outputs["pred_boxes"]

    logit_diff = torch.max(torch.abs(hf_logits - original_logits)).item()
    box_diff = torch.max(torch.abs(hf_boxes - original_boxes)).item()
    return {
        "max_abs_diff_logits": float(logit_diff),
        "max_abs_diff_boxes": float(box_diff),
        "hf_logits_shape": list(hf_logits.shape),
        "original_logits_shape": list(original_logits.shape),
    }


@torch.no_grad()
def evaluate_score_calibration(
    model: torch.nn.Module,
    model_kind: str,
    dataloader: DataLoader,
    image_processor: AutoImageProcessor,
    device: torch.device,
    hf_use_pixel_mask: bool = False,
) -> dict[str, Any]:
    model.eval()
    aggregate = {
        "0.0": {"num_predictions": 0.0, "max_score": 0.0, "mean_score_sum": 0.0},
        "0.5": {"num_predictions": 0.0, "max_score": 0.0, "mean_score_sum": 0.0},
        "num_images": 0.0,
    }
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
        labels = to_device_labels(batch["labels"], device)
        for idx in range(pixel_values.shape[0]):
            single_pixel_values = pixel_values[idx : idx + 1]
            single_pixel_mask = pixel_mask[idx : idx + 1] if pixel_mask is not None else None
            if model_kind == "hf":
                if hf_use_pixel_mask and single_pixel_mask is not None:
                    outputs = model(pixel_values=single_pixel_values, pixel_mask=single_pixel_mask)
                else:
                    outputs = model(pixel_values=single_pixel_values)
            else:
                outputs = model([single_pixel_values[0]])
            target_sizes = labels[idx]["orig_size"].unsqueeze(0)
            score_stats = post_process_scores(
                image_processor=image_processor,
                model_outputs=outputs,
                target_sizes=target_sizes,
            )
            aggregate["num_images"] += 1.0
            for threshold_key in ["0.0", "0.5"]:
                aggregate[threshold_key]["num_predictions"] += score_stats[threshold_key]["num_predictions"]
                aggregate[threshold_key]["max_score"] = max(
                    aggregate[threshold_key]["max_score"], score_stats[threshold_key]["max_score"]
                )
                aggregate[threshold_key]["mean_score_sum"] += score_stats[threshold_key]["mean_score"]

    num_images = max(1.0, aggregate["num_images"])
    return {
        "threshold_0.0": {
            "avg_num_predictions": aggregate["0.0"]["num_predictions"] / num_images,
            "max_score": aggregate["0.0"]["max_score"],
            "avg_mean_score": aggregate["0.0"]["mean_score_sum"] / num_images,
        },
        "threshold_0.5": {
            "avg_num_predictions": aggregate["0.5"]["num_predictions"] / num_images,
            "max_score": aggregate["0.5"]["max_score"],
            "avg_mean_score": aggregate["0.5"]["mean_score_sum"] / num_images,
        },
    }


def safe_evaluate_score_calibration(
    model: torch.nn.Module,
    model_kind: str,
    dataloader: DataLoader,
    image_processor: AutoImageProcessor,
    device: torch.device,
    hf_use_pixel_mask: bool = False,
) -> dict[str, Any]:
    try:
        return evaluate_score_calibration(
            model=model,
            model_kind=model_kind,
            dataloader=dataloader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=hf_use_pixel_mask,
        )
    except Exception as error:  # noqa: BLE001
        return {"error": str(error)}


def summarize_loss_series(series: list[float]) -> dict[str, Any]:
    if not series:
        return {"first": None, "last": None, "min": None, "series": []}
    return {
        "first": series[0],
        "last": series[-1],
        "min": min(series),
        "series": series,
    }


def compute_hf_loss_on_original_outputs(
    config: LwDetrConfig,
    original_outputs: dict[str, Any],
    labels: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    matcher = LwDetrHungarianMatcher(
        class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost
    )
    criterion = LwDetrImageLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        focal_alpha=config.focal_alpha,
        losses=["labels", "boxes", "cardinality"],
        group_detr=config.group_detr,
    ).to(device)
    criterion.train()

    outputs_loss = {
        "logits": original_outputs["pred_logits"],
        "pred_boxes": original_outputs["pred_boxes"],
    }
    if "enc_outputs" in original_outputs:
        outputs_loss["enc_outputs"] = {
            "logits": original_outputs["enc_outputs"]["pred_logits"],
            "pred_boxes": original_outputs["enc_outputs"]["pred_boxes"],
        }
    if "aux_outputs" in original_outputs:
        outputs_loss["auxiliary_outputs"] = [
            {"logits": aux_output["pred_logits"], "pred_boxes": aux_output["pred_boxes"]}
            for aux_output in original_outputs["aux_outputs"]
        ]

    hf_loss_dict = criterion(outputs_loss, labels)

    base_weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": float(config.bbox_loss_coefficient),
        "loss_giou": float(config.giou_loss_coefficient),
    }
    weight_dict = dict(base_weight_dict)
    for decoder_layer in range(config.decoder_layers - 1):
        for key, value in base_weight_dict.items():
            weight_dict[f"{key}_{decoder_layer}"] = value
    for key, value in base_weight_dict.items():
        weight_dict[f"{key}_enc"] = value

    hf_total_loss = sum(hf_loss_dict[key] * weight_dict[key] for key in hf_loss_dict if key in weight_dict)
    return hf_total_loss, hf_loss_dict, weight_dict


@torch.no_grad()
def evaluate_loss_implementation_parity(
    hf_model: torch.nn.Module,
    original_model: torch.nn.Module,
    original_criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    batch = next(iter(dataloader))
    pixel_values = batch["pixel_values"].to(device)
    labels = to_device_labels(batch["labels"], device)
    targets = to_original_targets(labels)

    original_model.train()
    original_outputs = original_model(list(pixel_values), targets=targets)
    original_loss_dict = original_criterion(original_outputs, targets)
    original_total_loss = sum(
        original_loss_dict[key] * original_criterion.weight_dict[key]
        for key in original_loss_dict
        if key in original_criterion.weight_dict
    )

    hf_total_loss, hf_loss_dict, hf_weight_dict = compute_hf_loss_on_original_outputs(
        config=hf_model.config,
        original_outputs=original_outputs,
        labels=labels,
        device=device,
    )

    per_key_abs_diffs: dict[str, float] = {}
    for key in sorted(set(hf_loss_dict.keys()) & set(original_loss_dict.keys())):
        if key in hf_weight_dict and torch.is_tensor(hf_loss_dict[key]) and torch.is_tensor(original_loss_dict[key]):
            per_key_abs_diffs[key] = float(torch.abs(hf_loss_dict[key] - original_loss_dict[key]).item())

    return {
        "abs_diff_total_loss": float(torch.abs(hf_total_loss - original_total_loss).item()),
        "hf_total_loss_on_original_outputs": float(hf_total_loss.item()),
        "original_total_loss": float(original_total_loss.item()),
        "per_key_abs_diffs": per_key_abs_diffs,
    }


def _global_grad_norm(grad_dict: dict[str, torch.Tensor]) -> float:
    if not grad_dict:
        return 0.0
    total = 0.0
    for grad in grad_dict.values():
        total += float(grad.detach().float().pow(2).sum().item())
    return float(math.sqrt(total))


def evaluate_gradient_parity(
    hf_model: torch.nn.Module,
    original_model: torch.nn.Module,
    original_criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    hf_use_pixel_mask: bool = False,
) -> dict[str, Any]:
    batch = next(iter(dataloader))
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)
    labels = to_device_labels(batch["labels"], device)
    targets = to_original_targets(labels)

    hf_model.train()
    original_model.train()
    original_criterion.train()
    hf_model.zero_grad(set_to_none=True)
    original_model.zero_grad(set_to_none=True)
    original_criterion.zero_grad(set_to_none=True)

    if hf_use_pixel_mask and pixel_mask is not None:
        hf_outputs = hf_model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    else:
        hf_outputs = hf_model(pixel_values=pixel_values, labels=labels)
    hf_loss = hf_outputs.loss
    hf_loss.backward()

    original_outputs = original_model(list(pixel_values), targets=targets)
    original_loss_dict = original_criterion(original_outputs, targets)
    original_loss = sum(
        original_loss_dict[key] * original_criterion.weight_dict[key]
        for key in original_loss_dict
        if key in original_criterion.weight_dict
    )
    original_loss.backward()

    hf_grads = {
        name: param.grad.detach().cpu() for name, param in hf_model.named_parameters() if param.grad is not None
    }
    original_grads = {
        name: param.grad.detach().cpu() for name, param in original_model.named_parameters() if param.grad is not None
    }

    converted_original_grads = convert_original_checkpoint_state_dict(original_grads, hf_model.config)
    shared_keys = sorted(set(hf_grads.keys()) & set(converted_original_grads.keys()))

    per_key_max_abs_diff: dict[str, float] = {}
    for key in shared_keys:
        if hf_grads[key].shape == converted_original_grads[key].shape:
            per_key_max_abs_diff[key] = float(
                torch.max(torch.abs(hf_grads[key] - converted_original_grads[key])).item()
            )

    top_keys = sorted(per_key_max_abs_diff.items(), key=lambda item: item[1], reverse=True)[:20]
    max_abs_diff = max(per_key_max_abs_diff.values()) if per_key_max_abs_diff else 0.0
    mean_abs_diff = (
        float(sum(per_key_max_abs_diff.values()) / len(per_key_max_abs_diff)) if per_key_max_abs_diff else 0.0
    )

    # Control check: verify key conversion correctness on current parameter values.
    original_weights = {name: param.detach().cpu() for name, param in original_model.named_parameters()}
    converted_original_weights = convert_original_checkpoint_state_dict(original_weights, hf_model.config)
    hf_weights = {name: param.detach().cpu() for name, param in hf_model.named_parameters()}
    shared_weight_keys = sorted(set(hf_weights.keys()) & set(converted_original_weights.keys()))
    weight_diffs = [
        float(torch.max(torch.abs(hf_weights[key] - converted_original_weights[key])).item())
        for key in shared_weight_keys
        if hf_weights[key].shape == converted_original_weights[key].shape
    ]
    control_weight_max_abs_diff = max(weight_diffs) if weight_diffs else 0.0
    control_weight_mean_abs_diff = float(sum(weight_diffs) / len(weight_diffs)) if weight_diffs else 0.0

    return {
        "hf_loss_first_batch": float(hf_loss.detach().cpu().item()),
        "original_loss_first_batch": float(original_loss.detach().cpu().item()),
        "hf_grad_param_count": len(hf_grads),
        "original_grad_param_count": len(original_grads),
        "shared_grad_param_count": len(per_key_max_abs_diff),
        "hf_global_grad_norm": _global_grad_norm(hf_grads),
        "original_global_grad_norm": _global_grad_norm(converted_original_grads),
        "shared_max_abs_diff": float(max_abs_diff),
        "shared_mean_abs_diff": float(mean_abs_diff),
        "control_weight_max_abs_diff": float(control_weight_max_abs_diff),
        "control_weight_mean_abs_diff": float(control_weight_mean_abs_diff),
        "top20_shared_max_abs_diff": [{"name": key, "max_abs_diff": value} for key, value in top_keys],
    }


def train_side_by_side(
    hf_model: torch.nn.Module,
    original_model: torch.nn.Module,
    original_criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    train_steps: int,
    hf_use_pixel_mask: bool = False,
) -> dict[str, Any]:
    hf_model.train()
    original_model.train()
    hf_optimizer = torch.optim.AdamW(hf_model.parameters(), lr=lr, weight_decay=weight_decay)
    original_optimizer = torch.optim.AdamW(original_model.parameters(), lr=lr, weight_decay=weight_decay)
    hf_losses: list[float] = []
    original_losses: list[float] = []
    hf_error: str | None = None
    original_error: str | None = None
    completed_steps = 0
    data_iter = iter(dataloader)

    for step in range(train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
        labels = to_device_labels(batch["labels"], device)

        # HF step
        hf_optimizer.zero_grad(set_to_none=True)
        try:
            if hf_use_pixel_mask and pixel_mask is not None:
                hf_outputs = hf_model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            else:
                hf_outputs = hf_model(pixel_values=pixel_values, labels=labels)
            hf_loss = hf_outputs.loss
            if not torch.isfinite(hf_loss):
                raise RuntimeError(f"Non-finite HF loss encountered: {hf_loss.detach().cpu().item()}")
            hf_loss.backward()
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), max_grad_norm)
            hf_optimizer.step()
            hf_losses.append(float(hf_loss.detach().cpu().item()))
        except Exception as error:  # noqa: BLE001
            hf_error = str(error)
            log(f"HF training error at step {step + 1}: {hf_error}")
            break

        # Original step
        original_optimizer.zero_grad(set_to_none=True)
        try:
            targets = to_original_targets(labels)
            original_outputs = original_model(list(pixel_values), targets=targets)
            original_loss_dict = original_criterion(original_outputs, targets)
            weighted_losses = [
                loss_value * original_criterion.weight_dict[loss_name]
                for loss_name, loss_value in original_loss_dict.items()
                if loss_name in original_criterion.weight_dict
            ]
            original_loss = sum(weighted_losses)
            if not torch.isfinite(original_loss):
                raise RuntimeError(f"Non-finite original loss encountered: {original_loss.detach().cpu().item()}")
            original_loss.backward()
            torch.nn.utils.clip_grad_norm_(original_model.parameters(), max_grad_norm)
            original_optimizer.step()
            original_losses.append(float(original_loss.detach().cpu().item()))
        except Exception as error:  # noqa: BLE001
            original_error = str(error)
            log(f"Original training error at step {step + 1}: {original_error}")
            break

        completed_steps += 1

        if (step + 1) % 5 == 0 or step == 0:
            log(f"step={step + 1}/{train_steps} hf_loss={hf_losses[-1]:.4f} original_loss={original_losses[-1]:.4f}")

    return {
        "completed_steps": completed_steps,
        "hf_error": hf_error,
        "original_error": original_error,
        "hf_loss": summarize_loss_series(hf_losses),
        "original_loss": summarize_loss_series(original_losses),
    }


def summarize_processed_bbox(samples: list[dict[str, Any]]) -> dict[str, Any]:
    all_boxes = []
    for sample in samples:
        boxes = sample["labels"]["boxes"]
        all_boxes.append(boxes)
    if not all_boxes:
        return {"num_boxes": 0, "min": 0.0, "max": 0.0}
    boxes = torch.cat(all_boxes, dim=0)
    return {
        "num_boxes": int(boxes.shape[0]),
        "min_value": float(boxes.min().item()),
        "max_value": float(boxes.max().item()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.disable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        log("TF32 disabled for CUDA matmul/cudnn.")

    log(f"Using device: {device}")
    if device.type != "cuda":
        log("Warning: CUDA is not available; this run will not use GPU.")

    dataset = load_dataset(args.dataset_id)
    train_split = dataset["train"].select(range(min(args.num_samples, len(dataset["train"]))))
    examples = [train_split[idx] for idx in range(len(train_split))]
    label_names = train_split.features["objects"]["category"].feature.names

    raw_bbox_summary = validate_raw_bbox_format(examples)
    log(f"Raw bbox summary: {raw_bbox_summary}")

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        do_resize=True,
        size={"max_height": args.image_size, "max_width": args.image_size},
        do_pad=True,
        pad_size={"height": args.image_size, "width": args.image_size},
    )
    processed_samples = build_processed_samples(examples, image_processor=image_processor, category_offset=1)
    processed_bbox_summary = summarize_processed_bbox(processed_samples)
    log(f"Processed bbox summary (normalized boxes): {processed_bbox_summary}")

    dataloader = DataLoader(
        processed_samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    hf_init_summary = get_hf_initialization_summary(args.model_id, label_names)
    log(f"HF initialization summary: {json.dumps(hf_init_summary, indent=2)}")

    # Comparison model uses unchanged classifier dimensions to match original checkpoint setup.
    if args.hf_load_from_original_checkpoint:
        hf_model, hf_loading_summary = load_hf_model_from_original_checkpoint(
            checkpoint_repo_id=args.checkpoint_repo_id,
            checkpoint_filename=args.checkpoint_filename,
            checkpoint_model_name=args.hf_checkpoint_model_name,
        )
        hf_model_source = "original_checkpoint_converted"
    else:
        comparison_config = AutoConfig.from_pretrained(args.model_id)
        comparison_config.dropout = 0.0
        hf_model, hf_loading_info = AutoModelForObjectDetection.from_pretrained(
            args.model_id, config=comparison_config, output_loading_info=True
        )
        hf_loading_summary = summarize_loading_info(hf_loading_info)
        hf_model_source = "hf_pretrained"

    hf_original_two_stage_head_init = {"applied": False, "reason": "flag_disabled"}
    if args.hf_copy_original_two_stage_head_init:
        hf_original_two_stage_head_init = apply_original_two_stage_head_init(hf_model)

    hf_attn_implementation = args.hf_attn_implementation
    if hf_attn_implementation != "auto":
        hf_model.config._attn_implementation = hf_attn_implementation

    hf_model.to(device)

    with tempfile.TemporaryDirectory(prefix="lw_detr_compare_") as tmpdir:
        tmp_path = Path(tmpdir)
        original_repo_dir = clone_original_repo(args.original_repo_url, args.original_repo_commit, tmp_path)
        original_model, original_criterion, original_loading_summary = init_original_model(
            repo_dir=original_repo_dir,
            checkpoint_repo_id=args.checkpoint_repo_id,
            checkpoint_filename=args.checkpoint_filename,
            device=device,
        )
        pretrain_forward_parity = compute_forward_parity(
            hf_model=hf_model,
            original_model=original_model,
            dataloader=dataloader,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )
        train_forward_parity = compute_train_forward_parity(
            hf_model=hf_model,
            original_model=original_model,
            dataloader=dataloader,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )
        loss_impl_parity = evaluate_loss_implementation_parity(
            hf_model=hf_model,
            original_model=original_model,
            original_criterion=original_criterion,
            dataloader=dataloader,
            device=device,
        )
        gradient_parity = {"enabled": False, "reason": "flag_disabled"}
        if args.compute_gradient_parity:
            gradient_parity = evaluate_gradient_parity(
                hf_model=hf_model,
                original_model=original_model,
                original_criterion=original_criterion,
                dataloader=dataloader,
                device=device,
                hf_use_pixel_mask=args.hf_use_pixel_mask,
            )

        pretrain_calibration_hf = safe_evaluate_score_calibration(
            hf_model,
            "hf",
            dataloader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )
        pretrain_calibration_original = safe_evaluate_score_calibration(
            original_model, "original", dataloader, image_processor=image_processor, device=device
        )

        training_summary = train_side_by_side(
            hf_model=hf_model,
            original_model=original_model,
            original_criterion=original_criterion,
            dataloader=dataloader,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            train_steps=args.train_steps,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )

        posttrain_calibration_hf = safe_evaluate_score_calibration(
            hf_model,
            "hf",
            dataloader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )
        posttrain_calibration_original = safe_evaluate_score_calibration(
            original_model, "original", dataloader, image_processor=image_processor, device=device
        )

    summary = {
        "dataset_id": args.dataset_id,
        "model_id": args.model_id,
        "num_samples": args.num_samples,
        "train_steps": args.train_steps,
        "hf_use_pixel_mask": bool(args.hf_use_pixel_mask),
        "hf_model_source": hf_model_source,
        "hf_checkpoint_model_name": args.hf_checkpoint_model_name if args.hf_load_from_original_checkpoint else None,
        "hf_copy_original_two_stage_head_init": bool(args.hf_copy_original_two_stage_head_init),
        "hf_attn_implementation": hf_attn_implementation,
        "disable_tf32": bool(args.disable_tf32),
        "device": str(device),
        "raw_bbox_summary": raw_bbox_summary,
        "processed_bbox_summary": processed_bbox_summary,
        "hf_initialization_summary": hf_init_summary,
        "hf_loading_summary_for_comparison_model": hf_loading_summary,
        "hf_original_two_stage_head_init": hf_original_two_stage_head_init,
        "original_loading_summary": original_loading_summary,
        "pretrain_forward_parity": pretrain_forward_parity,
        "train_forward_parity": train_forward_parity,
        "loss_impl_parity_on_original_outputs": loss_impl_parity,
        "gradient_parity_first_batch": gradient_parity,
        "pretrain_calibration_hf": pretrain_calibration_hf,
        "pretrain_calibration_original": pretrain_calibration_original,
        "training_summary": training_summary,
        "posttrain_calibration_hf": posttrain_calibration_hf,
        "posttrain_calibration_original": posttrain_calibration_original,
    }

    print("FINAL_SUMMARY_JSON_START")
    print(json.dumps(summary, indent=2))
    print("FINAL_SUMMARY_JSON_END")


if __name__ == "__main__":
    main()
