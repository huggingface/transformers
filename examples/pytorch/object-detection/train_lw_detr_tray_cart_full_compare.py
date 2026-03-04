#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "datasets>=3.0.0",
#   "numpy",
#   "huggingface-hub",
#   "timm",
#   "scipy",
#   "torchmetrics",
#   "pycocotools",
#   "transformers",
# ]
# ///

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection, LwDetrConfig
from transformers.image_transforms import center_to_corners_format
from transformers.models.lw_detr.convert_lw_detr_to_hf import (
    convert_original_checkpoint_state_dict,
    get_checkpoint_state_dict,
    get_model_config,
)


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LW-DETR HF and original side-by-side on nielsr/tray-cart-detection using "
            "original-like optimizer/scheduler settings."
        )
    )
    parser.add_argument("--dataset-id", type=str, default="nielsr/tray-cart-detection")
    parser.add_argument("--model-id", type=str, default="AnnaZhang/lwdetr_tiny_60e_coco")
    parser.add_argument("--checkpoint-repo-id", type=str, default="xbsu/LW-DETR")
    parser.add_argument("--checkpoint-filename", type=str, default="pretrain_weights/LWDETR_tiny_60e_coco.pth")
    parser.add_argument("--checkpoint-model-name", type=str, default="lwdetr_tiny_60e_coco")

    parser.add_argument("--original-repo-url", type=str, default="https://github.com/NielsRogge/LW-DETR.git")
    parser.add_argument("--original-repo-commit", type=str, default="d5e6e6c4add2d24dafb965ced8b50163c50b9788")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-drop", type=int, default=60)
    parser.add_argument("--lr-vit-layer-decay", type=float, default=0.8)
    parser.add_argument("--lr-component-decay", type=float, default=0.7)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)

    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--category-offset", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument(
        "--hf-init-source",
        type=str,
        default="original_checkpoint",
        choices=["original_checkpoint", "hf_pretrained"],
    )
    parser.add_argument(
        "--hf-use-pixel-mask",
        action="store_true",
        help="If set, pass pixel_mask to HF model. Default is False to follow original list-input behavior.",
    )
    parser.add_argument(
        "--hf-attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa"],
    )
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-original", action="store_true", default=True)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_cmd(command: list[str], cwd: Path | None = None) -> None:
    import subprocess

    log(f"Running command: {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, check=True)


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


def build_processed_samples(
    split,
    image_processor: AutoImageProcessor,
    category_offset: int,
) -> list[dict[str, Any]]:
    images = []
    annotations = []
    for sample in split:
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
    for idx in range(len(images)):
        sample = {
            "pixel_values": outputs["pixel_values"][idx],
            "labels": outputs["labels"][idx],
        }
        if pixel_mask is not None:
            sample["pixel_mask"] = pixel_mask[idx]
        samples.append(sample)
    return samples


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in batch]),
        "labels": [sample["labels"] for sample in batch],
    }
    if "pixel_mask" in batch[0]:
        collated["pixel_mask"] = torch.stack([sample["pixel_mask"] for sample in batch])
    return collated


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


def clone_original_repo(repo_url: str, commit: str, root_dir: Path) -> Path:
    repo_dir = root_dir / "lw_detr_original"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run_cmd(["git", "clone", repo_url, str(repo_dir)])
    run_cmd(["git", "checkout", commit], cwd=repo_dir)
    return repo_dir


def build_original_args(device: str, checkpoint_args: Any | None = None) -> argparse.Namespace:
    defaults = {
        "lr": 1e-4,
        "lr_encoder": 1.5e-4,
        "batch_size": 4,
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
        "num_queries": 100,
        "group_detr": 13,
        "two_stage": True,
        "projector_scale": ["P4"],
        "lite_refpoint_refine": True,
        "num_select": 100,
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
        "num_feature_levels": 1,
        "use_ema": False,
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
) -> tuple[torch.nn.Module, torch.nn.Module, argparse.Namespace, dict[str, Any]]:
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
        from util.get_param_dicts import get_param_dict
    finally:
        sys.path.pop(0)

    checkpoint_path = hf_hub_download(repo_id=checkpoint_repo_id, filename=checkpoint_filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    args = build_original_args(device=str(device), checkpoint_args=checkpoint_args)
    args.pretrained_encoder = None
    args.pretrain_weights = None
    args.resume = ""

    model, criterion, _ = build_original_model(args)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=False)

    fallback_count = 0
    for module in model.modules():
        if module.__class__.__name__ == "MSDeformAttn":
            module._export = True
            fallback_count += 1

    model.to(device)
    criterion.to(device)

    param_dicts = get_param_dict(args, model)

    loading_summary = {
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
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

    return model, criterion, args, {"param_dicts": param_dicts, **loading_summary}


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
        "source": "original_checkpoint_converted",
        "checkpoint_repo_id": checkpoint_repo_id,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_model_name": checkpoint_model_name,
        "checkpoint_path": checkpoint_path,
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
    }


def _backbone_layer_id(name: str, num_layers: int) -> int:
    layer_id = num_layers + 1
    if ".embeddings.position_embeddings" in name or ".embeddings.projection" in name:
        layer_id = 0
    elif ".encoder.layer." in name:
        marker = ".encoder.layer."
        layer_part = name.split(marker, maxsplit=1)[1]
        layer_idx = int(layer_part.split(".", maxsplit=1)[0])
        layer_id = layer_idx + 1
    return layer_id


def _weight_decay_multiplier(name: str) -> float:
    lowered = name.lower()
    if (
        "gamma" in name
        or "position_embeddings" in name
        or "rel_pos" in name
        or name.endswith(".bias")
        or "norm" in lowered
    ):
        return 0.0
    return 1.0


def build_hf_param_groups_like_original(
    model: torch.nn.Module,
    lr: float,
    lr_encoder: float,
    lr_vit_layer_decay: float,
    lr_component_decay: float,
    weight_decay: float,
    vit_encoder_num_layers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    param_groups: list[dict[str, Any]] = []
    counts = {"backbone_encoder": 0, "decoder": 0, "other": 0}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("model.backbone.backbone."):
            layer_id = _backbone_layer_id(name, vit_encoder_num_layers)
            lr_value = (
                lr_encoder * (lr_vit_layer_decay ** (vit_encoder_num_layers + 1 - layer_id)) * (lr_component_decay**2)
            )
            wd_value = weight_decay * _weight_decay_multiplier(name)
            counts["backbone_encoder"] += 1
        elif "model.decoder" in name:
            lr_value = lr * lr_component_decay
            wd_value = weight_decay
            counts["decoder"] += 1
        else:
            lr_value = lr
            wd_value = weight_decay
            counts["other"] += 1

        param_groups.append(
            {
                "params": [param],
                "lr": float(lr_value),
                "weight_decay": float(wd_value),
            }
        )

    unique_lrs = sorted({round(group["lr"], 12) for group in param_groups})
    unique_wd = sorted({round(group["weight_decay"], 12) for group in param_groups})
    summary = {
        "group_count": len(param_groups),
        "parameter_partition_counts": counts,
        "unique_lrs": unique_lrs,
        "unique_weight_decays": unique_wd,
    }
    return param_groups, summary


def summarize_loss_series(series: list[float]) -> dict[str, Any]:
    if not series:
        return {"first": None, "last": None, "mean": None, "min": None}
    return {
        "first": float(series[0]),
        "last": float(series[-1]),
        "mean": float(sum(series) / len(series)),
        "min": float(min(series)),
    }


def _outputs_for_postprocess(outputs: Any) -> SimpleNamespace:
    if isinstance(outputs, dict):
        logits = outputs.get("pred_logits", outputs.get("logits"))
        pred_boxes = outputs.get("pred_boxes", outputs.get("boxes"))
    else:
        logits = getattr(outputs, "pred_logits", None)
        if logits is None:
            logits = getattr(outputs, "logits")
        pred_boxes = getattr(outputs, "pred_boxes", None)
        if pred_boxes is None:
            pred_boxes = getattr(outputs, "boxes")
    if logits is None or pred_boxes is None:
        raise ValueError("Could not extract logits/pred_boxes for post-processing.")
    return SimpleNamespace(logits=logits, pred_boxes=pred_boxes)


def _map_targets_from_labels(labels: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    targets = []
    for label in labels:
        boxes_cxcywh = label["boxes"]
        orig_h, orig_w = label["orig_size"]
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype)
        boxes_xyxy = center_to_corners_format(boxes_cxcywh) * scale
        targets.append(
            {
                "boxes": boxes_xyxy.detach().cpu(),
                "labels": label["class_labels"].detach().cpu(),
            }
        )
    return targets


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    model_kind: str,
    dataloader: DataLoader,
    image_processor: AutoImageProcessor,
    device: torch.device,
    hf_use_pixel_mask: bool,
) -> dict[str, Any]:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    aggregate = {
        "0.0": {"num_predictions": 0.0, "max_score": 0.0, "mean_score_sum": 0.0},
        "0.5": {"num_predictions": 0.0, "max_score": 0.0, "mean_score_sum": 0.0},
        "num_images": 0.0,
    }

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = to_device_labels(batch["labels"], device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        if model_kind == "hf":
            if hf_use_pixel_mask and pixel_mask is not None:
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            else:
                outputs = model(pixel_values=pixel_values)
        else:
            outputs = model(list(pixel_values))

        wrapped_outputs = _outputs_for_postprocess(outputs)
        target_sizes = torch.stack([label["orig_size"] for label in labels], dim=0)

        detections_0 = image_processor.post_process_object_detection(
            wrapped_outputs, target_sizes=target_sizes, threshold=0.0
        )
        detections_05 = image_processor.post_process_object_detection(
            wrapped_outputs, target_sizes=target_sizes, threshold=0.5
        )

        preds = [
            {
                "boxes": det["boxes"].detach().cpu(),
                "scores": det["scores"].detach().cpu(),
                "labels": det["labels"].detach().cpu(),
            }
            for det in detections_0
        ]
        targets = _map_targets_from_labels(labels)
        metric.update(preds, targets)

        for det_0, det_05 in zip(detections_0, detections_05):
            scores_0 = det_0["scores"].detach().cpu().tolist()
            scores_05 = det_05["scores"].detach().cpu().tolist()
            aggregate["num_images"] += 1.0

            aggregate["0.0"]["num_predictions"] += float(len(scores_0))
            aggregate["0.0"]["max_score"] = max(
                aggregate["0.0"]["max_score"], float(max(scores_0) if scores_0 else 0.0)
            )
            aggregate["0.0"]["mean_score_sum"] += float(sum(scores_0) / len(scores_0) if scores_0 else 0.0)

            aggregate["0.5"]["num_predictions"] += float(len(scores_05))
            aggregate["0.5"]["max_score"] = max(
                aggregate["0.5"]["max_score"], float(max(scores_05) if scores_05 else 0.0)
            )
            aggregate["0.5"]["mean_score_sum"] += float(sum(scores_05) / len(scores_05) if scores_05 else 0.0)

    metric_values = metric.compute()
    num_images = max(1.0, aggregate["num_images"])

    return {
        "map": float(metric_values["map"].item()),
        "map_50": float(metric_values["map_50"].item()),
        "map_75": float(metric_values["map_75"].item()),
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


def train_one_epoch_side_by_side(
    hf_model: torch.nn.Module,
    original_model: torch.nn.Module,
    original_criterion: torch.nn.Module,
    dataloader: DataLoader,
    hf_optimizer: torch.optim.Optimizer,
    original_optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
    hf_use_pixel_mask: bool,
) -> dict[str, Any]:
    hf_model.train()
    original_model.train()
    original_criterion.train()

    hf_losses: list[float] = []
    original_losses: list[float] = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = to_device_labels(batch["labels"], device)

        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        hf_optimizer.zero_grad(set_to_none=True)
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

        original_optimizer.zero_grad(set_to_none=True)
        targets = to_original_targets(labels)
        original_outputs = original_model(list(pixel_values), targets=targets)
        original_loss_dict = original_criterion(original_outputs, targets)
        original_loss = sum(
            loss_value * original_criterion.weight_dict[loss_name]
            for loss_name, loss_value in original_loss_dict.items()
            if loss_name in original_criterion.weight_dict
        )
        if not torch.isfinite(original_loss):
            raise RuntimeError(f"Non-finite original loss encountered: {original_loss.detach().cpu().item()}")
        original_loss.backward()
        torch.nn.utils.clip_grad_norm_(original_model.parameters(), max_grad_norm)
        original_optimizer.step()

        hf_losses.append(float(hf_loss.detach().cpu().item()))
        original_losses.append(float(original_loss.detach().cpu().item()))

    return {
        "hf": summarize_loss_series(hf_losses),
        "original": summarize_loss_series(original_losses),
    }


def maybe_select(split, max_samples: int | None):
    if max_samples is None:
        return split
    return split.select(range(min(max_samples, len(split))))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.disable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        log("TF32 disabled for CUDA matmul/cudnn.")

    if device.type != "cuda":
        log("Warning: CUDA is not available. This full run will be slow.")

    dataset = load_dataset(args.dataset_id)
    train_split = maybe_select(dataset["train"], args.max_train_samples)
    val_split = maybe_select(dataset["validation"], args.max_val_samples)
    test_split = maybe_select(dataset["test"], args.max_test_samples)

    log(
        f"Loaded dataset splits train={len(train_split)} val={len(val_split)} test={len(test_split)} "
        f"from {args.dataset_id}"
    )

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        do_resize=True,
        size={"max_height": args.image_size, "max_width": args.image_size},
        do_pad=True,
        pad_size={"height": args.image_size, "width": args.image_size},
    )

    log("Preprocessing dataset with AutoImageProcessor...")
    train_samples = build_processed_samples(train_split, image_processor, category_offset=args.category_offset)
    val_samples = build_processed_samples(val_split, image_processor, category_offset=args.category_offset)
    test_samples = build_processed_samples(test_split, image_processor, category_offset=args.category_offset)

    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_samples,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_samples,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if args.hf_init_source == "original_checkpoint":
        hf_model, hf_loading_summary = load_hf_model_from_original_checkpoint(
            checkpoint_repo_id=args.checkpoint_repo_id,
            checkpoint_filename=args.checkpoint_filename,
            checkpoint_model_name=args.checkpoint_model_name,
        )
    else:
        hf_config = AutoConfig.from_pretrained(args.model_id)
        hf_config.dropout = 0.0
        hf_model, loading_info = AutoModelForObjectDetection.from_pretrained(
            args.model_id,
            config=hf_config,
            output_loading_info=True,
        )
        hf_loading_summary = {
            "source": "hf_pretrained",
            "missing_keys_count": len(loading_info.get("missing_keys", [])),
            "unexpected_keys_count": len(loading_info.get("unexpected_keys", [])),
            "mismatched_keys_count": len(loading_info.get("mismatched_keys", [])),
        }

    if args.hf_attn_implementation != "auto":
        hf_model.config._attn_implementation = args.hf_attn_implementation
    hf_model.to(device)

    with tempfile.TemporaryDirectory(prefix="lw_detr_full_train_") as tmpdir:
        tmp_path = Path(tmpdir)
        original_repo_dir = clone_original_repo(args.original_repo_url, args.original_repo_commit, tmp_path)

        original_model, original_criterion, original_args, original_loading_summary = init_original_model(
            repo_dir=original_repo_dir,
            checkpoint_repo_id=args.checkpoint_repo_id,
            checkpoint_filename=args.checkpoint_filename,
            device=device,
        )

        hf_param_groups, hf_optimizer_summary = build_hf_param_groups_like_original(
            model=hf_model,
            lr=args.lr,
            lr_encoder=args.lr_encoder,
            lr_vit_layer_decay=args.lr_vit_layer_decay,
            lr_component_decay=args.lr_component_decay,
            weight_decay=args.weight_decay,
            vit_encoder_num_layers=getattr(original_args, "vit_encoder_num_layers", 10),
        )

        hf_optimizer = torch.optim.AdamW(hf_param_groups, lr=args.lr, weight_decay=args.weight_decay)
        hf_scheduler = torch.optim.lr_scheduler.StepLR(hf_optimizer, step_size=args.lr_drop)

        original_optimizer = torch.optim.AdamW(
            original_loading_summary["param_dicts"],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        original_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=args.lr_drop)

        del original_loading_summary["param_dicts"]

        initial_val_hf = evaluate_model(
            model=hf_model,
            model_kind="hf",
            dataloader=val_loader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )
        initial_val_original = evaluate_model(
            model=original_model,
            model_kind="original",
            dataloader=val_loader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )

        epoch_logs = []
        best_val_map50 = {
            "hf": float(initial_val_hf["map_50"]),
            "original": float(initial_val_original["map_50"]),
        }

        start_time = time.time()
        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_summary = train_one_epoch_side_by_side(
                hf_model=hf_model,
                original_model=original_model,
                original_criterion=original_criterion,
                dataloader=train_loader,
                hf_optimizer=hf_optimizer,
                original_optimizer=original_optimizer,
                device=device,
                max_grad_norm=args.max_grad_norm,
                hf_use_pixel_mask=args.hf_use_pixel_mask,
            )

            hf_scheduler.step()
            original_scheduler.step()

            epoch_log: dict[str, Any] = {
                "epoch": epoch + 1,
                "hf_train_loss": train_summary["hf"],
                "original_train_loss": train_summary["original"],
                "epoch_seconds": float(time.time() - epoch_start),
            }

            do_eval = (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs
            if do_eval:
                val_hf = evaluate_model(
                    model=hf_model,
                    model_kind="hf",
                    dataloader=val_loader,
                    image_processor=image_processor,
                    device=device,
                    hf_use_pixel_mask=args.hf_use_pixel_mask,
                )
                val_original = evaluate_model(
                    model=original_model,
                    model_kind="original",
                    dataloader=val_loader,
                    image_processor=image_processor,
                    device=device,
                    hf_use_pixel_mask=args.hf_use_pixel_mask,
                )

                best_val_map50["hf"] = max(best_val_map50["hf"], float(val_hf["map_50"]))
                best_val_map50["original"] = max(best_val_map50["original"], float(val_original["map_50"]))

                epoch_log["val_hf"] = val_hf
                epoch_log["val_original"] = val_original

                log(
                    "epoch=%d/%d hf_loss=%.4f orig_loss=%.4f hf_val_map50=%.4f orig_val_map50=%.4f "
                    "hf_val_preds@0.5=%.2f orig_val_preds@0.5=%.2f"
                    % (
                        epoch + 1,
                        args.epochs,
                        train_summary["hf"]["last"],
                        train_summary["original"]["last"],
                        val_hf["map_50"],
                        val_original["map_50"],
                        val_hf["threshold_0.5"]["avg_num_predictions"],
                        val_original["threshold_0.5"]["avg_num_predictions"],
                    )
                )
            else:
                log(
                    "epoch=%d/%d hf_loss=%.4f orig_loss=%.4f"
                    % (
                        epoch + 1,
                        args.epochs,
                        train_summary["hf"]["last"],
                        train_summary["original"]["last"],
                    )
                )

            epoch_logs.append(epoch_log)

        final_val_hf = evaluate_model(
            model=hf_model,
            model_kind="hf",
            dataloader=val_loader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )
        final_val_original = evaluate_model(
            model=original_model,
            model_kind="original",
            dataloader=val_loader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )
        final_test_hf = evaluate_model(
            model=hf_model,
            model_kind="hf",
            dataloader=test_loader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )
        final_test_original = evaluate_model(
            model=original_model,
            model_kind="original",
            dataloader=test_loader,
            image_processor=image_processor,
            device=device,
            hf_use_pixel_mask=args.hf_use_pixel_mask,
        )

        total_seconds = float(time.time() - start_time)

    summary = {
        "dataset_id": args.dataset_id,
        "model_id": args.model_id,
        "checkpoint": {
            "repo_id": args.checkpoint_repo_id,
            "filename": args.checkpoint_filename,
            "model_name": args.checkpoint_model_name,
        },
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "eval_every": args.eval_every,
            "image_size": args.image_size,
            "lr": args.lr,
            "lr_encoder": args.lr_encoder,
            "lr_drop": args.lr_drop,
            "lr_vit_layer_decay": args.lr_vit_layer_decay,
            "lr_component_decay": args.lr_component_decay,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "hf_use_pixel_mask": bool(args.hf_use_pixel_mask),
            "hf_init_source": args.hf_init_source,
            "hf_attn_implementation": args.hf_attn_implementation,
            "disable_tf32": bool(args.disable_tf32),
        },
        "device": str(device),
        "dataset_sizes": {
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
        },
        "hf_loading_summary": hf_loading_summary,
        "hf_optimizer_summary": hf_optimizer_summary,
        "original_loading_summary": original_loading_summary,
        "initial_validation": {
            "hf": initial_val_hf,
            "original": initial_val_original,
        },
        "final_validation": {
            "hf": final_val_hf,
            "original": final_val_original,
        },
        "final_test": {
            "hf": final_test_hf,
            "original": final_test_original,
        },
        "best_val_map50": best_val_map50,
        "runtime_seconds": total_seconds,
        "epoch_logs": epoch_logs,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        log(f"Saved summary JSON to {output_path}")

    print("FINAL_SUMMARY_JSON_START")
    print(json.dumps(summary, indent=2))
    print("FINAL_SUMMARY_JSON_END")


if __name__ == "__main__":
    main()
