#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "torchvision",
#   "datasets>=3.0.0",
#   "numpy",
#   "albumentations>=1.4.16",
#   "torchmetrics",
#   "pycocotools",
#   "huggingface-hub",
#   "transformers",
#   "accelerate>=1.1.0",
# ]
# ///

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    EarlyStoppingCallback,
    LwDetrConfig,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.models.lw_detr.convert_lw_detr_to_hf import (
    ORIGINAL_TO_CONVERTED_KEY_MAPPING,
    backbone_read_in_q_k_v,
    convert_old_keys_to_new_keys,
    get_backbone_projector_sampling_key_mapping,
    get_model_config,
    read_in_q_k_v,
)
from transformers.trainer import EvalPrediction


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LW-DETR Trainer run with early stopping on tray-cart-detection.")
    parser.add_argument("--dataset-id", type=str, default="nielsr/tray-cart-detection")
    parser.add_argument("--model-id", type=str, default="AnnaZhang/lwdetr_tiny_60e_coco")

    parser.add_argument("--init-from-original-checkpoint", action="store_true", default=True)
    parser.add_argument("--checkpoint-repo-id", type=str, default="xbsu/LW-DETR")
    parser.add_argument("--checkpoint-filename", type=str, default="pretrain_weights/LWDETR_tiny_60e_coco.pth")
    parser.add_argument("--checkpoint-model-name", type=str, default="lwdetr_tiny_60e_coco")

    parser.add_argument("--output-dir", type=str, default="/tmp/lw_detr_trainer_early_stop")
    parser.add_argument("--output-json", type=str, default="")

    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--category-offset", type=int, default=1)
    parser.add_argument("--use-fast", action="store_true", default=False)
    parser.add_argument("--use-augmentation", action="store_true", default=False)

    parser.add_argument("--num-train-epochs", type=float, default=200.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--metric-for-best-model", type=str, default="eval_map_50")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--save-total-limit", type=int, default=3)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_select(split, max_samples: int | None):
    if max_samples is None:
        return split
    return split.select(range(min(max_samples, len(split))))


def format_image_annotations_as_coco(
    image_id: str,
    categories: list[int],
    bboxes: list[tuple[float]],
) -> dict[str, Any]:
    annotations = []
    for category, bbox in zip(categories, bboxes):
        w = float(bbox[2])
        h = float(bbox[3])
        annotations.append(
            {
                "image_id": image_id,
                "category_id": int(category),
                "iscrowd": 0,
                "area": w * h,
                "bbox": list(bbox),
            }
        )
    return {"image_id": image_id, "annotations": annotations}


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    category_offset: int = 1,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    images = []
    annotations = []

    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))
        categories = [int(c) + category_offset for c in objects["category"]]
        output = transform(image=image, bboxes=objects["bbox"], category=categories)

        images.append(output["image"])
        annotations.append(
            format_image_annotations_as_coco(
                image_id=str(image_id),
                categories=output["category"],
                bboxes=output["bboxes"],
            )
        )

    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    if not return_pixel_mask:
        result.pop("pixel_mask", None)
    return result


def collate_fn(batch: list[BatchFeature]) -> Mapping[str, torch.Tensor | list[Any]]:
    data = {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


def _extract_hw(image_size: Any) -> tuple[float, float] | None:
    size = torch.as_tensor(image_size).reshape(-1).to(dtype=torch.float32)
    if size.numel() >= 2:
        height = float(size[-2].item())
        width = float(size[-1].item())
        return height, width
    if size.numel() == 1:
        value = float(size[-1].item())
        return value, value
    return None


def _extract_hw_from_target(image_target: Mapping[str, Any]) -> tuple[float, float]:
    for key in ("orig_size", "size"):
        if key in image_target:
            hw = _extract_hw(image_target[key])
            if hw is not None:
                return hw
    raise ValueError(f"Could not infer image size from target keys: {list(image_target.keys())}")


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Any) -> torch.Tensor:
    boxes = center_to_corners_format(boxes)
    height, width = _extract_hw(image_size)
    scale = torch.tensor([[width, height, width, height]], dtype=boxes.dtype)
    boxes = boxes * scale
    return boxes


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
) -> Mapping[str, float]:
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    for batch in targets:
        batch_image_sizes = torch.tensor([_extract_hw_from_target(x) for x in batch], dtype=torch.float32)
        image_sizes.append(batch_image_sizes)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, _extract_hw_from_target(image_target))
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        processed = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(processed)

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    output_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                output_metrics[key] = float(value.item())
        elif isinstance(value, (float, int)):
            output_metrics[key] = float(value)

    return output_metrics


def load_model_from_original_checkpoint(
    checkpoint_repo_id: str,
    checkpoint_filename: str,
    checkpoint_model_name: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = hf_hub_download(repo_id=checkpoint_repo_id, filename=checkpoint_filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config_dict = get_model_config(checkpoint_model_name)
    model_config = LwDetrConfig(**model_config_dict)
    model_config.dropout = 0.0

    model = AutoModelForObjectDetection.from_config(model_config)
    if "state_dict" in checkpoint:
        original_state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        original_state_dict = checkpoint["model"]
    else:
        original_state_dict = checkpoint

    backbone_projector_sampling_key_mapping = get_backbone_projector_sampling_key_mapping(model_config)
    state_dict = backbone_read_in_q_k_v(original_state_dict, model_config)
    state_dict = read_in_q_k_v(state_dict, model_config)
    key_mapping = ORIGINAL_TO_CONVERTED_KEY_MAPPING | backbone_projector_sampling_key_mapping
    all_keys = list(state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys, key_mapping)

    converted_state_dict = {}
    for key in all_keys:
        if not any(key.startswith(prefix) for prefix in ["class_embed", "bbox_embed"]):
            converted_state_dict["model." + new_keys[key]] = state_dict[key]
        else:
            converted_state_dict[key] = state_dict[key]

    load_result = model.load_state_dict(converted_state_dict, strict=False)

    return model, {
        "source": "original_checkpoint_converted",
        "checkpoint_repo_id": checkpoint_repo_id,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_model_name": checkpoint_model_name,
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.disable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        log("TF32 disabled for CUDA matmul/cudnn.")

    dataset = load_dataset(args.dataset_id)
    dataset["train"] = maybe_select(dataset["train"], args.max_train_samples)
    dataset["validation"] = maybe_select(dataset["validation"], args.max_val_samples)
    dataset["test"] = maybe_select(dataset["test"], args.max_test_samples)

    log(f"Loaded dataset train={len(dataset['train'])} val={len(dataset['validation'])} test={len(dataset['test'])}")

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        do_resize=True,
        size={"max_height": args.image_size, "max_width": args.image_size},
        do_pad=True,
        pad_size={"height": args.image_size, "width": args.image_size},
        use_fast=args.use_fast,
    )

    if args.use_augmentation:
        train_transform = A.Compose(
            [
                A.Compose(
                    [
                        A.SmallestMaxSize(max_size=args.image_size, p=1.0),
                        A.RandomSizedBBoxSafeCrop(height=args.image_size, width=args.image_size, p=1.0),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.Blur(blur_limit=7, p=0.5),
                        A.MotionBlur(blur_limit=7, p=0.5),
                        A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                    ],
                    p=0.1,
                ),
                A.Perspective(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1),
        )
    else:
        train_transform = A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1),
        )

    eval_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    dataset["train"] = dataset["train"].with_transform(
        partial(
            augment_and_transform_batch,
            transform=train_transform,
            image_processor=image_processor,
            category_offset=args.category_offset,
        )
    )
    dataset["validation"] = dataset["validation"].with_transform(
        partial(
            augment_and_transform_batch,
            transform=eval_transform,
            image_processor=image_processor,
            category_offset=args.category_offset,
        )
    )
    dataset["test"] = dataset["test"].with_transform(
        partial(
            augment_and_transform_batch,
            transform=eval_transform,
            image_processor=image_processor,
            category_offset=args.category_offset,
        )
    )

    if args.init_from_original_checkpoint:
        model, model_loading_summary = load_model_from_original_checkpoint(
            checkpoint_repo_id=args.checkpoint_repo_id,
            checkpoint_filename=args.checkpoint_filename,
            checkpoint_model_name=args.checkpoint_model_name,
        )
    else:
        model_config = AutoConfig.from_pretrained(args.model_id)
        model_config.dropout = 0.0
        model = AutoModelForObjectDetection.from_pretrained(args.model_id, config=model_config)
        model_loading_summary = {"source": "hf_pretrained"}

    log(f"Model loading summary: {model_loading_summary}")

    compute_metrics_fn = partial(compute_metrics, image_processor=image_processor, threshold=0.0)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        bf16=bool(args.bf16 and torch.cuda.is_available()),
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )

    start = time.time()
    train_result = trainer.train()
    runtime = time.time() - start

    val_metrics = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")

    summary = {
        "dataset_id": args.dataset_id,
        "model_id": args.model_id,
        "device": str(device),
        "runtime_seconds": runtime,
        "train_samples": len(dataset["train"]),
        "validation_samples": len(dataset["validation"]),
        "test_samples": len(dataset["test"]),
        "init_from_original_checkpoint": bool(args.init_from_original_checkpoint),
        "model_loading_summary": model_loading_summary,
        "training_args": {
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "metric_for_best_model": args.metric_for_best_model,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "category_offset": args.category_offset,
            "use_augmentation": bool(args.use_augmentation),
            "disable_tf32": bool(args.disable_tf32),
        },
        "trainer_state": {
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "epoch": trainer.state.epoch,
            "global_step": trainer.state.global_step,
        },
        "train_metrics": train_result.metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
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
