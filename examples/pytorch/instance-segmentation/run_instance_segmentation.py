import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
    MaskFormerImageProcessor,
    MaskFormerForInstanceSegmentation,
)

from transformers import AutoModelForInstanceSegmentation, AutoImageProcessor


import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction, nested_numpify
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def augment_and_transform_batch(
    examples: Mapping[str, Any], transform: A.Compose, image_processor: Mask2FormerImageProcessor
) -> BatchFeature:
    
    batch = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }

    for pil_image, pil_annotation in zip(examples["image"], examples["annotation"]):
        
        image = np.array(pil_image)
        semantic_and_instance_masks = np.array(pil_annotation)[..., :2]

        # Apply augmentations
        output = transform(image=image, mask=semantic_and_instance_masks)

        aug_image = output["image"]
        aug_semantic_and_instance_masks = output["mask"]
        aug_instance_mask = aug_semantic_and_instance_masks[..., 1]

        # Create mapping from instance id to semantic id
        unique_semantic_id_instance_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
        instance_id_to_semantic_id = {
            instance_id: semantic_id 
            for semantic_id, instance_id in unique_semantic_id_instance_id_pairs
        }

        # Apply the image processor transformations: resizing, rescaling, normalization
        model_inputs = image_processor(
            images=[aug_image],
            segmentation_maps=[aug_instance_mask],
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            return_tensors="pt",
        )

        batch["pixel_values"].append(model_inputs.pixel_values[0])
        batch["mask_labels"].append(model_inputs.mask_labels[0])
        batch["class_labels"].append(model_inputs.class_labels[0])

    return batch


def collate_fn(examples):
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    return batch


@dataclass
class ModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor


def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    t = tensors.cpu().detach()
    return t


class Evaluator:

    def __init__(self, image_processor: Mask2FormerImageProcessor, id2label: Mapping[int, str], threshold: float = 0.5):
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = self.get_metric()

    def get_metric(self):
        return MeanAveragePrecision(iou_type="segm", class_metrics=True)

    @torch.no_grad()
    def __call__(self, evaluation_results: EvalPrediction, compute_result: bool = False) -> Mapping[str, float]:
        """
        Compute mean average mAP, mAR and their variants for the object detection task.

        Args:
            evaluation_results (EvalPrediction): Predictions and targets from evaluation.
            threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
            id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

        Returns:
            Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
        """

        prediction_batch = nested_cpu(evaluation_results.predictions)
        target_batch = nested_cpu(evaluation_results.label_ids)
        del evaluation_results

        # For metric computation we need to provide:
        #  - targets in a form of list of dictionaries with keys "masks", "labels"
        #  - predictions in a form of list of dictionaries with keys "masks", "labels", "scores"

        post_processed_targets = []
        post_processed_predictions = []
        target_sizes = []

        # Collect targets
        for masks, labels in zip(target_batch[0], target_batch[1]):
            post_processed_targets.append({
                "masks": torch.tensor(masks, dtype=torch.bool),
                "labels": torch.tensor(labels),
            })
            # keep = [i for i, label in enumerate(labels) if label.item() != 0]
            # post_processed_targets[-1]["masks"] = post_processed_targets[-1]["masks"][keep]
            # post_processed_targets[-1]["labels"] = post_processed_targets[-1]["labels"][keep]
            target_sizes.append(masks.shape[-2:])
        
        # Collect predictions
        class_queries_logits = torch.tensor(prediction_batch[0])
        masks_queries_logits = torch.tensor(prediction_batch[1])
        model_output = ModelOutput(class_queries_logits=class_queries_logits, masks_queries_logits=masks_queries_logits)
        post_processed_output = self.image_processor.post_process_instance_segmentation(
            model_output,
            threshold=self.threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )
        for image_predictions, target_size in zip(post_processed_output, target_sizes):
            has_masks = bool(image_predictions["segments_info"])
            if has_masks:
                post_processed_image_prediction = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                    "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]]),
                    "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]]),
                }
            else:
                # for void predictions, we need to provide empty tensors
                post_processed_image_prediction = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                    "labels": torch.tensor([]),
                    "scores": torch.tensor([]),
                }
            post_processed_predictions.append(post_processed_image_prediction)

        # Compute metrics
        self.metric.update(post_processed_predictions, post_processed_targets)

        if not compute_result:
            return

        metrics = self.metric.compute()
        self.metric = self.get_metric()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar
        
        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics


if __name__ == "__main__":

    # Dataset
    dataset = load_dataset("qubvel-hf/ade20k-mini")
    label2id = dataset["train"][0]["semantic_class_to_id"]


    # checkpoint = "facebook/mask2former-swin-tiny-coco-instance"
    checkpoint = "facebook/maskformer-swin-tiny-ade"

    # Image transformations
    image_processor = MaskFormerImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"height": 256, "width": 256},
        do_reduce_labels=True,
    )

    augmentation_transform = A.Compose([])

    train_augment_and_transform_batch = partial(augment_and_transform_batch, transform=augmentation_transform, image_processor=image_processor)
    valid_transform_batch = partial(augment_and_transform_batch, transform=augmentation_transform, image_processor=image_processor)
    dataset["train"] = dataset["train"].with_transform(train_augment_and_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(valid_transform_batch)

    # dataset["validation"] = dataset["train"].select(range(16))
    dataset["train"] = dataset["train"].select(list(range(8)))


    # Model
    if image_processor.do_reduce_labels:
        label2id.pop("background")
        label2id = {k: i - 1 for k, i in label2id.items()}

    id2label = {v: k for k, v in label2id.items()}

    model = MaskFormerForInstanceSegmentation.from_pretrained(
        checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir="finetune-instance-segmentation-ade20k-mini-maskformer",
        num_train_epochs=40,
        do_train=True,
        do_eval=True,
        fp16=True,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        save_total_limit=2,
        batch_eval_metrics=True,
        # max_steps=1,
    )

    eval_compute_metrics = Evaluator(image_processor=image_processor, id2label=id2label, threshold=0.0)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], # overfit for debugging
        data_collator=collate_fn,
        compute_metrics=dummy_compute_metrics, #eval_compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
