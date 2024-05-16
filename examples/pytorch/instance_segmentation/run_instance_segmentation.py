import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerConfig, MaskFormerImageProcessor



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
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


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


dataset = load_dataset("scene_parse_150", "instance_segmentation")
dataset_df = pd.read_csv(
    "https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/instanceInfo100_train.txt",
    sep="\t",
)

id2label = {row["Idx"]: row["Object Names"].strip() for _, row in dataset_df.iterrows()}
label2id = {label: id for id, label in id2label.items()}
print(id2label)


image_processor = Mask2FormerImageProcessor(
    do_resize=True,
    size={"height": 512, "width": 512},
    do_pad=False,
    do_rescale=True,
    do_normalize=True,
    reduce_labels=True,
    ignore_index=255,
)
augmentation_transform = A.Compose([])

train_augment_and_transform_batch = partial(augment_and_transform_batch, transform=augmentation_transform, image_processor=image_processor)
valid_transform_batch = partial(augment_and_transform_batch, transform=augmentation_transform, image_processor=image_processor)
test_transform_batch = partial(augment_and_transform_batch, transform=augmentation_transform, image_processor=image_processor)

dataset["train"] = dataset["train"].with_transform(train_augment_and_transform_batch)
dataset["validation"] = dataset["validation"].with_transform(valid_transform_batch)
dataset["test"] = dataset["test"].with_transform(test_transform_batch)

max_steps = 1000
if max_steps:
    dataset["train"] = dataset["train"].select(range(max_steps))
    dataset["validation"] = dataset["validation"].select(range(max_steps))
    dataset["test"] = dataset["test"].select(range(max_steps))

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-tiny-coco-instance",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=100,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    dataloader_num_workers=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    learning_rate=1e-5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collate_fn,
)

trainer.train()

# example = dataset["train"][1]

# image = np.array(example["image"])
# annotation = np.array(example["annotation"])[..., :2]

# semantic_seg = annotation[..., 0]
# instance_seg = annotation[..., 1]

# unique_semantic_instance_ids_pairs = np.unique(annotation.reshape(-1, 2), axis=0)
# instance2semantic = {pair[1]: pair[0] for pair in unique_semantic_instance_ids_pairs}

# inputs = processor(images=[image], segmentation_maps=[instance_seg], instance_id_to_semantic_id=instance2semantic, return_tensors="pt")
# print()