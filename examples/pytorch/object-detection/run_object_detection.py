#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.data.data_collator import torch_default_data_collator
from transformers import DetrForObjectDetection, DetrImageProcessor
from typing import List, Tuple
from transformers.image_transforms import center_to_corners_format

"""Finetuning any 🤗 Transformers model supported by AutoModelForObjectDetection for object detection leveraging the Trainer API."""

os.environ["WANDB_PROJECT"] = "detr-resnet50-cppe5-finetuning"

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.40.0.dev0")
require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/semantic-segmentation/requirements.txt")


def format_image_annotations_as_coco(image_id: str, category: List[int], area: List[float], bbox: List[Tuple[float]]) -> dict:
    """Format one image annotations to COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        category (List[int]): list of categories/class labels corresponding to provided bounding boxes
        area (List[float]): list of corresponding areas to provided bounding boxes
        bbox (List[Tuple[float]]): list of bounding boxes provided in COCO format 
            ([center_x, center_y, width, height] in absoulute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formated annotations
        }
    """
    annotations = []
    for i in range(len(category)):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category[i],
            "iscrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(examples, transform, image_processor):

    images = []
    annotations = []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    return result


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "pixel_mask": torch.stack([x["pixel_mask"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }


def convert_boxes_to_absolute_coordinates(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    """Image size: [height, width] tensor"""
    height, width = image_size
    scale_factor = torch.stack([width, height, width, height])
    # convert shape for multiplication: (4,) -> (1, 4)
    scale_factor = scale_factor.unsqueeze(0).to(boxes.device)  
    boxes = boxes * scale_factor
    return boxes


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default="cppe-5",
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    shortest_edge: Optional[int] = field(
        default=800, metadata={"help": "Processing image maximum shortest edge size"}
    )
    longest_edge: Optional[int] = field(
        default=1333, metadata={"help": "Processing image maximum longest edge size"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/detr-resnet-50",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    ignore_mismatched_sizes: bool = field(default=False, metadata={"help": "Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model (if for instance, you are instantiating a model with 10 labels from a checkpoint with 3 labels)."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_object_dection", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits
    # ------------------------------------------------------------------------------------------------

    dataset = load_dataset(data_args.dataset_name, cache_dir=model_args.cache_dir)

    if data_args.dataset_name == "cppe-5":
        # Remove of bad annotated images, this is dataset specific option
        # Some image have annotation boxes outside of the image, remove them for simplicity
        remove_idx = [590, 821, 822, 875, 876, 878, 879]
        keep = [i for i in range(len(dataset["train"])) if i not in remove_idx]
        dataset["train"] = dataset["train"].select(keep)

    # If we don't have a validation split, split off a percentage of train as validation
    data_args.train_val_split = None if "valid" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["valid"] = split["test"]

    # Get dataset categories and prepare mappings for label_name <-> label_id
    categories = dataset["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories)}
    label2id = {v: k for k, v in id2label.items()}

    # ------------------------------------------------------------------------------------------------
    # Load pretrained config, model and image processor
    # ------------------------------------------------------------------------------------------------

    common_pretrained_args = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        **common_pretrained_args
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        **common_pretrained_args
    )
    image_processor = DetrImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        size={"longest_edge": data_args.longest_edge, "shortest_edge": data_args.shortest_edge},
        **common_pretrained_args
    )

    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------

    train_augmentation_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
    )
    valid_augmentation_transform = A.Compose(
        [
            # empty transform, but you can add something here, e.g. resizing/padding
            A.NoOp(), 
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(augment_and_transform_batch, transform=train_augmentation_transform, image_processor=image_processor)
    valid_transform_batch = partial(augment_and_transform_batch, transform=valid_augmentation_transform, image_processor=image_processor)

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["valid"] = dataset["valid"].with_transform(valid_transform_batch)
    dataset["test"] = dataset["test"].with_transform(valid_transform_batch)

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def compute_metrics(evaluation_results):
        predictions, targets = evaluation_results
        return {"dummy": 1.0}

    training_args.include_inputs_for_metrics = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["valid"] if training_args.do_eval else None,
        tokenizer=image_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # ------------------------------------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------------------------------------

    dataloader = DataLoader(dataset["test"], batch_size=4, shuffle=False, collate_fn=collate_fn)
    mAP = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    model = model.cpu().eval()

    for batch in dataloader:
        with torch.no_grad():
            # model predict boxes in YOLO format (center_x, center_y, width, height) 
            # with coordinates *noramlized* to [0..1] (relative coordinates)
            output = model(batch["pixel_values"].cpu())
        
        # For metric computation we need to collect ground truth and predicted boxes in the same format
        
        # 1. Collect predicted boxes, classes, scores
        # image_processor convert boxes from YOLO format to Pascal VOC format 
        # ([x_min, y_min, x_max, y_max] in absolute coordinates)
        image_size = torch.stack([example["size"] for example in batch["labels"]], dim=0)
        predictions = image_processor.post_process_object_detection(output, threshold=0.0, target_sizes=image_size)

        # 2. Collect ground truth boxes in the same format for metric computation
        target = []
        for label in batch["labels"]:
            boxes = center_to_corners_format(label["boxes"])
            boxes = convert_boxes_to_absolute_coordinates(boxes, label["size"])
            labels = label["class_labels"]
            target.append({"boxes": boxes.cpu(), "labels": labels.cpu()})

        mAP.update(predictions, target)

    # compute and post-process metrics
    metrics = mAP.compute()
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()]
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar
    metrics = {f"eval_{k}": round(v.item(), 4) for k, v in metrics.items()}

    # save metrics
    trainer.log_metrics("eval", train_result.metrics)
    trainer.save_metrics("eval", train_result.metrics)



if __name__ == "__main__":
    main()
