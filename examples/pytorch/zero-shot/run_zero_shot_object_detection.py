#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Finetuning any 🤗 Transformers model supported by AutoModelForZeroShotObjectDetection for object detection leveraging the Trainer API."""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoConfig,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
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


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.44.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/zero_shot/requirements.txt")


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class ZeroShotTrainer(Trainer):
    def _select_inputs_for_validation(self, inputs):
        return inputs["input_ids"]


def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


def convert_zero_shot_to_coco_format(predictions, label2id):
    """
    Convert zershot format output to typical object detection format in order to calculate mAP.

    Args:
        predictions (Dict): Output of zero-shot object detection
            e.g. {'scores': tensor([0.4786, 0.4379, 0.4760], device='cuda:0'), 'labels': ['a cat', 'a cat', 'a remote control'], 'boxes': tensor([[344.6973,  23.1085, 637.1817, 374.2748],[ 12.2690,  51.9104, 316.8564, 472.4341],[ 38.5870,  70.0092, 176.7755, 118.1748]], device='cuda:0')}
        label2id (Dict): Dictionary of label to id mapping

    Returns:
        Dict: Output of zero-shot object detection
            e.g. {'scores': tensor([0.4786, 0.4379, 0.4760], device='cuda:0'), 'labels': [1, 1, 2], 'boxes': tensor([[344.6973,  23.1085, 637.1817, 374.2748],[ 12.2690,  51.9104, 316.8564, 472.4341],[ 38.5870,  70.0092, 176.7755, 118.1748]], device='cuda:0')}

    """
    # convert center to corners format
    torch_label = []
    for prediction in predictions:
        scores = prediction["scores"]
        device = scores.device
        labels = prediction["labels"]
        for label in labels:
            if label in label2id:
                torch_label.append(label2id[label])
            else:
                # Give background class
                torch_label.append(0)
        prediction["labels"] = torch.Tensor(torch_label).to(dtype=torch.int32).to(device)

    return predictions


def to_label_list(id2label):
    return list(id2label.values())


def concat_func(id2label):
    return ". ".join(to_label_list(id2label)) + "."


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    processor: AutoProcessor,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    random_text_prompt: bool = False,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """
    Apply augmentations and format annotations in COCO format for object detection task.
    Generates the text prompt used. If `random_text_prompt` is False
        then the prompt will follow the same ordering in `id2label` if set to
        True a new ordering will be created and the prompt will be build accordingly
        and labels will be updated as well.

        Example:
            `id2label` -> {'0': 'fish', '1': 'jellyfish', '2': 'penguins', '3':
                        'sharks', '4': 'puffins', '5': 'stingrays', '6': 'starfish'}

            If `random_text_prompt` -> False
                `text` -> "fish. jellyfish. penguins. sharks. puffins. stingrays. starfish."

            If `random_text_prompt` -> True
                `id2label` gets shuffled e.g. {0: 'fish', 1: 'penguins', 2: 'stingrays', 3:
                                            'jellyfish', 4: 'sharks', 5: 'starfish', 6: 'puffins'}
                `text` -> "fish. penguins. stingrays. jellyfish. sharks. starfish. puffins."
    """

    images = []
    annotations = []
    text = []

    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        if random_text_prompt:
            # Original ordering label list
            label_list = to_label_list(id2label)
            # Shuffle label list
            random.shuffle(label_list)
            # Create shuffled id2label
            shuffled_id2label = dict(enumerate(label_list))

            # Mapping of original to shuffled id to update annotations
            old2new = {label2id[label]: new_id for new_id, label in shuffled_id2label.items()}
            prompt = concat_func(shuffled_id2label)
            category = [old2new[category] for category in objects["category"]]
        else:
            prompt = concat_func(id2label)
            category = objects["category"]

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=category)
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)
        text.append(prompt)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = processor(images=images, text=text, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def collate_fn(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["input_ids"] = torch.stack([x["input_ids"] for x in batch])
    data["token_type_ids"] = torch.stack([x["token_type_ids"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    if "attention_mask" in batch[0]:
        data["attention_mask"] = torch.stack([x["attention_mask"] for x in batch])
    return data


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    processor: AutoProcessor,
    box_threshold: float = 0.15,
    text_threshold: float = 0.1,
    id2label: Optional[Mapping[int, str]] = None,
    label2id: Optional[Mapping[str, int]] = None,
) -> Mapping[str, float]:
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        box_threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.15.
        text_threshold (float, optional): Threshold to filter predicted text by confidence. Defaults to 0.1.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets, inputs = (
        evaluation_results.predictions,
        evaluation_results.label_ids,
        evaluation_results.inputs,
    )
    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor([x["orig_size"] for x in batch])
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then processor convert them to Pascal VOC format
    for batch, target_sizes, input_ids in zip(predictions, image_sizes, inputs):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = processor.post_process_grounded_object_detection(
            output, input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=target_sizes
        )
        post_processed_output = convert_zero_shot_to_coco_format(post_processed_output, label2id)
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: str = field(
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
    image_square_size: Optional[int] = field(
        default=600,
        metadata={"help": "Image longest size will be resized to this value, then image will be padded to square."},
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


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="IDEA-Research/grounding-dino-tiny",
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
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model (if for instance, you are instantiating a model with 10 labels from a checkpoint with 3 labels)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_backbone: bool = field(
        default=True,
        metadata={"help": ("Whether freeze the image backbone.")},
    )
    freeze_text_backbone: bool = field(
        default=True,
        metadata={"help": ("Whether freeze the text encoder.")},
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

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_object_detection", model_args, data_args)

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
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits
    # ------------------------------------------------------------------------------------------------

    dataset = load_dataset(
        data_args.dataset_name, cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code
    )

    # If we don't have a validation split, split off a percentage of train as validation
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Get dataset categories and prepare mappings for label_name <-> label_id
    categories = dataset["train"].features["objects"].feature["category"].names
    id2label = dict(enumerate(categories))
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
        **common_pretrained_args,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        **common_pretrained_args,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
    )

    # Freeze both text_backbone
    if model_args.freeze_backbone:
        model.model.freeze_backbone()
    if model_args.freeze_text_backbone:
        for name, param in model.model.text_backbone.named_parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------
    max_size = data_args.image_square_size
    train_augment_and_transform = A.Compose(
        [
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=max_size, p=1.0),
                    A.RandomSizedBBoxSafeCrop(height=max_size, width=max_size, p=1.0),
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
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch,
        transform=train_augment_and_transform,
        processor=processor,
        id2label=id2label,
        label2id=label2id,
        random_text_prompt=False,
    )
    validation_transform_batch = partial(
        augment_and_transform_batch,
        transform=validation_transform,
        processor=processor,
        id2label=id2label,
        label2id=label2id,
        random_text_prompt=True,
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------

    eval_compute_metrics_fn = partial(compute_metrics, processor=processor, id2label=id2label, label2id=label2id)

    trainer = ZeroShotTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Final evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["object-detection", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
