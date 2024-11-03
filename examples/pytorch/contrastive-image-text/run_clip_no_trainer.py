#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
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
Training a CLIP like dual encoder models using text and vision encoders in the library
without using HuggingFace Trainer.

The script can be used to train CLIP like models for languages other than English by using
a text encoder pre-trained in the desired language. Currently this script supports the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=fill-mask)
"""

import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional
import argparse

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
import datasets
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionTextDualEncoderModel,
    SchedulerType,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.47.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training a CLIP-like dual encoder model using text and vision encoders "
    )

    parser.add_argument(
        "--vision_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--text_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        "--image_processor_name",
        type=str,
        default=None,
        help="Name or path of preprocessor config."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to store the pretrained models downloaded from s3"
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="Specific model version to use (can be branch name, tag name, or commit id)"
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Use one of the fast tokenizers backed by the tokenizers library"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HTTP bearer authorization token for remote files"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust execution of code from datasets/models defined on the Hub. Only use for trusted repositories"
    )
    parser.add_argument(
        "--freeze_vision_model",
        action="store_true",
        help="Freeze the vision model parameters"
    )
    parser.add_argument(
        "--freeze_text_model",
        action="store_true",
        help="Freeze the text model parameters"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset to use (via the datasets library)"
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Configuration name of the dataset to use (via the datasets library)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing input data files"
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image_path",
        help="Column name in the dataset containing the image file paths"
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="Column name in the dataset containing image captions"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to the input training data file (jsonlines format)"
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="Optional input evaluation data file (jsonlines format)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Max input sequence length after tokenization"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit the number of training examples (for debugging or quicker training)"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit the number of evaluation examples (for debugging or quicker training)"
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="Number of processes to use for data preprocessing"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The initial learning rate for [`AdamW`] optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )

    args = parser.parse_args()

    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clip_no_trainer", args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            keep_in_memory=False,
            data_dir=args.data_dir,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=args.cache_dir,
            token=args.token,
        )

    # Load tokenizer 
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            cache_dir=args.cache_dir,
            use_fast=args.use_fast_tokenizer,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.text_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_model_name_or_path,
            cache_dir=args.cache_dir,
            use_fast=args.use_fast_tokenizer,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    image_processor = AutoImageProcessor.from_pretrained(
        args.image_processor_name or args.vision_model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )

    model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        args.vision_model_name_or_path,
        args.text_model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    config = model.config

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if args.freeze_text_model:
        _freeze_params(model.text_model)

    column_names = dataset["train"].column_names

    # Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
    )
    image_transformations = torch.jit.script(image_transformations)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def transform_images(examples):
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[image_column]:
            try:
                Image.open(image_file)
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    train_dataset = dataset["train"]
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    train_dataset = train_dataset.filter(
        filter_corrupt_images, batched=True, num_proc=args.preprocessing_num_workers
    )
    train_dataset = train_dataset.map(
        function=tokenize_captions,
        batched=True,
        remove_columns=[col for col in column_names if col != image_column],
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    # Transform images on the fly as doing it on the whole dataset takes too much time.
    train_dataset.set_transform(transform_images)

    eval_dataset = dataset["validation"]
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    eval_dataset = eval_dataset.filter(
        filter_corrupt_images, batched=True, num_proc=args.preprocessing_num_workers
    )
    eval_dataset = eval_dataset.map(
        function=tokenize_captions,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[col for col in column_names if col != image_column],
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    # Transform images on the fly as doing it on the whole dataset takes too much time.
    eval_dataset.set_transform(transform_images)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collate_fn,
        batch_size=args.per_device_eval_batch_size,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mim_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(int(args.max_train_steps)), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        logger.info(f"epoch {epoch}: eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            image_processor.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )


if __name__ == "__main__":
    main()
