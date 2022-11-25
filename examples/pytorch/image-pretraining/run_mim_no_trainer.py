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

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import datasets
from datasets import load_dataset
from pathlib import Path


import numpy as np
import torch
from datasets import load_dataset
from accelerate.utils import set_seed
from huggingface_hub import Repository


from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor

import transformers
from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForMaskedImageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version


""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM) 
without using HuggingFace Trainer.

Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help="Name of a dataset from the datasets package",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--image_column_name",
        type=str,
        default=None,
        help="The column name of the images in the files. If not set, will try to use 'image' or 'img'.",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--validation_dir",
        type=None,
        default=None,
        help="A folder containing the validation data.",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation.",
    )
    parser.add_argument(
        "--mask_patch_size",
        type=int,
        default=32,
        help="The size of the square patches to use for masking.",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.6,
        help="Percentage of patches to mask.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help=(
            "The model checkpoint for weights initialization. Can be a local path to a pytorch_model.bin or a "
            "checkpoint identifier on the hub. "
            "Don't set if you want to train a model from scratch."
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--config_name_or_path",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--config_overrides",
        type=str,
        default=None,
        help=(
            "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store (cache) the pretrained models/datasets downloaded from the hub",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--feature_extractor_name",
        type=str,
        default=None,
        help="Name or path of preprocessor config.",
    )
    parser.add_argument(
        "--use_auth_token",
        type=bool,
        default=False,
        help=(
            "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
            "with private models)."
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="The size (resolution) of each image. If not specified, will use `image_size` of the configuration.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="The size (resolution) of each patch. If not specified, will use `patch_size` of the configuration.",
    )
    parser.add_argument(
        "--encoder_stride",
        type=int,
        default=None,
        help={"help": "Stride to use for the encoder."},
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
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    data_files = dict()
    if args.train_dir is not None:
        data_files["train"] = args.train_dir
    if args.validation_dir is not None:
        data_files["val"] = args.validation_dir
    args.data_files = data_files if data_files else None

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mim_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
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
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize our dataset.
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        data_files=args.data_files,
        cache_dir=args.cache_dir,
        use_auth_token=True if args.use_auth_token else None,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    args.train_val_split = None if "validation" in ds.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = ds["train"].train_test_split(args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Create config
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.model_revision,
        "use_auth_token": True if args.use_auth_token else None,
    }
    if args.config_name_or_path:
        config = AutoConfig.from_pretrained(args.config_name_or_path, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.config_overrides is not None:
            logger.info(f"Overriding config: {args.config_overrides}")
            config.update_from_string(args.config_overrides)
            logger.info(f"New config: {config}")

    # make sure the decoder_type is "simmim" (only relevant for BEiT)
    if hasattr(config, "decoder_type"):
        config.decoder_type = "simmim"

    # adapt config
    args.image_size = args.image_size if args.image_size is not None else config.image_size
    args.patch_size = args.patch_size if args.patch_size is not None else config.patch_size
    args.encoder_stride = args.encoder_stride if args.encoder_stride is not None else config.encoder_stride

    config.update(
        {
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "encoder_stride": args.encoder_stride,
        }
    )
    
    # create feature extractor
    if args.feature_extractor_name:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.feature_extractor_name, **config_kwargs)
    elif args.model_name_or_path:
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        FEATURE_EXTRACTOR_TYPES = {
            conf.model_type: feature_extractor_class
            for conf, feature_extractor_class in FEATURE_EXTRACTOR_MAPPING.items()
        }
        feature_extractor = FEATURE_EXTRACTOR_TYPES[args.model_type]()


if __name__ == "__main__":
    main()
