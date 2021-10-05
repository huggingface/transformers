#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

""" Pre-Training a 🤗 Wav2Vec2 model on unlabeled audio data """

import os
import torch
import math
import datasets
import wandb
import argparse
import logging
import torchaudio
import transformers

from datasets import DatasetDict, load_dataset, concatenate_datasets
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
from transformers import (
    MODEL_MAPPING,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    AdamW,
    get_scheduler,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices


logger = logging.getLogger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        nargs="+",
        type=str,
        required=True,
        help="The configuration names of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_split_names",
        nargs="+",
        type=str,
        required=True,
        help="The names of the training data set splits to use (via the datasets library).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--preprocessing_only", type=bool, default=None, help="Only run the preprocessing script to be cached for future use"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=1,
        help="Percentage of training data that should be used for validation if no validation is present in dataset.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--saving_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--audio_file_column",
        type=str,
        default="file",
        help="Column in the dataset that contains speech file path. Defaults to 'file'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    parser.add_argument(
        "--max_duration_in_seconds",
        type=float,
        default=5.0,
        help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--min_duration_in_seconds",
        type=float,
        default=3.0,
        help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help="If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])

        # make sure that no loss is computed on padded inputs
        if batch["attention_mask"] is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(mask_indices_seq_length, batch["attention_mask"])

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        batch["mask_time_indices"] = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch["sub_attention_mask"],
            device=device,
        )

        # sample negative indices
        batch["sampled_negative_indices"] = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=batch["mask_time_indices"].numpy(),
        )
        batch["sampled_negative_indices"] = torch.tensor(batch["sampled_negative_indices"], dtype=torch.long, device=device)

        return batch


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    datasets_splits = []
    # Downloading and loading a dataset from the hub.
    for dataset_config_name, train_split_name in zip(args.dataset_config_names, args.dataset_train_split_names):
        # load dataset
        dataset_split = load_dataset(args.dataset_name, dataset_config_name, split=train_split_name, cache_dir=args.cache_dir)
        datasets_splits.append(dataset_split)

    # concatenate all datasets if necessary
    raw_datasets = DatasetDict()
    if len(datasets_splits) > 1:
        raw_datasets["train"] = concatenate_datasets(datasets_splits).shuffle(seed=args.seed)
    else:
        raw_datasets["train"] = datasets_splits[0]

    # create validation dataset
    num_validation_samples = raw_datasets["train"].num_rows * args.validation_split_percentage // 100
    raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
    raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))

    # only normalized-inputs-training is supported
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)

    if not feature_extractor.do_normalize:
        raise ValueError(
            "Training is only supported for normalized inputs. "
            "Make sure ``feature_extractor.do_normalize == True``"
        )

    # set max & min audio length in number of samples
    max_length = int(args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_length = int(args.min_duration_in_seconds * feature_extractor.sampling_rate)

    resampler = None
    if raw_datasets["train"][args.audio_column_name][0].split(".")[-1] == "mp3":
        # TODO(PVP) - remove hard-coded 48_000 after audio feature is merged
        resampler = torchaudio.transforms.Resample(48_000, feature_extractor.sampling_rate)

    def prepare_dataset(batch):
        speech_array, sampling_rate = torchaudio.load(batch[args.audio_column_name])
        speech_array = speech_array.squeeze()

        # if necessary resample audio
        if resampler is not None:
            # TODO(PVP) - remove hard-coded 48_000 after audio feature is merged
            speech_array = resampler(speech_array)
            sampling_rate = resampler.new_freq

        speech_array = speech_array.numpy()
        inputs = feature_extractor(
            speech_array, sampling_rate=sampling_rate, max_length=max_length, truncation=True
        )
        batch["input_values"] = inputs.input_values[0]
        return batch

    # load audio files into numpy arrays
    with accelerator.main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset, num_proc=args.preprocessing_num_workers, remove_columns=raw_datasets["train"].column_names, load_from_cache_file=not args.overwrite_cache,
        )
        vectorized_datasets = vectorized_datasets.filter(lambda x: len(x["input_values"]) > min_length, load_from_cache_file=not args.overwrite_cache)

    # pretraining is only supported for "newer" stable layer norm architecture
    # apply_spec_augment has to be True, mask_feature_prob has to be 0.0
    config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)

    if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
        raise ValueError(
            "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and ``config.feat_extract_norm='layer'"
        )

    # initialize random model
    model = Wav2Vec2ForPreTraining(config)

    # data collator
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model, feature_extractor=feature_extractor, pad_to_multiple_of=args.pad_to_multiple_of
    )
    train_dataloader = DataLoader(
        vectorized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        vectorized_datasets["validation"], collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    optimizer = AdamW(list(model.parameters()), lr=args.learning_rate, betas=[args.adam_beta1, args.adam_beta2], eps=args.adam_epsilon)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(vectorized_datasets['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # sub attention mask in only required for logging
            sub_attention_mask = batch.pop("sub_attention_mask")

            # forward
            outputs = model(**batch)

            num_losses = batch["mask_time_indices"].sum()
            percent_masked = num_losses / sub_attention_mask.sum()

            # divide loss by gradient accumulation steps
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            # make sure that `num_losses` is summed for distributed training
            # average gradients over losses (and nodes in distributed training)
            if accelerator.state.num_processes > 1:
                dist.all_reduce(num_losses)
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1 / num_losses)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                # compute grad norm for monitoring
                scale = accelerator.scaler._scale.item() if hasattr(accelerator, "scaler") and accelerator.scaler is not None else 1
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)

                optimizer.step()
                optimizer.zero_grad()

                if not optimizer.is_overflow:
                    lr_scheduler.step()
                else:
                    progress_bar.write(
                        "Gradients have overflown - skipping update step... "
                        f"Updating gradient scale to {scale}..."
                    )

                # update gumbel temperature
                gumbel_temperature = max(args.max_gumbel_temperature * args.gumbel_temperature_decay ** completed_steps, args.min_gumbel_temperature)

                if hasattr(model, "module"):
                    model.module.set_gumbel_temperature(gumbel_temperature)
                else:
                    model.set_gumbel_temperature(gumbel_temperature)

                progress_bar.update(1)
                completed_steps += 1

            # log results
            if step % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                loss.detach()
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()

                if accelerator.state.num_processes > 1:
                    dist.all_reduce(loss)
                    dist.all_reduce(outputs.contrastive_loss)
                    dist.all_reduce(outputs.diversity_loss)
                    dist.all_reduce(percent_masked)

                train_logs = {
                    "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                    "constrast_loss": outputs.contrastive_loss / num_losses,
                    "div_loss": outputs.diversity_loss / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,
                    "ppl": outputs.codevector_perplexity,
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                    "temp": torch.tensor(gumbel_temperature),
                    "grad_norm": torch.tensor(grad_norm),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)

            # save model every `args.saving_steps` steps
            if step % (args.gradient_accumulation_steps * args.saving_steps) == 0:
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

        model.eval()
        val_logs = {
            "val_loss": 0,
            "val_contrastive_loss": 0,
            "val_diversity_loss": 0,
            "val_num_losses": 0,
        }
        for step, batch in enumerate(eval_dataloader):
            sub_attention_mask = batch.pop("sub_attention_mask")
            with torch.no_grad():
                outputs = model(**batch)

            val_logs["val_loss"] += outputs.loss
            val_logs["val_contrastive_loss"] += outputs.contrastive_loss
            val_logs["val_diversity_loss"] += outputs.diversity_loss
            val_logs["val_num_losses"] += batch["mask_time_indices"].sum()

        if accelerator.num_processes > 1:
            [dist.all_reduce(v) for v in val_logs.values()]

        val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

        log_str = ""
        for k, v in val_logs.items():
            log_str += "| {}: {:.3e}".format(k, v.item())

        if accelerator.is_local_main_process:
            wandb.log(val_logs)
            progress_bar.write(log_str)


if __name__ == "__main__":
    main()
