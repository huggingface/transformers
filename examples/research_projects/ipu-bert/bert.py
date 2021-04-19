# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
import wandb
import warnings
from tqdm import tqdm
import logging

import torch
import transformers
from poptorch import trainingModel, DataLoader
from poptorch.enums import DataLoaderMode
from bert_data import get_dataset, get_generated_datum
from bert_model import PipelinedBertWithLoss
from bert_ipu import get_options
from bert_optimization import get_lr_scheduler, get_optimizer
from bert_checkpoint import save_checkpoint, restore_checkpoint, checkpoints_exist
from utils import parse_bert_args, cycle, logger


if __name__ == "__main__":

    # Ignore known warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    logging.getLogger("poptorch::python").setLevel(logging.ERROR)

    # Build config from args
    config = transformers.BertConfig(**(vars(parse_bert_args())))

    # Checkpoints should be saved to a directory with no existing checkpoints
    if config.checkpoint_dir and checkpoints_exist(config):
        raise RuntimeError("Found previously saved checkpoint(s) at checkpoint-dir. "
                           "Overwriting checkpoints is not supported. "
                           "Please specify a different checkpoint-dir to "
                           "save checkpoints from this run.")
    # Restore from checkpoint if necessary
    checkpoint = restore_checkpoint(config) if config.checkpoint_file else None

    # Execution parameters
    opts = get_options(config)

    # W&B
    if config.wandb:
        wandb.init(project="torch-bert")
        wandb.config.update(vars(config))

    # Dataset selection
    dataset = get_dataset(config)

    # Dataloader
    logger("------------------- Data Loading Started ------------------")
    start_loading = time.perf_counter()
    loader = DataLoader(opts,
                        dataset,
                        batch_size=config.batch_size,
                        num_workers=config.dataloader_workers,
                        mode=DataLoaderMode.Async if config.async_dataloader else DataLoaderMode.Sync)

    steps_per_epoch = len(loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration, "
                           "try reducing deviceIterations or gradientAccumulation.")
    duration_loader = time.perf_counter() - start_loading
    logger(f"Data loaded in {duration_loader} secs")
    logger("-----------------------------------------------------------")

    # IPU Model and Optimizer
    model = PipelinedBertWithLoss(config).half().train()
    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(optimizer, config.lr_schedule,
                                 config.lr_warmup, config.training_steps)

    # Restore model from checkpoint
    steps_finished = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.restore_steps_and_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.last_epoch = steps_finished = checkpoint["step"]
            checkpoint_metrics = checkpoint["metrics"]
        else:
            # Checkpoint model with epochs and optimizer state reset
            # for further training
            save_checkpoint(config, model, optimizer, steps_finished)
    else:
        # Checkpoint model at start of run
        save_checkpoint(config, model, optimizer, steps_finished)

    poptorch_model = trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    logger("---------- Compilation/Loading from Cache Started ---------")
    start_compile = time.perf_counter()
    datum = get_generated_datum(config)
    poptorch_model.compile(*datum)
    duration_compilation = time.perf_counter() - start_compile
    logger(f"Compiled/Loaded model in {duration_compilation} secs")
    logger("-----------------------------------------------------------")

    # Training loop
    logger("--------------------- Training Started --------------------")
    factor = config.gradient_accumulation * config.batches_per_step
    start_train = time.perf_counter()
    loader = cycle(loader)
    train_iterator = tqdm(range(steps_finished, config.training_steps),
                          desc="Training", disable=config.disable_progress_bar)
    for step in train_iterator:
        start_step = time.perf_counter()
        outputs = poptorch_model(*next(loader))
        scheduler.step()
        poptorch_model.setOptimizer(optimizer)
        step_length = time.perf_counter() - start_step
        step_throughput = config.samples_per_step / step_length

        train_iterator.set_description(
            f"Step: {step} / {config.training_steps-1} - "
            f"LR: {scheduler.get_last_lr()[0]:.2e} - "
            f"Loss: {outputs[0].div(factor).mean().item():3.3f} - "
            f"Loss/MLM: {outputs[1].div(factor).mean().item():3.3f} - "
            f"Loss/NSP: {outputs[2].div(factor).mean().item():3.3f} - "
            f"Acc/MLM: {outputs[3].div(factor).mean().item():3.3f} - "
            f"Acc/NSP: {outputs[4].div(factor).mean().item():3.3f}")
        train_iterator.set_postfix_str(f"{step_throughput:.1f} sequences/s")
        if config.disable_progress_bar:
            print(train_iterator.desc, train_iterator.postfix, file=sys.stderr)

        if config.wandb:
            wandb.log({"Loss": outputs[0].div(factor).mean().item(),
                       "Loss/MLM": outputs[1].div(factor).mean().item(),
                       "Loss/NSP": outputs[2].div(factor).mean().item(),
                       "Acc/MLM": outputs[3].div(factor).mean().item(),
                       "Acc/NSP": outputs[4].div(factor).mean().item(),
                       "LR": scheduler.get_last_lr()[0],
                       "Step": step,
                       "Throughput": step_throughput})

            if config.wandb_param_every_nsteps and (step % config.wandb_param_every_nsteps) == 0:
                for name, parameter in poptorch_model.named_parameters():
                    wandb.run.history.torch.log_tensor_stats(parameter.data, name)

        if config.checkpoint_every_nsteps and (step % config.checkpoint_every_nsteps) == 0:
            save_checkpoint(config, model, optimizer, step,
                            metrics={"Loss": outputs[0].div(factor).mean().item(),
                                     "Acc/MLM": outputs[3].div(factor).mean().item(),
                                     "Acc/NSP": outputs[4].div(factor).mean().item()})

        if step + 1 == config.training_steps:
            break  # Training finished mid-epoch

    stop_train = time.perf_counter()
    # Checkpoint at end of run
    save_checkpoint(config, model, optimizer, step,
                    metrics={"Loss": outputs[0].div(factor).mean().item(),
                             "Acc/MLM": outputs[3].div(factor).mean().item(),
                             "Acc/NSP": outputs[4].div(factor).mean().item()})
    logger("-----------------------------------------------------------")

    logger("-------------------- Training Metrics ---------------------")
    logger(f"global_batch_size: {config.global_batch_size}")
    logger(f"batches_per_step: {config.batches_per_step}")
    logger(f"training_steps: {config.training_steps}")
    duration_run = stop_train - start_train
    num_samples = config.samples_per_step * config.training_steps
    logger(f"Training time: {duration_run:.3f} secs")
    logger("-----------------------------------------------------------")
