# Copyright (c) 2019-present, the HuggingFace Inc. authors.
# All rights reserved. This source code is licensed under the BSD-style
# license found in the LICENSE file in the root directory of this source tree.
import logging
import os
from tqdm import tqdm
from pprint import pformat

import torch

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler, OutputHandler, TensorboardLogger


def average_distributed_scalar(scalar, args):
    """ Average a scalar over nodes if we are in distributed training.
        We use this for distributed evaluation.
        Beware, such averages only works for metrics which are additive with regard
        to the evaluation dataset, e.g. accuracy, log probabilities.
        Doesn't work for ratio metrics like F1.
    """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def add_logging_and_checkpoint_saving(trainer, evaluator, metrics, model, optimizer, args, prefix=""):
    """ Add to a PyTorch ignite training engine tensorboard logging,
        progress bar with average loss, checkpoint saving and save training config.
    """
    # Add progress bar with average loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, prefix + "loss")
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=[prefix + "loss"])
    evaluator.add_event_handler(Events.COMPLETED, lambda _:
                                pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

    # Add tensorboard logging with training and evaluation metrics
    tb_logger = TensorboardLogger(log_dir=None)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=[prefix + "loss"]),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer),
                     event_name=Events.ITERATION_STARTED)
    @evaluator.on(Events.COMPLETED)
    def tb_log_metrics(engine):
        for name in metrics.keys():
            tb_logger.writer.add_scalar(name, engine.state.metrics[name], trainer.state.iteration)

    # Add checkpoint saving after each epoch - take care of distributed encapsulation ('getattr()')
    checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})

    # Save training configuration
    torch.save(args, os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))

    return checkpoint_handler, tb_logger
