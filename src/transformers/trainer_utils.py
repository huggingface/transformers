import logging
import os
from typing import Dict, NamedTuple, Optional

import numpy as np
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False
    logger.info(
        "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
        "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
    )


def is_wandb_available():
    return _has_wandb


def setup_wandb(trainer):
    """
    Setup the optional Weights & Biases (`wandb`) integration.

    One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
    You can also override the following environment variables:

    Environment:
        WANDB_WATCH:
            (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
            or "all" to log gradients and parameters
        WANDB_PROJECT:
            (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
        WANDB_DISABLED:
            (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
    """
    logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')

    wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(trainer.args))
    # keep track of model topology and gradients (only with Pytorch)
    if os.getenv("WANDB_WATCH") != "false" and trainer.__class__.__name__ == "Trainer":
        wandb.watch(
            trainer.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, trainer.args.logging_steps)
        )
    # give access to wandb module
    trainer._wandb = wandb


def log_metrics(trainer, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
    """
    Log metrics with available loggers.
    """
    if trainer.epoch_logging is not None:
        logs["epoch"] = trainer.epoch_logging
    if trainer.tb_writer:
        trainer._log_tb(logs)
    if is_wandb_available():
        trainer._wandb.log(logs, step=trainer.global_step)
    output = {**logs, **{"step": trainer.global_step}}
    if iterator is not None:
        iterator.write(output)
    else:
        logger.info(output)


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used
    to compute metrics.
    """

    predictions: np.ndarray
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


PREFIX_CHECKPOINT_DIR = "checkpoint"
