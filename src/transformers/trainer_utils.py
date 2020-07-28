import os
import random
from typing import Dict, NamedTuple, Optional
import warnings

import numpy as np

from .file_utils import is_tf_available, is_torch_available


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``
    (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


def estimate_tokens(inputs):
    """
    Helper function to estimate the batch size and sequence length from the model inputs

    Args:
        inputs (:obj:`dict`): The model inputs.

    Returns:
        seed (:obj:`tuple`): The batch size and sequence length.
    """
    inputs_ids = inputs.get("input_ids")
    input_embeds = inputs.get("input_embeds")
    if inputs is not None:
        return inputs_ids.shape[0], inputs_ids.shape[1]
    if input_embeds is not None:
        return input_embeds.shape[0], input_embeds.shape[1]
    warnings.warn(
        "Could not estimate the number of tokens of the input, floating-point operations will" "not be computed"
    )
    return 0, 0


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
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
