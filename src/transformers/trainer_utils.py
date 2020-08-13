import random
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np

from .file_utils import is_tf_available, is_torch_available
from .tokenization_utils_base import ExplicitEnum


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


class FinalActivation(ExplicitEnum):
    """
    Possible values for the ``final_activation`` argument in :meth:`Trainer.from_nlp_dataset`.
    Useful for tab-completion in an IDE.
    """

    NONE = "none"
    ARGMAX = "argmax"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


ACTIVATION_NAME_TO_FUNCTION = {
    "none": lambda x: x,
    "argmax": lambda x: x.argmax(axis=-1),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "softmax": lambda x: np.exp(x) / np.exp(x).sum(-1, keepdims=True),
}


@dataclass
class ComputeNLPMetrics:
    metrics: Union["nlp.Metric", List["nlp.Metric"]]  # noqa: F821
    activation: Optional[Union[str, FinalActivation]] = None

    def __call__(self, eval_pred):
        preds, labels = eval_pred
        metrics = self.metrics if isinstance(self.metrics, list) else [self.metrics]
        result = {}
        for metric in metrics:
            if self.activation is not None:
                preds = ACTIVATION_NAME_TO_FUNCTION[self.activation](preds)
            # TODO: when https://github.com/huggingface/nlp/pull/466 is merged, remove the `tolist`.
            result = {**result, **metric.compute(preds.tolist(), references=labels.tolist())}
        return result
