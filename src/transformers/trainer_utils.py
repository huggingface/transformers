import random
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import numpy as np

from .file_utils import is_tf_available, is_torch_available
from .tokenization_utils_base import ExplicitEnum


SEQUENCE_CLASSIFICATION_MODELS = []
if is_torch_available():
    from .modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    SEQUENCE_CLASSIFICATION_MODELS = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.values()
elif is_tf_available():
    from .modeling_tf_auto import TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    SEQUENCE_CLASSIFICATION_MODELS = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.values()


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
    FinalActivation.NONE: lambda x: x,
    FinalActivation.ARGMAX: lambda x: x.argmax(axis=-1),
    FinalActivation.SIGMOID: lambda x: 1 / (1 + np.exp(-x)),
    FinalActivation.SOFTMAX: lambda x: np.exp(x) / np.exp(x).sum(-1, keepdims=True),
}


def auto_activation(model) -> FinalActivation:
    model_class = model.__class__
    if model_class in SEQUENCE_CLASSIFICATION_MODELS:
        return FinalActivation.ARGMAX
    return FinalActivation.NONE


@dataclass
class ComputeNLPMetrics:
    metrics: Union["nlp.Metric", List["nlp.Metric"]]  # noqa: F821
    activation: Optional[Union[str, FinalActivation, Callable]] = None

    def __post_init__(self):
        if isinstance(self.activation, str):
            self.activation = FinalActivation(self.activation)

    def __call__(self, eval_pred):
        preds, labels = eval_pred
        metrics = self.metrics if isinstance(self.metrics, list) else [self.metrics]
        result = {}
        for metric in metrics:
            if self.activation is not None:
                activation_function = (
                    self.activation if callable(self.activation) else ACTIVATION_NAME_TO_FUNCTION[self.activation]
                )
                preds = activation_function(preds)
            # TODO: when https://github.com/huggingface/nlp/pull/466 is merged, remove the `tolist`.
            result = {**result, **metric.compute(preds.tolist(), references=labels.tolist())}
        return result
