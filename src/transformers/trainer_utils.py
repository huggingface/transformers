from typing import Dict, NamedTuple, Optional, Union

import numpy as np


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used
    to compute metrics.
    """

    predictions: Union[np.ndarray, Dict[str, np.ndarray]]
    label_ids: Union[np.ndarray, Dict[str, np.ndarray]]


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


PREFIX_CHECKPOINT_DIR = "checkpoint"
