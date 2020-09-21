import random
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from .file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from .tokenization_utils_base import ExplicitEnum


if is_torch_available():
    import torch


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

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


PREFIX_CHECKPOINT_DIR = "checkpoint"


class BestRun(NamedTuple):
    """
    The best run found by an hyperparameter search (see :class:`~transformers.Trainer.hyperparameter_search`).

    Parameters:
        run_id (:obj:`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (:obj:`float`):
            The objective that was obtained for this run.
        hyperparameters (:obj:`Dict[str, Any]`):
            The hyperparameters picked to get this run.
    """

    run_id: str
    objective: float
    hyperparameters: Dict[str, Any]


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the :class:`~transformers.Trainer`, the sum of all metrics otherwise.

    Args:
        metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        :obj:`float`: The objective to minimize or maximize
    """
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    return loss if len(metrics) == 0 else sum(metrics.values())


def default_hp_space_optuna(trial) -> Dict[str, float]:
    from .integrations import is_optuna_available

    assert is_optuna_available(), "This function needs Optuna installed: `pip install optuna`"
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
    }


def default_hp_space_ray(trial) -> Dict[str, float]:
    from .integrations import is_ray_available

    assert is_ray_available(), "This function needs ray installed: `pip install ray[tune]`"
    from ray import tune

    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(list(range(1, 6))),
        "seed": tune.uniform(1, 40),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
    }


class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"


default_hp_space = {
    HPSearchBackend.OPTUNA: default_hp_space_optuna,
    HPSearchBackend.RAY: default_hp_space_ray,
}


def nested_concat(tensors, new_tensors, dim=0):
    "Concat the `new_tensors` to `tensors` on `dim`. Works for tensors or nested list/tuples of tensors."
    if is_torch_available():
        assert type(tensors) == type(
            new_tensors
        ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
        if isinstance(tensors, (list, tuple)):
            return type(tensors)(nested_concat(t, n, dim) for t, n in zip(tensors, new_tensors))
        return torch.cat((tensors, new_tensors), dim=dim)
    else:
        raise ImportError("Torch must be installed to use `nested_concat`")


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def nested_xla_mesh_reduce(tensors, name):
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm

        if isinstance(tensors, (list, tuple)):
            return type(tensors)(nested_xla_mesh_reduce(t, f"{name}_{i}") for i, t in enumerate(tensors))
        return xm.mesh_reduce(name, tensors, torch.cat)
    else:
        raise ImportError("Torch xla must be installed to use `nested_xla_mesh_reduce`")


def distributed_concat(tensor: "torch.Tensor", num_total_examples: Optional[int] = None) -> "torch.Tensor":
    if is_torch_available():
        try:
            if isinstance(tensor, (tuple, list)):
                return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
            output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(output_tensors, tensor)
            concat = torch.cat(output_tensors, dim=0)

            # truncate the dummy elements added by SequentialDistributedSampler
            if num_total_examples is not None:
                concat = concat[:num_total_examples]
            return concat
        except AssertionError:
            raise AssertionError("Not currently using distributed training")
    else:
        raise ImportError("Torch must be installed to use `distributed_concat`")


def distributed_broadcast_scalars(
    scalars: List[Union[int, float]], num_total_examples: Optional[int] = None
) -> "torch.Tensor":
    if is_torch_available():
        try:
            tensorized_scalar = torch.Tensor(scalars).cuda()
            output_tensors = [tensorized_scalar.clone() for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(output_tensors, tensorized_scalar)
            concat = torch.cat(output_tensors, dim=0)

            # truncate the dummy elements added by SequentialDistributedSampler
            if num_total_examples is not None:
                concat = concat[:num_total_examples]
            return concat
        except AssertionError:
            raise AssertionError("Not currently using distributed training")
    else:
        raise ImportError("Torch must be installed to use `distributed_broadcast_scalars`")
