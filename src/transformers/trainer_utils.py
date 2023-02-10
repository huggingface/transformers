# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
# limitations under the License.
"""
Utilities for the Trainer and TFTrainer class. Should be independent from PyTorch and TensorFlow.
"""

import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from .utils import (
    ExplicitEnum,
    is_psutil_available,
    is_tf_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_tpu_available,
    requires_backends,
)


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def enable_full_determinism(seed: int):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    # set seed first
    set_seed(seed)

    if is_torch_available():
        # Enable PyTorch deterministic mode. This potentially requires either the environment
        # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
        # depending on the CUDA version, so we set them both here
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if is_tf_available():
        tf.config.experimental.enable_op_determinism()


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        tf.random.set_seed(seed)


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EvaluationStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class HubStrategy(ExplicitEnum):
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"


class BestRun(NamedTuple):
    """
    The best run found by an hyperparameter search (see [`~Trainer.hyperparameter_search`]).

    Parameters:
        run_id (`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (`float`):
            The objective that was obtained for this run.
        hyperparameters (`Dict[str, Any]`):
            The hyperparameters picked to get this run.
    """

    run_id: str
    objective: float
    hyperparameters: Dict[str, Any]


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the [`Trainer`], the sum of all metrics otherwise.

    Args:
        metrics (`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        `float`: The objective to minimize or maximize
    """
    metrics = copy.deepcopy(metrics)
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    # Remove speed metrics
    speed_metrics = [
        m
        for m in metrics.keys()
        if m.endswith("_runtime") or m.endswith("_per_second") or m.endswith("_compilation_time")
    ]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
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
    from .integrations import is_ray_tune_available

    assert is_ray_tune_available(), "This function needs ray installed: `pip install ray[tune]`"
    from ray import tune

    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(list(range(1, 6))),
        "seed": tune.uniform(1, 40),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
    }


def default_hp_space_sigopt(trial):
    return [
        {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double", "transformamtion": "log"},
        {"bounds": {"min": 1, "max": 6}, "name": "num_train_epochs", "type": "int"},
        {"bounds": {"min": 1, "max": 40}, "name": "seed", "type": "int"},
        {
            "categorical_values": ["4", "8", "16", "32", "64"],
            "name": "per_device_train_batch_size",
            "type": "categorical",
        },
    ]


def default_hp_space_wandb(trial) -> Dict[str, float]:
    from .integrations import is_wandb_available

    if not is_wandb_available():
        raise ImportError("This function needs wandb installed: `pip install wandb`")

    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "num_train_epochs": {"distribution": "int_uniform", "min": 1, "max": 6},
            "seed": {"distribution": "int_uniform", "min": 1, "max": 40},
            "per_device_train_batch_size": {"values": [4, 8, 16, 32, 64]},
        },
    }


class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"
    SIGOPT = "sigopt"
    WANDB = "wandb"


default_hp_space = {
    HPSearchBackend.OPTUNA: default_hp_space_optuna,
    HPSearchBackend.RAY: default_hp_space_ray,
    HPSearchBackend.SIGOPT: default_hp_space_sigopt,
    HPSearchBackend.WANDB: default_hp_space_wandb,
}


def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    if is_torch_tpu_available(check_device=True):
        import torch_xla.core.xla_model as xm

        return xm.get_ordinal() == 0
    return local_rank in [-1, 0]


def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    if is_torch_tpu_available(check_device=True):
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()
    elif local_rank != -1 and is_torch_available():
        import torch

        return torch.distributed.get_world_size()
    return 1


def speed_metrics(split, start_time, num_samples=None, num_steps=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    return result


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"


class TrainerMemoryTracker:
    """
    A helper class that tracks cpu and gpu memory.

    This class will silently skip unless `psutil` is available. Install with `pip install psutil`.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example :

    ```python
    self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
    self._memory_tracker.start()
    # code ...
    metrics = {"train_runtime": 10.5}
    self._memory_tracker.stop_and_update_metrics(metrics)
    ```

    At the moment GPU tracking is only for `pytorch`, but can be extended to support `tensorflow`.

    To understand this class' intricacies please read the documentation of [`~Trainer.log_metrics`].
    """

    # map trainer methods to metrics prefix
    stages = {
        "__init__": "init",
        "train": "train",
        "_inner_training_loop": "train",
        "evaluate": "eval",
        "predict": "test",
    }

    def __init__(self, skip_memory_metrics=False):
        self.skip_memory_metrics = skip_memory_metrics

        if not is_psutil_available():
            # soft dependency on psutil
            self.skip_memory_metrics = True

        if self.skip_memory_metrics:
            return

        import psutil  # noqa

        if is_torch_cuda_available():
            import torch

            self.torch = torch
            self.gpu = {}
        else:
            self.torch = None

        self.process = psutil.Process()

        self.cur_stage = None
        self.cpu = {}
        self.init_reported = False

    def derive_stage(self):
        """derives the stage/caller name automatically"""
        caller = inspect.currentframe().f_back.f_back.f_code.co_name
        if caller in self.stages:
            return self.stages[caller]
        else:
            raise ValueError(
                f"was called from {caller}, but only expect to be called from one of {self.stages.keys()}"
            )

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1

        while True:
            self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def start(self):
        """start tracking for the caller's stage"""
        if self.skip_memory_metrics:
            return

        stage = self.derive_stage()
        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        self.cur_stage = stage

        gc.collect()

        if self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.empty_cache()

        # gpu
        if self.torch is not None:
            self.gpu_mem_used_at_start = self.torch.cuda.memory_allocated()

        # cpu
        self.cpu_mem_used_at_start = self.cpu_mem_used()

        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def stop(self, stage):
        """stop tracking for the passed stage"""

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # this sends a signal to peak_monitor_func to complete its loop
        self.peak_monitoring = False

        # first ensure all objects get collected and their memory is freed
        gc.collect()

        if self.torch is not None:
            self.torch.cuda.empty_cache()

        # concepts:
        # - alloc_delta:  the difference of allocated memory between the end and the start
        # - peaked_delta: the difference between the peak memory and the current memory
        # in order to know how much memory the measured code consumed one needs to sum these two

        # gpu
        if self.torch is not None:
            self.gpu_mem_used_now = self.torch.cuda.memory_allocated()
            self.gpu_mem_used_peak = self.torch.cuda.max_memory_allocated()
            self.gpu[self.cur_stage] = dict(
                begin=self.gpu_mem_used_at_start,
                end=self.gpu_mem_used_now,
                alloc=(self.gpu_mem_used_now - self.gpu_mem_used_at_start),
                peaked=max(0, self.gpu_mem_used_peak - self.gpu_mem_used_now),
            )

        # cpu
        self.cpu_mem_used_now = self.cpu_mem_used()
        self.cpu[self.cur_stage] = dict(
            begin=self.cpu_mem_used_at_start,
            end=self.cpu_mem_used_now,
            alloc=(self.cpu_mem_used_now - self.cpu_mem_used_at_start),
            peaked=max(0, self.cpu_mem_used_peak - self.cpu_mem_used_now),
        )

        # reset - cycle finished
        self.cur_stage = None

    def update_metrics(self, stage, metrics):
        """updates the metrics"""
        if self.skip_memory_metrics:
            return

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # since we don't have a way to return init metrics, we push them into the first of train/val/predict
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        for stage in stages:
            for t in ["alloc", "peaked"]:
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]
            # if we need additional debug info, enable the following
            # for t in ["begin", "end"]:
            #     if stage in self.cpu and t in self.cpu[stage]:
            #         metrics[f"{stage}_mem_cpu_{t}"] = self.cpu[stage][t]
            #     if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
            #         metrics[f"{stage}_mem_gpu_{t}"] = self.gpu[stage][t]

        # since memory can be allocated before init, and it might be difficult to track overall
        # memory usage, in particular for GPU, let's report memory usage at the point init was called
        if stages[0] == "init":
            metrics["before_init_mem_cpu"] = self.cpu["init"]["begin"]
            if self.torch is not None:
                metrics["before_init_mem_gpu"] = self.gpu["init"]["begin"]
            # if we also wanted to report any additional memory allocations in between init and
            # whatever the next stage was we could also report this:
            # if self.cpu["init"]["end"] != self.cpu[stage]["begin"]:
            #     metrics[f"after_init_mem_cpu_delta"] = self.cpu[stage]["begin"] - self.cpu["init"]["end"]
            # if self.torch is not None and self.gpu["init"]["end"] != self.gpu[stage]["begin"]:
            #     metrics[f"after_init_mem_gpu_delta"] = self.gpu[stage]["begin"] - self.gpu["init"]["end"]

    def stop_and_update_metrics(self, metrics=None):
        """combine stop and metrics update in one call for simpler code"""
        if self.skip_memory_metrics:
            return

        stage = self.derive_stage()
        self.stop(stage)

        # init doesn't have metrics to update so we just save that data for later stages to retrieve
        if metrics is not None:
            self.update_metrics(stage, metrics)


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


def denumpify_detensorize(metrics):
    """
    Recursively calls `.item()` on the element of the dictionary passed
    """
    if isinstance(metrics, (list, tuple)):
        return type(metrics)(denumpify_detensorize(m) for m in metrics)
    elif isinstance(metrics, dict):
        return type(metrics)({k: denumpify_detensorize(v) for k, v in metrics.items()})
    elif isinstance(metrics, np.generic):
        return metrics.item()
    elif is_torch_available() and isinstance(metrics, torch.Tensor) and metrics.numel() == 1:
        return metrics.item()
    return metrics


def number_of_arguments(func):
    """
    Return the number of arguments of the passed function, even if it's a partial function.
    """
    if isinstance(func, functools.partial):
        total_args = len(inspect.signature(func.func).parameters)
        return total_args - len(func.args) - len(func.keywords)
    return len(inspect.signature(func).parameters)


class ShardedDDPOption(ExplicitEnum):
    SIMPLE = "simple"
    ZERO_DP_2 = "zero_dp_2"
    ZERO_DP_3 = "zero_dp_3"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"


def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128, auto_find_batch_size: bool = False
):
    """
    Args:
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function` `function` must take in a `batch_size` parameter as
    its first argument.
        function (`callable`, *optional*)
            A function to wrap
        starting_batch_size (`int`, *optional*)
            The batch size to try and fit into memory
        auto_find_batch_size (`bool`, *optional*)
            If False, will just execute `function`
    """
    if function is None:
        return functools.partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            auto_find_batch_size=auto_find_batch_size,
        )

    if auto_find_batch_size:
        requires_backends(find_executable_batch_size, "accelerate")
        from accelerate.utils import find_executable_batch_size as accelerate_find_executable_batch_size

        return accelerate_find_executable_batch_size(function=function, starting_batch_size=starting_batch_size)

    return functools.partial(function, batch_size=starting_batch_size)


class FSDPOption(ExplicitEnum):
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"


class RemoveColumnsCollator:
    """Wrap the data collator to remove unused columns before they are passed to the collator."""

    def __init__(
        self,
        data_collator,
        signature_columns,
        logger=None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.data_collator = data_collator
        self.signature_columns = signature_columns
        self.logger = logger
        self.description = description
        self.model_name = model_name
        self.message_logged = False

    def _remove_columns(self, feature: dict) -> dict:
        if not isinstance(feature, dict):
            return feature
        if not self.message_logged and self.logger and self.model_name:
            ignored_columns = list(set(feature.keys()) - set(self.signature_columns))
            if len(ignored_columns) > 0:
                dset_description = "" if self.description is None else f"in the {self.description} set"
                self.logger.info(
                    f"The following columns {dset_description} don't have a corresponding argument in "
                    f"`{self.model_name}.forward` and have been ignored: {', '.join(ignored_columns)}."
                    f" If {', '.join(ignored_columns)} are not expected by `{self.model_name}.forward`, "
                    " you can safely ignore this message."
                )
                self.message_logged = True
        return {k: v for k, v in feature.items() if k in self.signature_columns}

    def __call__(self, features: List[dict]):
        features = [self._remove_columns(feature) for feature in features]
        return self.data_collator(features)
