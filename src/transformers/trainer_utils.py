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
PyTorch-independent utilities for the Trainer class.
"""

import copy
import functools
import gc
import inspect
import json
import os
import random
import re
import shutil
import threading
import time
from collections.abc import Callable, Sized
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, TypeGuard

import numpy as np

from .utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_INDEX_NAME,
    ExplicitEnum,
    check_torch_load_is_safe,
    is_peft_available,
    is_psutil_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
    requires_backends,
)


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch
    from safetensors.torch import load_file as safe_load_file

if is_peft_available():
    from peft import PeftMixedModel, PeftModel


def _is_peft_model(model):
    if is_peft_available():
        return isinstance(model, (PeftModel, PeftMixedModel))
    return False


def unwrap_peft_model(model):
    """
    Extract the base model from a PEFT-wrapped model.

    If the model is not a PEFT model, returns it unchanged. Otherwise, attempts to
    unwrap the base model using ``get_base_model()`` or the ``base_model.model`` attribute.

    Args:
        model: The model to unwrap.

    Returns:
        The unwrapped base model.

    Raises:
        AttributeError: If the model is a PEFT model but cannot be unwrapped safely.
    """
    if not _is_peft_model(model):
        return model
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # PeftMixedModel do not provide a `get_base_model` method
        return model.base_model.model
    else:
        raise AttributeError("Cannot extract base model safely from this PEFT wrapper.")


def validate_quantization_for_training(model):
    """
    Validate that a quantized model is set up correctly for training.

    Raises `ValueError` when:
    - A quantized + compiled model is used (torch.compile is not supported with PEFT fine-tuning).
    - A purely quantized model has no trainable adapters attached (unless it supports QAT).
    - The quantization method does not support training.

    Args:
        model: The model to validate.
    """
    _is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
        model, "_hf_peft_config_loaded", False
    )
    _quantization_method_supports_training = (
        getattr(model, "hf_quantizer", None) is not None and model.hf_quantizer.is_trainable
    )
    _is_model_quantized_and_qat_trainable = getattr(model, "hf_quantizer", None) is not None and getattr(
        model.hf_quantizer, "is_qat_trainable", False
    )

    # Filter out quantized + compiled models
    if _is_quantized_and_base_model and hasattr(model, "_orig_mod"):
        raise ValueError(
            "You cannot fine-tune quantized model with `torch.compile()` make sure to pass a non-compiled model when fine-tuning a quantized model with PEFT"
        )

    # At this stage the model is already loaded
    if _is_quantized_and_base_model and not _is_peft_model(model) and not _is_model_quantized_and_qat_trainable:
        raise ValueError(
            "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
            " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft"
            " for more details"
        )
    elif _is_quantized_and_base_model and not _quantization_method_supports_training:
        raise ValueError(
            f"The model you are trying to fine-tune is quantized with {model.hf_quantizer.quantization_config.quant_method}"
            " but that quantization method do not support training. Please open an issue on GitHub: https://github.com/huggingface/transformers"
            f" to request the support for training support for {model.hf_quantizer.quantization_config.quant_method}"
        )


def seed_worker(worker_id: int, num_workers: int, rank: int):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    init_seed = torch.initial_seed() % 2**32
    worker_seed = num_workers * rank + init_seed
    set_seed(worker_seed)


def enable_full_determinism(seed: int, warn_only: bool = False):
    """
    Helper function for reproducible behavior during distributed training. See
    https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    # set seed first
    set_seed(seed)

    if is_torch_available():
        # Enable PyTorch deterministic mode. This potentially requires either the environment
        # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
        # depending on the CUDA version, so we set them both here
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
        os.environ["HCCL_DETERMINISTIC"] = "1"

        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
        if deterministic:
            torch.use_deterministic_algorithms(True)
    if is_torch_mlu_available():
        torch.mlu.manual_seed_all(seed)
    if is_torch_musa_available():
        torch.musa.manual_seed_all(seed)
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    if is_torch_hpu_available():
        torch.hpu.manual_seed_all(seed)
    if is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*): Input data passed to the model.
        losses (`np.ndarray`, *optional*): Loss values computed during evaluation.
    """

    def __init__(
        self,
        predictions: np.ndarray | tuple[np.ndarray],
        label_ids: np.ndarray | tuple[np.ndarray],
        inputs: np.ndarray | tuple[np.ndarray] | None = None,
        losses: np.ndarray | tuple[np.ndarray] | None = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs
        self.losses = losses
        self.elements = (self.predictions, self.label_ids)
        if self.inputs is not None:
            self.elements += (self.inputs,)
        if self.losses is not None:
            self.elements += (self.losses,)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.elements):
            raise IndexError("tuple index out of range")
        return self.elements[idx]


class EvalLoopOutput(NamedTuple):
    predictions: np.ndarray | tuple[np.ndarray]
    label_ids: np.ndarray | tuple[np.ndarray] | None
    metrics: dict[str, float] | None
    num_samples: int | None


class PredictionOutput(NamedTuple):
    predictions: np.ndarray | tuple[np.ndarray]
    label_ids: np.ndarray | tuple[np.ndarray] | None
    metrics: dict[str, float] | None


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: dict[str, float]


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


def sort_checkpoints(
    output_dir: str,
    checkpoint_prefix: str = PREFIX_CHECKPOINT_DIR,
    use_mtime: bool = False,
    best_model_checkpoint: str | None = None,
) -> list[str]:
    """
    Return checkpoint directories sorted by step number (oldest first).

    Args:
        output_dir (`str`):
            The directory containing the checkpoints.
        checkpoint_prefix (`str`, *optional*, defaults to `"checkpoint"`):
            The prefix used for checkpoint directory names.
        use_mtime (`bool`, *optional*, defaults to `False`):
            Whether to sort by modification time instead of step number.
        best_model_checkpoint (`str`, *optional*):
            If provided, this checkpoint is moved to second-to-last position to protect
            it from deletion while keeping the most recent checkpoint last for resuming.

    Returns:
        `list[str]`: Sorted list of checkpoint directory paths (oldest first).
    """
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)

    # mtime is not reliable on some filesystems (e.g., cloud fuse filesystems)
    # so we check if the mtime is fake and fall back to numerical ordering
    if use_mtime and len(checkpoints_sorted) > 1:
        mtime_diff = checkpoints_sorted[-1][0] - checkpoints_sorted[0][0]
        if mtime_diff < 1.0:
            logger.warning("mtime may not be reliable on this filesystem, falling back to numerical ordering")
            return sort_checkpoints(
                output_dir, checkpoint_prefix, use_mtime=False, best_model_checkpoint=best_model_checkpoint
            )

    checkpoints_sorted = [path for _, path in checkpoints_sorted]

    # Move best_model_checkpoint to second-to-last position to protect it from deletion
    # while keeping the most recent checkpoint at the end for resuming training.
    if best_model_checkpoint is not None:
        best_model_checkpoint = str(Path(best_model_checkpoint))
        if best_model_checkpoint in checkpoints_sorted and checkpoints_sorted[-1] != best_model_checkpoint:
            most_recent = checkpoints_sorted[-1]
            checkpoints_sorted = [c for c in checkpoints_sorted if c not in {best_model_checkpoint, most_recent}]
            checkpoints_sorted += [best_model_checkpoint, most_recent]

    return checkpoints_sorted


def rotate_checkpoints(
    output_dir: str,
    save_total_limit: int | None = None,
    best_model_checkpoint: str | None = None,
    use_mtime: bool = False,
    checkpoint_prefix: str = PREFIX_CHECKPOINT_DIR,
) -> None:
    """
    Delete older checkpoints, keeping at most `save_total_limit`.

    Always preserves the most recent checkpoint and the best model checkpoint (if provided).

    Args:
        output_dir (`str`):
            The directory containing the checkpoints.
        save_total_limit (`int`, *optional*):
            Maximum number of checkpoints to keep. No deletion if `None` or <= 0.
        best_model_checkpoint (`str`, *optional*):
            Path to best checkpoint (will always be preserved).
        use_mtime (`bool`, *optional*, defaults to `False`):
            Whether to sort by modification time instead of step number.
        checkpoint_prefix (`str`, *optional*, defaults to `"checkpoint"`):
            The prefix used for checkpoint directory names.
    """
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoints = sort_checkpoints(output_dir, checkpoint_prefix, use_mtime)
    if len(checkpoints) <= save_total_limit:
        return

    # Checkpoints that must not be deleted
    protected = {checkpoints[-1]}  # most recent, for resuming
    if best_model_checkpoint is not None:
        protected.add(str(Path(best_model_checkpoint)))

    # Delete oldest non-protected checkpoints until we have save_total_limit left
    num_to_keep = max(save_total_limit, len(protected))
    remaining = len(checkpoints)
    for checkpoint in checkpoints:
        if remaining <= num_to_keep:
            break
        if checkpoint not in protected:
            shutil.rmtree(checkpoint, ignore_errors=True)
            remaining -= 1


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SaveStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"
    BEST = "best"


class HubStrategy(ExplicitEnum):
    END = "end"
    EVERY_SAVE = "every_save"
    CHECKPOINT = "checkpoint"
    ALL_CHECKPOINTS = "all_checkpoints"


class BestRun(NamedTuple):
    """
    The best run found by a hyperparameter search (see [`~Trainer.hyperparameter_search`]).

    Parameters:
        run_id (`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (`float`):
            The objective that was obtained for this run.
        hyperparameters (`dict[str, Any]`):
            The hyperparameters picked to get this run.
        run_summary (`Optional[Any]`):
            A summary of tuning experiments. `ray.tune.ExperimentAnalysis` object for Ray backend.
    """

    run_id: str
    objective: float | list[float]
    hyperparameters: dict[str, Any]
    run_summary: Any | None = None


def default_compute_objective(metrics: dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the [`Trainer`], the sum of all metrics otherwise.

    Args:
        metrics (`dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        `float`: The objective to minimize or maximize
    """
    metrics = copy.deepcopy(metrics)
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    # Remove speed metrics
    speed_metrics = [m for m in metrics if m.endswith("_runtime") or m.endswith("_per_second")]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return loss if len(metrics) == 0 else sum(metrics.values())


def default_hp_space_optuna(trial) -> dict[str, float]:
    from .integrations import is_optuna_available

    assert is_optuna_available(), "This function needs Optuna installed: `pip install optuna`"
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
    }


def default_hp_space_ray(trial) -> dict[str, Any]:
    from .integrations import is_ray_tune_available

    assert is_ray_tune_available(), "This function needs ray installed: `pip install ray[tune]`"
    from ray import tune

    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(list(range(1, 6))),
        "seed": tune.uniform(1, 40),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
    }


def default_hp_space_wandb(trial) -> dict[str, Any]:
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
    WANDB = "wandb"


def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xr.global_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    if is_torch_xla_available():
        import torch_xla.runtime as xr

        return xr.global_ordinal() == 0
    return local_rank in [-1, 0]


def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    if is_torch_xla_available():
        import torch_xla.runtime as xr

        return xr.world_size()
    elif local_rank != -1 and is_torch_available():
        import torch

        return torch.distributed.get_world_size()
    return 1


def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    return result


class SchedulerType(ExplicitEnum):
    """
    Scheduler names for the parameter `lr_scheduler_type` in [`TrainingArguments`].
    By default, it uses "linear". Internally, this retrieves `get_linear_schedule_with_warmup` scheduler from [`Trainer`].
    Scheduler types:
       - "linear" = [`get_linear_schedule_with_warmup`]
       - "cosine" = [`get_cosine_schedule_with_warmup`]
       - "cosine_with_restarts" = [`get_cosine_with_hard_restarts_schedule_with_warmup`]
       - "polynomial" = [`get_polynomial_decay_schedule_with_warmup`]
       - "constant" =  [`get_constant_schedule`]
       - "constant_with_warmup" = [`get_constant_schedule_with_warmup`]
       - "inverse_sqrt" = [`get_inverse_sqrt_schedule`]
       - "reduce_lr_on_plateau" = [`get_reduce_on_plateau_schedule`]
       - "cosine_with_min_lr" = [`get_cosine_with_min_lr_schedule_with_warmup`]
       - "cosine_warmup_with_min_lr" = [`get_cosine_with_min_lr_schedule_with_warmup_lr_rate`]
       - "warmup_stable_decay" = [`get_wsd_schedule`]
    """

    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    COSINE_WARMUP_WITH_MIN_LR = "cosine_warmup_with_min_lr"
    WARMUP_STABLE_DECAY = "warmup_stable_decay"


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

        import psutil

        if is_torch_cuda_available() or is_torch_mlu_available() or is_torch_musa_available():
            import torch

            self.torch = torch
            self.gpu = {}
        elif is_torch_mps_available():
            import torch

            self.torch = torch
            self.gpu = {}
        elif is_torch_xpu_available():
            import torch

            self.torch = torch
            self.gpu = {}
        elif is_torch_npu_available():
            import torch

            self.torch = torch
            self.gpu = {}
        elif is_torch_hpu_available():
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
            if torch.cuda.is_available():
                self.torch.cuda.reset_peak_memory_stats()
                self.torch.cuda.empty_cache()
            elif is_torch_mlu_available():
                self.torch.mlu.reset_peak_memory_stats()
                self.torch.mlu.empty_cache()
            elif is_torch_musa_available():
                self.torch.musa.reset_peak_memory_stats()
                self.torch.musa.empty_cache()
            elif is_torch_xpu_available():
                self.torch.xpu.reset_peak_memory_stats()
                self.torch.xpu.empty_cache()
            elif is_torch_npu_available():
                self.torch.npu.reset_peak_memory_stats()
                self.torch.npu.empty_cache()
            elif is_torch_hpu_available():
                self.torch.hpu.reset_peak_memory_stats()
                # not available on hpu as it reserves all device memory for the current process
                # self.torch.hpu.empty_cache()
            elif is_torch_mps_available():
                self.torch.mps.empty_cache()

        # gpu
        if self.torch is not None:
            if torch.cuda.is_available():
                self.gpu_mem_used_at_start = self.torch.cuda.memory_allocated()
            elif is_torch_mlu_available():
                self.gpu_mem_used_at_start = self.torch.mlu.memory_allocated()
            elif is_torch_musa_available():
                self.gpu_mem_used_at_start = self.torch.musa.memory_allocated()
            elif is_torch_xpu_available():
                self.gpu_mem_used_at_start = self.torch.xpu.memory_allocated()
            elif is_torch_npu_available():
                self.gpu_mem_used_at_start = self.torch.npu.memory_allocated()
            elif is_torch_hpu_available():
                self.gpu_mem_used_at_start = self.torch.hpu.memory_allocated()
            elif is_torch_mps_available():
                self.gpu_mem_used_at_start = self.torch.mps.current_allocated_memory()

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
            if torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            elif is_torch_mlu_available():
                self.torch.mlu.empty_cache()
            elif is_torch_musa_available():
                self.torch.musa.empty_cache()
            elif is_torch_xpu_available():
                self.torch.xpu.empty_cache()
            elif is_torch_npu_available():
                self.torch.npu.empty_cache()
            elif is_torch_hpu_available():
                # not available on hpu as it reserves all device memory for the current process
                # self.torch.npu.empty_cache()
                pass
            elif is_torch_mps_available():
                self.torch.mps.empty_cache()

        # concepts:
        # - alloc_delta:  the difference of allocated memory between the end and the start
        # - peaked_delta: the difference between the peak memory and the current memory
        # in order to know how much memory the measured code consumed one needs to sum these two

        # gpu
        if self.torch is not None:
            if torch.cuda.is_available():
                self.gpu_mem_used_now = self.torch.cuda.memory_allocated()
                self.gpu_mem_used_peak = self.torch.cuda.max_memory_allocated()
            elif is_torch_mlu_available():
                self.gpu_mem_used_now = self.torch.mlu.memory_allocated()
                self.gpu_mem_used_peak = self.torch.mlu.max_memory_allocated()
            elif is_torch_musa_available():
                self.gpu_mem_used_now = self.torch.musa.memory_allocated()
                self.gpu_mem_used_peak = self.torch.musa.max_memory_allocated()
            elif is_torch_xpu_available():
                self.gpu_mem_used_now = self.torch.xpu.memory_allocated()
                self.gpu_mem_used_peak = self.torch.xpu.max_memory_allocated()
            elif is_torch_npu_available():
                self.gpu_mem_used_now = self.torch.npu.memory_allocated()
                self.gpu_mem_used_peak = self.torch.npu.max_memory_allocated()
            elif is_torch_hpu_available():
                self.gpu_mem_used_now = self.torch.hpu.memory_allocated()
                self.gpu_mem_used_peak = self.torch.hpu.max_memory_allocated()
            elif is_torch_mps_available():
                self.gpu_mem_used_now = self.torch.mps.current_allocated_memory()
                # self.torch.mps.max_memory_allocated() does not exist yet
                self.gpu_mem_used_peak = None

            else:
                raise ValueError("No available GPU device found!")

            self.gpu[self.cur_stage] = {
                "begin": self.gpu_mem_used_at_start,
                "end": self.gpu_mem_used_now,
                "alloc": (self.gpu_mem_used_now - self.gpu_mem_used_at_start),
            }
            if self.gpu_mem_used_peak is not None:
                self.gpu[self.cur_stage]["peaked"] = max(0, self.gpu_mem_used_peak - self.gpu_mem_used_now)
            else:
                self.gpu[self.cur_stage]["peaked"] = "Not available"

        # cpu
        self.cpu_mem_used_now = self.cpu_mem_used()
        self.cpu[self.cur_stage] = {
            "begin": self.cpu_mem_used_at_start,
            "end": self.cpu_mem_used_now,
            "alloc": (self.cpu_mem_used_now - self.cpu_mem_used_at_start),
            "peaked": max(0, self.cpu_mem_used_peak - self.cpu_mem_used_now),
        }

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


def has_length(dataset: Any) -> TypeGuard[Sized]:
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False
    except AttributeError:
        # Ray DataSets raises an AttributeError: https://github.com/ray-project/ray/blob/master/python/ray/data/dataset.py#L5616
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


def find_executable_batch_size(
    function: Callable | None = None, starting_batch_size: int = 128, auto_find_batch_size: bool = False
):
    """
    Args:
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is multiplied by 0.9 and passed to `function`. `function` must take in a `batch_size` parameter as
    its first argument.
        function (`Callable`, *optional*)
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
    HYBRID_SHARD = "hybrid_shard"
    HYBRID_SHARD_ZERO2 = "hybrid_shard_zero2"
    OFFLOAD = "offload"
    AUTO_WRAP = "auto_wrap"


class RemoveColumnsCollator:
    """Wrap the data collator to remove unused columns before they are passed to the collator."""

    def __init__(
        self,
        data_collator,
        signature_columns,
        logger=None,
        model_name: str | None = None,
        description: str | None = None,
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

    def __call__(self, features: list[dict]):
        features = [self._remove_columns(feature) for feature in features]
        return self.data_collator(features)


def check_target_module_exists(optim_target_modules, key: str, return_is_regex: bool = False):
    """A helper method to check if the passed module's key name matches any of the target modules in the optim_target_modules.

    Args:
        optim_target_modules (`Union[str, list[str]]`):
            A list of strings to try to match. Can be also a full string.
        key (`str`):
            A key to search any matches in optim_target_modules
        return_is_regex (`bool`):
            If set to `True`, the method will return whether the passed `optim_target_modules`
            is a regex or not.

    Returns:
        `bool` : True of match object if key matches any target modules from config, False or
        None if no match found
        `bool` : If the matched target module is a regex to silence out the warnings in Trainer
        for extra modules being found (only if `target_module_found=True` for an array of regex).
    """
    target_module_found = False
    is_regex = False

    if isinstance(optim_target_modules, str):
        target_module_found = bool(re.fullmatch(optim_target_modules, key))
        is_regex = optim_target_modules != key
    elif key in optim_target_modules:  # from here, target_module_found must be a list of str
        # this module is specified directly in target_modules
        target_module_found = True
    elif any(target_key in key for target_key in optim_target_modules):
        target_module_found = True
    elif any(bool(re.fullmatch(optim_target_module, key)) for optim_target_module in optim_target_modules):
        target_module_found = True
        is_regex = True

    if return_is_regex:
        return target_module_found, is_regex

    return target_module_found


def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    """
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional*, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`):
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not safe_index_present:
        filenames = (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME)
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = safe_index_present and (prefer_safe or not index_present)
    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    # TODO: Here, update the weight map with the config.dynamic_weight_conversion
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    if load_safe:
        loader = safe_load_file
    else:
        check_torch_load_is_safe()
        loader = partial(torch.load, map_location="cpu", weights_only=True)

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PyTorch load_state_dict function.
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)


def compare_trainer_and_checkpoint_args(training_args, trainer_state):
    """
    Compare training arguments with those stored in a checkpoint's trainer state.

    Logs a warning if there are mismatches between the current training arguments
    and the ones saved in the checkpoint.

    Args:
        training_args: The current training arguments.
        trainer_state: The trainer state loaded from a checkpoint.
    """
    attributes_map = {
        "logging_steps": "logging_steps",
        "eval_steps": "eval_steps",
        "save_steps": "save_steps",
    }

    has_warning = False
    warning_str = "Warning: The following arguments do not match the ones in the `trainer_state.json` within the checkpoint directory: "
    for arg_attr, state_attr in attributes_map.items():
        arg_value = getattr(training_args, arg_attr, None)
        state_value = getattr(trainer_state, state_attr, None)

        if arg_value is not None and state_value is not None and arg_value != state_value:
            warning_str += f"\n\t{arg_attr}: {arg_value} (from args) != {state_value} (from trainer_state.json)"
            has_warning = True

    # train bs is special as we need to account for multi-GPU
    train_bs_args = training_args.per_device_train_batch_size
    train_bs_state = trainer_state.train_batch_size // max(1, training_args.n_gpu)

    if train_bs_args != train_bs_state:
        warning_str += f"\n\tper_device_train_batch_size: {train_bs_args} (from args) != {train_bs_state} (from trainer_state.json)"
        has_warning = True

    if has_warning:
        logger.warning_once(warning_str)


def align_special_tokens(model, processing_class):
    """
    Aligns the special tokens of the tokenizer with the model configs.

    A new tokens may be defined in the tokenizer for fine-tuning purposes, e.g. an "end of turn" token may be
    added on chat models. In that case, we want the model configs to be aligned with the tokenizer, so that all
    downstream uses work as expected. This alignment should happen before training, to ensure the prediction step
    uses the new tokens as well.
    """
    from .processing_utils import ProcessorMixin
    from .tokenization_utils_base import PreTrainedTokenizerBase

    if isinstance(processing_class, ProcessorMixin):
        tokenizer: PreTrainedTokenizerBase = processing_class.tokenizer
    else:
        tokenizer = processing_class
    model_has_generation_config = hasattr(model, "generation_config") and model.generation_config is not None
    updated_tokens = {}

    # 1 - Align EOS token. EOS is more complex than the others, as `generation_config` may hold more than one EOS
    # token.
    tokenizer_has_new_eos = tokenizer.eos_token_id != getattr(model.config, "eos_token_id", None)
    if model_has_generation_config:
        # `generation_config.eos_token_id` is None: direct comparison
        if model.generation_config.eos_token_id is None:
            tokenizer_has_new_eos |= tokenizer.eos_token_id != model.generation_config.eos_token_id
        else:
            # `generation_config.eos_token_id` is an `int`: convert it to list (and continue below)
            if isinstance(model.generation_config.eos_token_id, int):
                model.generation_config.eos_token_id = [model.generation_config.eos_token_id]
            # `generation_config.eos_token_id` is a `list`: check if the tokenizer's EOS token is in the list
            tokenizer_has_new_eos |= tokenizer.eos_token_id not in model.generation_config.eos_token_id

    if tokenizer_has_new_eos:
        updated_tokens["eos_token_id"] = tokenizer.eos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        # The generation config may hold more than one EOS token. We preserve the original EOS tokens: any of the
        # EOS tokens defined here will halt generation.
        if model_has_generation_config:
            all_eos_tokens = [tokenizer.eos_token_id]
            if model.generation_config.eos_token_id is not None:
                all_eos_tokens += list(model.generation_config.eos_token_id)
            model.generation_config.eos_token_id = [token for token in all_eos_tokens if token is not None]

    # 2 - Align BOS
    tokenizer_has_new_bos = tokenizer.bos_token_id != getattr(model.config, "bos_token_id", None)
    if model_has_generation_config:
        tokenizer_has_new_bos |= tokenizer.bos_token_id != model.generation_config.bos_token_id

    if tokenizer_has_new_bos:
        updated_tokens["bos_token_id"] = tokenizer.bos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        if model_has_generation_config:
            model.generation_config.bos_token_id = tokenizer.bos_token_id

    # 3 - Align PAD
    tokenizer_has_new_pad = tokenizer.pad_token_id != getattr(model.config, "pad_token_id", None)
    if model_has_generation_config:
        tokenizer_has_new_pad |= tokenizer.pad_token_id != model.generation_config.pad_token_id

    if tokenizer_has_new_pad:
        updated_tokens["pad_token_id"] = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        if model_has_generation_config:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 4 - Warn users about the changes
    if len(updated_tokens) > 0:
        logger.warning(
            "The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. "
            "The model config and generation config were aligned accordingly, being updated with the tokenizer's "
            f"values. Updated tokens: {updated_tokens}."
        )
