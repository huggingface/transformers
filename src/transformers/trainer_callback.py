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
Callbacks to use with the Trainer class and customize the training loop.
"""

import dataclasses
import json
import math
import re
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from .trainer_utils import IntervalStrategy, SaveStrategy, has_length
from .training_args import TrainingArguments
from .utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.distributed as dist


logger = logging.get_logger(__name__)


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    <Tip>

    In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
    step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
    step requires going through *n* batches.

    </Tip>

    Args:
        epoch (`float`, *optional*):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (`int`, *optional*, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (`int`, *optional*, defaults to 0):
            The number of update steps to do during the current training.
        logging_steps (`int`, *optional*, defaults to 500):
            Log every X updates steps
        eval_steps (`int`, *optional*):
            Run an evaluation every X steps.
        save_steps (`int`, *optional*, defaults to 500):
            Save checkpoint every X updates steps.
        train_batch_size (`int`, *optional*):
            The batch size for the training dataloader. Only needed when
            `auto_find_batch_size` has been used.
        num_input_tokens_seen (`int`, *optional*, defaults to 0):
            When tracking the inputs tokens, the number of tokens seen during training (number of input tokens, not the
            number of prediction tokens).
        total_flos (`float`, *optional*, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (`list[dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.
        best_metric (`float`, *optional*):
            When tracking the best model, the value of the best metric encountered so far.
        best_global_step (`int`, *optional*):
            When tracking the best model, the step at which the best metric was encountered.
            Used for setting `best_model_checkpoint`.
        best_model_checkpoint (`str`, *optional*):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be `True` for one process).
        is_hyper_param_search (`bool`, *optional*, defaults to `False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way data will be logged in TensorBoard.
        stateful_callbacks (`list[StatefulTrainerCallback]`, *optional*):
            Callbacks attached to the `Trainer` that should have their states be saved or restored.
            Relevant callbacks should implement a `state` and `from_state` function.
    """

    epoch: float = 0
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    train_batch_size: int | None = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    total_flos: float = 0
    log_history: list[dict[str, float]] = None
    best_metric: float | None = None
    best_global_step: int | None = None
    best_model_checkpoint: str | None = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str | None = None
    trial_params: dict[str, str | float | int | bool] | None = None
    stateful_callbacks: list["TrainerCallback"] | None = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        if self.stateful_callbacks is None:
            self.stateful_callbacks = {}
        elif isinstance(self.stateful_callbacks, dict):
            # We are loading the callbacks in from the state file, no need to process them
            pass
        else:
            # Saveable callbacks get stored as dict of kwargs
            stateful_callbacks = {}
            for callback in self.stateful_callbacks:
                if not isinstance(callback, (ExportableState)):
                    raise TypeError(
                        f"All callbacks passed to be saved must inherit `ExportableState`, but received {type(callback)}"
                    )
                name = callback.__class__.__name__
                if name in stateful_callbacks:
                    # We can have multiple versions of the same callback
                    # if so, we store them as a list of states to restore
                    if not isinstance(stateful_callbacks[name], list):
                        stateful_callbacks[name] = [stateful_callbacks[name]]
                    stateful_callbacks[name].append(callback.state())
                else:
                    stateful_callbacks[name] = callback.state()
            self.stateful_callbacks = stateful_callbacks

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    def compute_steps(self, args, max_steps):
        """
        Calculates and stores the absolute value for logging,
        eval, and save steps based on if it was a proportion
        or not.
        """
        for step_kind in ("logging", "eval", "save"):
            num_steps = getattr(args, f"{step_kind}_steps")
            if num_steps is not None:
                if num_steps < 1:
                    num_steps = math.ceil(max_steps * num_steps)
                setattr(self, f"{step_kind}_steps", num_steps)

    def init_training_references(self, trainer, max_steps, num_train_epochs, trial):
        """
        Stores the initial training references needed in `self`
        """
        if trainer.hp_name is not None and trainer._trial is not None:
            # use self._trial because the Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.trial_name = trainer.hp_name(trainer._trial)
        self.trial_params = None
        if trial is not None:
            from transformers.integrations import hp_params

            self.trial_params = hp_params(trial)

        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        self.is_local_process_zero = trainer.is_local_process_zero()
        self.is_world_process_zero = trainer.is_world_process_zero()


class ExportableState:
    """
    A class for objects that include the ability to have its state
    be saved during `Trainer._save_checkpoint` and loaded back in during
    `Trainer._load_from_checkpoint`.

    These must implement a `state` function that gets called during the respective
    Trainer function call. It should only include parameters and attributes needed to
    recreate the state at a particular time, to avoid utilizing pickle/maintain standard
    file IO writing.

    Example:

    ```python
    class EarlyStoppingCallback(TrainerCallback, ExportableState):
        def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_threshold = early_stopping_threshold
            # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
            self.early_stopping_patience_counter = 0

        def state(self) -> dict:
            return {
                "args": {
                    "early_stopping_patience": self.early_stopping_patience,
                    "early_stopping_threshold": self.early_stopping_threshold,
                },
                "attributes": {
                    "early_stopping_patience_counter": self.early_stopping_patience_counter,
                }
            }
    ```"""

    def state(self) -> dict:
        raise NotImplementedError("You must implement a `state` function to utilize this class.")

    @classmethod
    def from_state(cls, state):
        instance = cls(**state["args"])
        for k, v in state["attributes"].items():
            setattr(instance, k, v)
        return instance


@dataclass
class TrainerControl(ExportableState):
    """
    A class that handles the [`Trainer`] control flow. This class is used by the [`TrainerCallback`] to activate some
    switches in the training loop.

    Args:
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False

    def state(self) -> dict:
        return {
            "args": {
                "should_training_stop": self.should_training_stop,
                "should_epoch_stop": self.should_epoch_stop,
                "should_save": self.should_save,
                "should_evaluate": self.should_evaluate,
                "should_log": self.should_log,
            },
            "attributes": {},
        }


class TrainerCallback:
    # no-format
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        processing_class ([`PreTrainedTokenizer` or `BaseImageProcessor` or `ProcessorMixin` or `FeatureExtractionMixin`]):
            The processing class used for encoding the data. Can be a tokenizer, a processor, an image processor or a feature extractor.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for evaluation.
        metrics (`dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.

    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformers.PrinterCallback`].

    Example:

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```"""

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called before the optimizer step but after gradient clipping. Useful for monitoring gradients.
        """

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients.
        """

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """

    def on_push_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called before pushing the model to the hub, at the beginning of Trainer.push_to_hub and Trainer._push_from_checkpoint.
        """


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(self, callbacks, model, processing_class, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_pre_optimizer_step", args, state, control)

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_optimizer_step", args, state, control)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_substep_end", args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        return self.call_event("on_predict", args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_prediction_step", args, state, control)

    def on_push_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_push_begin", args, state, control, **kwargs)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class MoERouterHealthCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that records MoE router health metrics during training.

    The callback installs forward hooks on router modules advertised through the model's `can_record_outputs`
    metadata. It prefers exact selected expert indices when a router surfaces them and otherwise falls back to
    deriving top-k assignments from router logits. Metrics are emitted as flat trainer logs.

    Args:
        prefix (`str`, *optional*, defaults to `"moe"`):
            Prefix used for the logged metric keys.
        reduction_mode (`str`, *optional*, defaults to `"auto"`):
            How to reduce expert counts before logging. Use `"auto"` to reduce across the world process group for
            normal distributed replicas and to skip implicit reduction for tensor-parallel models. Use `"world"` to
            always all-reduce across the default process group, or `"none"` to disable implicit reduction entirely.
        log_aux_loss (`bool`, *optional*, defaults to `True`):
            Whether to log model-level auxiliary routing losses such as `aux_loss` when present in model outputs.
    """

    def __init__(self, prefix: str = "moe", reduction_mode: str = "auto", log_aux_loss: bool = True):
        if reduction_mode not in {"auto", "world", "none"}:
            raise ValueError(
                f"`reduction_mode` must be one of 'auto', 'world', or 'none', but got {reduction_mode!r}."
            )
        self.prefix = prefix.rstrip("/")
        self.reduction_mode = reduction_mode
        self.log_aux_loss = log_aux_loss
        self._layer_counts = {}
        self._layer_order = []
        self._router_handles = []
        self._model_handle = None
        self._last_aux_metrics = {}
        self._resolved_reduction_mode = reduction_mode

    @staticmethod
    def _safe_metric_divide(numerator, denominator) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def _format_layer_name(module_name: str, layer_idx: int) -> str:
        if module_name:
            sanitized_name = re.sub(r"[^a-zA-Z0-9]+", "_", module_name).strip("_")
            if sanitized_name:
                return sanitized_name
        return f"layer_{layer_idx}"

    @staticmethod
    def _compute_routing_metrics(expert_counts) -> dict[str, float]:
        counts = expert_counts.to(dtype=torch.float64)
        total_assignments = counts.sum()
        if total_assignments <= 0:
            return {
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "load_cv": 0.0,
                "max_load_ratio": 0.0,
                "active_experts": 0.0,
                "dead_experts": float(counts.numel()),
                "total_assignments": 0.0,
            }

        fractions = counts / total_assignments
        nonzero_fractions = fractions[fractions > 0]
        entropy = float(-(nonzero_fractions * nonzero_fractions.log()).sum().item())
        max_entropy = math.log(counts.numel()) if counts.numel() > 1 else 0.0
        normalized_entropy = MoERouterHealthCallback._safe_metric_divide(entropy, max_entropy)

        load_mean = counts.mean()
        load_std = counts.std(unbiased=False)
        load_cv = MoERouterHealthCallback._safe_metric_divide(load_std.item(), load_mean.item())
        max_load_ratio = MoERouterHealthCallback._safe_metric_divide(counts.max().item(), load_mean.item())

        active_experts = float((counts > 0).sum().item())
        dead_experts = float((counts == 0).sum().item())

        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "load_cv": load_cv,
            "max_load_ratio": max_load_ratio,
            "active_experts": active_experts,
            "dead_experts": dead_experts,
            "total_assignments": float(total_assignments.item()),
        }

    @staticmethod
    def _reduce_counts(expert_counts):
        if not dist.is_available() or not dist.is_initialized():
            return expert_counts

        reduced_counts = expert_counts.clone()
        dist.all_reduce(reduced_counts)
        return reduced_counts

    @staticmethod
    def _extract_tensor(output, index: int | None = None):
        if index is None:
            return output
        if isinstance(output, (tuple, list)) and len(output) > index:
            return output[index]
        return None

    @staticmethod
    def _extract_selected_experts(output):
        if not isinstance(output, (tuple, list)):
            return None

        for candidate in output:
            if torch.is_tensor(candidate) and candidate.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                return candidate
        return None

    @staticmethod
    def _infer_top_k(module, model) -> int:
        for candidate in (
            getattr(module, "top_k", None),
            getattr(module, "num_experts_per_tok", None),
            getattr(model.config, "num_experts_per_tok", None),
            getattr(model.config, "moe_topk", None),
            getattr(model.config, "num_experts_per_tok", None),
            1,
        ):
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        return 1

    @staticmethod
    def _compute_counts_from_routing(selected_experts, num_experts: int):
        flattened_experts = selected_experts.reshape(-1)
        valid_experts = flattened_experts[(flattened_experts >= 0) & (flattened_experts < num_experts)]
        if valid_experts.numel() == 0:
            return torch.zeros(num_experts, device=selected_experts.device, dtype=torch.float64)
        return torch.bincount(valid_experts, minlength=num_experts).to(dtype=torch.float64)

    @staticmethod
    def _get_base_model(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    @staticmethod
    def _resolve_reduction_mode(model, reduction_mode: str) -> str:
        if reduction_mode != "auto":
            return reduction_mode

        tp_size = getattr(model, "tp_size", None)
        if tp_size is None:
            tp_size = getattr(model, "_tp_size", None)

        if tp_size is not None and tp_size > 1:
            return "none"

        return "world"

    def _reset_state(self):
        self._layer_counts = {}
        self._layer_order = []
        self._last_aux_metrics = {}

    def _remove_hooks(self):
        for handle in self._router_handles:
            handle.remove()
        self._router_handles = []
        if self._model_handle is not None:
            self._model_handle.remove()
            self._model_handle = None

    def _accumulate_router_counts(self, module_name: str, output, recorder_index: int, model, module) -> None:
        selected_experts = self._extract_selected_experts(output)
        router_logits = self._extract_tensor(output, recorder_index)

        if selected_experts is None:
            if router_logits is None or not torch.is_tensor(router_logits):
                return
            top_k = min(self._infer_top_k(module=module, model=model), router_logits.shape[-1])
            selected_experts = torch.topk(router_logits.detach().float(), k=top_k, dim=-1).indices

        if not torch.is_tensor(selected_experts):
            return

        if router_logits is not None and torch.is_tensor(router_logits):
            num_experts = int(router_logits.shape[-1])
        else:
            max_selected = int(selected_experts.max().item()) if selected_experts.numel() > 0 else -1
            num_experts = max_selected + 1
        if num_experts <= 0:
            return

        counts = self._compute_counts_from_routing(selected_experts.detach(), num_experts=num_experts)
        if module_name not in self._layer_counts:
            self._layer_counts[module_name] = counts
            self._layer_order.append(module_name)
        else:
            self._layer_counts[module_name] = self._layer_counts[module_name] + counts.to(
                device=self._layer_counts[module_name].device
            )

    def _capture_model_aux_metrics(self, outputs) -> None:
        if not self.log_aux_loss or outputs is None:
            return

        metrics = {}
        for attribute_name in ("aux_loss", "router_aux_loss", "z_loss"):
            value = getattr(outputs, attribute_name, None)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "item"):
                value = value.item()
            metrics[f"{self.prefix}/{attribute_name}"] = float(value)
        self._last_aux_metrics = metrics

    def _iter_router_modules(self, model):
        capture_specs = getattr(model, "can_record_outputs", {})
        router_specs = capture_specs.get("router_logits")
        if router_specs is None:
            return
        if not isinstance(router_specs, list):
            router_specs = [router_specs]

        for spec in router_specs:
            for module_name, module in model.named_modules():
                target_class = getattr(spec, "target_class", None)
                class_name = getattr(spec, "class_name", None)
                layer_name = getattr(spec, "layer_name", None)
                matches_class = target_class is not None and isinstance(module, target_class)
                matches_name = class_name is not None and module_name.endswith(class_name)
                if not (matches_class or matches_name):
                    continue
                if layer_name is not None and layer_name not in module_name:
                    continue
                yield module_name, module, getattr(spec, "index", 0)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._remove_hooks()
        self._reset_state()
        if model is None:
            return
        model = self._get_base_model(model)
        self._resolved_reduction_mode = self._resolve_reduction_mode(model, self.reduction_mode)

        router_modules = list(self._iter_router_modules(model))
        if len(router_modules) == 0:
            logger.warning_once(
                "MoERouterHealthCallback did not find any router modules exposed through `can_record_outputs`."
            )
            return

        for module_name, module, recorder_index in router_modules:
            handle = module.register_forward_hook(
                lambda current_module,
                module_args,
                output,
                name=module_name,
                idx=recorder_index: self._accumulate_router_counts(name, output, idx, model, current_module)
            )
            self._router_handles.append(handle)

        if self.log_aux_loss:
            self._model_handle = model.register_forward_hook(
                lambda current_module, module_args, outputs: self._capture_model_aux_metrics(outputs)
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        per_layer_metrics = []
        for layer_idx, module_name in enumerate(self._layer_order):
            expert_counts = self._layer_counts.get(module_name)
            if expert_counts is None:
                continue
            if self._resolved_reduction_mode == "world":
                expert_counts = self._reduce_counts(expert_counts)
            metric_prefix = f"{self.prefix}/{self._format_layer_name(module_name, layer_idx)}"
            layer_metrics = self._compute_routing_metrics(expert_counts)
            for metric_name, metric_value in layer_metrics.items():
                logs[f"{metric_prefix}/{metric_name}"] = metric_value
            per_layer_metrics.append(layer_metrics)

        if per_layer_metrics:
            metric_names = per_layer_metrics[0].keys()
            for metric_name in metric_names:
                logs[f"{self.prefix}/global/mean_{metric_name}"] = float(
                    np.mean([layer_metrics[metric_name] for layer_metrics in per_layer_metrics])
                )

        logs.update(self._last_aux_metrics)
        self._reset_state()

    def on_train_end(self, args, state, control, **kwargs):
        self._remove_hooks()
        self._reset_state()


class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.eval_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == SaveStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
            # Save the model at the end if we have a save strategy
            if args.save_strategy == SaveStrategy.STEPS:
                control.should_save = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.eval_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if args.save_strategy == SaveStrategy.EPOCH:
            control.should_save = True

        return control


class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, max_str_len: int = 100):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        self.training_bar = None
        self.prediction_bar = None
        self.max_str_len = max_str_len

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                if isinstance(v, float):
                    # Format floats for better readability
                    shallow_logs[k] = f"{v:.4g}"
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            self.training_bar.write(str(shallow_logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            if logs is not None:
                logs = {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in logs.items()}
            print(logs)


class EarlyStoppingCallback(TrainerCallback, ExportableState):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float | None = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        if not args.load_best_model_at_end:
            logger.warning(
                "Using EarlyStoppingCallback without load_best_model_at_end=True. "
                "Once training is finished, the best model will not be loaded automatically."
            )
        assert args.metric_for_best_model is not None, (
            "EarlyStoppingCallback requires metric_for_best_model to be defined"
        )
        assert args.eval_strategy != IntervalStrategy.NO, (
            "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"
        )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

    def state(self) -> dict:
        return {
            "args": {
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_threshold": self.early_stopping_threshold,
            },
            "attributes": {
                "early_stopping_patience_counter": self.early_stopping_patience_counter,
            },
        }
