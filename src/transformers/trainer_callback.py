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
import datetime
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Union, List, Callable
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity, ProfilerAction
from tqdm.auto import tqdm

from .trainer_utils import HPSearchBackend, IntervalStrategy, SaveStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging

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
        log_history (`List[Dict[str, float]]`, *optional*):
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
        stateful_callbacks (`List[StatefulTrainerCallback]`, *optional*):
            Callbacks attached to the `Trainer` that should have their states be saved or restored.
            Relevant callbacks should implement a `state` and `from_state` function.
    """

    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    train_batch_size: Optional[int] = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    total_flos: float = 0
    log_history: list[dict[str, float]] = None
    best_metric: Optional[float] = None
    best_global_step: Optional[int] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: Optional[str] = None
    trial_params: dict[str, Union[str, float, int, bool]] = None
    stateful_callbacks: list["TrainerCallback"] = None

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
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.trial_name = trainer.hp_name(trainer._trial)
        self.trial_params = None
        if trial is not None:
            from transformers.integrations import hp_params

            assignments = trial.assignments if trainer.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.trial_params = hp_params(assignments)

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
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the data. This is deprecated in favour of `processing_class`.
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
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
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
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called before the optimizer step but after gradient clipping. Useful for monitoring gradients.
        """
        pass

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients.
        """
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass


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
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
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

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
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


class ProfilerCallback(TrainerCallback):
    """
    A callback that profiles the training process using PyTorch Profiler.
    Supports both step-level and epoch-level profiling.
    """

    def __init__(
            self,
            profile_steps=10,
            warmup_steps=3,
            log_dir=None,  # Will use args.output_dir if None
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            export_chrome_trace=True,
            profile_level="step",  # New parameter: "step" or "epoch"
            profile_epochs=1,  # New parameter: number of epochs to profile
            warmup_epochs=0  # New parameter: number of epochs to warmup
    ):
        """
        Initialize the ProfilerCallback.

        Args:
            profile_steps (int): Number of steps to profile after warmup
            warmup_steps (int): Number of warmup steps before profiling
            log_dir (str, optional): Directory to save profiler logs. If None, will use args.output_dir/profiler_logs
            activities (list): List of activities to profile
            record_shapes (bool): Whether to record tensor shapes
            profile_memory (bool): Whether to profile memory usage
            with_stack (bool): Whether to record stack traces
            with_flops (bool): Whether to attempt to record FLOPs
            export_chrome_trace (bool): Whether to export a Chrome trace file
            profile_level (str): Level of profiling - "step" or "epoch"
            profile_epochs (int): Number of epochs to profile (when profile_level="epoch")
            warmup_epochs (int): Number of epochs to warmup before profiling (when profile_level="epoch")
        """
        self.profile_steps = profile_steps
        self.warmup_steps = warmup_steps
        self.log_dir = log_dir  # Will be set in on_train_begin
        self.activities = activities
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.export_chrome_trace = export_chrome_trace

        self.profile_level = profile_level
        self.profile_epochs = profile_epochs
        self.warmup_epochs = warmup_epochs

        self.profiler = None
        self.step_count = 0
        self.epoch_count = 0
        self.is_profiling = False
        self.start_time = None

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero and not state.is_local_process_zero:
            return

        # Set up log directory if not provided
        if self.log_dir is None:
            self.log_dir = os.path.join(args.output_dir, "profiler_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        if self.profile_level == "step":
            logger.info(
                f"🔍 Step-level profiler initialized. Will profile after {self.warmup_steps} warmup steps for {self.profile_steps} steps.")

            schedule = torch.profiler.schedule(
                warmup=self.warmup_steps,
                active=self.profile_steps,
                repeat=1
            )

            self._start_profiler(schedule)
        else:
            logger.info(
                f"🔍 Epoch-level profiler initialized. Will profile after {self.warmup_epochs} warmup epochs for {self.profile_epochs} epochs.")

    def on_step_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero and not state.is_local_process_zero:
            return

        if self.is_profiling and self.profile_level == "step":
            self.step_count += 1
            if self.step_count <= self.warmup_steps:
                logger.info(f"🔥 Warming up: Step {self.step_count}/{self.warmup_steps}")
            elif self.step_count <= self.warmup_steps + self.profile_steps:
                profile_step = self.step_count - self.warmup_steps
                logger.info(f"📊 Profiling: Step {profile_step}/{self.profile_steps}")

            with record_function(f"step_{self.step_count}"):
                pass

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero and not state.is_local_process_zero:
            return

        if self.is_profiling:
            try:
                self.profiler.step()

                if self.profile_level == "step":
                    total_profile_steps = self.warmup_steps + self.profile_steps
                    if self.step_count >= total_profile_steps:
                        self.stop_profiler(f"step_level_profile", state)
            except Exception as e:
                logger.info(f"❌ Error in profiler step: {e}")
                self.stop_profiler(state=state)

    def stop_profiler(self, profile_name="profile", state=None):
        if state is not None:
            logger.info(f"Stopping profiler. Current operation: is_local_process_zero={state.is_local_process_zero}, is_world_process_zero={state.is_world_process_zero}")
            if not state.is_world_process_zero and not state.is_local_process_zero:
                return
        elif self.is_profiling and self.profiler is not None:
            logger.info("Stopping profiler without state information")
        
        if self.is_profiling and self.profiler is not None:
            duration = time.time() - self.start_time
            logger.info(f"⏱️ Profiling completed in {duration:.2f} seconds")

            try:
                self.profiler.stop()

                logger.info(f"\n===== PROFILER SUMMARY =====")
                logger.info(self.profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20))

                self.print_optimization_tips()

                logger.info(f"\n📊 Profiler logs saved to {self.log_dir}")
                logger.info(f"📈 View results with: tensorboard --logdir={self.log_dir}")
            except Exception as e:
                logger.error(f"❌ Error stopping profiler", e)
            self.is_profiling = False

    def print_optimization_tips(self):
        logger.info("\n===== OPTIMIZATION TIPS =====")
        
        logger.info("\n----- TOP CUDA OPERATIONS -----")
        cuda_table = self.profiler.key_averages().table(
            sort_by="cuda_time_total", row_limit=5)
        logger.info(cuda_table)

        logger.info("\n----- TOP CPU OPERATIONS -----")
        cpu_table = self.profiler.key_averages().table(
            sort_by="cpu_time_total", row_limit=5)
        logger.info(cpu_table)

        if self.profile_memory:
            try:
                logger.info("\n----- TOP MEMORY-CONSUMING OPERATIONS -----")
                memory_table = self.profiler.key_averages().table(
                    sort_by="self_cuda_memory_usage", row_limit=5)
                logger.info(memory_table)
                logger.info("💾 Check for memory-intensive operations in the trace")
            except:
                pass

        logger.info("\n----- OPTIMIZATION SUGGESTIONS -----")
        logger.info("⚡ Focus on optimizing the most time-consuming operations")
        logger.info("💡 Consider using torch.compile() for performance improvements")
        logger.info("🧠 Check for unnecessary CPU-GPU synchronization")
        logger.info("📏 Consider optimizing batch size for better GPU utilization")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"Train end. Current operation: is_local_process_zero={state.is_local_process_zero}, is_world_process_zero={state.is_world_process_zero}")
        if not state.is_world_process_zero and not state.is_local_process_zero:
            return

        self.stop_profiler(state=state)
        logger.info("✅ Profiling session completed")




class SimpleProfilerCallback(TrainerCallback):

    def __init__(self, profiler: str):
        if str == "simple":
            from pytorch_lightning.profilers import SimpleProfiler
            self.profiler = SimpleProfiler()
        elif str == "pytorch":
            from pytorch_lightning.profilers import PyTorchProfiler
            self.profiler = PyTorchProfiler()
        else:
            from pytorch_lightning.profilers import SimpleProfiler
            self.profiler = SimpleProfiler()

    def on_train_begin(self, args, state, control, **kwargs):
        print("set up profiler, rank:" + (os.getenv("RANK", 0)))
        self.profiler.setup(stage="training", local_rank=int(os.getenv("RANK", 0)))

    def on_epoch_begin(self, args, state, control, **kwargs):
        if args.logging_strategy == "epoch":
            print(f"start epoch: {state.epoch + 1}")
            local_rank=int(os.getenv("RANK", 0))
            action_name = f"inner_training_loop.during_training.epoch{state.epoch + 1}.rank{local_rank}.iter"
            if action_name not in self.profiler.current_actions:
                print(action_name)
                self.profiler.start(action_name)


    def on_epoch_end(self, args, state, control, **kwargs):
        if args.logging_strategy == "epoch":
            print(f"end epoch: {state.epoch + 1}")
            local_rank=int(os.getenv("RANK", 0))
            action_name = f"inner_training_loop.during_training.epoch{state.epoch + 1}.rank{local_rank}.iter"
            if action_name in self.profiler.current_actions:
                print(action_name)
                self.profiler.stop(action_name)


    def on_step_begin(self, args, state, control, **kwargs):
        print(f"start step: {state.global_step + 1}")
        local_rank=int(os.getenv("RANK", 0))
        action_name = f"inner_training_loop.during_training.step{state.global_step + 1}.rank{local_rank}.iter"
        if action_name not in self.profiler.current_actions:
            print(action_name)
            self.profiler.start(action_name)

    def on_step_end(self, args, state, control, **kwargs):
        print(f"end step: {state.global_step}")
        local_rank=int(os.getenv("RANK", 0))
        action_name = f"inner_training_loop.during_training.step{state.global_step}.rank{local_rank}.iter"
        if action_name in self.profiler.current_actions:
            print(action_name)
            self.profiler.stop(action_name)


    def on_train_end(self, args, state, control, **kwargs):
        print("summary at train end")
        print(self.profiler.summary())


