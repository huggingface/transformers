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

import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .trainer_utils import EvaluationStrategy
from .training_args import TrainingArguments


@dataclass
class TrainerState:
    """
    A class containing the `Trainer` inner state that will be saved along the model and optimizer.

    .. note::

        In all this class, one step is to be understood as one update step. When using gradient accumulation, one
        update step may require several forward and backward passes: if you use :obj:`gradient_accumulation_steps=n`,
        then one update step requires going throuch `n` batches.

    Args:
        epoch (:obj:`float`, `optional`):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (:obj:`int`, `optional`, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (:obj:`int`, `optional`, defaults to 0):
            The number of update steps to do during the current training.
        total_flos (:obj:`int`, `optional`, defaults to 0):
            The total number of floating operations done by the model since the beginning of training.
        log_history (:obj:`List[Dict[str, float]]`, `optional`):
            The list of logs done since the beginning of training.
        best_metric (:obj:`float`, `optional`):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (:obj:`str`, `optional`):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not this process is the global main process (when training in a distributed fashion on
            several machines, this is only going to be :obj:`True` for one process).
    """

    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    total_flos: int = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_world_process_zero: bool = True

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path: str):
        """ Save the content of this instance in JSON format inside :obj:`json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """ Create an instance from the content of :obj:`json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


@dataclass
class TrainerControl:
    """
    A class that handles the :class:`~transformers.Trainer` controlflow.

    Args:
        should_training_stop (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the training should be interrupted.

            If :obj:`True`, this variable will not be set back to :obj:`False`. The training will just stop.
        should_epoch_stop (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the current epoch should be interrupted.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next epoch.
        should_save (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should be saved at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
        should_evaluate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should be evaluated at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
        should_log (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the logs should be reported at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """ Internal method that resets the variable for a new training. """
        self.should_training_stop = False

    def _new_epoch(self):
        """ Internal method that resets the variable for a new epoch. """
        self.should_epoch_stop = False

    def _new_step(self):
        """ Internal method that resets the variable for a new step. """
        self.should_save_model = False
        self.should_evaluate = False
        self.should_log = False


class TrainerCallback:
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control


class CallbackHandler(TrainerCallback):
    """ Internal class that just calls the list of callbacks in order. """

    def __init__(self, callbacks, model, optimizer, lr_scheduler):
        self.callbacks = [cb() if isinstance(cb, type) else cb for cb in callbacks]
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

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

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            control = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                **kwargs,
            )
        return control


class DefaultFlowCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_steps > 0 and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == EvaluationStrategy.STEPS and state.global_step % args.eval_steps == 0:
            control.should_evaluate = True
            if args.load_best_model_at_end:
                control.should_save = True

        # Save
        if not args.load_best_model_at_end and args.save_steps > 0 and state.global_step % args.save_steps == 0:
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.evaluation_strategy == EvaluationStrategy.EPOCH:
            control.should_evaluate = True
            if args.load_best_model_at_end:
                control.should_save = True
        return control
