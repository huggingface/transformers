# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest
from unittest.mock import patch

from transformers import (
    DefaultFlowCallback,
    IntervalStrategy,
    PrinterCallback,
    ProgressCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_torch_available,
)
from transformers.testing_utils import require_torch


if is_torch_available():
    from transformers.trainer import DEFAULT_CALLBACKS

    from .test_trainer import RegressionDataset, RegressionModelConfig, RegressionPreTrainedModel


class MyTestTrainerCallback(TrainerCallback):
    "A callback that registers the events that goes through."

    def __init__(self):
        self.events = []

    def on_init_end(self, args, state, control, **kwargs):
        self.events.append("on_init_end")

    def on_train_begin(self, args, state, control, **kwargs):
        self.events.append("on_train_begin")

    def on_train_end(self, args, state, control, **kwargs):
        self.events.append("on_train_end")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.events.append("on_epoch_begin")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.events.append("on_epoch_end")

    def on_step_begin(self, args, state, control, **kwargs):
        self.events.append("on_step_begin")

    def on_step_end(self, args, state, control, **kwargs):
        self.events.append("on_step_end")

    def on_evaluate(self, args, state, control, **kwargs):
        self.events.append("on_evaluate")

    def on_save(self, args, state, control, **kwargs):
        self.events.append("on_save")

    def on_log(self, args, state, control, **kwargs):
        self.events.append("on_log")

    def on_prediction_step(self, args, state, control, **kwargs):
        self.events.append("on_prediction_step")


@require_torch
class TrainerCallbackTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def get_trainer(self, a=0, b=0, train_len=64, eval_len=64, callbacks=None, disable_tqdm=False, **kwargs):
        # disable_tqdm in TrainingArguments has a flaky default since it depends on the level of logging. We make sure
        # its set to False since the tests later on depend on its value.
        train_dataset = RegressionDataset(length=train_len)
        eval_dataset = RegressionDataset(length=eval_len)
        config = RegressionModelConfig(a=a, b=b)
        model = RegressionPreTrainedModel(config)

        args = TrainingArguments(self.output_dir, disable_tqdm=disable_tqdm, report_to=[], **kwargs)
        return Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

    def check_callbacks_equality(self, cbs1, cbs2):
        self.assertEqual(len(cbs1), len(cbs2))

        # Order doesn't matter
        cbs1 = list(sorted(cbs1, key=lambda cb: cb.__name__ if isinstance(cb, type) else cb.__class__.__name__))
        cbs2 = list(sorted(cbs2, key=lambda cb: cb.__name__ if isinstance(cb, type) else cb.__class__.__name__))

        for cb1, cb2 in zip(cbs1, cbs2):
            if isinstance(cb1, type) and isinstance(cb2, type):
                self.assertEqual(cb1, cb2)
            elif isinstance(cb1, type) and not isinstance(cb2, type):
                self.assertEqual(cb1, cb2.__class__)
            elif not isinstance(cb1, type) and isinstance(cb2, type):
                self.assertEqual(cb1.__class__, cb2)
            else:
                self.assertEqual(cb1, cb2)

    def get_expected_events(self, trainer):
        expected_events = ["on_init_end", "on_train_begin"]
        step = 0
        train_dl_len = len(trainer.get_eval_dataloader())
        evaluation_events = ["on_prediction_step"] * len(trainer.get_eval_dataloader()) + ["on_log", "on_evaluate"]
        for _ in range(trainer.state.num_train_epochs):
            expected_events.append("on_epoch_begin")
            for _ in range(train_dl_len):
                step += 1
                expected_events += ["on_step_begin", "on_step_end"]
                if step % trainer.args.logging_steps == 0:
                    expected_events.append("on_log")
                if trainer.args.evaluation_strategy == IntervalStrategy.STEPS and step % trainer.args.eval_steps == 0:
                    expected_events += evaluation_events.copy()
                if step % trainer.args.save_steps == 0:
                    expected_events.append("on_save")
            expected_events.append("on_epoch_end")
            if trainer.args.evaluation_strategy == IntervalStrategy.EPOCH:
                expected_events += evaluation_events.copy()
        expected_events += ["on_log", "on_train_end"]
        return expected_events

    def test_init_callback(self):
        trainer = self.get_trainer()
        expected_callbacks = DEFAULT_CALLBACKS.copy() + [ProgressCallback]
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

        # Callbacks passed at init are added to the default callbacks
        trainer = self.get_trainer(callbacks=[MyTestTrainerCallback])
        expected_callbacks.append(MyTestTrainerCallback)
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

        # TrainingArguments.disable_tqdm controls if use ProgressCallback or PrinterCallback
        trainer = self.get_trainer(disable_tqdm=True)
        expected_callbacks = DEFAULT_CALLBACKS.copy() + [PrinterCallback]
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

    def test_add_remove_callback(self):
        expected_callbacks = DEFAULT_CALLBACKS.copy() + [ProgressCallback]
        trainer = self.get_trainer()

        # We can add, pop, or remove by class name
        trainer.remove_callback(DefaultFlowCallback)
        expected_callbacks.remove(DefaultFlowCallback)
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

        trainer = self.get_trainer()
        cb = trainer.pop_callback(DefaultFlowCallback)
        self.assertEqual(cb.__class__, DefaultFlowCallback)
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

        trainer.add_callback(DefaultFlowCallback)
        expected_callbacks.insert(0, DefaultFlowCallback)
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

        # We can also add, pop, or remove by instance
        trainer = self.get_trainer()
        cb = trainer.callback_handler.callbacks[0]
        trainer.remove_callback(cb)
        expected_callbacks.remove(DefaultFlowCallback)
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

        trainer = self.get_trainer()
        cb1 = trainer.callback_handler.callbacks[0]
        cb2 = trainer.pop_callback(cb1)
        self.assertEqual(cb1, cb2)
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

        trainer.add_callback(cb1)
        expected_callbacks.insert(0, DefaultFlowCallback)
        self.check_callbacks_equality(trainer.callback_handler.callbacks, expected_callbacks)

    def test_event_flow(self):
        import warnings

        # XXX: for now ignore scatter_gather warnings in this test since it's not relevant to what's being tested
        warnings.simplefilter(action="ignore", category=UserWarning)

        trainer = self.get_trainer(callbacks=[MyTestTrainerCallback])
        trainer.train()
        events = trainer.callback_handler.callbacks[-2].events
        self.assertEqual(events, self.get_expected_events(trainer))

        # Independent log/save/eval
        trainer = self.get_trainer(callbacks=[MyTestTrainerCallback], logging_steps=5)
        trainer.train()
        events = trainer.callback_handler.callbacks[-2].events
        self.assertEqual(events, self.get_expected_events(trainer))

        trainer = self.get_trainer(callbacks=[MyTestTrainerCallback], save_steps=5)
        trainer.train()
        events = trainer.callback_handler.callbacks[-2].events
        self.assertEqual(events, self.get_expected_events(trainer))

        trainer = self.get_trainer(callbacks=[MyTestTrainerCallback], eval_steps=5, evaluation_strategy="steps")
        trainer.train()
        events = trainer.callback_handler.callbacks[-2].events
        self.assertEqual(events, self.get_expected_events(trainer))

        trainer = self.get_trainer(callbacks=[MyTestTrainerCallback], evaluation_strategy="epoch")
        trainer.train()
        events = trainer.callback_handler.callbacks[-2].events
        self.assertEqual(events, self.get_expected_events(trainer))

        # A bit of everything
        trainer = self.get_trainer(
            callbacks=[MyTestTrainerCallback],
            logging_steps=3,
            save_steps=10,
            eval_steps=5,
            evaluation_strategy="steps",
        )
        trainer.train()
        events = trainer.callback_handler.callbacks[-2].events
        self.assertEqual(events, self.get_expected_events(trainer))

        # warning should be emitted for duplicated callbacks
        with patch("transformers.trainer_callback.logger.warning") as warn_mock:
            trainer = self.get_trainer(
                callbacks=[MyTestTrainerCallback, MyTestTrainerCallback],
            )
            assert str(MyTestTrainerCallback) in warn_mock.call_args[0][0]
