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

"""
Tests for trainer callbacks.

This module tests:
- Callback registration (add, remove, pop)
- Event firing order during training
- Stateful callback persistence across checkpoints
- TrainerState and TrainerControl behavior
- Built-in callbacks (DefaultFlowCallback, EarlyStoppingCallback, etc.)
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers import (
    DefaultFlowCallback,
    EarlyStoppingCallback,
    IntervalStrategy,
    PrinterCallback,
    ProgressCallback,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainingArguments,
    is_torch_available,
)
from transformers.integrations.integration_utils import SwanLabCallback
from transformers.testing_utils import require_torch
from transformers.trainer_callback import CallbackHandler, ExportableState, TrainerControl


if is_torch_available():
    from transformers.trainer import DEFAULT_CALLBACKS, TRAINER_STATE_NAME

    from .test_trainer import RegressionDataset, RegressionModelConfig, RegressionPreTrainedModel


# =============================================================================
# Test Callback Implementations
# =============================================================================


class EventRecorderCallback(TrainerCallback):
    """
    A callback that records all events it receives.

    Used to verify that callbacks are called at the right times
    and in the right order during training.
    """

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

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.events.append("on_pre_optimizer_step")

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.events.append("on_optimizer_step")

    def on_substep_end(self, args, state, control, **kwargs):
        self.events.append("on_substep_end")

    def on_step_end(self, args, state, control, **kwargs):
        self.events.append("on_step_end")

    def on_evaluate(self, args, state, control, **kwargs):
        self.events.append("on_evaluate")

    def on_predict(self, args, state, control, **kwargs):
        self.events.append("on_predict")

    def on_save(self, args, state, control, **kwargs):
        self.events.append("on_save")

    def on_log(self, args, state, control, **kwargs):
        self.events.append("on_log")

    def on_prediction_step(self, args, state, control, **kwargs):
        self.events.append("on_prediction_step")

    def on_push_begin(self, args, state, control, **kwargs):
        self.events.append("on_push_begin")


class StatefulTestCallback(TrainerCallback, ExportableState):
    """
    A stateful callback that can save and restore its state.

    Used to test checkpoint persistence of callback state.
    """

    def __init__(self, my_value="default"):
        self.my_value = my_value

    def state(self):
        return {
            "args": {"my_value": self.my_value},
            "attributes": {},
        }


class StopTrainingCallback(TrainerCallback):
    """A callback that stops training after a specified number of steps."""

    def __init__(self, stop_after_steps=1):
        self.stop_after_steps = stop_after_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_steps:
            control.should_training_stop = True
        return control


class ModifyControlCallback(TrainerCallback):
    """A callback that modifies control flags to test control flow."""

    def __init__(self):
        self.control_modifications = []

    def on_step_end(self, args, state, control, **kwargs):
        self.control_modifications.append(
            {
                "step": state.global_step,
                "should_log": control.should_log,
                "should_save": control.should_save,
                "should_evaluate": control.should_evaluate,
            }
        )
        return control


# =============================================================================
# Helper Functions
# =============================================================================


def get_callback_names(callbacks):
    """Extract callback class names from a list of callbacks (classes or instances)."""
    names = []
    for cb in callbacks:
        if isinstance(cb, type):
            names.append(cb.__name__)
        else:
            names.append(cb.__class__.__name__)
    return sorted(names)


# =============================================================================
# Test Classes
# =============================================================================


@require_torch
class TrainerCallbackTest(unittest.TestCase):
    """Tests for callback registration and lifecycle with Trainer."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def _create_trainer(self, callbacks=None, **kwargs):
        """
        Create a Trainer instance with a simple regression model.

        Args:
            callbacks: List of callbacks to add to the trainer.
            **kwargs: Additional arguments passed to TrainingArguments.

        Returns:
            A configured Trainer instance.
        """
        train_dataset = RegressionDataset(length=64)
        eval_dataset = RegressionDataset(length=64)
        config = RegressionModelConfig(a=0, b=0)
        model = RegressionPreTrainedModel(config)

        # disable_tqdm must be explicit since it depends on logging level
        kwargs.setdefault("disable_tqdm", False)
        kwargs.setdefault("report_to", [])

        args = TrainingArguments(self.output_dir, **kwargs)
        return Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

    def _get_callback(self, trainer, callback_class):
        """Get a callback instance from the trainer by class type."""
        for cb in trainer.callback_handler.callbacks:
            if isinstance(cb, callback_class):
                return cb
        return None

    # -------------------------------------------------------------------------
    # Callback Registration Tests
    # -------------------------------------------------------------------------

    def test_default_callbacks_are_present(self):
        """Trainer should have default callbacks plus ProgressCallback."""
        trainer = self._create_trainer()

        expected = get_callback_names(DEFAULT_CALLBACKS + [ProgressCallback])
        actual = get_callback_names(trainer.callback_handler.callbacks)

        self.assertEqual(actual, expected)

    def test_custom_callback_added_at_init(self):
        """Custom callbacks passed at init should be added to defaults."""
        trainer = self._create_trainer(callbacks=[EventRecorderCallback])

        expected = get_callback_names(DEFAULT_CALLBACKS + [ProgressCallback, EventRecorderCallback])
        actual = get_callback_names(trainer.callback_handler.callbacks)

        self.assertEqual(actual, expected)

    def test_printer_callback_when_tqdm_disabled(self):
        """PrinterCallback should replace ProgressCallback when tqdm is disabled."""
        trainer = self._create_trainer(disable_tqdm=True)

        expected = get_callback_names(DEFAULT_CALLBACKS + [PrinterCallback])
        actual = get_callback_names(trainer.callback_handler.callbacks)

        self.assertEqual(actual, expected)

    def test_add_callback_by_class(self):
        """Adding a callback by class should instantiate and add it."""
        trainer = self._create_trainer()
        initial_count = len(trainer.callback_handler.callbacks)

        trainer.add_callback(EventRecorderCallback)

        self.assertEqual(len(trainer.callback_handler.callbacks), initial_count + 1)
        self.assertIsNotNone(self._get_callback(trainer, EventRecorderCallback))

    def test_add_callback_by_instance(self):
        """Adding a callback instance should add that exact instance."""
        trainer = self._create_trainer()
        callback = EventRecorderCallback()

        trainer.add_callback(callback)

        self.assertIn(callback, trainer.callback_handler.callbacks)

    def test_remove_callback_by_class(self):
        """Removing by class should remove the first matching callback."""
        trainer = self._create_trainer()
        self.assertIsNotNone(self._get_callback(trainer, DefaultFlowCallback))

        trainer.remove_callback(DefaultFlowCallback)

        self.assertIsNone(self._get_callback(trainer, DefaultFlowCallback))

    def test_remove_callback_by_instance(self):
        """Removing by instance should remove that exact callback."""
        trainer = self._create_trainer()
        callback = trainer.callback_handler.callbacks[0]

        trainer.remove_callback(callback)

        self.assertNotIn(callback, trainer.callback_handler.callbacks)

    def test_pop_callback_returns_instance(self):
        """Pop should remove and return the callback instance."""
        trainer = self._create_trainer()
        original_callback = self._get_callback(trainer, DefaultFlowCallback)

        popped = trainer.pop_callback(DefaultFlowCallback)

        self.assertEqual(popped, original_callback)
        self.assertIsNone(self._get_callback(trainer, DefaultFlowCallback))

    def test_duplicate_callback_warning(self):
        """Adding a duplicate callback class should emit a warning."""
        with patch("transformers.trainer_callback.logger.warning") as warn_mock:
            self._create_trainer(callbacks=[EventRecorderCallback, EventRecorderCallback])

            self.assertTrue(warn_mock.called)
            self.assertIn("EventRecorderCallback", warn_mock.call_args[0][0])

    # -------------------------------------------------------------------------
    # Event Flow Tests
    # -------------------------------------------------------------------------

    def _get_expected_events(self, trainer):
        """Compute the exact expected event sequence for a training run."""
        expected_events = ["on_init_end", "on_train_begin"]
        step = 0
        train_dl_len = len(trainer.get_eval_dataloader())
        evaluation_events = ["on_prediction_step"] * len(trainer.get_eval_dataloader()) + ["on_log", "on_evaluate"]
        for _ in range(trainer.state.num_train_epochs):
            expected_events.append("on_epoch_begin")
            for _ in range(train_dl_len):
                step += 1
                expected_events += ["on_step_begin", "on_pre_optimizer_step", "on_optimizer_step", "on_step_end"]
                if step % trainer.args.logging_steps == 0:
                    expected_events.append("on_log")
                if trainer.args.eval_strategy == IntervalStrategy.STEPS and step % trainer.args.eval_steps == 0:
                    expected_events += evaluation_events.copy()
                if step % trainer.args.save_steps == 0 or step == trainer.state.max_steps:
                    expected_events.append("on_save")
            expected_events.append("on_epoch_end")
            if trainer.args.eval_strategy == IntervalStrategy.EPOCH:
                expected_events += evaluation_events.copy()
        expected_events += ["on_log", "on_train_end"]
        return expected_events

    def test_event_flow(self):
        """Test exact event sequence across multiple training configurations."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)

            # Default configuration
            trainer = self._create_trainer(callbacks=[EventRecorderCallback])
            trainer.train()
            events = self._get_callback(trainer, EventRecorderCallback).events
            self.assertEqual(events, self._get_expected_events(trainer))

            # Independent log/save/eval steps
            trainer = self._create_trainer(callbacks=[EventRecorderCallback], logging_steps=5)
            trainer.train()
            events = self._get_callback(trainer, EventRecorderCallback).events
            self.assertEqual(events, self._get_expected_events(trainer))

            trainer = self._create_trainer(callbacks=[EventRecorderCallback], save_steps=5)
            trainer.train()
            events = self._get_callback(trainer, EventRecorderCallback).events
            self.assertEqual(events, self._get_expected_events(trainer))

            trainer = self._create_trainer(callbacks=[EventRecorderCallback], eval_steps=5, eval_strategy="steps")
            trainer.train()
            events = self._get_callback(trainer, EventRecorderCallback).events
            self.assertEqual(events, self._get_expected_events(trainer))

            trainer = self._create_trainer(callbacks=[EventRecorderCallback], eval_strategy="epoch")
            trainer.train()
            events = self._get_callback(trainer, EventRecorderCallback).events
            self.assertEqual(events, self._get_expected_events(trainer))

            # A bit of everything
            trainer = self._create_trainer(
                callbacks=[EventRecorderCallback],
                logging_steps=3,
                save_steps=10,
                eval_steps=5,
                eval_strategy="steps",
            )
            trainer.train()
            events = self._get_callback(trainer, EventRecorderCallback).events
            self.assertEqual(events, self._get_expected_events(trainer))

    def test_on_push_begin_event(self):
        """on_push_begin should be callable and fire correctly."""
        trainer = self._create_trainer(callbacks=[EventRecorderCallback], max_steps=1)
        trainer.train()

        callback = self._get_callback(trainer, EventRecorderCallback)
        initial_count = len(callback.events)

        # Manually trigger push_begin event
        trainer.callback_handler.on_push_begin(trainer.args, trainer.state, trainer.control)

        self.assertIn("on_push_begin", callback.events)
        self.assertEqual(callback.events.count("on_push_begin"), 1)
        self.assertEqual(len(callback.events), initial_count + 1)

    def test_no_duplicate_save_on_epoch_strategy(self):
        """Save should only happen once per epoch with epoch strategy."""
        save_count = 0

        class SaveCounterCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                nonlocal save_count
                if control.should_save:
                    save_count += 1

            def on_epoch_end(self, args, state, control, **kwargs):
                nonlocal save_count
                if control.should_save:
                    save_count += 1

        trainer = self._create_trainer(
            callbacks=[SaveCounterCallback()],
            max_steps=2,
            save_strategy="epoch",
        )
        trainer.train()

        self.assertEqual(save_count, 1)

    # -------------------------------------------------------------------------
    # Control Flow Tests
    # -------------------------------------------------------------------------

    def test_callback_can_stop_training(self):
        """A callback should be able to stop training by setting control flag."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)

            trainer = self._create_trainer(
                callbacks=[StopTrainingCallback(stop_after_steps=1)],
                max_steps=10,
                logging_steps=1,
                save_strategy="no",
            )
            trainer.train()

        # Training should have stopped after 1 step, not 10
        self.assertEqual(trainer.state.global_step, 1)

    def test_callback_receives_control_flags(self):
        """Callbacks should receive current control flags."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)

            callback = ModifyControlCallback()
            trainer = self._create_trainer(
                callbacks=[callback],
                max_steps=2,
                logging_steps=1,
                save_strategy="no",
            )
            trainer.train()

        # Should have recorded control state for each step
        self.assertEqual(len(callback.control_modifications), 2)


@require_torch
class StatefulCallbackTest(unittest.TestCase):
    """Tests for stateful callback persistence across checkpoints."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def _create_trainer(self, callbacks=None, **kwargs):
        """Create a Trainer for stateful callback tests."""
        train_dataset = RegressionDataset(length=64)
        eval_dataset = RegressionDataset(length=64)
        config = RegressionModelConfig(a=0, b=0)
        model = RegressionPreTrainedModel(config)

        kwargs.setdefault("disable_tqdm", False)
        kwargs.setdefault("report_to", [])

        args = TrainingArguments(self.output_dir, **kwargs)
        return Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

    def test_early_stopping_state_persists(self):
        """EarlyStoppingCallback state should persist across checkpoint resume."""
        # First training run with custom patience
        cb = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.2)
        trainer = self._create_trainer(
            callbacks=[cb],
            load_best_model_at_end=True,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=2,
            eval_steps=2,
            max_steps=2,
        )
        trainer.train()

        # Resume with default callback - should load saved state
        trainer = self._create_trainer(
            callbacks=[EarlyStoppingCallback()],
            load_best_model_at_end=True,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=2,
            eval_steps=2,
            max_steps=2,
            restore_callback_states_from_checkpoint=True,
        )
        checkpoint = os.path.join(self.output_dir, "checkpoint-2")
        trainer.train(resume_from_checkpoint=checkpoint)

        # Find the callback and verify state was restored
        restored_cb = None
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                restored_cb = callback
                break

        self.assertIsNotNone(restored_cb)
        self.assertEqual(restored_cb.early_stopping_patience, 5)
        self.assertEqual(restored_cb.early_stopping_threshold, 0.2)

    def test_mixed_stateful_and_regular_callbacks(self):
        """Stateful and regular callbacks should coexist correctly."""
        cbs = [
            EventRecorderCallback(),
            EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.2),
        ]
        trainer = self._create_trainer(
            callbacks=cbs,
            load_best_model_at_end=True,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=2,
            eval_steps=2,
            max_steps=2,
        )
        trainer.train()

        # Resume with fresh callbacks
        trainer = self._create_trainer(
            callbacks=[EarlyStoppingCallback(), EventRecorderCallback()],
            load_best_model_at_end=True,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=2,
            eval_steps=2,
            max_steps=2,
            restore_callback_states_from_checkpoint=True,
        )
        checkpoint = os.path.join(self.output_dir, "checkpoint-2")
        trainer.train(resume_from_checkpoint=checkpoint)

        # Stateful callback should be restored
        early_stopping = None
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                early_stopping = callback
                break

        self.assertEqual(early_stopping.early_stopping_patience, 5)

    def test_multiple_instances_of_same_stateful_callback(self):
        """Multiple instances of the same stateful callback should each persist."""
        cbs = [StatefulTestCallback("first"), StatefulTestCallback("second")]
        trainer = self._create_trainer(
            callbacks=cbs,
            load_best_model_at_end=True,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=2,
            eval_steps=2,
            max_steps=2,
        )
        trainer.train()

        # Resume with default values
        trainer = self._create_trainer(
            callbacks=[StatefulTestCallback(), StatefulTestCallback()],
            load_best_model_at_end=True,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=2,
            eval_steps=2,
            max_steps=2,
            restore_callback_states_from_checkpoint=True,
        )
        checkpoint = os.path.join(self.output_dir, "checkpoint-2")
        trainer.train(resume_from_checkpoint=checkpoint)

        restored = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, StatefulTestCallback)]

        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].my_value, "first")
        self.assertEqual(restored[1].my_value, "second")

    def test_missing_stateful_callback_warning(self):
        """Warning should be emitted when a stateful callback is missing on resume."""
        cb = EarlyStoppingCallback()
        trainer = self._create_trainer(
            callbacks=[cb],
            load_best_model_at_end=True,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=2,
            eval_steps=2,
            max_steps=2,
        )
        trainer.train()

        # Resume WITHOUT the EarlyStoppingCallback
        trainer = self._create_trainer(
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=2,
            eval_steps=2,
            max_steps=2,
            restore_callback_states_from_checkpoint=True,
        )
        checkpoint = os.path.join(self.output_dir, "checkpoint-2")

        with patch("transformers.trainer.logger.warning") as warn_mock:
            trainer.train(resume_from_checkpoint=checkpoint)

            self.assertTrue(warn_mock.called)
            self.assertIn("EarlyStoppingCallback", warn_mock.call_args[0][0])

    def test_trainer_control_state_persists(self):
        """TrainerControl state should persist across checkpoint resume."""
        trainer = self._create_trainer(
            max_steps=2,
            save_strategy="steps",
            save_steps=2,
        )
        trainer.train()

        # Load state and verify
        trainer = self._create_trainer(max_steps=2, restore_callback_states_from_checkpoint=True)
        checkpoint = os.path.join(self.output_dir, "checkpoint-2")
        trainer.state = TrainerState.load_from_json(os.path.join(checkpoint, TRAINER_STATE_NAME))
        trainer._load_callback_state()

        self.assertTrue(trainer.control.should_training_stop)


class TrainerStateTest(unittest.TestCase):
    """Tests for TrainerState functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_json(self):
        """TrainerState should serialize and deserialize to JSON."""
        state = TrainerState(
            epoch=1.5,
            global_step=100,
            max_steps=200,
            best_metric=0.95,
        )
        json_path = os.path.join(self.temp_dir, "state.json")

        state.save_to_json(json_path)
        loaded = TrainerState.load_from_json(json_path)

        self.assertEqual(loaded.epoch, 1.5)
        self.assertEqual(loaded.global_step, 100)
        self.assertEqual(loaded.max_steps, 200)
        self.assertEqual(loaded.best_metric, 0.95)

    def test_log_history_initialized(self):
        """log_history should be initialized as empty list."""
        state = TrainerState()

        self.assertEqual(state.log_history, [])

    def test_stateful_callbacks_initialized(self):
        """stateful_callbacks should be initialized as empty dict."""
        state = TrainerState()

        self.assertEqual(state.stateful_callbacks, {})

    def test_compute_steps_from_proportion(self):
        """compute_steps should convert proportions to absolute values."""
        state = TrainerState()

        class MockArgs:
            logging_steps = 0.1  # 10% of max_steps
            eval_steps = 0.2  # 20% of max_steps
            save_steps = 0.5  # 50% of max_steps

        state.compute_steps(MockArgs(), max_steps=100)

        self.assertEqual(state.logging_steps, 10)
        self.assertEqual(state.eval_steps, 20)
        self.assertEqual(state.save_steps, 50)

    def test_compute_steps_from_integers(self):
        """compute_steps should preserve integer values."""
        state = TrainerState()

        class MockArgs:
            logging_steps = 10
            eval_steps = 20
            save_steps = 50

        state.compute_steps(MockArgs(), max_steps=100)

        self.assertEqual(state.logging_steps, 10)
        self.assertEqual(state.eval_steps, 20)
        self.assertEqual(state.save_steps, 50)


class SwanLabCallbackTest(unittest.TestCase):
    def _create_callback(self, fake_swanlab):
        with patch("transformers.integrations.integration_utils.is_swanlab_available", return_value=True):
            with patch.dict("sys.modules", {"swanlab": fake_swanlab}):
                callback = SwanLabCallback()
        return callback

    @staticmethod
    def _create_args():
        class SwanLabArgs:
            run_name = "swanlab-run"
            resume_from_checkpoint = False

            @staticmethod
            def to_dict():
                return {}

        return SwanLabArgs()

    @staticmethod
    def _create_state():
        return SimpleNamespace(is_world_process_zero=True, trial_name=None)

    @staticmethod
    def _create_model():
        class DummyConfig:
            @staticmethod
            def to_dict():
                return {}

        class DummyModel:
            config = DummyConfig()
            peft_config = None

            @staticmethod
            def num_parameters():
                return 1

        return DummyModel()

    def test_setup_does_not_forward_id_or_resume_by_default(self):
        fake_swanlab = Mock()
        fake_swanlab.get_run.return_value = None
        fake_swanlab.config = {}
        callback = self._create_callback(fake_swanlab)

        with patch.dict(os.environ, {}, clear=True):
            callback.setup(self._create_args(), self._create_state(), self._create_model())

        init_kwargs = fake_swanlab.init.call_args.kwargs
        self.assertNotIn("id", init_kwargs)
        self.assertNotIn("resume", init_kwargs)

    def test_setup_forwards_id_and_resume_from_env(self):
        fake_swanlab = Mock()
        fake_swanlab.get_run.return_value = None
        fake_swanlab.config = {}
        callback = self._create_callback(fake_swanlab)

        with patch.dict(os.environ, {"SWANLAB_RUN_ID": "run-123", "SWANLAB_RESUME": "must"}, clear=True):
            callback.setup(self._create_args(), self._create_state(), self._create_model())

        init_kwargs = fake_swanlab.init.call_args.kwargs
        self.assertEqual(init_kwargs["id"], "run-123")
        self.assertEqual(init_kwargs["resume"], "must")


class TrainerControlTest(unittest.TestCase):
    """Tests for TrainerControl functionality."""

    def test_default_values(self):
        """TrainerControl should have all flags False by default."""
        control = TrainerControl()

        self.assertFalse(control.should_training_stop)
        self.assertFalse(control.should_epoch_stop)
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_evaluate)
        self.assertFalse(control.should_log)

    def test_new_training_resets_stop_flag(self):
        """_new_training should reset should_training_stop."""
        control = TrainerControl(should_training_stop=True)

        control._new_training()

        self.assertFalse(control.should_training_stop)

    def test_new_epoch_resets_epoch_stop_flag(self):
        """_new_epoch should reset should_epoch_stop."""
        control = TrainerControl(should_epoch_stop=True)

        control._new_epoch()

        self.assertFalse(control.should_epoch_stop)

    def test_new_step_resets_step_flags(self):
        """_new_step should reset save, evaluate, and log flags."""
        control = TrainerControl(
            should_save=True,
            should_evaluate=True,
            should_log=True,
        )

        control._new_step()

        self.assertFalse(control.should_save)
        self.assertFalse(control.should_evaluate)
        self.assertFalse(control.should_log)

    def test_state_export(self):
        """state() should return all control flags."""
        control = TrainerControl(
            should_training_stop=True,
            should_save=True,
        )

        state = control.state()

        self.assertEqual(state["args"]["should_training_stop"], True)
        self.assertEqual(state["args"]["should_save"], True)
        self.assertEqual(state["attributes"], {})


class CallbackHandlerTest(unittest.TestCase):
    """Tests for CallbackHandler functionality."""

    def test_callback_list_property(self):
        """callback_list should return newline-separated callback names."""
        handler = CallbackHandler(
            callbacks=[DefaultFlowCallback(), ProgressCallback()],
            model=None,
            processing_class=None,
            optimizer=None,
            lr_scheduler=None,
        )

        callback_list = handler.callback_list

        self.assertIn("DefaultFlowCallback", callback_list)
        self.assertIn("ProgressCallback", callback_list)

    def test_warning_without_default_flow_callback(self):
        """Warning should be emitted if DefaultFlowCallback is missing."""
        with patch("transformers.trainer_callback.logger.warning") as warn_mock:
            CallbackHandler(
                callbacks=[ProgressCallback()],
                model=None,
                processing_class=None,
                optimizer=None,
                lr_scheduler=None,
            )

            self.assertTrue(warn_mock.called)
            self.assertIn("DefaultFlowCallback", warn_mock.call_args[0][0])

    def test_pop_callback_returns_none_if_not_found(self):
        """pop_callback should return None if callback not found."""
        handler = CallbackHandler(
            callbacks=[DefaultFlowCallback()],
            model=None,
            processing_class=None,
            optimizer=None,
            lr_scheduler=None,
        )

        result = handler.pop_callback(ProgressCallback)

        self.assertIsNone(result)

    def test_call_event_passes_kwargs(self):
        """call_event should pass kwargs to all callbacks."""
        received_kwargs = {}

        class KwargsRecorderCallback(TrainerCallback):
            def on_log(self, args, state, control, **kwargs):
                received_kwargs.update(kwargs)

        handler = CallbackHandler(
            callbacks=[DefaultFlowCallback(), KwargsRecorderCallback()],
            model="test_model",
            processing_class="test_processor",
            optimizer="test_optimizer",
            lr_scheduler="test_scheduler",
        )
        handler.train_dataloader = "test_train_dl"
        handler.eval_dataloader = "test_eval_dl"

        control = TrainerControl()
        handler.call_event("on_log", None, TrainerState(), control, logs={"loss": 1.0})

        self.assertEqual(received_kwargs["model"], "test_model")
        self.assertEqual(received_kwargs["processing_class"], "test_processor")
        self.assertEqual(received_kwargs["logs"], {"loss": 1.0})


class EarlyStoppingCallbackTest(unittest.TestCase):
    """Tests for EarlyStoppingCallback logic."""

    def test_patience_counter_increments_when_metric_does_not_improve(self):
        """Patience counter should increment when metric doesn't improve."""
        callback = EarlyStoppingCallback(early_stopping_patience=3)
        state = TrainerState(best_metric=0.9)
        control = TrainerControl()

        class MockArgs:
            greater_is_better = True

        # Metric is worse (0.8 < 0.9), counter should increment
        callback.check_metric_value(MockArgs(), state, control, 0.8)

        self.assertEqual(callback.early_stopping_patience_counter, 1)

    def test_patience_counter_resets_when_metric_improves(self):
        """Patience counter should reset when metric improves."""
        callback = EarlyStoppingCallback(early_stopping_patience=3)
        callback.early_stopping_patience_counter = 2
        state = TrainerState(best_metric=0.8)
        control = TrainerControl()

        class MockArgs:
            greater_is_better = True

        # Metric is better (0.95 > 0.8), counter should reset
        callback.check_metric_value(MockArgs(), state, control, 0.95)

        self.assertEqual(callback.early_stopping_patience_counter, 0)

    def test_threshold_prevents_small_improvements(self):
        """Small improvements within threshold should not reset counter."""
        callback = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.1,
        )
        state = TrainerState(best_metric=0.8)
        control = TrainerControl()

        class MockArgs:
            greater_is_better = True

        # Improvement of 0.05 is less than threshold of 0.1
        callback.check_metric_value(MockArgs(), state, control, 0.85)

        self.assertEqual(callback.early_stopping_patience_counter, 1)

    def test_state_includes_all_attributes(self):
        """state() should include patience, threshold, and counter."""
        callback = EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.1,
        )
        callback.early_stopping_patience_counter = 3

        state = callback.state()

        self.assertEqual(state["args"]["early_stopping_patience"], 5)
        self.assertEqual(state["args"]["early_stopping_threshold"], 0.1)
        self.assertEqual(state["attributes"]["early_stopping_patience_counter"], 3)


class ExportableStateTest(unittest.TestCase):
    """Tests for ExportableState interface."""

    def test_from_state_creates_instance(self):
        """from_state should create instance with correct args and attributes."""
        state = {
            "args": {"my_value": "restored"},
            "attributes": {},
        }

        instance = StatefulTestCallback.from_state(state)

        self.assertEqual(instance.my_value, "restored")

    def test_from_state_sets_attributes(self):
        """from_state should set attributes from state dict."""

        class CallbackWithAttributes(TrainerCallback, ExportableState):
            def __init__(self, name="default"):
                self.name = name
                self.counter = 0

            def state(self):
                return {
                    "args": {"name": self.name},
                    "attributes": {"counter": self.counter},
                }

        state = {
            "args": {"name": "test"},
            "attributes": {"counter": 5},
        }

        instance = CallbackWithAttributes.from_state(state)

        self.assertEqual(instance.name, "test")
        self.assertEqual(instance.counter, 5)
