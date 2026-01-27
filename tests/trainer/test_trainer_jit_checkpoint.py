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

import os
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from transformers import TrainingArguments, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    from transformers.trainer_jit_checkpoint import CheckpointManager, JITCheckpointCallback

    from .test_trainer import RegressionDataset, RegressionModelConfig, RegressionPreTrainedModel


@require_torch
class JITCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def get_trainer(self, enable_jit=True):
        """Helper method to create a trainer with JIT checkpointing enabled."""
        from transformers import Trainer

        model_config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(model_config)

        args = TrainingArguments(
            output_dir=self.test_dir,
            enable_jit_checkpoint=enable_jit,
            per_device_train_batch_size=16,
            learning_rate=0.1,
            logging_steps=1,
            num_train_epochs=1,
            max_steps=10,
            save_steps=10,
        )

        train_dataset = RegressionDataset(length=64)

        return Trainer(model=model, args=args, train_dataset=train_dataset)

    def test_checkpoint_manager_initialization(self):
        """Test CheckpointManager initialization with different configurations."""
        trainer = self.get_trainer()

        # Test with default parameters
        manager = CheckpointManager(trainer)
        self.assertEqual(manager.trainer, trainer)
        self.assertEqual(manager.kill_wait, 3)
        self.assertFalse(manager.is_checkpoint_requested)

        # Test with custom parameters
        manager_custom = CheckpointManager(trainer, kill_wait=5)
        self.assertEqual(manager_custom.kill_wait, 5)

    def test_signal_handler_setup(self):
        """Test signal handler setup and restoration."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Store original handler
        original_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        try:
            # Setup JIT signal handler
            manager.setup_signal_handler()

            # Verify handler is set
            current_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self.assertNotEqual(current_handler, signal.SIG_DFL)

            # Verify original handler is stored
            self.assertIsNotNone(manager._original_sigterm_handler)

        finally:
            # Restore original handler
            signal.signal(signal.SIGTERM, original_handler)

    @patch("threading.Timer")
    def test_sigterm_handler_flow(self, mock_timer):
        """Test SIGTERM handler execution flow."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer, kill_wait=2)

        # Mock timer to prevent actual threading
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        # Test first SIGTERM call
        self.assertFalse(manager.is_checkpoint_requested)
        manager._sigterm_handler(signal.SIGTERM, None)

        # Verify checkpoint was NOT immediately requested (timer is used)
        self.assertFalse(manager.is_checkpoint_requested)

        # Verify timer was created with kill_wait period and correct callback
        mock_timer.assert_called_once_with(2, manager._enable_checkpoint)
        mock_timer_instance.start.assert_called_once()

        # Manually trigger the timer callback to test flag setting
        manager._enable_checkpoint()

        # Verify checkpoint is now requested
        self.assertTrue(manager.is_checkpoint_requested)

        # Test second SIGTERM call (should be ignored)
        mock_timer.reset_mock()
        manager._sigterm_handler(signal.SIGTERM, None)

        # Verify no additional timer was created
        mock_timer.assert_not_called()

    def test_toggle_checkpoint_flag(self):
        """Test the toggle checkpoint flag method."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Initially should not be requested
        self.assertFalse(manager.is_checkpoint_requested)

        # Toggle flag
        manager._enable_checkpoint()

        # Should now be requested
        self.assertTrue(manager.is_checkpoint_requested)

    def test_execute_jit_checkpoint(self):
        """Test the checkpoint execution logic with sentinel file."""
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Mock trainer's save checkpoint method
        trainer._save_checkpoint = Mock()
        trainer.state.global_step = 42

        # Set checkpoint requested flag
        manager.is_checkpoint_requested = True

        # Execute checkpoint
        manager.execute_jit_checkpoint()

        # Verify checkpoint was called
        trainer._save_checkpoint.assert_called_once_with(trainer.model, trial=None)

        # Verify checkpoint flag was reset
        self.assertFalse(manager.is_checkpoint_requested)

        # Verify sentinel file was removed (should be in checkpoint-42 folder)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-42"
        sentinel_file = os.path.join(self.test_dir, checkpoint_folder, "checkpoint-is-incomplete.txt")
        self.assertFalse(os.path.exists(sentinel_file))

    def test_execute_jit_checkpoint_sentinel_file_cleanup(self):
        """Test that sentinel file is cleaned up after successful checkpoint."""
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Mock trainer's save checkpoint method
        trainer._save_checkpoint = Mock()
        trainer.state.global_step = 42

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-42"
        sentinel_file = os.path.join(self.test_dir, checkpoint_folder, "checkpoint-is-incomplete.txt")

        # Execute checkpoint
        manager.execute_jit_checkpoint()

        # Verify sentinel file doesn't exist after successful checkpoint
        self.assertFalse(os.path.exists(sentinel_file))

    def test_execute_jit_checkpoint_with_exception(self):
        """Test checkpoint execution with exception handling."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Mock trainer's save checkpoint method to raise exception
        trainer._save_checkpoint = Mock(side_effect=Exception("Checkpoint failed"))
        trainer.state.global_step = 42

        # Test that exception is re-raised
        with self.assertRaises(Exception) as context:
            manager.execute_jit_checkpoint()

        self.assertEqual(str(context.exception), "Checkpoint failed")

        # Verify checkpoint flag was still reset to avoid multiple attempts
        self.assertFalse(manager.is_checkpoint_requested)

    def test_jit_checkpoint_callback_initialization(self):
        """Test JITCheckpointCallback initialization."""
        callback = JITCheckpointCallback()

        self.assertIsNone(callback.trainer)
        self.assertIsNone(callback.jit_manager)

    def test_jit_checkpoint_callback_set_trainer_enabled(self):
        """Test setting trainer with JIT checkpointing enabled."""
        trainer = self.get_trainer(enable_jit=True)
        callback = JITCheckpointCallback()

        with patch.object(CheckpointManager, "setup_signal_handler") as mock_setup:
            callback.set_trainer(trainer)

            self.assertEqual(callback.trainer, trainer)
            self.assertIsNotNone(callback.jit_manager)
            self.assertIsInstance(callback.jit_manager, CheckpointManager)
            mock_setup.assert_called_once()

    def test_jit_checkpoint_callback_set_trainer_disabled(self):
        """Test setting trainer with JIT checkpointing disabled."""
        trainer = self.get_trainer(enable_jit=False)
        callback = JITCheckpointCallback()

        callback.set_trainer(trainer)

        self.assertEqual(callback.trainer, trainer)
        self.assertIsNone(callback.jit_manager)

    def test_jit_checkpoint_callback_on_pre_optimizer_step(self):
        """Test callback behavior during pre-optimizer step."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_training_stop = False

        # Mock execute method
        with patch.object(callback.jit_manager, "execute_jit_checkpoint") as mock_execute:
            # Test when checkpoint not requested
            callback.jit_manager.is_checkpoint_requested = False
            callback.on_pre_optimizer_step(trainer.args, trainer.state, control)
            self.assertFalse(control.should_training_stop)
            mock_execute.assert_not_called()

            # Test when checkpoint requested
            callback.jit_manager.is_checkpoint_requested = True
            callback.on_pre_optimizer_step(trainer.args, trainer.state, control)
            self.assertTrue(control.should_training_stop)
            mock_execute.assert_called_once()

    def test_jit_checkpoint_callback_on_step_begin(self):
        """Test callback behavior at step begin."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_training_stop = False

        # Mock execute method
        with patch.object(callback.jit_manager, "execute_jit_checkpoint") as mock_execute:
            # Test when checkpoint not requested
            callback.jit_manager.is_checkpoint_requested = False
            callback.on_step_begin(trainer.args, trainer.state, control)
            self.assertFalse(control.should_training_stop)
            mock_execute.assert_not_called()

            # Test when checkpoint requested
            callback.jit_manager.is_checkpoint_requested = True
            callback.on_step_begin(trainer.args, trainer.state, control)
            self.assertTrue(control.should_training_stop)
            mock_execute.assert_called_once()

    def test_jit_checkpoint_callback_on_step_end(self):
        """Test callback behavior at step end."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_training_stop = False
        control.should_save = True

        # Mock execute method
        with patch.object(callback.jit_manager, "execute_jit_checkpoint") as mock_execute:
            # Test when checkpoint not requested
            callback.jit_manager.is_checkpoint_requested = False
            callback.on_step_end(trainer.args, trainer.state, control)
            self.assertFalse(control.should_training_stop)
            mock_execute.assert_not_called()

            # Reset control
            control.should_save = True

            # Test when checkpoint requested
            callback.jit_manager.is_checkpoint_requested = True
            callback.on_step_end(trainer.args, trainer.state, control)
            self.assertTrue(control.should_training_stop)
            self.assertFalse(control.should_save)
            mock_execute.assert_called_once()

    def test_jit_checkpoint_callback_on_epoch_end(self):
        """Test callback behavior at epoch end."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_save = True
        control.should_training_stop = False

        # Mock execute method
        with patch.object(callback.jit_manager, "execute_jit_checkpoint") as mock_execute:
            # Test when checkpoint not requested
            callback.jit_manager.is_checkpoint_requested = False
            callback.on_epoch_end(trainer.args, trainer.state, control)
            # should_save should remain unchanged when checkpoint not requested
            self.assertTrue(control.should_save)
            self.assertFalse(control.should_training_stop)
            mock_execute.assert_not_called()

            # Reset control
            control.should_save = True
            control.should_training_stop = False

            # Test when checkpoint requested
            callback.jit_manager.is_checkpoint_requested = True
            callback.on_epoch_end(trainer.args, trainer.state, control)
            self.assertFalse(control.should_save)
            self.assertTrue(control.should_training_stop)
            mock_execute.assert_called_once()

    def test_jit_checkpoint_callback_on_train_end(self):
        """Test signal handler restoration on training end."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()

        # Store original SIGTERM handler
        original_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)

        try:
            callback.set_trainer(trainer)

            # Verify signal handler was set up
            self.assertIsNotNone(callback.jit_manager._original_sigterm_handler)

            # Mock control object
            control = Mock()

            # Call on_train_end
            callback.on_train_end(trainer.args, trainer.state, control)

            # Verify signal handler was restored
            current_handler = signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self.assertEqual(current_handler, callback.jit_manager._original_sigterm_handler)

        finally:
            # Restore original handler for cleanup
            signal.signal(signal.SIGTERM, original_handler)

    @patch("threading.Timer")
    def test_kill_wait_period(self, mock_timer):
        """Test the kill wait period functionality."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer, kill_wait=5)

        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        manager._sigterm_handler(signal.SIGTERM, None)

        # Verify Timer was created with the correct kill_wait period and callback
        mock_timer.assert_called_once_with(5, manager._enable_checkpoint)
        mock_timer_instance.start.assert_called_once()

    def test_integration_with_trainer(self):
        """Test integration of JIT checkpointing with Trainer."""
        trainer = self.get_trainer(enable_jit=True)

        # Check that JIT callback was added
        jit_callbacks = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, JITCheckpointCallback)]
        self.assertEqual(len(jit_callbacks), 1)

        jit_callback = jit_callbacks[0]
        self.assertIsNotNone(jit_callback.jit_manager)
        self.assertEqual(jit_callback.trainer, trainer)
