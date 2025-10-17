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

import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from transformers import TrainingArguments, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

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
        self.assertFalse(manager.checkpoint_requested)
        self.assertIsNone(manager.checkpoint_thread)

        # Test with custom parameters
        manager_custom = CheckpointManager(trainer, kill_wait=5)
        self.assertEqual(manager_custom.kill_wait, 5)

        # Test CUDA stream creation if available
        if torch.cuda.is_available():
            self.assertIsNotNone(manager.checkpoint_stream)
            self.assertIsInstance(manager.checkpoint_stream, torch.cuda.Stream)
        else:
            self.assertIsNone(manager.checkpoint_stream)

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

    @patch("time.sleep")
    @patch("threading.Thread")
    def test_sigterm_handler_flow(self, mock_thread, mock_sleep):
        """Test SIGTERM handler execution flow."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer, kill_wait=2)

        # Mock thread to prevent actual threading
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        # Test first SIGTERM call
        self.assertFalse(manager.checkpoint_requested)
        manager._sigterm_handler(signal.SIGTERM, None)

        # Verify checkpoint was requested
        self.assertTrue(manager.checkpoint_requested)

        # Verify sleep was called with kill_wait period
        mock_sleep.assert_called_once_with(2)

        # Verify thread was created and started
        mock_thread.assert_called_once_with(target=manager._immediate_async_checkpoint, daemon=True)
        mock_thread_instance.start.assert_called_once()

        # Test second SIGTERM call (should be ignored)
        mock_thread.reset_mock()
        mock_sleep.reset_mock()
        manager._sigterm_handler(signal.SIGTERM, None)

        # Verify no additional calls were made
        mock_thread.assert_not_called()
        mock_sleep.assert_not_called()

    def test_should_checkpoint_now(self):
        """Test checkpoint condition checking."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Initially should not checkpoint
        self.assertFalse(manager.should_checkpoint_now())

        # After requesting checkpoint
        manager.checkpoint_requested = True
        self.assertTrue(manager.should_checkpoint_now())

    @patch("torch.cuda.is_available", return_value=False)
    def test_checkpoint_manager_without_cuda(self, mock_cuda_available):
        """Test CheckpointManager behavior when CUDA is not available."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        self.assertIsNone(manager.checkpoint_stream)

    def test_execute_jit_checkpoint(self):
        """Test the checkpoint execution logic."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Mock trainer's save checkpoint method
        trainer._save_checkpoint = Mock()
        trainer.state.global_step = 42

        # Mock CUDA stream for assertion
        if torch.cuda.is_available():
            with patch("torch.cuda.current_stream", return_value=manager.checkpoint_stream):
                manager._execute_jit_checkpoint()
        else:
            # Skip CUDA stream assertion when CUDA not available
            with patch.object(manager, "checkpoint_stream", None):
                manager._execute_jit_checkpoint()

        # Verify checkpoint was called
        trainer._save_checkpoint.assert_called_once_with(trainer.model, trial=None)

    def test_execute_jit_checkpoint_with_exception(self):
        """Test checkpoint execution with exception handling."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Mock trainer's save checkpoint method to raise exception
        trainer._save_checkpoint = Mock(side_effect=Exception("Checkpoint failed"))
        trainer.state.global_step = 42

        # Test that exception is re-raised
        with self.assertRaises(Exception) as context:
            if torch.cuda.is_available():
                with patch("torch.cuda.current_stream", return_value=manager.checkpoint_stream):
                    manager._execute_jit_checkpoint()
            else:
                with patch.object(manager, "checkpoint_stream", None):
                    manager._execute_jit_checkpoint()

        self.assertEqual(str(context.exception), "Checkpoint failed")

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_immediate_async_checkpoint_cuda_streams(
        self, mock_stream_class, mock_current_stream, mock_cuda_available
    ):
        """Test async checkpoint with CUDA streams."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer)

        # Setup mocks
        mock_checkpoint_stream = Mock()
        mock_current_stream_instance = Mock()
        mock_stream_class.return_value = mock_checkpoint_stream
        mock_current_stream.return_value = mock_current_stream_instance
        manager.checkpoint_stream = mock_checkpoint_stream

        # Mock the context manager behavior
        mock_cuda_stream_context = Mock()

        with patch("torch.cuda.stream", return_value=mock_cuda_stream_context):
            with patch.object(manager, "_execute_jit_checkpoint") as mock_execute:
                mock_cuda_stream_context.__enter__ = Mock()
                mock_cuda_stream_context.__exit__ = Mock()

                manager._immediate_async_checkpoint()

                # Verify stream operations
                mock_current_stream_instance.wait_stream.assert_called_once_with(mock_checkpoint_stream)
                mock_checkpoint_stream.synchronize.assert_called_once()
                mock_execute.assert_called_once()

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

        # Test when checkpoint not requested
        callback.jit_manager.checkpoint_requested = False
        callback.on_pre_optimizer_step(trainer.args, trainer.state, control)
        self.assertFalse(control.should_training_stop)

        # Test when checkpoint requested
        callback.jit_manager.checkpoint_requested = True
        callback.on_pre_optimizer_step(trainer.args, trainer.state, control)
        self.assertTrue(control.should_training_stop)

    def test_jit_checkpoint_callback_on_step_end(self):
        """Test callback behavior at step end."""
        trainer = self.get_trainer()
        callback = JITCheckpointCallback()
        callback.set_trainer(trainer)

        # Mock control object
        control = Mock()
        control.should_training_stop = False

        # Test when checkpoint not requested
        callback.jit_manager.checkpoint_requested = False
        callback.on_step_end(trainer.args, trainer.state, control)
        self.assertFalse(control.should_training_stop)

        # Test when checkpoint requested
        callback.jit_manager.checkpoint_requested = True
        callback.on_step_end(trainer.args, trainer.state, control)
        self.assertTrue(control.should_training_stop)

    def test_jit_checkpoint_callback_without_manager(self):
        """Test callback behavior when manager is not set."""
        callback = JITCheckpointCallback()
        control = Mock()
        control.should_training_stop = False

        # Should not raise exception and not modify control
        callback.on_pre_optimizer_step(None, None, control)
        callback.on_step_end(None, None, control)
        self.assertFalse(control.should_training_stop)

    @patch("time.sleep")
    def test_kill_wait_period(self, mock_sleep):
        """Test the kill wait period functionality."""
        trainer = self.get_trainer()
        manager = CheckpointManager(trainer, kill_wait=5)

        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            manager._sigterm_handler(signal.SIGTERM, None)

            # Verify sleep was called with the correct kill_wait period
            mock_sleep.assert_called_once_with(5)

    def test_integration_with_trainer(self):
        """Test integration of JIT checkpointing with Trainer."""
        trainer = self.get_trainer(enable_jit=True)

        # Check that JIT callback was added
        jit_callbacks = [cb for cb in trainer.callback_handler.callbacks if isinstance(cb, JITCheckpointCallback)]
        self.assertEqual(len(jit_callbacks), 1)

        jit_callback = jit_callbacks[0]
        self.assertIsNotNone(jit_callback.jit_manager)
        self.assertEqual(jit_callback.trainer, trainer)
