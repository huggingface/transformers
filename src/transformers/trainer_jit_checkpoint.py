import signal
import threading
from typing import Optional

import torch

from .trainer_callback import TrainerCallback
from .utils import logging


logger = logging.get_logger(__name__)


class CheckpointManager:
    def __init__(self, trainer, kill_wait: int = 3):
        self.trainer = trainer
        self.checkpoint_thread = None
        self.checkpoint_stream = None
        self.checkpoint_requested = False
        self._original_sigterm_handler = None
        self.kill_wait = kill_wait
        self._checkpoint_timer = None

        if torch.cuda.is_available():
            self.checkpoint_stream = torch.cuda.Stream()

    def setup_signal_handler(self):
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
        logger.info("JIT checkpoint signal handler registered for SIGTERM")

    def _sigterm_handler(self, signum, frame):
        if self.checkpoint_requested:
            return

        logger.info("SIGTERM received, initiating JIT checkpoint")
        self.checkpoint_requested = True

        #  Sleep for specified seconds to avoid checkpointing if SIGKILL
        logger.info(f"Waiting for {self.kill_wait}s to allow checkpointing to start...")
        self._checkpoint_timer = threading.Timer(self.kill_wait, self._start_checkpoint_thread)
        self._checkpoint_timer.start()

    def _start_checkpoint_thread(self):
        """Helper method to start checkpoint thread after delay"""
        self.checkpoint_thread = threading.Thread(target=self._immediate_async_checkpoint, daemon=True)
        self.checkpoint_thread.start()

    def _immediate_async_checkpoint(self):
        """Immediate checkpoint using CUDA streams to avoid blocking training"""
        try:
            logger.info("Starting immediate async JIT checkpoint")

            # Capture the current stream before switching
            current_stream = torch.cuda.current_stream()

            # Wait for current CUDA operations to complete
            current_stream.wait_stream(self.checkpoint_stream)

            # Switch to checkpoint stream for all checkpoint operations
            with torch.cuda.stream(self.checkpoint_stream):
                self._execute_jit_checkpoint()

            # Synchronize checkpoint stream
            self.checkpoint_stream.synchronize()

            # Switch back to the original stream
            with torch.cuda.stream(current_stream):
                pass

            logger.info("Immediate async JIT checkpoint completed successfully")

        except Exception as e:
            logger.error(f"Failed to complete immediate async JIT checkpoint: {e}")

    def _execute_jit_checkpoint(self):
        try:
            original_step = self.trainer.state.global_step
            logger.info(f"Saving JIT checkpoint at step {original_step}")

            # Ensure we're on the checkpoint stream
            if torch.cuda.is_available() and torch.cuda.current_stream() != self.checkpoint_stream:
                logger.warning("Checkpoint not running on expected CUDA stream")

            # Call the trainer's checkpoint method directly
            self.trainer._save_checkpoint(self.trainer.model, trial=None)

        except Exception as e:
            logger.error(f"Failed to save JIT checkpoint: {e}")
            raise

    def should_checkpoint_now(self) -> bool:
        return self.checkpoint_requested


class JITCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None
        self.jit_manager: Optional[CheckpointManager] = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        if trainer.args.enable_jit_checkpoint:
            self.jit_manager = CheckpointManager(trainer=trainer)
            self.jit_manager.setup_signal_handler()
            logger.info("JIT checkpointing enabled")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            control.should_save = False
            control.should_training_stop = True

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            control.should_save = False
