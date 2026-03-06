import os
import signal
import threading

from .trainer_callback import TrainerCallback
from .trainer_utils import PREFIX_CHECKPOINT_DIR
from .utils import logging


logger = logging.get_logger(__name__)


class CheckpointManager:
    def __init__(self, trainer, kill_wait: int = 3):
        """
        Initialize the CheckpointManager for Just-In-Time checkpoint handling.

        Args:
            trainer: The Trainer instance that will be used to save checkpoints when SIGTERM is received.
            kill_wait (`int`, *optional*, defaults to 3): Grace period to distinguish between SIGTERM and SIGKILL.
        """
        self.trainer = trainer
        self.is_checkpoint_requested = False
        self._original_sigterm_handler = None
        self.kill_wait = kill_wait

    def setup_signal_handler(self):
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
        logger.info("JIT checkpoint signal handler registered for SIGTERM")

    def _sigterm_handler(self, signum, frame):
        if self.is_checkpoint_requested:
            return

        logger.info(f"SIGTERM received, will request JIT checkpoint after {self.kill_wait}s")
        threading.Timer(self.kill_wait, self._enable_checkpoint).start()

    def _enable_checkpoint(self):
        logger.info("Kill wait period elapsed, requesting checkpoint")
        self.is_checkpoint_requested = True

    def execute_jit_checkpoint(self):
        try:
            # Set checkpoint flag to False to avoid multiple checkpoints getting triggered by other callbacks
            self.is_checkpoint_requested = False

            logger.info("Starting JIT checkpointing...")
            current_step = self.trainer.state.global_step
            logger.info(f"Saving JIT checkpoint at step {current_step}")

            output_dir = self.trainer._get_output_dir(trial=None)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{current_step}"
            checkpoint_path = os.path.join(output_dir, checkpoint_folder)

            # Create checkpoint directory
            os.makedirs(checkpoint_path, exist_ok=True)

            # Create a sentinel file to indicate checkpointing is in progress
            sentinel_file = os.path.join(output_dir, checkpoint_folder, "checkpoint-is-incomplete.txt")
            with open(sentinel_file, "w") as f:
                f.write(f"Checkpoint started at step {current_step} and in progress...")
            logger.info(f"Created checkpoint progress sentinel marker file: {sentinel_file}")

            # Invoke the trainer's checkpoint method directly
            self.trainer._save_checkpoint(self.trainer.model, trial=None)

            # Remove sentinel file upon successful checkpointing
            if os.path.exists(sentinel_file):
                os.remove(sentinel_file)
                logger.info("Sentinel marker file removed")

            logger.info("Immediate JIT checkpoint completed successfully")

        except Exception as e:
            logger.error(f"Failed to save JIT checkpoint: {e}")
            raise


class JITCheckpointCallback(TrainerCallback):
    """
    Callback for Just-In-Time checkpointing on SIGTERM signals.

    When SIGTERM is received, the checkpoint manager sets `is_checkpoint_requested=True`.
    The callbacks detect this flag and set `control.should_training_stop=True`, which signals
    the Trainer's training loop to exit gracefully after saving the checkpoint.
    """

    def __init__(self):
        self.trainer = None
        self.jit_manager: CheckpointManager | None = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        if trainer.args.enable_jit_checkpoint:
            self.jit_manager = CheckpointManager(trainer=trainer)
            self.jit_manager.setup_signal_handler()
            logger.info("JIT checkpointing enabled")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.is_checkpoint_requested:
            control.should_training_stop = True
            self.jit_manager.execute_jit_checkpoint()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.is_checkpoint_requested:
            control.should_training_stop = True
            self.jit_manager.execute_jit_checkpoint()

    def on_step_end(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.is_checkpoint_requested:
            control.should_save = False
            control.should_training_stop = True
            self.jit_manager.execute_jit_checkpoint()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.is_checkpoint_requested:
            control.should_save = False
            control.should_training_stop = True
            self.jit_manager.execute_jit_checkpoint()

    def on_train_end(self, args, state, control, **kwargs):
        #  Restore original SIGTERM handler
        if self.jit_manager and self.jit_manager._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self.jit_manager._original_sigterm_handler)
            logger.info("Restored original SIGTERM handler after training completion")
