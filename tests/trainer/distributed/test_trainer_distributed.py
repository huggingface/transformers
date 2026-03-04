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
Shared constants, helpers, and reusable test logic for distributed trainer tests.
"""

import json
import os
from abc import ABC, abstractmethod

from transformers import is_torch_available
from transformers.testing_utils import execute_subprocess_async
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint


if is_torch_available():
    import torch

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
DISTRIBUTED_DIR = os.path.dirname(__file__)
CONFIGS_DIR = os.path.join(DISTRIBUTED_DIR, "accelerate_configs")
SCRIPTS_DIR = os.path.join(DISTRIBUTED_DIR, "scripts")
TRAIN_SCRIPT = os.path.join(SCRIPTS_DIR, "train.py")


class TrainerDistributedCommon(ABC):
    """Reusable helpers and test logic shared across DDP, FSDP, and DeepSpeed.

    Subclasses must implement ``get_accelerate_cmd`` and define the following
    test methods (parameterized over framework-specific dimensions as needed)::

        test_training               → self.run_training(dtype, ...)
        test_training_mixed_precision → self.run_mixed_precision(dtype, ...)
        test_training_with_gradient_accumulation → self.run_gradient_accumulation(...)
        test_training_and_can_resume_normally    → self.run_resume(...)
        test_eval                   → self.run_eval(...)

    These test methods can't be defined here as ``@abstractmethod`` because
    ``@parameterized.expand`` removes the original method name from the
    subclass, which would cause ABC to raise ``TypeError`` at instantiation.
    """

    @abstractmethod
    def get_accelerate_cmd(self, script, config_file, launch_args=None, script_args=None, **kwargs):
        """Build the full ``accelerate launch`` command list.

        ``config_file`` is always required so that every call site is
        explicit about the configuration being used.
        """
        ...

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _get_train_args(self, output_dir, num_epochs=1, logging_steps=5, save_steps=None):
        args = [
            "--output_dir",
            output_dir,
            "--num_train_epochs",
            str(num_epochs),
            "--logging_steps",
            str(logging_steps),
            "--per_device_train_batch_size",
            "4",
            "--learning_rate",
            "5e-5",
        ]
        if save_steps is not None:
            args += ["--save_steps", str(save_steps)]
        else:
            args += ["--save_strategy", "no"]
        return args

    def _run_and_get_logs(self, cmd, output_dir):
        execute_subprocess_async(cmd, env=self.get_env())
        checkpoint = get_last_checkpoint(output_dir)
        state_file = os.path.join(checkpoint, "trainer_state.json")
        return TrainerState.load_from_json(state_file).log_history

    # -------------------------------------------------------------------
    # Reusable test logic — called from subclass test methods
    # -------------------------------------------------------------------
    def run_training(self, dtype="bf16", **cmd_kwargs):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_train_args(output_dir) + ["--model_dtype", dtype]
        execute_subprocess_async(
            self.get_accelerate_cmd(TRAIN_SCRIPT, script_args=args, **cmd_kwargs),
            env=self.get_env(),
        )

    def run_mixed_precision(self, dtype="bf16", **cmd_kwargs):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_train_args(output_dir) + ["--model_dtype", "fp32", f"--{dtype}"]
        if dtype == "fp16":
            args += ["--optim", "adamw_torch"]
        execute_subprocess_async(
            self.get_accelerate_cmd(TRAIN_SCRIPT, script_args=args, **cmd_kwargs),
            env=self.get_env(),
        )

    def run_gradient_accumulation(self, **cmd_kwargs):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_train_args(output_dir) + ["--bf16", "--gradient_accumulation_steps", "2"]
        execute_subprocess_async(
            self.get_accelerate_cmd(TRAIN_SCRIPT, script_args=args, **cmd_kwargs),
            env=self.get_env(),
        )

    def run_resume(self, **cmd_kwargs):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_train_args(output_dir, num_epochs=2, logging_steps=2, save_steps=2) + ["--bf16"]

        logs = self._run_and_get_logs(
            self.get_accelerate_cmd(TRAIN_SCRIPT, script_args=args, **cmd_kwargs),
            output_dir,
        )

        checkpoint = os.path.join(output_dir, "checkpoint-2")
        self.assertTrue(os.path.isdir(checkpoint), f"Checkpoint dir not found: {checkpoint}")

        resume_args = args + ["--resume_from_checkpoint", checkpoint]
        logs_resume = self._run_and_get_logs(
            self.get_accelerate_cmd(TRAIN_SCRIPT, script_args=resume_args, **cmd_kwargs),
            output_dir,
        )

        for log, log1 in zip(logs, logs_resume):
            if "learning_rate" in log:
                self.assertAlmostEqual(log["learning_rate"], log1["learning_rate"], delta=1e-5)

    def run_eval(self, **cmd_kwargs):
        output_dir = self.get_auto_remove_tmp_dir()
        eval_output = os.path.join(output_dir, "eval_metrics.json")
        args = self._get_train_args(output_dir) + ["--eval_output_file", eval_output]
        execute_subprocess_async(
            self.get_accelerate_cmd(TRAIN_SCRIPT, script_args=args, **cmd_kwargs),
            env=self.get_env(),
        )

        with open(eval_output) as f:
            eval_metrics = json.load(f)
        self.assertIn("eval_loss", eval_metrics)
        self.assertTrue(torch.isfinite(torch.tensor(eval_metrics["eval_loss"])))
