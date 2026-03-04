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
DDP-specific distributed trainer tests.
"""

import json
import os

import yaml
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_torch_multi_accelerator,
    run_first,
    slow,
    torch_device,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from .test_trainer_distributed import CONFIGS_DIR, SCRIPTS_DIR


if is_torch_available():
    import torch

DDP_CONFIG_FILE = os.path.join(CONFIGS_DIR, "ddp.yaml")
TRAIN_SCRIPT = os.path.join(SCRIPTS_DIR, "train.py")

dtypes = []
if is_torch_bf16_available_on_device(torch_device):
    dtypes += ["bf16"]
if is_torch_fp16_available_on_device(torch_device):
    dtypes += ["fp16"]

pure_dtype_params = ["fp32"] + dtypes
mixed_precision_params = list(dtypes)


def _parameterized_custom_name_func(func, param_num, param):
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


def _load_ddp_config():
    with open(DDP_CONFIG_FILE) as f:
        return yaml.safe_load(f)


class DDPCommandsMixin:
    """Provides ``get_torchrun_cmd`` and ``get_accelerate_cmd`` for DDP."""

    def get_torchrun_cmd(self, script, extra_args=None, num_processes=None):
        config = _load_ddp_config()
        if num_processes is None:
            num_processes = backend_device_count(torch_device)
        port = get_torch_dist_unique_port()
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_processes}",
            f"--nnodes={config.get('num_machines', 1)}",
            f"--master_port={port}",
            script,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def get_accelerate_cmd(self, script, extra_args=None, num_processes=None):
        if num_processes is None:
            num_processes = backend_device_count(torch_device)
        port = get_torch_dist_unique_port()
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            DDP_CONFIG_FILE,
            "--num_processes",
            str(num_processes),
            "--main_process_port",
            str(port),
            script,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd


class TestTrainerDistributedDDP(DDPCommandsMixin, TestCasePlus):
    # -----------------------------------------------------------------------
    # accelerate launch tests
    # -----------------------------------------------------------------------
    @run_first
    @require_torch_multi_accelerator
    def test_eval_order(self):
        output_dir = self.get_auto_remove_tmp_dir()
        script = os.path.join(SCRIPTS_DIR, "eval_ddp.py")
        cmd = self.get_accelerate_cmd(
            script,
            extra_args=["--output_dir", output_dir],
        )
        execute_subprocess_async(cmd, env=self.get_env())

    @run_first
    @require_torch_multi_accelerator
    def test_loss_averaging(self):
        device_count = backend_device_count(torch_device)
        min_bs = 2
        output_dir = self.get_auto_remove_tmp_dir()
        script = os.path.join(SCRIPTS_DIR, "loss_averaging.py")

        # Launch 1: single-GPU baseline
        cmd = self.get_accelerate_cmd(
            script,
            extra_args=[
                "--output_dir",
                f"{output_dir}/base",
                "--per_device_train_batch_size",
                str(min_bs * device_count),
                "--average_tokens_across_devices",
                "True",
            ],
            num_processes=1,
        )
        execute_subprocess_async(cmd, env=self.get_env())

        # Launch 2: multi-GPU with both averaging modes in one process
        cmd = self.get_accelerate_cmd(
            script,
            extra_args=[
                "--output_dir",
                f"{output_dir}/multi",
                "--per_device_train_batch_size",
                str(min_bs),
                "--run_both_averaging_modes",
            ],
            num_processes=device_count,
        )
        execute_subprocess_async(cmd, env=self.get_env())

        with open(f"{output_dir}/base_losses.json") as f:
            base_loss = json.load(f)
        with open(f"{output_dir}/multi/broken_losses.json") as f:
            broken_loss = json.load(f)
        with open(f"{output_dir}/multi/fixed_losses.json") as f:
            fixed_loss = json.load(f)

        broken_diff = [abs(base_loss[i] - broken_loss[i]) for i in range(len(base_loss))]
        fixed_diff = [abs(base_loss[i] - fixed_loss[i]) for i in range(len(base_loss))]
        sum_base = sum(base_loss)
        sum_broken = sum(broken_loss)
        relative_broken = abs(sum_base - sum_broken) / max(sum_base, sum_broken)

        self.assertGreater(max(broken_diff), 0.5)
        self.assertLess(max(fixed_diff), 0.005)
        self.assertLess(relative_broken, 0.1)

    @run_first
    @require_torch_multi_accelerator
    def test_worker_seed(self):
        output_dir = self.get_auto_remove_tmp_dir()
        script = os.path.join(SCRIPTS_DIR, "worker_seed.py")
        cmd = self.get_accelerate_cmd(
            script,
            extra_args=["--output_dir", output_dir],
        )
        execute_subprocess_async(cmd, env=self.get_env())

    # -----------------------------------------------------------------------
    # torchrun vs accelerate parity
    # -----------------------------------------------------------------------
    @run_first
    @require_torch_multi_accelerator
    def test_torchrun_accelerate_training_parity(self):
        """Verify that torchrun and accelerate launch produce identical training losses."""
        script = os.path.join(SCRIPTS_DIR, "training_parity.py")

        torchrun_dir = self.get_auto_remove_tmp_dir()
        cmd = self.get_torchrun_cmd(script, extra_args=["--output_dir", torchrun_dir])
        execute_subprocess_async(cmd, env=self.get_env())

        accelerate_dir = self.get_auto_remove_tmp_dir()
        cmd = self.get_accelerate_cmd(script, extra_args=["--output_dir", accelerate_dir])
        execute_subprocess_async(cmd, env=self.get_env())

        with open(os.path.join(torchrun_dir, "losses.json")) as f:
            torchrun_losses = json.load(f)
        with open(os.path.join(accelerate_dir, "losses.json")) as f:
            accelerate_losses = json.load(f)

        self.assertEqual(len(torchrun_losses), len(accelerate_losses))
        for step, (t_loss, a_loss) in enumerate(zip(torchrun_losses, accelerate_losses)):
            self.assertAlmostEqual(t_loss, a_loss, places=4, msg=f"Loss mismatch at step {step}")


# ---------------------------------------------------------------------------
# DDP training integration tests (using train.py)
# ---------------------------------------------------------------------------


@slow
@run_first
@require_torch_multi_accelerator
class TestTrainerDistributedDDPTraining(DDPCommandsMixin, TestCasePlus):
    """
    Distributed DDP training tests using ``accelerate launch`` with the shared
    train.py script. Mirrors the test structure used in FSDP and DeepSpeed.
    """

    # -------------------------------------------------------------------
    # Pure dtype training: model loaded in target dtype, no mixed precision
    # -------------------------------------------------------------------
    @parameterized.expand(pure_dtype_params, name_func=_parameterized_custom_name_func)
    def test_training(self, dtype):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_train_args(output_dir) + ["--model_dtype", dtype]
        cmd = self.get_accelerate_cmd(TRAIN_SCRIPT, extra_args=args)
        execute_subprocess_async(cmd, env=self.get_env())

    # -------------------------------------------------------------------
    # Mixed precision: model loaded in fp32, training with --bf16/--fp16
    # -------------------------------------------------------------------
    @parameterized.expand(mixed_precision_params, name_func=_parameterized_custom_name_func)
    def test_training_mixed_precision(self, dtype):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_train_args(output_dir) + ["--model_dtype", "fp32", f"--{dtype}"]
        if dtype == "fp16":
            args += ["--optim", "adamw_torch"]
        cmd = self.get_accelerate_cmd(TRAIN_SCRIPT, extra_args=args)
        execute_subprocess_async(cmd, env=self.get_env())

    # -------------------------------------------------------------------
    # Gradient accumulation
    # -------------------------------------------------------------------
    def test_training_with_gradient_accumulation(self):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_train_args(output_dir) + ["--bf16", "--gradient_accumulation_steps", "2"]
        cmd = self.get_accelerate_cmd(TRAIN_SCRIPT, extra_args=args)
        execute_subprocess_async(cmd, env=self.get_env())

    # -------------------------------------------------------------------
    # Checkpoint save and resume
    # -------------------------------------------------------------------
    def test_training_and_can_resume_normally(self):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_train_args(output_dir, num_epochs=2, logging_steps=2, save_steps=2) + ["--bf16"]

        # First training run
        logs = self._run_and_get_logs(
            self.get_accelerate_cmd(TRAIN_SCRIPT, extra_args=args),
            output_dir,
        )

        # Resume from checkpoint
        checkpoint = os.path.join(output_dir, "checkpoint-2")
        self.assertTrue(os.path.isdir(checkpoint), f"Checkpoint dir not found: {checkpoint}")

        resume_args = args + ["--resume_from_checkpoint", checkpoint]
        logs_resume = self._run_and_get_logs(
            self.get_accelerate_cmd(TRAIN_SCRIPT, extra_args=resume_args),
            output_dir,
        )

        for log, log1 in zip(logs, logs_resume):
            if "learning_rate" in log:
                self.assertAlmostEqual(log["learning_rate"], log1["learning_rate"], delta=1e-5)

    # -------------------------------------------------------------------
    # Eval
    # -------------------------------------------------------------------
    def test_eval(self):
        output_dir = self.get_auto_remove_tmp_dir()
        eval_output = os.path.join(output_dir, "eval_metrics.json")
        args = self._get_train_args(output_dir) + ["--eval_output_file", eval_output]
        cmd = self.get_accelerate_cmd(TRAIN_SCRIPT, extra_args=args)
        execute_subprocess_async(cmd, env=self.get_env())

        eval_metrics = json.loads(open(eval_output).read())
        self.assertIn("eval_loss", eval_metrics)
        self.assertTrue(torch.isfinite(torch.tensor(eval_metrics["eval_loss"])))

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _run_and_get_logs(self, cmd, output_dir):
        execute_subprocess_async(cmd, env=self.get_env())
        checkpoint = get_last_checkpoint(output_dir)
        state_file = os.path.join(checkpoint, "trainer_state.json")
        return TrainerState.load_from_json(state_file).log_history

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
            "--report_to",
            "none",
        ]
        if save_steps is not None:
            args += ["--save_steps", str(save_steps)]
        else:
            args += ["--save_strategy", "no"]
        return args
