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

from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_torch_multi_accelerator,
    run_first,
    torch_device,
)

from .test_trainer_distributed import CONFIGS_DIR, SCRIPTS_DIR


DDP_CONFIG_FILE = os.path.join(CONFIGS_DIR, "ddp.yaml")


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
