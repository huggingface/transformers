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

from parameterized import parameterized

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
from transformers.utils import is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from .test_trainer_distributed import CONFIGS_DIR, SCRIPTS_DIR, TrainerDistributedCommon


DDP_CONFIG_FILE = os.path.join(CONFIGS_DIR, "ddp.yaml")

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


class DDPCommandsMixin:
    """Provides ``get_torchrun_cmd`` and ``get_accelerate_cmd`` for DDP."""

    def get_torchrun_cmd(self, script, script_args=None, num_processes=None):
        if num_processes is None:
            num_processes = backend_device_count(torch_device)
        port = get_torch_dist_unique_port()
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_processes}",
            "--nnodes=1",
            f"--master_port={port}",
            script,
        ]
        if script_args:
            cmd.extend(script_args)
        return cmd

    def get_accelerate_cmd(
        self, script, config_file, launch_args=None, script_args=None, num_processes=None, **kwargs
    ):
        if num_processes is None:
            num_processes = backend_device_count(torch_device)
        port = get_torch_dist_unique_port()
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            config_file,
            "--num_processes",
            str(num_processes),
            "--main_process_port",
            str(port),
        ]
        if launch_args:
            cmd.extend(launch_args)
        cmd.append(script)
        if script_args:
            cmd.extend(script_args)
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
            DDP_CONFIG_FILE,
            script_args=["--output_dir", output_dir],
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
            DDP_CONFIG_FILE,
            script_args=[
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
            DDP_CONFIG_FILE,
            script_args=[
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
            DDP_CONFIG_FILE,
            script_args=["--output_dir", output_dir],
        )
        execute_subprocess_async(cmd, env=self.get_env())

    # -----------------------------------------------------------------------
    # torchrun vs accelerate env parity
    # -----------------------------------------------------------------------
    @run_first
    @require_torch_multi_accelerator
    def test_torchrun_accelerate_env_parity(self):
        """Verify torchrun and accelerate launch produce the same distributed environment for DDP."""
        script = os.path.join(SCRIPTS_DIR, "torchrun_env_check.py")
        num_processes = backend_device_count(torch_device)

        torchrun_dir = self.get_auto_remove_tmp_dir()
        cmd = self.get_torchrun_cmd(script, script_args=["--output_dir", torchrun_dir], num_processes=num_processes)
        execute_subprocess_async(cmd, env=self.get_env())

        accelerate_dir = self.get_auto_remove_tmp_dir()
        cmd = self.get_accelerate_cmd(
            script, DDP_CONFIG_FILE, script_args=["--output_dir", accelerate_dir], num_processes=num_processes
        )
        execute_subprocess_async(cmd, env=self.get_env())

        for rank in range(num_processes):
            with open(os.path.join(torchrun_dir, f"env_rank{rank}.json")) as f:
                tr = json.load(f)
            with open(os.path.join(accelerate_dir, f"env_rank{rank}.json")) as f:
                ac = json.load(f)

            for info in (tr, ac):
                # Rank consistency: env vars, TrainingArguments, and accelerator all agree
                self.assertEqual(info["env_world_size"], str(num_processes))
                self.assertEqual(info["env_rank"], str(rank))
                self.assertEqual(info["env_local_rank"], str(rank))
                self.assertEqual(info["args_process_index"], rank)
                self.assertEqual(info["args_local_process_index"], rank)
                self.assertIn(info["args_local_rank"], (rank, -1))  # may be -1 before framework consumes it
                self.assertEqual(info["accelerator_process_index"], rank)
                self.assertEqual(info["accelerator_local_process_index"], rank)
                self.assertIsNotNone(info["env_master_addr"])
                self.assertIsNotNone(info["env_master_port"])

                # World size and parallel mode
                self.assertEqual(info["args_world_size"], num_processes)
                self.assertEqual(info["args_n_gpu"], 1)
                self.assertEqual(info["args_parallel_mode"], "ParallelMode.DISTRIBUTED")
                self.assertEqual(info["accelerator_num_processes"], num_processes)
                self.assertTrue(info["accelerator_use_distributed"])
                self.assertEqual(info["accelerator_is_main_process"], rank == 0)
                self.assertEqual(info["accelerator_is_local_main_process"], rank == 0)

                # DDP: distributed type is MULTI_GPU
                self.assertEqual(info["accelerator_distributed_type"], "DistributedType.MULTI_GPU")

                # Each rank on its own device
                self.assertIn(f"cuda:{rank}", info["accelerator_device"])

                # DDP should not activate FSDP or DeepSpeed
                self.assertFalse(info["trainer_is_fsdp_enabled"])
                self.assertFalse(info["trainer_is_deepspeed_enabled"])
                self.assertNotIn("fsdp_version", info)
                self.assertNotIn("deepspeed_zero_stage", info)


# ---------------------------------------------------------------------------
# DDP training integration tests (using train.py)
# ---------------------------------------------------------------------------


@slow
@run_first
@require_torch_multi_accelerator
class TestTrainerDistributedDDPCommon(DDPCommandsMixin, TrainerDistributedCommon, TestCasePlus):
    """
    Distributed DDP training tests using ``accelerate launch`` with the shared
    train.py script. Mirrors the test structure used in FSDP and DeepSpeed.
    """

    @parameterized.expand(pure_dtype_params, name_func=_parameterized_custom_name_func)
    def test_training(self, dtype):
        self.check_training(dtype, config_file=DDP_CONFIG_FILE)

    @parameterized.expand(mixed_precision_params, name_func=_parameterized_custom_name_func)
    def test_training_mixed_precision(self, dtype):
        self.check_mixed_precision(dtype, config_file=DDP_CONFIG_FILE)

    def test_training_with_gradient_accumulation(self):
        self.check_gradient_accumulation(config_file=DDP_CONFIG_FILE)

    def test_training_and_can_resume_normally(self):
        self.check_resume(config_file=DDP_CONFIG_FILE)

    def test_eval(self):
        self.check_eval(config_file=DDP_CONFIG_FILE)
