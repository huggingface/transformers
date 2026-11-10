# Copyright 2026 The HuggingFace Team. All rights reserved.
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
Trainer tests for models sharded at load time by transformers' `DistributedConfig`.
"""

import json
import os
import tempfile

import torch
from parameterized import parameterized

from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_peft,
    require_torch_greater_or_equal,
    require_torch_multi_accelerator,
)

from .test_trainer_distributed import SCRIPTS_DIR


PEFT_SAVE_SCRIPT = os.path.join(SCRIPTS_DIR, "peft_save_distributed_config.py")
GRAD_ACCUM_SCRIPT = os.path.join(SCRIPTS_DIR, "grad_accum_distributed_config.py")
CP_SCRIPT = os.path.join(SCRIPTS_DIR, "cp_distributed_config.py")


@require_torch_greater_or_equal("2.7")
@require_torch_multi_accelerator
class TestTrainerDistributedConfig(TestCasePlus):
    def get_torchrun_cmd(self, script, script_args=None, nproc_per_node=2):
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            "--nnodes=1",
            f"--master_port={get_torch_dist_unique_port()}",
            script,
        ]
        return cmd + (script_args or [])

    @require_peft
    def test_peft_save_fsdp(self):
        """Save and reload an adapter whose weights are sharded with FSDP."""
        cmd = self.get_torchrun_cmd(
            PEFT_SAVE_SCRIPT, ["--output_dir", self.get_auto_remove_tmp_dir(), "--parallelism", "fsdp"]
        )
        execute_subprocess_async(cmd, env=self.get_env())

    @require_peft
    def test_peft_save_tp(self):
        """Save and reload an adapter whose weights are tensor-parallel DTensors."""
        cmd = self.get_torchrun_cmd(
            PEFT_SAVE_SCRIPT, ["--output_dir", self.get_auto_remove_tmp_dir(), "--parallelism", "tp"]
        )
        execute_subprocess_async(cmd, env=self.get_env())

    @require_peft
    def test_peft_save_trained_bias(self):
        """Same with sharded base-model parameters trained next to the adapter (the attention biases, `bias="lora_only"`)."""
        cmd = self.get_torchrun_cmd(
            PEFT_SAVE_SCRIPT,
            [
                "--output_dir",
                self.get_auto_remove_tmp_dir(),
                "--model",
                "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--bias",
                "lora_only",
                "--parallelism",
                "fsdp",
            ],
        )
        execute_subprocess_async(cmd, env=self.get_env())

    @parameterized.expand([("fsdp", 1), ("hsdp", 2)])
    def test_grad_accumulation_reduces_on_last_micro_batch(self, _, dp_replicate_size):
        """Gradients of a model sharded at load time are only reduced on the last micro-batch."""
        script_args = ["--output_dir", self.get_auto_remove_tmp_dir(), "--dp_replicate_size", str(dp_replicate_size)]
        execute_subprocess_async(self.get_torchrun_cmd(GRAD_ACCUM_SCRIPT, script_args), env=self.get_env())

    @parameterized.expand(
        [
            ("cp", 2, 1, 1, 1, True),
            ("cp_grad_accumulation", 2, 1, 1, 2, True),
            ("fsdp_cp", 2, 2, 1, 2, True),
            ("tp_cp", 2, 1, 2, 1, True),
            ("cp_tokens_not_averaged_across_devices", 2, 1, 1, 1, False),
        ]
    )
    def test_context_parallel(self, _, cp_size, fsdp_size, tp_size, gradient_accumulation_steps, average_tokens):
        """A model split over cp and tp groups takes the same steps as a single device on the same samples."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            single_device_file = os.path.join(tmp_dir, "single_device.json")
            # the data parallel groups hold a distinct batch each, so a single device accumulates one micro-batch per
            # group to cover the same samples in a step
            reference_args = [
                "--output_dir",
                os.path.join(tmp_dir, "single_device"),
                "--metrics_file",
                single_device_file,
                "--gradient_accumulation_steps",
                str(gradient_accumulation_steps * fsdp_size),
            ] + ([] if average_tokens else ["--no_average_tokens_across_devices"])
            # one process: a plain run would use `DataParallel` over every visible GPU
            reference_cmd = self.get_torchrun_cmd(CP_SCRIPT, reference_args, nproc_per_node=1)
            execute_subprocess_async(reference_cmd, env=self.get_env())

            cp_file = os.path.join(tmp_dir, "context_parallel.json")
            script_args = [
                "--output_dir",
                os.path.join(tmp_dir, "context_parallel"),
                "--metrics_file",
                cp_file,
                "--cp_size",
                str(cp_size),
                "--fsdp_size",
                str(fsdp_size),
                "--tp_size",
                str(tp_size),
                "--gradient_accumulation_steps",
                str(gradient_accumulation_steps),
            ] + ([] if average_tokens else ["--no_average_tokens_across_devices"])
            cmd = self.get_torchrun_cmd(CP_SCRIPT, script_args, nproc_per_node=cp_size * fsdp_size * tp_size)
            execute_subprocess_async(cmd, env=self.get_env())

            with open(single_device_file, encoding="utf-8") as f:
                single_device = json.load(f)
            with open(cp_file, encoding="utf-8") as f:
                context_parallel = json.load(f)

        torch.testing.assert_close(
            torch.tensor(context_parallel["losses"]),
            torch.tensor(single_device["losses"]),
            rtol=1e-5,
            atol=1e-6,
            msg="Loss mismatch",
        )
        torch.testing.assert_close(
            torch.tensor(context_parallel["grad_norms"]),
            torch.tensor(single_device["grad_norms"]),
            rtol=1e-5,
            atol=1e-6,
            msg="Gradient norm mismatch",
        )
        torch.testing.assert_close(
            torch.tensor(context_parallel["eval_loss"]),
            torch.tensor(single_device["eval_loss"]),
            rtol=1e-5,
            atol=1e-6,
            msg="Evaluation loss mismatch",
        )
        assert context_parallel["num_input_tokens_seen"] == single_device["num_input_tokens_seen"]
