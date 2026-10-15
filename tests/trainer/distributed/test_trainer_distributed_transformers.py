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

import os

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


@require_torch_greater_or_equal("2.7")
@require_torch_multi_accelerator
class TestTrainerDistributedConfig(TestCasePlus):
    def get_torchrun_cmd(self, script, script_args=None):
        cmd = ["torchrun", "--nproc_per_node=2", "--nnodes=1", f"--master_port={get_torch_dist_unique_port()}", script]
        return cmd + (script_args or [])

    @require_peft
    def test_peft_save(self):
        """`Trainer` saves the LoRA adapter of a model sharded at load time, and the saved adapter is the trained one."""
        cmd = self.get_torchrun_cmd(PEFT_SAVE_SCRIPT, ["--output_dir", self.get_auto_remove_tmp_dir()])
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
            ],
        )
        execute_subprocess_async(cmd, env=self.get_env())
