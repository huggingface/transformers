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


@require_torch_greater_or_equal("2.7")
@require_torch_multi_accelerator
class TestTrainerDistributedTP(TestCasePlus):
    def get_torchrun_cmd(self, script, script_args=None):
        cmd = ["torchrun", "--nproc_per_node=2", "--nnodes=1", f"--master_port={get_torch_dist_unique_port()}", script]
        return cmd + (script_args or [])

    @require_peft
    def test_save_peft_model(self):
        cmd = self.get_torchrun_cmd(
            os.path.join(SCRIPTS_DIR, "save_peft_model.py"),
            [
                "--model_name",
                "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--output_dir",
                self.get_auto_remove_tmp_dir(),
            ],
        )
        execute_subprocess_async(cmd, env=self.get_env())
