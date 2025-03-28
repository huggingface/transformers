# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import subprocess
import tempfile
import textwrap

from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    get_torch_dist_unique_port,
    require_torch_multi_gpu,
)


if is_torch_available():
    import torch


# RUN_SLOW=1 pytest -sv tests/tensor_parallel/test_tensor_parallel.py
class TestTensorParallel(TestCasePlus):
    nproc_per_node = 2

    def torchrun(self, script: str):
        """Run the `script` using `torchrun` command for multi-processing in a subprocess. Captures errors as necessary."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py") as tmp:
            tmp.write(script)
            tmp.flush()
            tmp.seek(0)
            cmd = (
                f"torchrun --nproc_per_node {self.nproc_per_node} --master_port {get_torch_dist_unique_port()} {tmp.name}"
            ).split()

            # Note that the subprocess will be waited for here, and raise an error if not successful
            try:
                _ = subprocess.run(cmd, capture_output=True, env=self.get_env(), text=True, check=True)
            except subprocess.CalledProcessError as e:
                raise Exception(f"The following error was captured: {e.stderr}")

    def test_model_forward(self):
        script_to_run = textwrap.dedent(
            """
            import torch
            import os
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "JackFram/llama-68m"

            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", tp_plan="auto")
            torch.distributed.barrier()

            has_dtensor = 0
            for name, parameter in model.named_parameters():
                if isinstance(parameter.data, torch.distributed.tensor.DTensor):
                    has_dtensor = 1
                    break

            assert has_dtensor == 1, "TP model must has DTensor"

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            prompt = "Can I help"

            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            outputs = model(inputs)

            next_token_logits = outputs[0][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            response = tokenizer.decode(next_token)
            assert response == "with"

            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            """
        )
        self.torchrun(script_to_run)


@require_torch_multi_gpu
class TestTensorParallelCuda(TestTensorParallel):
    nproc_per_node = torch.cuda.device_count()
