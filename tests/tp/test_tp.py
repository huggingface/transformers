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

import os
import subprocess
import tempfile
import textwrap

#  TORCH_LOGS=+dtensor CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 PYTHONPATH="src" python -m torch.distributed.run --nproc_per_node 2 ./tests/tp/test_tp.py
from transformers import is_torch_available
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_torch_multi_gpu,
)


if is_torch_available():
    import torch


class TestTensorParallel(TestCasePlus):
    def torchrun(self, script: str):
        """Run the `script` using `torchrun` command for multi-processing in a subprocess. Captures errors as necesary."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py") as tmp:
            tmp.write(script)
            tmp.flush()
            tmp.seek(0)
            cmd = (
                f"torchrun --nproc_per_node {torch.cuda.device_count()} --master_port {get_torch_dist_unique_port()} {tmp.name}"
            ).split()

            # Note that the subprocess will be waited for here, and raise an error if not successful
            try:
                _ = subprocess.run(cmd, capture_output=True, env=self.get_env(), text=True, check=True)
            except subprocess.CalledProcessError as e:
                raise Exception(f"The following error was captured: {e.stderr}")

    @require_torch_multi_gpu
    def test_tp(self):
        distributed_args = f"""--nproc_per_node={torch.cuda.device_count()}
            --master_port={get_torch_dist_unique_port()}
            {self.test_file_dir}/test_tp.py
        """.split()
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"--output_dir {output_dir} --report_to none".split()
        cmd = ["torchrun"] + distributed_args + args
        print(cmd)
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call

    @require_torch_multi_gpu
    def test_loading_memory_consumption(self):
        script_to_run = textwrap.dedent(
            """
            import torch
            import os
            from transformers import AutoModelForCausalLM

            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            device = torch.device(f"cuda:{rank}")
            torch.distributed.init_process_group("nccl", device_id=device)

            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, tp_plan="auto")
            torch.distributed.barrier()

            # The expected full model memory footprint
            expected_model_memory = 16
            overhead_factor = 1.2

            # Assert we did not use more than the full model expected memory (with some overhead)
            if not torch.cuda.max_memory_allocated(device) / 1024**3 < expected_model_memory * overhead_factor:
                raise ValueError("Loading the model used more than the full model size")

            # Assert we correctly handled the sharding between devices
            if not torch.cuda.memory_allocated(device) / 1024**3 < (expected_model_memory / world_size) * overhead_factor:
                raise ValueError("Each model shard is larger than what is expected.")

            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            """
        )
        self.torchrun(script_to_run)


if __name__ == "__main__":
    # The script below is meant to be run under torch.distributed, on a machine with multiple GPUs:
    # CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/tp/test_tp.py
    # or
    # PYTHONPATH="src" python -m torch.distributed.run --nproc_per_node 2 ./tests/tp/test_tp.py

    if not is_torch_available():
        exit(0)

    # Test settings
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    bs = 1
    seqlen = 4096
    # Get distributed settings
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize distributed
    device = torch.device(f"cuda:{rank}")
    torch.distributed.init_process_group("nccl", device_id=device)
    device_mesh = torch.distributed.init_device_mesh("cuda", (world_size,))

    # Get model config
    config = LlamaConfig.from_pretrained(model_id)
    config.hidden_size = 2048
    config.attention_bias = False
    # Instantiate model
    with device:
        model = LlamaModel(config).to(dtype=torch.float16)

    model.eval()
    # Tensor Parallel
    if world_size > 1:
        model.tensor_parallel(device_mesh)
    # Run model

    inputs = torch.randint(config.vocab_size, (bs, seqlen), device=device)

    # Test cuda graphing explicitly
    with torch.cuda.device(device):
        print("Cuda graphing")
        with torch.no_grad():
            inputs = torch.randint(config.vocab_size, (bs, seqlen), device=device)
            # CUDA Graph setup
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(3):
                    out = model(inputs)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = model(inputs)

            for _ in range(2):
                g.replay()
            s.synchronize()

    assert out.last_hidden_state.shape == torch.Size([bs, seqlen, config.hidden_size])

    # Test compile
    with torch.no_grad():
        out = model(inputs)
        model.forward = torch.compile(model.forward, mode="reduce-overhead")
        out = model(inputs)
        out = model(inputs)
