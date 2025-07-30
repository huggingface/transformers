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

# Run the test: CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/tensor_parallel/test_tensor_parallel.py

import os
import subprocess
import tempfile
import textwrap

from transformers import is_torch_available
from transformers.integrations.tensor_parallel import get_packed_weights, repack_weights
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    get_torch_dist_unique_port,
    require_huggingface_hub_greater_or_equal,
    require_torch_multi_accelerator,
    torch_device,
)


if is_torch_available():
    import torch


class TestTensorParallelUtils(TestCasePlus):
    def test_packed_unpacked_conversion(self):
        WORLD_SIZE = 2
        PACKED_BLOCK_SIZE = 800
        SHARDING_DIM = 2
        NUM_BLOCKS = 2

        original_packed_weights = torch.randn(4, 512, 2 * PACKED_BLOCK_SIZE)
        original_packed_weights.get_dtype = lambda: "F32"  # get_packed_weights expects PySlice object
        empty_param = torch.empty(4, 512, 2 * PACKED_BLOCK_SIZE)

        class MockDeviceMesh:
            def size(self):
                return WORLD_SIZE

        mock_mesh = (
            MockDeviceMesh()
        )  # get_packed_weights only calls `.size()`, do this to avoid doing actual distributed run

        packed_weights_0 = get_packed_weights(original_packed_weights, empty_param, mock_mesh, 0, SHARDING_DIM)
        packed_weights_1 = get_packed_weights(original_packed_weights, empty_param, mock_mesh, 1, SHARDING_DIM)

        # simulate all gather of sharded weights
        packed_weights = torch.cat([packed_weights_0, packed_weights_1], dim=SHARDING_DIM)
        unpacked_weights = repack_weights(packed_weights, SHARDING_DIM, WORLD_SIZE, NUM_BLOCKS)

        assert torch.allclose(unpacked_weights, original_packed_weights)


class TestTensorParallel(TestCasePlus):
    nproc_per_node = 2

    def torchrun(self, script: str, is_torchrun: bool = True):
        """Run the `script` using `torchrun` command for multi-processing in a subprocess. Captures errors as necessary."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py") as tmp:
            tmp.write(script)
            tmp.flush()
            tmp.seek(0)
            if is_torchrun:
                cmd = (
                    f"torchrun --nproc_per_node {self.nproc_per_node} --master_port {get_torch_dist_unique_port()} {tmp.name}"
                ).split()
            else:
                cmd = ["python3", tmp.name]

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

            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", tp_plan="auto")
            torch.distributed.barrier()

            has_dtensor = 0
            for name, parameter in model.named_parameters():
                if isinstance(parameter.data, torch.distributed.tensor.DTensor):
                    has_dtensor = 1
                    break

            assert has_dtensor == 1, "TP model must has DTensor"

            tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
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

    def test_model_generate(self):
        script_to_run = textwrap.dedent(
            """
            import torch
            import os
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "JackFram/llama-68m"

            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", tp_plan="auto")
            torch.distributed.barrier()

            model.forward = torch.compile(model.forward)

            has_dtensor = 0
            for name, parameter in model.named_parameters():
                if isinstance(parameter.data, torch.distributed.tensor.DTensor):
                    has_dtensor = 1
                    break

            assert has_dtensor == 1, "TP model must has DTensor"

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            prompt = "Can I help"

            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            outputs = model.generate(inputs, max_new_tokens=10, cache_implementation="static")

            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            assert output_text[0].startswith(prompt), f"Expected output to start with '{prompt}', got '{output_text[0]}'"

            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            """
        )
        self.torchrun(script_to_run)

    @require_huggingface_hub_greater_or_equal("0.31.4")
    def test_model_save(self):
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as tmp_dir:
            for is_torchrun in [True, False]:
                script_to_run = textwrap.dedent(
                    f"""
                    import torch
                    import os
                    from transformers import AutoModelForCausalLM

                    model_id = "JackFram/llama-68m"
                    kwargs = dict()

                    if os.environ.get("RANK", None) is not None:
                        kwargs["tp_plan"] = "auto"
                        result_dir = "{tmp_dir}/tp"
                    else:
                        result_dir = "{tmp_dir}/nontp"

                    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
                    model.save_pretrained(result_dir)
                    """
                )
                self.torchrun(script_to_run, is_torchrun=is_torchrun)

            non_tp_model_path = os.path.join(tmp_dir, "nontp")
            tp_model_path = os.path.join(tmp_dir, "tp")

            for filename in os.listdir(non_tp_model_path):
                if not filename.endswith(".safetensors"):
                    continue

                non_tp_model = safe_open(os.path.join(non_tp_model_path, filename), device="cpu", framework="pt")
                tp_model = safe_open(os.path.join(tp_model_path, filename), device="cpu", framework="pt")
                for non_tp_key in non_tp_model.keys():
                    non_tp_tensor = non_tp_model.get_tensor(non_tp_key)
                    tp_tensor = tp_model.get_tensor(non_tp_key)
                    assert torch.allclose(non_tp_tensor, tp_tensor), f"Tensor with key: {non_tp_key} does not match"
                    del non_tp_tensor, tp_tensor


@require_torch_multi_accelerator
class TestTensorParallelAccelerator(TestTensorParallel):
    nproc_per_node = backend_device_count(torch_device)
