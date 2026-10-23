# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
import sys

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_deepspeed,
    require_torch_gpu,
    require_torch_multi_accelerator,
    slow,
)


if is_torch_available():
    import torch


@slow
@require_torch_gpu
@require_torch_multi_accelerator
class TrainerSSTDistributedTest(TestCasePlus):
    def run_backend(self, backend, optim, precision="fp32", extra_args=(), processes=2, script_name="sst.py"):
        script = os.path.join(os.path.dirname(__file__), "scripts", script_name)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={processes}",
            f"--master_port={get_torch_dist_unique_port()}",
            script,
            "--backend",
            backend,
            "--optim",
            optim,
            "--precision",
            precision,
            "--output_dir",
            self.get_auto_remove_tmp_dir(),
            *extra_args,
        ]
        execute_subprocess_async(command, env=self.get_env())

    @parameterized.expand(
        [(backend, optim) for backend in ("ddp", "fsdp1", "fsdp2") for optim in ("sgd", "adamw", "torch_adafactor")]
    )
    def test_replicated_and_sharded_optimizers(self, backend, optim):
        self.run_backend(backend, optim)

    def test_transformers_adafactor_with_ddp(self):
        # Transformers Adafactor's factored buffers and RMS checkpoint metadata are not FSDP-compatible.
        self.run_backend("ddp", "adafactor")

    def test_custom_optimizer_with_fsdp2(self):
        self.run_backend("fsdp2", "rmsprop")

    def test_independent_submesh_bases(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("Independent two-rank submeshes require four CUDA devices.")
        self.run_backend("ddp", "sgd", processes=4, script_name="sst_submeshes.py")

    def test_fsdp2_hybrid_mesh(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("A two-by-two hybrid mesh requires four CUDA devices.")
        self.run_backend("fsdp2", "adamw", processes=4, extra_args=("--dp_replicate_size", "2"))

    @parameterized.expand(["fsdp1", "fsdp2"])
    def test_bf16_compute_with_fp32_optimizer_weights(self, backend):
        self.run_backend(backend, "adamw", precision="bf16")

    def test_fsdp1_flat_parameters(self):
        self.run_backend("fsdp1", "adamw", extra_args=("--flat_parameters",))

    @parameterized.expand(["fsdp1", "fsdp2"])
    def test_cpu_offload(self, backend):
        self.run_backend(backend, "adamw", extra_args=("--offload",))

    @parameterized.expand(["fsdp1", "fsdp2"])
    def test_shard_grad_op(self, backend):
        self.run_backend(backend, "sgd", extra_args=("--sharding", "shard_grad_op"))

    def test_fused_optimizer_and_fp16_scaling(self):
        self.run_backend("ddp", "fused_adamw", precision="fp16")

    def test_flat_parameters_with_bf16(self):
        self.run_backend("fsdp1", "adamw", precision="bf16", extra_args=("--flat_parameters",))

    @parameterized.expand(["zero1", "zero2", "zero3"])
    @require_deepspeed
    def test_deepspeed_master_parameter_writeback(self, backend):
        self.run_backend(backend, "adamw", precision="bf16")
