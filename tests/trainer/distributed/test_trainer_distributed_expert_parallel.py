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

import json
import os

from parameterized import parameterized

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch
    from safetensors.torch import load_file


SCRIPT = os.path.join(os.path.dirname(__file__), "scripts", "expert_parallel_train.py")
LAYOUTS = ("masked", "dispatch", "dispatch_tp")


@slow
@require_torch_multi_accelerator
class TestTrainerExpertParallel(TestCasePlus):
    """
    `Trainer` with a model sharded at load time by `DistributedConfig`: router masking with all-reduce on a
    `(fsdp, tp)` mesh, token dispatch with an independent batch per rank, and token dispatch with TP groups sharing a
    batch. Every layout consumes the same global batch per step as the single-process reference, so the logged
    losses and gradient norms and the saved weights have to match it.
    """

    def _run(self, layout, model_dir, world_size):
        output_dir = self.get_auto_remove_tmp_dir()
        cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",
            "--nnodes=1",
            f"--master_port={get_torch_dist_unique_port()}",
            SCRIPT,
            f"--layout={layout}",
            f"--model_dir={model_dir}",
            f"--output_dir={output_dir}",
        ]
        execute_subprocess_async(cmd, env=self.get_env())
        with open(os.path.join(output_dir, "results.json")) as f:
            results = json.load(f)
        return results, load_file(os.path.join(output_dir, "model", "model.safetensors"))

    def _model_dir(self):
        torch.manual_seed(0)
        config = Qwen3MoeConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
        )
        model_dir = self.get_auto_remove_tmp_dir()
        Qwen3MoeForCausalLM(config).save_pretrained(model_dir)
        return model_dir

    @parameterized.expand([(layout,) for layout in LAYOUTS])
    def test_matches_single_process_reference(self, layout):
        if backend_device_count(torch_device) < 4:
            self.skipTest("Requires 4 accelerators")
        model_dir = self._model_dir()
        reference, reference_weights = self._run("reference", model_dir, world_size=1)
        results, weights = self._run(layout, model_dir, world_size=4)

        self.assertEqual(len(results["loss"]), len(reference["loss"]))
        torch.testing.assert_close(
            torch.tensor(results["loss"]), torch.tensor(reference["loss"]), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            torch.tensor(results["grad_norm"]), torch.tensor(reference["grad_norm"]), rtol=1e-3, atol=1e-4
        )
        self.assertEqual(set(weights), set(reference_weights))
        # Adam turns rounding-level gradient differences into `lr`-sized weight differences where the gradient is
        # near zero, so the weights get a tolerance above the learning rate.
        for key, tensor in reference_weights.items():
            torch.testing.assert_close(weights[key], tensor, rtol=1e-3, atol=2e-3, msg=lambda m: f"{key}: {m}")
