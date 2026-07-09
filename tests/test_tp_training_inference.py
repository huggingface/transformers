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

"""End-to-end tests for the unified tensor-parallel path with a mode-aware runtime.

A single ``base_model_tp_plan`` drives both regimes over DTensor-sharded parameters:
- ``model.eval()`` (inference): the styles unwrap to plain tensors and run raw collectives
  (all-reduce / all-gather), skipping DTensor redistribute.
- ``model.train()`` (training): the styles run the autograd-aware DTensor redistribute path.
"""

import json
import os
import socket
import tempfile
import traceback
import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    backend_device_count,
    is_tensor_parallel_test,
    require_torch_greater_or_equal,
    torch_device,
)


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    from transformers import LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM
    from transformers.distributed import DistributedConfig


WORLD_SIZE = 2
SEED = 0


def _has_dtensor_params(model) -> bool:
    return any(hasattr(p, "placements") for p in model.parameters())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _tiny_llama() -> "LlamaConfig":
    return LlamaConfig(
        vocab_size=320,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        tie_word_embeddings=False,
    )


def _tiny_mixtral() -> "MixtralConfig":
    return MixtralConfig(
        vocab_size=320,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
    )


def _worker(rank, world_size, port, save_dir, impl_name, results_file):
    os.environ.update(
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
    )
    torch.manual_seed(SEED)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    error = None
    try:
        globals()[impl_name](rank, world_size, save_dir)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    flag = torch.tensor([1 if error else 0], device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if rank == 0:
        with open(results_file, "w") as f:
            json.dump({"error": error or ("Failed on another rank" if flag.item() else None)}, f)
    dist.barrier()
    dist.destroy_process_group()


# =============================================================================
# Test implementations (top-level for pickling by mp.spawn)
# =============================================================================


def _impl_dense_inference(rank, world_size, save_dir):
    device = f"cuda:{rank}"
    input_ids = torch.arange(1, 9, device=device).unsqueeze(0)

    ref = LlamaForCausalLM.from_pretrained(save_dir, dtype=torch.bfloat16).to(device).eval()
    with torch.no_grad():
        ref_logits = ref(input_ids).logits
    del ref
    torch.cuda.empty_cache()

    # from_pretrained leaves the model in eval() → inference (plain-collective) path.
    model = LlamaForCausalLM.from_pretrained(
        save_dir, dtype=torch.bfloat16, distributed_config=DistributedConfig(tp_size=world_size)
    )
    assert _has_dtensor_params(model), "TP params should be DTensors (mode-aware runtime)"
    with torch.no_grad():
        tp_logits = model(input_ids).logits
    torch.testing.assert_close(tp_logits.float(), ref_logits.float(), atol=3e-2, rtol=1e-2)

    # Cross-rank agreement (rowwise all-reduce makes the output replicated).
    gathered = [torch.empty_like(tp_logits) for _ in range(world_size)]
    dist.all_gather(gathered, tp_logits.contiguous())
    for g in gathered:
        torch.testing.assert_close(gathered[0].float(), g.float(), atol=1e-4, rtol=0)


def _impl_moe_inference(rank, world_size, save_dir):
    device = f"cuda:{rank}"
    input_ids = torch.arange(1, 9, device=device).unsqueeze(0)

    ref = MixtralForCausalLM.from_pretrained(save_dir, dtype=torch.bfloat16).to(device).eval()
    with torch.no_grad():
        ref_logits = ref(input_ids).logits
    del ref
    torch.cuda.empty_cache()

    model = MixtralForCausalLM.from_pretrained(
        save_dir, dtype=torch.bfloat16, distributed_config=DistributedConfig(tp_size=world_size)
    )
    assert _has_dtensor_params(model)
    with torch.no_grad():
        tp_logits = model(input_ids).logits
    torch.testing.assert_close(tp_logits.float(), ref_logits.float(), atol=3e-2, rtol=1e-2)


def _impl_dense_training(rank, world_size, save_dir):
    device = f"cuda:{rank}"
    model = LlamaForCausalLM.from_pretrained(
        save_dir, dtype=torch.bfloat16, distributed_config=DistributedConfig(tp_size=world_size)
    )
    assert _has_dtensor_params(model), "TP params should be DTensors for training"

    # Switching to train() activates the autograd-aware DTensor redistribute path.
    model.train()
    input_ids = torch.randint(0, 320, (1, 8), device=device)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    assert torch.isfinite(loss), "loss must be finite"
    assert any(p.grad is not None for p in model.parameters()), "backward should populate grads"


@is_tensor_parallel_test
@require_torch_greater_or_equal("2.5")
class TPTrainingInferenceTest(unittest.TestCase):
    """Multi-process (torch.distributed) tests. Require >= 2 accelerators."""

    @classmethod
    def setUpClass(cls):
        if backend_device_count(torch_device) < WORLD_SIZE:
            raise unittest.SkipTest(f"needs >= {WORLD_SIZE} accelerators")

    def _spawn(self, impl_name, config_factory, model_cls):
        with tempfile.TemporaryDirectory() as save_dir:
            torch.manual_seed(SEED)
            model_cls(config_factory()).to(torch.bfloat16).save_pretrained(save_dir)
            results_file = os.path.join(save_dir, "results.json")
            mp.spawn(
                _worker,
                args=(WORLD_SIZE, _free_port(), save_dir, impl_name, results_file),
                nprocs=WORLD_SIZE,
                join=True,
            )
            with open(results_file) as f:
                error = json.load(f)["error"]
            self.assertIsNone(error, msg=error)

    def test_dense_tp_inference_matches_reference(self):
        self._spawn("_impl_dense_inference", _tiny_llama, LlamaForCausalLM)

    def test_moe_tp_inference_matches_reference(self):
        self._spawn("_impl_moe_inference", _tiny_mixtral, MixtralForCausalLM)

    def test_dense_tp_training_backward(self):
        self._spawn("_impl_dense_training", _tiny_llama, LlamaForCausalLM)


if __name__ == "__main__":
    unittest.main()
