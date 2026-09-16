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
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import patch

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.distributed import DistributedConfig
    from transformers.distributed.utils import clip_grad_norm_, load_optimizer_distributed, save_optimizer_distributed

    if dist.is_available():
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard, distribute_tensor


def _full_tensor(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _optimizer(model):
    # foreach requires homogeneous groups when TP leaves some parameters unsharded.
    groups = [
        [p for p in model.parameters() if isinstance(p, DTensor) == distributed] for distributed in (False, True)
    ]
    return torch.optim.AdamW([{"params": group} for group in groups if group], lr=0.003, foreach=True)


def _step(model, optimizer):
    for parameter in model.parameters():
        parameter.grad = parameter.detach().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _check_optimizer(model, optimizer, reference, reference_optimizer):
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        actual_state = optimizer.state[parameter]
        expected_state = reference_optimizer.state[expected]
        assert actual_state.keys() == expected_state.keys()
        for key in expected_state:
            torch.testing.assert_close(_full_tensor(actual_state[key]), expected_state[key])
    assert all(group["lr"] == reference_optimizer.param_groups[0]["lr"] for group in optimizer.param_groups)


@contextmanager
def _distributed_context(rank, directory):
    environment = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "4"}
    with patch.dict(os.environ, environment), patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/rendezvous",
            rank=rank,
            world_size=4,
            timeout=timedelta(seconds=60),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


def _load_model(directory, consolidate, config=None):
    if consolidate:
        return LlamaForCausalLM.from_pretrained(f"{directory}/saved", distributed_config=config)

    model = LlamaForCausalLM.from_pretrained(f"{directory}/seed", distributed_config=config)
    # Clear the destination so unchanged seed weights cannot make the round trip pass.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.load_distributed_checkpoint(f"{directory}/saved")
    return model


# Workers stay at module scope so multiprocessing.spawn can pickle them.
def _model_checkpoint_worker(rank, directory, consolidate):
    with _distributed_context(rank, directory):
        reference = LlamaForCausalLM.from_pretrained(f"{directory}/seed")
        model = LlamaForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(tp_size=2, fsdp_size=2)
        )
        model.save_pretrained(
            f"{directory}/saved",
            distributed_checkpoint=True,
            consolidate_distributed_checkpoint=consolidate,
        )
        for config in (DistributedConfig(tp_size=2, fsdp_size=2), DistributedConfig(tp_size=4)):
            restored = _load_model(directory, consolidate, config)
            for name, parameter in restored.state_dict().items():
                torch.testing.assert_close(_full_tensor(parameter), reference.state_dict()[name])


def _optimizer_checkpoint_worker(rank, directory, consolidate):
    with _distributed_context(rank, directory):
        reference = LlamaForCausalLM.from_pretrained(f"{directory}/seed")
        model = LlamaForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(tp_size=2, fsdp_size=2)
        )
        optimizer, reference_optimizer = _optimizer(model), _optimizer(reference)
        _step(model, optimizer)
        _step(reference, reference_optimizer)
        save_optimizer_distributed(model, optimizer, f"{directory}/saved", consolidate=consolidate)
        checkpoint = f"{directory}/saved/optimizer.pt" if consolidate else f"{directory}/saved"
        for config in (DistributedConfig(tp_size=2, fsdp_size=2), DistributedConfig(tp_size=4)):
            restored = LlamaForCausalLM.from_pretrained(f"{directory}/seed", distributed_config=config)
            restored_optimizer = _optimizer(restored)
            load_optimizer_distributed(restored, restored_optimizer, checkpoint)
            _check_optimizer(restored, restored_optimizer, reference, reference_optimizer)


def _gradient_clipping_worker(rank, directory):
    with _distributed_context(rank, directory):
        mesh = init_device_mesh("cpu", (4,))
        for distributed in ((False, False), (True, True), (False, True)):
            for max_norm in (1.0, 10000.0):
                parameters, reference = [], []
                for i, is_distributed in enumerate(distributed):
                    gradient = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8) * (i + 1)
                    expected = torch.nn.Parameter(torch.zeros_like(gradient))
                    expected.grad = gradient.clone()
                    reference.append(expected)
                    if is_distributed:
                        gradient = distribute_tensor(gradient, mesh, [Shard(0)])
                    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
                    parameter.grad = gradient
                    parameters.append(parameter)
                expected_norm = torch.nn.utils.clip_grad_norm_(reference, max_norm, foreach=True)
                actual_norm = clip_grad_norm_(parameters, max_norm, foreach=True)
                torch.testing.assert_close(_full_tensor(actual_norm), expected_norm)
                for parameter, expected in zip(parameters, reference):
                    torch.testing.assert_close(_full_tensor(parameter.grad), expected.grad)


@require_torch
@unittest.skipUnless(
    is_torch_available() and dist.is_available() and dist.is_gloo_available(), "Requires distributed Gloo"
)
class DistributedUtilsTest(unittest.TestCase):
    def setUp(self):
        self.config = LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

    def test_model_checkpoint(self):
        for consolidate in (False, True):
            with self.subTest(consolidate=consolidate):
                with tempfile.TemporaryDirectory() as directory:
                    reference = LlamaForCausalLM(self.config)
                    reference.save_pretrained(f"{directory}/seed")
                    mp.spawn(
                        _model_checkpoint_worker,
                        args=(directory, consolidate),
                        nprocs=4,
                        join=True,
                    )

                    # Reload without a process group or distributed configuration.
                    restored = _load_model(directory, consolidate)
                    for name, parameter in restored.state_dict().items():
                        torch.testing.assert_close(parameter, reference.state_dict()[name])

    def test_optimizer_checkpoint(self):
        for consolidate in (False, True):
            with self.subTest(consolidate=consolidate), tempfile.TemporaryDirectory() as directory:
                reference = LlamaForCausalLM(self.config)
                reference.save_pretrained(f"{directory}/seed")
                mp.spawn(_optimizer_checkpoint_worker, args=(directory, consolidate), nprocs=4, join=True)

                # Reload without a process group or distributed configuration.
                reference_optimizer = _optimizer(reference)
                _step(reference, reference_optimizer)
                restored = LlamaForCausalLM.from_pretrained(f"{directory}/seed")
                restored_optimizer = _optimizer(restored)
                checkpoint = f"{directory}/saved/optimizer.pt" if consolidate else f"{directory}/saved"
                load_optimizer_distributed(restored, restored_optimizer, checkpoint)
                _check_optimizer(restored, restored_optimizer, reference, reference_optimizer)

    def test_gradient_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_gradient_clipping_worker, args=(directory,), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
