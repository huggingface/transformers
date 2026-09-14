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
import warnings
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DTensor

from transformers.distributed import DistributedConfig
from transformers.distributed.mixin import DistributedMixin
from transformers.distributed.utils import initialize_fully_sharded_data_parallelism
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_greater_or_equal,
    torch_device,
)
from transformers.utils import is_torch_greater_or_equal


class DistributedConfigTest(unittest.TestCase):
    def test_defaults_and_non_ep_parallelism(self):
        for kwargs in ({}, {"tp_size": 4}, {"fsdp_size": 4}, {"pp_size": 4}, {"tp_size": 2, "fsdp_size": 2}):
            config = DistributedConfig(**kwargs)
            self.assertEqual(config.ep_size, 1)
            self.assertFalse(config.enable_expert_parallel)
            self.assertFalse(config.dispatches_tokens)
            self.assertEqual(config.edp_size, config.fsdp_size)

    def test_dispatch_topology_and_round_trip(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            config = DistributedConfig(ep_size=4)
        self.assertEqual((config.tp_size, config.fsdp_size, config.ep_size, config.edp_size), (1, 8, 4, 2))
        self.assertTrue(config.enable_expert_parallel)
        self.assertEqual(config.experts_dispatch, "all-to-all")
        with warnings.catch_warnings(record=True) as caught:
            self.assertEqual(DistributedConfig.from_dict(config.to_dict()), config)
        self.assertFalse(caught)

    def test_legacy_dispatch_preserves_batches_and_expert_groups(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = DistributedConfig(
                tp_size=4, fsdp_size=2, enable_expert_parallel=True, experts_dispatch="all-to-all"
            )
        self.assertTrue(any(w.category is FutureWarning for w in caught))
        self.assertEqual(legacy, DistributedConfig(tp_size=1, fsdp_size=8, ep_size=4))

    def test_legacy_all_reduce_preserves_plan_selection(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = DistributedConfig(tp_size=4, fsdp_size=2, enable_expert_parallel=True)
        self.assertTrue(any(w.category is FutureWarning for w in caught))
        self.assertEqual(legacy, DistributedConfig(tp_size=4, fsdp_size=2, ep_size=4))
        self.assertEqual(legacy.experts_dispatch, "all-reduce")

    def test_inferred_legacy_tp_size(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = DistributedConfig(
                tp_plan="auto", fsdp_size=2, enable_expert_parallel=True, experts_dispatch="all-to-all"
            )
        self.assertTrue(any(w.category is FutureWarning for w in caught))
        self.assertEqual((config.tp_size, config.fsdp_size, config.ep_size), (1, 8, 4))

    def test_invalid_topologies(self):
        cases = [
            ({"fsdp_size": 4, "ep_size": 3}, "must divide"),
            ({"tp_size": 2, "fsdp_size": 2, "ep_size": 4}, "trunk tensor parallelism"),
            ({"fsdp_size": 4, "ep_size": 4, "experts_dispatch": "all-reduce"}, "identical tokens"),
            ({"fsdp_size": 4, "experts_dispatch": "all-to-all"}, "ep_size > 1"),
            ({"experts_dispatch": "unknown"}, "Unknown"),
            ({"ep_size": 2, "fsdp_size": 2, "pp_size": 2}, "Pipeline parallelism"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                DistributedConfig(**kwargs)
        for name in ("tp_size", "fsdp_size", "pp_size", "ep_size"):
            for value in (0, -1, 1.5, True, "4"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                    DistributedConfig(**{name: value})


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.weight = nn.Parameter(torch.randn(4, 3, 3))

    def forward(self, hidden_states, indices, weights):
        selected = self.weight[indices]
        outputs = torch.einsum("ni,nkoi->nko", hidden_states, selected)
        return (outputs * weights.unsqueeze(-1)).sum(1)


class _MoE(DistributedMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.dense = nn.Linear(3, 3)
        self.experts = _Experts()
        self._tp_plan = {}
        self._ep_plan = {"experts.weight": "grouped_gemm", "experts": "moe_tp_experts"}
        self._fsdp_plan = {"dense": "free_full_weight"}

    def forward(self, x, indices, weights):
        return self.experts(self.dense(x), indices, weights)


def _mesh_worker(rank, rendezvous, check_backward, device_type):
    dist.init_process_group(
        "gloo" if device_type == "cpu" else "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        configs = [DistributedConfig(fsdp_size=4), DistributedConfig(tp_size=2, fsdp_size=2)]
        configs += [DistributedConfig(fsdp_size=4, ep_size=ep) for ep in (2, 4)]
        for config in configs:
            with patch("torch._C._get_accelerator", return_value=torch.device(device_type)):
                device, root = initialize_fully_sharded_data_parallelism(config)
            assert root["fsdp"].size() == config.fsdp_size
            assert root["tp"].size() == config.tp_size
            assert root["edp"].size() == config.edp_size
            if not config.dispatches_tokens:
                continue
            ep = config.ep_size
            assert dist.get_process_group_ranks(root["ep"].get_group()) == list(
                range(rank // ep * ep, (rank // ep + 1) * ep)
            )
            assert dist.get_process_group_ranks(root["edp"].get_group()) == list(range(rank % ep, 4, ep))
            if not check_backward:
                continue

            torch.manual_seed(42)
            reference = _MoE().to(device)
            model = deepcopy(reference)
            model = model.maybe_distribute_model(model, config, root)
            # Different inputs on every rank; the reference evaluates the concatenated global batch.
            inputs = torch.randn(8, 3, device=device)
            weights = torch.rand(8, 2, device=device)
            for empty_receivers in (False, True):
                indices = (
                    torch.zeros(8, 2, dtype=torch.long, device=device)
                    if empty_receivers
                    else torch.arange(16, device=device).reshape(8, 2) % 4
                )
                reference.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                x = inputs[rank * 2 : (rank + 1) * 2].clone().requires_grad_()
                scores = weights[rank * 2 : (rank + 1) * 2].clone().requires_grad_()
                ref_x, ref_scores = inputs.clone().requires_grad_(), weights.clone().requires_grad_()
                expected = reference(ref_x, indices, ref_scores)
                actual = model(x, indices[rank * 2 : (rank + 1) * 2], scores)
                torch.testing.assert_close(actual, expected[rank * 2 : (rank + 1) * 2])
                expected.square().mean().backward()
                actual.square().mean().backward()
                torch.testing.assert_close(x.grad / 4, ref_x.grad[rank * 2 : (rank + 1) * 2])
                torch.testing.assert_close(scores.grad / 4, ref_scores.grad[rank * 2 : (rank + 1) * 2])
                for (_, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
                    grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
                    torch.testing.assert_close(grad, ref_param.grad)
                # Use non-foreach SGD so optimizer groups do not need to mix different meshes.
                torch.optim.SGD(model.parameters(), lr=0.01, foreach=False).step()
                torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False).step()
                for (_, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
                    full = param.full_tensor() if isinstance(param, DTensor) else param
                    torch.testing.assert_close(full, ref_param)
    finally:
        dist.destroy_process_group()


def _trainer_worker(rank, rendezvous):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Trainer, TrainingArguments

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4", LOCAL_WORLD_SIZE="4")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        torch.manual_seed(42)
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=8,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
        )
        reference = Qwen3MoeForCausalLM(config)
        source = rendezvous + "_model"
        if rank == 0:
            reference.save_pretrained(source)
        dist.barrier()
        with patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
            model = Qwen3MoeForCausalLM.from_pretrained(
                source,
                distributed_config=DistributedConfig(fsdp_size=4, ep_size=2),
                experts_implementation="eager",
            )
        model.train()
        reference.train()
        inputs = torch.arange(64).reshape(8, 8) % 32
        labels = inputs.clone()
        labels[::2, 1:3] = -100
        dataset = [{"input_ids": row, "labels": label} for row, label in zip(inputs, labels)]
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=rendezvous + "_output",
                use_cpu=True,
                max_steps=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                per_device_eval_batch_size=2,
                learning_rate=0.01,
                max_grad_norm=0,
                lr_scheduler_type="constant",
                save_strategy="no",
                report_to="none",
                disable_tqdm=True,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            optimizers=(torch.optim.SGD(model.parameters(), lr=0.01, foreach=False), None),
        )
        trainer.train()
        reference(inputs, labels=labels).loss.backward()
        torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False).step()
        for (name, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
            full = param.full_tensor() if isinstance(param, DTensor) else param
            torch.testing.assert_close(full, ref_param, atol=1e-6, rtol=1e-4, msg=lambda msg: f"{name}: {msg}")
        assert torch.isfinite(torch.tensor(trainer.evaluate()["eval_loss"]))
    finally:
        dist.destroy_process_group()


@require_torch
@require_torch_greater_or_equal("2.7")
class ExpertMeshTest(unittest.TestCase):
    def test_mesh_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_mesh_worker, args=(os.path.join(directory, "init"), False, "cpu"), nprocs=4, join=True)

    def test_dispatch_backward_cpu(self):
        if not is_torch_greater_or_equal("2.13"):
            self.skipTest("CPU FSDP coverage requires torch>=2.13")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_mesh_worker, args=(os.path.join(directory, "init"), True, "cpu"), nprocs=4, join=True)

    def test_dispatch_trainer_cpu(self):
        if not is_torch_greater_or_equal("2.13"):
            self.skipTest("CPU FSDP coverage requires torch>=2.13")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_trainer_worker, args=(os.path.join(directory, "init"),), nprocs=4, join=True)

    @require_torch_accelerator
    def test_dispatch_backward(self):
        if torch_device != "cuda" or torch.cuda.device_count() < 4:
            self.skipTest("Requires four CUDA devices")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_mesh_worker, args=(os.path.join(directory, "init"), True, "cuda"), nprocs=4, join=True)
