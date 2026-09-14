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
from collections import OrderedDict
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
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, apply_tensor_parallelism
from transformers.distributed.utils import initialize_distributed_mesh
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
            self.assertEqual(config.experts_dispatch, "all-reduce")
            self.assertEqual(config.efsdp_size, config.fsdp_size * config.tp_size)

    def test_dispatch_topology_and_round_trip(self):
        config = DistributedConfig(fsdp_size=8, ep_size=4, ep_plan={"experts": "ep_dispatch_experts"})
        self.assertEqual((config.tp_size, config.fsdp_size, config.ep_size, config.efsdp_size), (1, 8, 4, 2))
        self.assertTrue(config.enable_expert_parallel)
        self.assertEqual(config.experts_dispatch, "all-to-all")
        with warnings.catch_warnings(record=True) as caught:
            self.assertEqual(DistributedConfig.from_dict(config.to_dict()), config)
        self.assertFalse(caught)

    def test_ep_does_not_infer_fsdp_size(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            with self.assertRaisesRegex(ValueError, "must divide"):
                DistributedConfig(ep_size=4, ep_plan={"experts": "ep_dispatch_experts"})
            self.assertEqual(DistributedConfig(tp_size=4, ep_size=4).fsdp_size, 1)

    def test_legacy_dispatch_preserves_parallel_sizes(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = DistributedConfig(
                tp_size=4, fsdp_size=2, enable_expert_parallel=True, ep_plan={"experts": "ep_dispatch_experts"}
            )
        self.assertTrue(any(w.category is FutureWarning for w in caught))
        self.assertEqual(
            legacy, DistributedConfig(tp_size=4, fsdp_size=2, ep_size=4, ep_plan={"experts": "ep_dispatch_experts"})
        )

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
                tp_plan="auto", fsdp_size=2, enable_expert_parallel=True, ep_plan={"experts": "ep_dispatch_experts"}
            )
        self.assertTrue(any(w.category is FutureWarning for w in caught))
        self.assertEqual((config.tp_size, config.fsdp_size, config.ep_size), (4, 2, 4))

    def test_explicit_ep_size_takes_precedence_over_legacy_flag(self):
        for ep_size, dispatcher in ((1, "all-reduce"), (4, "all-reduce"), (8, "all-to-all")):
            with self.subTest(ep_size=ep_size), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = DistributedConfig(
                    tp_size=4,
                    fsdp_size=2,
                    ep_size=ep_size,
                    enable_expert_parallel=True,
                    ep_plan={"experts": "ep_dispatch_experts"} if dispatcher == "all-to-all" else None,
                )
            self.assertFalse(caught)
            self.assertEqual((config.tp_size, config.fsdp_size, config.ep_size), (4, 2, ep_size))
            self.assertEqual(config.enable_expert_parallel, ep_size > 1)
            self.assertEqual(config.experts_dispatch, dispatcher)

    def test_invalid_topologies(self):
        cases = [
            ({"fsdp_size": 4, "ep_size": 3, "ep_plan": {"experts": "ep_dispatch_experts"}}, "must divide"),
            (
                {"tp_size": 4, "fsdp_size": 2, "ep_size": 2, "ep_plan": {"experts": "ep_dispatch_experts"}},
                "multiple of",
            ),
            ({"fsdp_size": 8, "ep_size": 4}, "identical tokens"),
            ({"fsdp_size": 4, "ep_size": 4}, "identical tokens"),
            (
                {"ep_size": 2, "fsdp_size": 2, "pp_size": 2, "ep_plan": {"experts": "ep_dispatch_experts"}},
                "pipeline parallelism",
            ),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                DistributedConfig(**kwargs)
        for name in ("tp_size", "fsdp_size", "pp_size", "ep_size"):
            for value in (0, -1):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "must be >= 1"):
                    DistributedConfig(**{name: value})

    def test_folded_tp_dispatch(self):
        for fsdp, tp, ep, efsdp in ((2, 2, 4, 1), (4, 2, 4, 2), (2, 2, 2, 2), (1, 4, 4, 1)):
            with self.subTest(fsdp=fsdp, tp=tp, ep=ep):
                config = DistributedConfig(
                    fsdp_size=fsdp, tp_size=tp, ep_size=ep, ep_plan={"experts": "ep_dispatch_experts"}
                )
                self.assertEqual(config.efsdp_size, efsdp)
                self.assertEqual(DistributedConfig.from_dict(config.to_dict()), config)

    def test_efsdp_size_is_independent_of_dispatcher(self):
        for dispatcher in ("all-reduce", "all-to-all"):
            config = DistributedConfig(
                tp_size=2,
                fsdp_size=4,
                ep_size=2,
                ep_plan={"experts": "ep_dispatch_experts"} if dispatcher == "all-to-all" else None,
            )
            self.assertEqual(config.efsdp_size, 4)

    def test_disabled_mesh_does_not_initialize_distributed(self):
        with patch("transformers.distributed.utils._ensure_torch_distributed") as initialize:
            self.assertEqual(initialize_distributed_mesh(DistributedConfig()), (None, None))
        initialize.assert_not_called()


class _Experts(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.randn(num_experts, 3, 3))

    def forward(self, hidden_states, indices, weights):
        if hidden_states.size(0) == 0:
            # Eager experts skip every matmul when no expert receives a token.
            return torch.zeros_like(hidden_states)
        selected = self.weight[indices]
        outputs = torch.einsum("ni,nkoi->nko", hidden_states, selected)
        return (outputs * weights.unsqueeze(-1)).sum(1)


class _MoE(DistributedMixin, nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.config = SimpleNamespace()
        self.dense = nn.Sequential(OrderedDict(up=nn.Linear(3, 4), down=nn.Linear(4, 3)))
        self.experts = _Experts(num_experts)
        self._tp_plan = {
            "dense.up": "colwise",
            "dense.down": "rowwise",
            "experts.weight": "packed_colwise",
            "experts": "moe_tp_experts",
        }
        self._ep_plan = {"experts.weight": "grouped_gemm", "experts": "moe_tp_experts"}
        self._fsdp_plan = {"dense": "free_full_weight"}

    def forward(self, x, indices, weights):
        return self.experts(self.dense(x), indices, weights)


class DistributedPlanTest(unittest.TestCase):
    def test_dispatch_uses_distinct_tp_and_ep_meshes(self):
        model = _MoE()
        tp_mesh, ep_mesh = object(), object()
        plan = {"experts.weight": "grouped_gemm", "experts": "ep_dispatch_experts"}
        with (
            patch.object(ALL_PARALLEL_STYLES["grouped_gemm"], "validate_param") as validate,
            patch.object(ALL_PARALLEL_STYLES["grouped_gemm"], "shard_param") as shard,
            patch.object(ALL_PARALLEL_STYLES["ep_dispatch_experts"], "install_forward") as install,
        ):
            apply_tensor_parallelism(model, tp_mesh, plan, ep_mesh=ep_mesh)
        validate.assert_called_once_with(model.experts, "weight", ep_mesh, parameter_name="experts.weight")
        shard.assert_called_once_with(model.experts, "weight", ep_mesh)
        install.assert_called_once_with(model.experts, ep_mesh=ep_mesh, tp_mesh=tp_mesh)

    def test_tp_override_preserves_defaults(self):
        model = _MoE()
        original_tp_plan = dict(model.tp_plan)
        override = {"dense.up": "colwise_rep"}
        config = DistributedConfig(tp_size=2, ep_size=2, tp_plan=override)
        with patch("transformers.distributed.tensor_parallel.apply_tensor_parallelism", return_value=model) as apply:
            model.maybe_distribute_model(model, config, SimpleNamespace(get_mesh=lambda dims: dims))
        self.assertEqual(model.tp_plan, original_tp_plan | override)
        self.assertEqual(
            apply.call_args_list[0].args, (model, "tp", {"dense.up": "colwise_rep", "dense.down": "rowwise"})
        )
        self.assertEqual(config.tp_plan, override)

    def test_invalid_override_style_fails_before_sharding(self):
        for plan_name in ("tp_plan", "ep_plan"):
            with self.subTest(plan=plan_name):
                model = _MoE()
                config = DistributedConfig(tp_size=2, ep_size=2, **{plan_name: {"experts.weight": "invalid_style"}})
                with patch("transformers.distributed.tensor_parallel.apply_tensor_parallelism") as apply:
                    with self.assertRaisesRegex(ValueError, "Unsupported tensor parallel styles.*invalid_style"):
                        model.maybe_distribute_model(model, config, SimpleNamespace(get_mesh=lambda dims: dims))
                apply.assert_not_called()

    def test_custom_ep_plan(self):
        for dispatcher, style in (("all-reduce", "moe_tp_experts"), ("all-to-all", "ep_dispatch_experts")):
            with self.subTest(dispatcher=dispatcher):
                model = _MoE()
                model._ep_plan = None
                original_tp_plan = dict(model.tp_plan)
                tp_plan = {"dense.up": "colwise", "dense.down": "rowwise"}
                ep_plan = {"experts.weight": "grouped_gemm", "experts": style}
                config = DistributedConfig(tp_size=2, ep_size=2, tp_plan=tp_plan, ep_plan=ep_plan)
                self.assertEqual(DistributedConfig.from_dict(config.to_dict()), config)
                meshes = SimpleNamespace(get_mesh=lambda dims: dims)
                with (
                    patch(
                        "transformers.distributed.tensor_parallel.apply_tensor_parallelism", return_value=model
                    ) as apply,
                    patch("transformers.distributed.mixin.apply_fully_sharded_data_parallelism", return_value=model),
                ):
                    model.maybe_distribute_model(model, config, meshes)
                self.assertEqual(model.tp_plan, original_tp_plan | tp_plan)
                self.assertEqual(model.ep_plan, ep_plan)
                self.assertEqual(apply.call_count, 2)
                self.assertEqual(apply.call_args_list[0].args, (model, "tp", tp_plan))
                if dispatcher == "all-to-all":
                    apply.assert_called_with(model, "tp", ep_plan, ep_mesh="ep")
                else:
                    apply.assert_called_with(model, "tp", ep_plan)

    def test_ep_override_preserves_default_weight_rules(self):
        model = _MoE()
        original_tp_plan = dict(model.tp_plan)
        original_ep_plan = dict(model.ep_plan)
        override = {"experts": "ep_dispatch_experts"}
        config = DistributedConfig(tp_size=2, ep_size=2, ep_plan=override)
        with (
            patch("transformers.distributed.tensor_parallel.apply_tensor_parallelism", return_value=model) as apply,
            patch("transformers.distributed.mixin.apply_fully_sharded_data_parallelism", return_value=model),
        ):
            model.maybe_distribute_model(model, config, SimpleNamespace(get_mesh=lambda dims: dims))
        self.assertEqual(model.tp_plan, original_tp_plan)
        self.assertEqual(model.ep_plan, original_ep_plan | override)
        self.assertEqual(apply.call_args_list[0].args[2], {"dense.up": "colwise", "dense.down": "rowwise"})
        apply.assert_called_with(model, "tp", {"experts.weight": "grouped_gemm", **override}, ep_mesh="ep")

    def test_auto_ep_plan_preserves_model_plan(self):
        model = _MoE()
        ep_plan = model.ep_plan
        config = DistributedConfig(tp_size=2, ep_size=2, ep_plan="auto")
        with patch("transformers.distributed.tensor_parallel.apply_tensor_parallelism", return_value=model):
            model.maybe_distribute_model(model, config, SimpleNamespace(get_mesh=lambda dims: dims))
        self.assertIs(model.ep_plan, ep_plan)

    def test_tp_plan_is_independent_of_ep_settings(self):
        model = _MoE()
        tp_plan = model._tp_plan
        for dispatcher in ("all-reduce", "all-to-all"):
            model.config.distributed_config = DistributedConfig(
                tp_size=2,
                ep_size=2,
                ep_plan={"experts": "ep_dispatch_experts"} if dispatcher == "all-to-all" else None,
            )
            for ep_plan in ({}, {"experts.weight": "grouped_gemm"}):
                with self.subTest(dispatcher=dispatcher, ep_plan=ep_plan):
                    model._ep_plan = ep_plan
                    self.assertIs(model.tp_plan, tp_plan)

    def test_all_reduce_selects_plan_when_applying_parallelism(self):
        for ep_size in (1, 2):
            with self.subTest(ep_size=ep_size):
                model = _MoE()
                model._ep_plan["router"] = "ep_router"
                tp_plan = model.tp_plan
                mesh = object()
                meshes = SimpleNamespace(get_mesh=lambda dims: mesh)
                config = DistributedConfig(tp_size=2, ep_size=ep_size)
                with patch(
                    "transformers.distributed.tensor_parallel.apply_tensor_parallelism", return_value=model
                ) as apply:
                    model.maybe_distribute_model(model, config, meshes)
                if ep_size > 1:
                    self.assertEqual(apply.call_count, 2)
                    self.assertEqual(
                        apply.call_args_list[0].args, (model, mesh, {"dense.up": "colwise", "dense.down": "rowwise"})
                    )
                    self.assertEqual(apply.call_args_list[1].args, (model, mesh, model.ep_plan))
                else:
                    apply.assert_called_once_with(model, mesh, tp_plan)
                self.assertIs(model.tp_plan, tp_plan)

    def test_disabled_tp_and_ep_do_not_apply_plans(self):
        model = _MoE()
        with patch("transformers.distributed.tensor_parallel.apply_tensor_parallelism") as apply:
            model.maybe_distribute_model(model, DistributedConfig(), SimpleNamespace(get_mesh=lambda dims: dims))
        apply.assert_not_called()

    def test_all_reduce_requires_ep_plan_when_applied(self):
        for ep_plan in (None, {}):
            with self.subTest(ep_plan=ep_plan):
                model = _MoE()
                model._ep_plan = ep_plan
                config = DistributedConfig(tp_size=2, ep_size=2)
                with self.assertRaisesRegex(ValueError, "does not define an expert-parallel plan"):
                    model.maybe_distribute_model(model, config, SimpleNamespace(get_mesh=lambda dims: None))


def _mesh_worker(rank, rendezvous, check_backward, device_type, world_size=4):
    dist.init_process_group(
        "gloo" if device_type == "cpu" else "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        configs = [
            DistributedConfig(fsdp_size=4),
            DistributedConfig(tp_size=4),
            DistributedConfig(pp_size=4),
            DistributedConfig(tp_size=2, fsdp_size=2),
            DistributedConfig(tp_size=2, pp_size=2),
            DistributedConfig(tp_size=2, fsdp_size=2, ep_size=2),
        ]
        configs += [
            DistributedConfig(fsdp_size=4, ep_size=ep, ep_plan={"experts": "ep_dispatch_experts"}) for ep in (2, 4)
        ]
        configs += [
            DistributedConfig(fsdp_size=2, tp_size=2, ep_size=ep, ep_plan={"experts": "ep_dispatch_experts"})
            for ep in (2, 4)
        ]
        configs += [DistributedConfig(fsdp_size=1, tp_size=4, ep_size=4, ep_plan={"experts": "ep_dispatch_experts"})]
        if world_size == 8:
            configs = [
                DistributedConfig(fsdp_size=4, tp_size=2, ep_size=4, ep_plan={"experts": "ep_dispatch_experts"})
            ]
        for config in configs:
            with patch("torch._C._get_accelerator", return_value=torch.device(device_type)):
                device, meshes = initialize_distributed_mesh(config)
            assert meshes.get_mesh(("pp", "fsdp", "tp")).mesh_dim_names == ("pp", "fsdp", "tp")
            for axes in (("pp", "fsdp", "tp"), ("pp", "efsdp", "ep")):
                assert meshes.get_mesh(axes).size() == world_size
                assert meshes.get_mesh(axes).mesh_dim_names == axes
                for name in axes:
                    assert meshes.get_mesh(name).size() == getattr(config, name + "_size")
            assert meshes.get_mesh(("fsdp", "tp")).mesh_dim_names == ("fsdp", "tp")
            assert meshes.get_mesh(("efsdp", "ep")).mesh_dim_names == ("efsdp", "ep")
            for invalid in ("missing", ("tp", "ep"), ("fsdp", "efsdp")):
                try:
                    meshes.get_mesh(invalid)
                except KeyError:
                    pass
                else:
                    raise AssertionError(f"Accepted invalid mesh dimensions: {invalid}")
            stage_size = config.fsdp_size * config.tp_size
            stage_start = rank // stage_size * stage_size
            expert_rank = (rank - stage_start) % config.ep_size
            ep_start = rank // config.ep_size * config.ep_size
            assert dist.get_process_group_ranks(meshes.get_mesh("ep").get_group()) == list(
                range(ep_start, ep_start + config.ep_size)
            )
            assert dist.get_process_group_ranks(meshes.get_mesh("efsdp").get_group()) == list(
                range(stage_start + expert_rank, stage_start + stage_size, config.ep_size)
            )
            if config.experts_dispatch != "all-to-all":
                continue
            ep = config.ep_size
            if not check_backward:
                continue

            torch.manual_seed(42)
            reference = _MoE(num_experts=world_size).to(device)
            model = deepcopy(reference)
            model = model.maybe_distribute_model(model, config, meshes)
            assert model.tp_plan == reference.tp_plan
            assert model.ep_plan == reference.ep_plan | config.ep_plan
            assert model._device_mesh is meshes.get_mesh(("pp", "fsdp", "tp"))
            assert model._mesh_manager is meshes
            expert_mesh = model.experts.weight.device_mesh
            assert dist.get_process_group_ranks(expert_mesh["ep"].get_group()) == list(
                range(rank // ep * ep, (rank // ep + 1) * ep)
            )
            if config.efsdp_size > 1:
                assert dist.get_process_group_ranks(expert_mesh["efsdp"].get_group()) == list(
                    range(rank % ep, world_size, ep)
                )
            # Different inputs per TP group; the reference evaluates the concatenated global batch.
            batch = rank // config.tp_size
            if config.tp_size > 1:
                assert model.dense[0].weight.device_mesh.mesh_dim_names[-1] == "tp"
                assert model.dense[0].weight.placements[-1].is_shard()
            for routing in ("balanced", "empty_receivers", "different_efsdp_receivers"):
                # Uneven TP slices and single-token decoding exercise empty senders as well as receivers.
                tokens_per_batch = {"balanced": 3, "empty_receivers": 1, "different_efsdp_receivers": 4}[routing]
                total_tokens = config.fsdp_size * tokens_per_batch
                inputs = torch.randn(total_tokens, 3, device=device)
                weights = torch.rand(total_tokens, 2, device=device)
                rows = slice(batch * tokens_per_batch, (batch + 1) * tokens_per_batch)
                if routing == "balanced":
                    indices = torch.arange(total_tokens * 2, device=device).reshape(total_tokens, 2) % world_size
                elif routing == "empty_receivers":
                    indices = torch.zeros(total_tokens, 2, dtype=torch.long, device=device)
                else:
                    # One expert replica receives tokens while another replica of that expert is empty.
                    indices = torch.arange(total_tokens, device=device) // (total_tokens // config.efsdp_size)
                    indices = (indices * (world_size // ep)).unsqueeze(-1).expand(-1, 2)
                reference.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                x = inputs[rows].clone().requires_grad_()
                scores = weights[rows].clone().requires_grad_()
                ref_x, ref_scores = inputs.clone().requires_grad_(), weights.clone().requires_grad_()
                expected = reference(ref_x, indices, ref_scores)
                actual = model(x, indices[rows], scores)
                torch.testing.assert_close(actual, expected[rows])
                expected.square().mean().backward()
                actual.square().mean().backward()
                torch.testing.assert_close(x.grad / config.fsdp_size, ref_x.grad[rows])
                torch.testing.assert_close(scores.grad / config.fsdp_size, ref_scores.grad[rows])
                for (_, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
                    grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
                    torch.testing.assert_close(grad, ref_param.grad)
                # Use non-foreach SGD so optimizer groups do not need to mix different meshes.
                torch.optim.SGD(model.parameters(), lr=0.01, foreach=False).step()
                torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False).step()
                for (_, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
                    full = param.full_tensor() if isinstance(param, DTensor) else param
                    torch.testing.assert_close(full, ref_param)
                with torch.no_grad():
                    torch.testing.assert_close(
                        model(inputs[rows], indices[rows], weights[rows]), reference(inputs, indices, weights)[rows]
                    )
    finally:
        dist.destroy_process_group()


def _dense_load_worker(rank, rendezvous):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        reference = Qwen2ForCausalLM(config).eval()
        source = rendezvous + "_model"
        if rank == 0:
            reference.save_pretrained(source)
        dist.barrier()
        inputs = torch.tensor([[1, 2, 3]])
        generation_kwargs = {
            "max_new_tokens": 2,
            "do_sample": False,
            "output_logits": True,
            "return_dict_in_generate": True,
        }
        expected = reference.generate(inputs, **generation_kwargs)
        for distributed_config in (DistributedConfig(tp_size=2), DistributedConfig(pp_size=2)):
            with patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
                model = Qwen2ForCausalLM.from_pretrained(source, distributed_config=distributed_config).eval()
            assert model._device_mesh is model._mesh_manager.get_mesh(("pp", "fsdp", "tp"))
            actual = model.generate(inputs, **generation_kwargs)
            torch.testing.assert_close(actual.sequences, expected.sequences)
            torch.testing.assert_close(torch.stack(actual.logits), torch.stack(expected.logits))
            if distributed_config.tp_size > 1:
                destination = rendezvous + "_saved"
                model.save_pretrained(destination)
                dist.barrier()
                restored = Qwen2ForCausalLM.from_pretrained(destination).eval()
                for name, param in restored.named_parameters():
                    torch.testing.assert_close(param, dict(reference.named_parameters())[name], atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def _trainer_worker(rank, rendezvous, tp_size=1, device_type="cpu"):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Trainer, TrainingArguments

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4", LOCAL_WORLD_SIZE="4")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = torch.device("cpu" if device_type == "cpu" else f"cuda:{rank}")
    dist.init_process_group(
        "gloo" if device_type == "cpu" else "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=4
    )
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
        reference = Qwen3MoeForCausalLM(config).to(device)
        source = rendezvous + "_model"
        if rank == 0:
            reference.save_pretrained(source)
        dist.barrier()
        with patch("torch._C._get_accelerator", return_value=torch.device(device_type)):
            model = Qwen3MoeForCausalLM.from_pretrained(
                source,
                distributed_config=DistributedConfig(
                    fsdp_size=4 // tp_size,
                    tp_size=tp_size,
                    ep_size=4 if tp_size > 1 else 2,
                    ep_plan={"model.layers.*.mlp.experts": "ep_dispatch_experts"},
                ),
                experts_implementation="eager",
            )
        model.train()
        reference.train()
        inputs = torch.arange(64).reshape(8, 8) % 32
        labels = inputs.clone()
        labels[::2, 1:3] = -100
        dataset = [{"input_ids": row, "labels": label} for row, label in zip(inputs, labels)]
        if tp_size > 1:
            # Compare every gradient before the optimizer can hide small scale errors in the update.
            rows = slice((rank // tp_size) * 4, (rank // tp_size + 1) * 4)

            def force_empty_receivers(module, args, output):
                logits, scores, indices = output
                return logits, scores, torch.arange(2, device=indices.device).expand_as(indices)

            for empty_receivers in (False, True):
                handles = []
                if empty_receivers:
                    handles = [
                        m.model.layers[0].mlp.gate.register_forward_hook(force_empty_receivers)
                        for m in (model, reference)
                    ]
                expected = reference(inputs.to(device), labels=labels.to(device))
                actual = model(inputs[rows].to(device), labels=labels[rows].to(device))
                torch.testing.assert_close(actual.logits, expected.logits[rows], atol=1e-6, rtol=1e-4)
                actual.loss.backward()
                expected.loss.backward()
                for (name, param), (_, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
                    grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
                    torch.testing.assert_close(
                        grad, ref_param.grad, atol=1e-7, rtol=1e-4, msg=lambda msg: f"{name}: {msg}"
                    )
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                for handle in handles:
                    handle.remove()
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=rendezvous + "_output",
                use_cpu=device_type == "cpu",
                max_steps=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2 * tp_size,
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
        reference(inputs.to(device), labels=labels.to(device)).loss.backward()
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
    def test_dense_load_generate_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_dense_load_worker, args=(os.path.join(directory, "init"),), nprocs=2, join=True)

    def test_dispatch_forward_rule_comes_from_override(self):
        model = _MoE()
        config = DistributedConfig(fsdp_size=4, ep_size=2, ep_plan={"experts": "ep_dispatch_experts"})
        with (
            patch("transformers.distributed.tensor_parallel.apply_tensor_parallelism", return_value=model) as apply,
            patch("transformers.distributed.mixin.apply_fully_sharded_data_parallelism", return_value=model),
        ):
            model.maybe_distribute_model(model, config, SimpleNamespace(get_mesh=lambda dims: dims))
        apply.assert_called_once_with(
            model, "tp", {"experts.weight": "grouped_gemm", "experts": "ep_dispatch_experts"}, ep_mesh="ep"
        )

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

    def test_tp_dispatch_folded_efsdp_cpu(self):
        if not is_torch_greater_or_equal("2.13"):
            self.skipTest("CPU FSDP coverage requires torch>=2.13")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_mesh_worker, args=(os.path.join(directory, "init"), True, "cpu", 8), nprocs=8, join=True)

    @require_torch_accelerator
    def test_tp_dispatch_trainer(self):
        if torch_device != "cuda" or torch.cuda.device_count() < 4:
            self.skipTest("Requires four CUDA devices")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_trainer_worker, args=(os.path.join(directory, "init"), 2, "cuda"), nprocs=4, join=True)

    @require_torch_accelerator
    def test_dispatch_backward(self):
        if torch_device != "cuda" or torch.cuda.device_count() < 4:
            self.skipTest("Requires four CUDA devices")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_mesh_worker, args=(os.path.join(directory, "init"), True, "cuda"), nprocs=4, join=True)
