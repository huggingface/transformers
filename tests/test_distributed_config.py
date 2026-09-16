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
from datetime import timedelta
from unittest.mock import patch

from transformers.distributed import DistributedConfig
from transformers.testing_utils import require_torch, require_torch_greater_or_equal
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    from transformers.distributed.mixin import DistributedMixin
    from transformers.distributed.utils import initialize_distributed_mesh


class DistributedConfigTest(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        for kwargs in ({}, {"tp_size": 4}, {"fsdp_size": 4}, {"pp_size": 4}, {"tp_size": 2, "fsdp_size": 2}):
            with self.subTest(kwargs=kwargs):
                config = DistributedConfig(**kwargs)
                self.assertEqual(config.ep_size, 1)
                self.assertFalse(config.enable_expert_parallel)
                self.assertEqual(config.efsdp_size, config.fsdp_size * config.tp_size)
                self.assertEqual(DistributedConfig.from_dict(config.to_dict()), config)

    def test_legacy_and_explicit_ep_sizes(self):
        legacy = DistributedConfig(tp_size=4, fsdp_size=2, enable_expert_parallel=True)
        explicit = DistributedConfig(tp_size=4, fsdp_size=2, ep_size=4)
        self.assertEqual(legacy, explicit)
        self.assertEqual(DistributedConfig.from_dict(explicit.to_dict()), explicit)
        for ep_size in (1, 4, 8):
            with self.subTest(ep_size=ep_size):
                config = DistributedConfig(tp_size=4, fsdp_size=2, ep_size=ep_size, enable_expert_parallel=True)
                self.assertEqual(config.ep_size, ep_size)
                self.assertEqual(config.enable_expert_parallel, ep_size > 1)
                self.assertEqual((config.tp_size, config.fsdp_size), (4, 2))

    def test_inferred_tp_size(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            config = DistributedConfig(tp_plan="auto", fsdp_size=2, enable_expert_parallel=True)
        self.assertEqual((config.tp_size, config.ep_size, config.efsdp_size), (4, 4, 2))

    def test_expert_mesh_sizes(self):
        for fsdp, tp, ep, efsdp in ((8, 1, 4, 2), (2, 2, 4, 1), (4, 2, 4, 2), (2, 2, 2, 2), (1, 4, 4, 1)):
            with self.subTest(fsdp=fsdp, tp=tp, ep=ep):
                config = DistributedConfig(fsdp_size=fsdp, tp_size=tp, ep_size=ep)
                self.assertEqual(config.efsdp_size, efsdp)
                self.assertEqual(DistributedConfig.from_dict(config.to_dict()), config)

    def test_invalid_sizes(self):
        for name in ("tp_size", "fsdp_size", "pp_size", "ep_size"):
            for value in (0, -1):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "must be >= 1"):
                    DistributedConfig(**{name: value})
        for kwargs, message in (
            ({"tp_size": 4, "ep_size": 2}, "multiple"),
            ({"fsdp_size": 4, "ep_size": 3}, "must divide"),
            ({"fsdp_size": 2, "pp_size": 2}, "pipeline parallelism"),
            ({"ep_size": 2}, "must divide"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                DistributedConfig(**kwargs)


@require_torch
class DistributedMeshValidationTest(unittest.TestCase):
    def test_disabled_mesh_does_not_initialize_distributed(self):
        with patch("transformers.distributed.utils._ensure_torch_distributed") as initialize:
            self.assertEqual(initialize_distributed_mesh(DistributedConfig()), (None, None))
            config, device_map, meshes = DistributedMixin.prepare_distribute_model({}, device_map="cpu")
        self.assertEqual(config, DistributedConfig())
        self.assertEqual(device_map, "cpu")
        self.assertIsNone(meshes)
        initialize.assert_not_called()

    def test_model_loading_rejects_unsupported_ep_layout_before_initialization(self):
        with patch("transformers.distributed.mixin.initialize_distributed_mesh") as initialize:
            with self.assertRaisesRegex(ValueError, "ep_size=tp_size"):
                DistributedMixin.prepare_distribute_model(DistributedConfig(fsdp_size=4, ep_size=2))
        initialize.assert_not_called()

    def test_world_size_mismatch(self):
        with (
            patch("transformers.distributed.utils._ensure_torch_distributed"),
            patch("torch._C._get_accelerator", return_value=torch.device("cpu")),
            patch("torch.distributed.get_world_size", return_value=2),
            self.assertRaisesRegex(RuntimeError, "requires 4 processes"),
        ):
            initialize_distributed_mesh(DistributedConfig(tp_size=4))


def _mesh_worker(rank, rendezvous):
    world_size = 4
    dist.init_process_group(
        "gloo",
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
        ]
        configs += [DistributedConfig(fsdp_size=4, ep_size=ep) for ep in (2, 4)]
        configs += [DistributedConfig(fsdp_size=2, tp_size=2, ep_size=ep) for ep in (2, 4)]
        configs += [DistributedConfig(fsdp_size=1, tp_size=4, ep_size=4)]
        for config in configs:
            with patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
                _, meshes = initialize_distributed_mesh(config)
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
            tp_start = rank // config.tp_size * config.tp_size
            assert dist.get_process_group_ranks(meshes.get_mesh("tp").get_group()) == list(
                range(tp_start, tp_start + config.tp_size)
            )
            assert dist.get_process_group_ranks(meshes.get_mesh("fsdp").get_group()) == list(
                range(stage_start + rank % config.tp_size, stage_start + stage_size, config.tp_size)
            )
            assert dist.get_process_group_ranks(meshes.get_mesh("pp").get_group()) == list(
                range(rank % stage_size, world_size, stage_size)
            )
            assert meshes.get_mesh("ep").get_group() is meshes.get_mesh("ep").get_group()
    finally:
        dist.destroy_process_group()


def _dense_load_worker(rank, rendezvous):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
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


@require_torch
@require_torch_greater_or_equal("2.5")
class DistributedMeshTest(unittest.TestCase):
    def test_mesh_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_mesh_worker, args=(os.path.join(directory, "init"),), nprocs=4, join=True)

    def test_dense_load_generate_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_dense_load_worker, args=(os.path.join(directory, "init"),), nprocs=2, join=True)
