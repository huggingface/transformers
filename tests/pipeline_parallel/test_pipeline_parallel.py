# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import socket
import tempfile
import unittest
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parameterized import parameterized

from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, set_seed
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.distributed.pipeline_parallel import PipelineIdentityLayer, PipelineStage, apply_pipeline_parallelism
from transformers.modeling_utils import LoadStateDictConfig
from transformers.core_model_loading import convert_and_load_state_dict_in_model
from transformers.testing_utils import TestCasePlus, is_pipeline_parallel_test, require_torch_greater_or_equal
from transformers.utils.loading_report import log_state_dict_report


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def init_process_group(rank, pp_size, port):
    os.environ.update(
        {
            "WORLD_SIZE": str(pp_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    dist.init_process_group(backend="gloo", rank=rank, world_size=pp_size)
    return dist.init_device_mesh("cpu", (pp_size,), mesh_dim_names=("pp",))


@contextmanager
def _shared_model_dir(rank):
    if rank == 0:
        tmpdir = tempfile.TemporaryDirectory()
        path = [tmpdir.name]
    else:
        tmpdir = None
        path = [None]
    dist.broadcast_object_list(path, src=0)
    try:
        yield path[0]
    finally:
        if rank == 0 and tmpdir is not None:
            tmpdir.cleanup()


def _verify_pp_split(model):
    stage = PipelineStage.from_device_mesh(model._device_mesh)
    pp_rank = stage.pp_rank
    pp_size = stage.pp_size
    base_model = model.model
    num_layers = len(base_model.layers)
    start_layer, end_layer = stage.layer_range_for_rank(pp_rank, num_layers)

    assert (pp_rank == 0) == (not isinstance(base_model.embed_tokens, PipelineIdentityLayer))
    assert (pp_rank == pp_size - 1) == (not isinstance(base_model.norm, PipelineIdentityLayer))
    assert (pp_rank == pp_size - 1) == (not isinstance(model.lm_head, PipelineIdentityLayer))

    for layer_idx, layer in enumerate(base_model.layers):
        is_local = start_layer <= layer_idx < end_layer
        assert is_local == (not isinstance(layer, PipelineIdentityLayer)), (
            f"layer {layer_idx} on rank {pp_rank}: expected {'real' if is_local else 'stub'}"
        )

    for name in model.state_dict():
        if name.startswith("model.layers."):
            layer_idx = int(name.split(".")[2])
            assert start_layer <= layer_idx < end_layer, f"{name} should not exist on rank {pp_rank}"
        elif name.startswith("model.embed_tokens."):
            assert pp_rank == 0, f"{name} should only exist on rank 0"
        elif name.startswith(("model.norm.", "lm_head.")):
            assert pp_rank == pp_size - 1, f"{name} should only exist on last rank"


def _pp_split(rank, config_dict, pp_size, port):
    pp_mesh = init_process_group(rank, pp_size, port)
    config = Qwen2Config.from_dict(config_dict)
    model = Qwen2ForCausalLM(config)
    model = apply_pipeline_parallelism(model, pp_mesh)
    model._device_mesh = pp_mesh
    _verify_pp_split(model)
    dist.barrier()
    dist.destroy_process_group()


def _pp_load_report(rank, config_dict, pp_size, port):
    pp_mesh = init_process_group(rank, pp_size, port)
    config = Qwen2Config.from_dict(config_dict)
    model = Qwen2ForCausalLM(config)
    model = apply_pipeline_parallelism(model, pp_mesh)
    model._device_mesh = pp_mesh

    if rank == 0:
        full_model = Qwen2ForCausalLM(config)
        load_config = LoadStateDictConfig()
        loading_info, _ = convert_and_load_state_dict_in_model(model, full_model.state_dict(), load_config)

        report = loading_info.create_loading_report(model)
        assert report is not None
        assert "OWNED" in report
        assert "SKIPPED" in report
        assert "PP rank 0" in report
        assert "PP rank 1" in report
        assert "model.embed_tokens.weight" in report
        assert "lm_head.weight" in report
        assert loading_info.unexpected_keys == set()

        with unittest.TestCase().assertLogs("transformers.utils.loading_report", level="WARNING") as logs:
            log_state_dict_report(model, "/tmp/pp-test", True, loading_info)

        log_text = "\n".join(logs.output)
        assert "LOAD REPORT" in log_text
        assert "OWNED" in log_text
        assert "SKIPPED" in log_text
        assert "PP rank 0" in log_text
        assert "PP rank 1" in log_text
        assert "model.layers.{0, 1, 2, 3, 4, 5}" in log_text
        assert "model.layers.{6, 7, 8, 9, 10, 11}" in log_text

    dist.barrier()
    dist.destroy_process_group()


def _pp_weight_loading(rank, config_dict, pp_size, port):
    init_process_group(rank, pp_size, port)
    config = Qwen2Config.from_dict(config_dict)

    with _shared_model_dir(rank) as model_dir:
        # Rank 0 saves a full (non-PP) checkpoint; all ranks wait, then read from it.
        if rank == 0:
            set_seed(42)
            tmp_model = Qwen2ForCausalLM(config)
            tmp_model.to(torch.float32).save_pretrained(model_dir)
            del tmp_model
        dist.barrier()

        # Reference: load the full model on CPU for later comparison.
        ref_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="cpu")
        ref_state = {name: param.detach().cpu() for name, param in ref_model.state_dict().items()}
        del ref_model

        # Under test: load with PP sharding (tie_word_embeddings=False: last rank owns lm_head).
        pp_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            distributed_config=DistributedConfig(pp_size=pp_size),
            tie_word_embeddings=False,  # TODO(3outeille): should test with tie_word_embeddings=True later
            torch_dtype=torch.float32,
        )
        dist.barrier()

        # Check if the model is split correctly
        _verify_pp_split(pp_model)

        # Check that each local weight matches the reference.
        for name, param in pp_model.named_parameters():
            torch.testing.assert_close(
                param.detach().cpu(),
                ref_state[name],
                rtol=0,
                atol=0,
                msg=f"weight mismatch for {name} on rank {rank}",
            )

    dist.barrier()
    dist.destroy_process_group()


def _pp_generation(rank, config_dict, pp_size, port, max_new_tokens):
    init_process_group(rank, pp_size, port)
    config = Qwen2Config.from_dict(config_dict)

    with _shared_model_dir(rank) as model_dir:
        if rank == 0:
            set_seed(42)
            tmp_model = Qwen2ForCausalLM(config)
            tmp_model.to(torch.float32).save_pretrained(model_dir)
            del tmp_model
        dist.barrier()

        pp_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            distributed_config=DistributedConfig(pp_size=pp_size),
            tie_word_embeddings=False,
            torch_dtype=torch.float32,
        )
        pp_model.eval()

        ref_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
        ref_model = ref_model.to(pp_model.device)
        ref_model.eval()
        dist.barrier()

        set_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "output_logits": True,
            "return_dict_in_generate": True,
            "use_cache": True,
        }

        with torch.no_grad():
            output_pp = pp_model.generate(input_ids.to(pp_model.device), **generation_kwargs)
            output_ref = ref_model.generate(input_ids.to(pp_model.device), **generation_kwargs)

        logits_pp = torch.stack(output_pp.logits).cpu()
        logits_ref = torch.stack(output_ref.logits).cpu()

        torch.testing.assert_close(
            logits_pp,
            logits_ref,
            rtol=0,
            atol=0,
            msg=f"PP generation logits differ from reference on rank {rank}",
        )
        assert torch.equal(output_pp.sequences, output_ref.sequences), (
            f"PP generated different token sequences than reference on rank {rank}. "
            f"PP: {output_pp.sequences.tolist()} | Ref: {output_ref.sequences.tolist()}"
        )

    dist.barrier()
    dist.destroy_process_group()


def _tiny_qwen2_config(num_hidden_layers):
    return Qwen2Config(
        num_hidden_layers=num_hidden_layers,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )


@is_pipeline_parallel_test
@require_torch_greater_or_equal("2.5")
class TestPipelineParallelLoadReport(TestCasePlus):
    def test_pp_loading_report_table(self):
        config = _tiny_qwen2_config(num_hidden_layers=12)

        mp.spawn(
            _pp_load_report,
            args=(config.to_dict(), 2, _find_free_port()),
            nprocs=2,
            join=True,
        )


@is_pipeline_parallel_test
@require_torch_greater_or_equal("2.5")
class TestPipelineParallelSplit(TestCasePlus):
    @parameterized.expand([(pp_size,) for pp_size in [2]])
    def test_pp_split(self, pp_size):
        config = _tiny_qwen2_config(num_hidden_layers=12)

        mp.spawn(
            _pp_split,
            args=(config.to_dict(), pp_size, _find_free_port()),
            nprocs=pp_size,
            join=True,
        )


@is_pipeline_parallel_test
@require_torch_greater_or_equal("2.5")
class TestPipelineParallelWeightLoading(TestCasePlus):
    @parameterized.expand([(pp_size,) for pp_size in [2]])
    def test_pp_weight_loading(self, pp_size):
        config = _tiny_qwen2_config(num_hidden_layers=12)

        mp.spawn(
            _pp_weight_loading,
            args=(config.to_dict(), pp_size, _find_free_port()),
            nprocs=pp_size,
            join=True,
        )


@is_pipeline_parallel_test
@require_torch_greater_or_equal("2.5")
class TestPipelineParallelGeneration(TestCasePlus):
    @parameterized.expand([(pp_size,) for pp_size in [2]])
    def test_pp_generation(self, pp_size):
        config = _tiny_qwen2_config(num_hidden_layers=12)
        max_new_tokens = 5

        mp.spawn(
            _pp_generation,
            args=(config.to_dict(), pp_size, _find_free_port(), max_new_tokens),
            nprocs=pp_size,
            join=True,
        )
