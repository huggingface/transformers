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
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parameterized import parameterized

from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, set_seed
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.distributed.pipeline_parallel import PPMissingLayer, apply_pipeline_parallelism
from transformers.testing_utils import TestCasePlus, is_pipeline_parallel_test, require_torch_greater_or_equal

class MockDeviceMesh:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank

    def size(self):
        return self.world_size

    def get_local_rank(self):
        return self.rank

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", 0))
        return s.getsockname()[1]

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

def _verify_pp_split(model, pp_rank, pp_size):
    base_model = model.model
    num_layers = len(base_model.layers)
    layers_per_rank = num_layers // pp_size
    start_layer = pp_rank * layers_per_rank
    end_layer = num_layers if pp_rank == pp_size - 1 else start_layer + layers_per_rank

    assert model._pp_rank == pp_rank
    assert model._pp_size == pp_size

    assert (pp_rank == 0) == (not isinstance(base_model.embed_tokens, PPMissingLayer))
    assert (pp_rank == pp_size - 1) == (not isinstance(base_model.norm, PPMissingLayer))
    assert (pp_rank == pp_size - 1) == (not isinstance(model.lm_head, PPMissingLayer))

    for layer_idx, layer in enumerate(base_model.layers):
        is_local = start_layer <= layer_idx < end_layer
        assert is_local == (not isinstance(layer, PPMissingLayer)), (
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

def _pp_weight_loading(rank, config_dict, pp_size, port):
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
            tie_word_embeddings=False, #TODO(3outeille): should test with tie_word_embeddings=True later
            torch_dtype=torch.float32,
        )
        dist.barrier()

        # Check if the model is split correctly
        _verify_pp_split(pp_model, pp_model._pp_rank, pp_size)

        # Check that each local weight matches the reference.
        local_names = set()
        for name, param in pp_model.named_parameters():
            local_names.add(name)
            torch.testing.assert_close(
                param.detach().cpu(),
                ref_state[name],
                rtol=0,
                atol=0,
                msg=f"weight mismatch for {name} on rank {rank}",
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
class TestPipelineParallelSplit(TestCasePlus):
    @parameterized.expand([(pp_size,) for pp_size in [2]])
    def test_pp_split(self, pp_size):
        config = _tiny_qwen2_config(num_hidden_layers=12)
        for pp_rank in range(pp_size):
            model = Qwen2ForCausalLM(config)
            model = apply_pipeline_parallelism(model, MockDeviceMesh(pp_size, pp_rank))
            _verify_pp_split(model, pp_rank, pp_size)

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

