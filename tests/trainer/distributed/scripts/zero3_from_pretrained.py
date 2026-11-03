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

"""Loads a tiny MoE checkpoint (with weight conversions and a persistent buffer) under ZeRO-3, and checks that only
rank 0 reads checkpoint values while every rank ends up with the right weights."""

import argparse
import os
from unittest.mock import patch

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, modeling_utils
from transformers.integrations.deepspeed import HfDeepSpeedConfig


class Qwen3MoeWithBuffer(Qwen3MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("scale", torch.ones(4))


def main(output_dir):
    get_accelerator().set_device(int(os.environ["LOCAL_RANK"]))
    deepspeed.init_distributed()
    rank = dist.get_rank()

    config = Qwen3MoeConfig(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
    )
    torch.manual_seed(0)
    model = Qwen3MoeWithBuffer(config)
    model.scale.fill_(42.0)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    if rank == 0:
        model.save_pretrained(output_dir)
    dist.barrier()

    # `from_pretrained` only sees the DeepSpeed config while this object is alive
    hf_ds_config = HfDeepSpeedConfig({"train_batch_size": dist.get_world_size(), "zero_optimization": {"stage": 3}})
    assert hf_ds_config.is_zero3()
    with patch.object(modeling_utils, "load_state_dict", wraps=modeling_utils.load_state_dict) as load_state_dict:
        model, loading_info = Qwen3MoeWithBuffer.from_pretrained(output_dir, output_loading_info=True)

    map_locations = {call.kwargs["map_location"] for call in load_state_dict.call_args_list}
    assert map_locations == {"cpu" if rank == 0 else "meta"}, f"rank {rank} loaded with {map_locations}"
    assert not loading_info["missing_keys"], loading_info["missing_keys"]
    with deepspeed.zero.GatheredParameters(list(model.parameters())):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[name], msg=f"rank {rank}: {name} was not loaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args().output_dir)
