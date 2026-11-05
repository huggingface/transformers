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

"""Two-rank ZeRO-3 checkpoint loading, without downloading model weights."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator
from safetensors.torch import save_file

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, modeling_utils
from transformers.integrations import deepspeed as ds_integration
from transformers.integrations.deepspeed import HfDeepSpeedConfig, unset_hf_deepspeed_config


class BufferedQwen3Moe(Qwen3MoeForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("scale", torch.zeros(2, 3).t())
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))


def main(output_dir):
    accelerator = get_accelerator()
    accelerator.set_device(int(os.environ["LOCAL_RANK"]))
    deepspeed.init_distributed(dist_backend=accelerator.communication_backend_name())
    rank = dist.get_rank()
    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
    )
    reference = BufferedQwen3Moe(config)
    reference.scale.copy_(torch.arange(6).reshape(2, 3).t() + 0.25)
    reference.count.fill_(7)
    expected = {k: v.clone() for k, v in reference.state_dict().items()}
    source = {}
    for key, value in expected.items():
        if key.endswith("experts.gate_up_proj"):
            prefix = key.removesuffix("gate_up_proj")
            for i, expert in enumerate(value):
                gate, up = expert.chunk(2, dim=0)
                source[f"{prefix}{i}.gate_proj.weight"] = gate.contiguous()
                source[f"{prefix}{i}.up_proj.weight"] = up.contiguous()
        elif key.endswith("experts.down_proj"):
            prefix = key.removesuffix("down_proj")
            for i, expert in enumerate(value):
                source[f"{prefix}{i}.down_proj.weight"] = expert.contiguous()
        else:
            source["checkpoint_scale" if key == "scale" else key] = value.contiguous()
    del reference

    checkpoint = Path(output_dir)
    if rank == 0:
        checkpoint.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(checkpoint)
        weight_map = {}
        # Split each layer's experts across files, so loading must preserve cross-file fusion.
        for i in range(3):
            shard = dict(list(source.items())[i::3])
            filename = f"model-{i:05d}.safetensors"
            save_file(shard, checkpoint / filename, metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(shard, filename))
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
        torch.save(source, checkpoint / "pytorch_model.bin")
    dist.barrier()

    ds_config = {"train_batch_size": dist.get_world_size(), "zero_optimization": {"stage": 3}}
    dschf = HfDeepSpeedConfig(ds_config)
    original_read = modeling_utils.load_state_dict
    original_convert = ds_integration._apply_weight_conversions_to_state_dict
    try:
        for mode in ("safetensors", "safetensors_no_mmap", "bin", "state_dict"):
            reads, conversions = [], []

            def read(*args, **kwargs):
                if kwargs.get("map_location", "cpu") != "meta":
                    assert rank == 0, "Nonzero rank read checkpoint payload"
                    reads.append(str(args[0]))
                return original_read(*args, **kwargs)

            def convert(*args, **kwargs):
                assert rank == 0, "Nonzero rank converted checkpoint weights"
                conversions.append(True)
                return original_convert(*args, **kwargs)

            kwargs = {"use_safetensors": mode.startswith("safetensors")}
            if mode == "safetensors_no_mmap":
                kwargs["disable_mmap"] = True
            model_path = str(checkpoint)
            if mode == "state_dict":
                model_path = None
                kwargs = {"state_dict": source.copy() if rank == 0 else {}}
            with (
                patch.object(modeling_utils, "load_state_dict", read),
                patch.object(ds_integration, "_apply_weight_conversions_to_state_dict", convert),
            ):
                model, info = BufferedQwen3Moe.from_pretrained(
                    model_path,
                    config=config,
                    dtype=torch.float32,
                    output_loading_info=True,
                    key_mapping={"checkpoint_scale": "scale"},
                    **kwargs,
                )
            assert not info["missing_keys"], info
            assert not info["error_msgs"], info
            assert len(conversions) == (1 if rank == 0 else 0)
            if rank == 0 and mode != "state_dict":
                assert reads
            with deepspeed.zero.GatheredParameters(list(model.parameters())):
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
            print(
                f"rank={rank} {mode}: values match; payload_reads={len(reads)}, conversions={len(conversions)}",
                flush=True,
            )
            del model

        # A rank-0 conversion failure must reach receivers instead of leaving them in a collective.
        with patch.object(
            ds_integration, "_apply_weight_conversions_to_state_dict", side_effect=ValueError("bad fusion")
        ):
            try:
                BufferedQwen3Moe.from_pretrained(
                    None,
                    config=config,
                    state_dict=source.copy() if rank == 0 else {},
                    dtype=torch.float32,
                )
            except RuntimeError as error:
                assert "bad fusion" in str(error), error
            else:
                raise AssertionError("Expected a checkpoint conversion failure on every rank")
        print(f"rank={rank} conversion error propagated", flush=True)

        def fail_read(*args, **kwargs):
            if kwargs.get("map_location", "cpu") != "meta":
                raise OSError("checkpoint read failed")
            return original_read(*args, **kwargs)

        with patch.object(modeling_utils, "load_state_dict", fail_read):
            try:
                BufferedQwen3Moe.from_pretrained(checkpoint, config=config, dtype=torch.float32)
            except RuntimeError as error:
                assert "checkpoint read failed" in str(error), error
            else:
                raise AssertionError("Expected a checkpoint read failure on every rank")
        print(f"rank={rank} read error propagated", flush=True)
    finally:
        del dschf
        unset_hf_deepspeed_config()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args().output_dir)
