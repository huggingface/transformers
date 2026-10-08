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

from parameterized import parameterized

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available, is_torch_distributed_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.distributed import DistributedConfig
    from transformers.distributed.checkpoint import load_model_checkpoint_distributed

    if is_torch_distributed_available():
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )


PARALLEL_CONFIGS = {
    "tp": {"tp_size": 4},
    "fsdp": {"fsdp_size": 4},
    "tp_fsdp": {"tp_size": 2, "fsdp_size": 2},
}

# The distributed format remains sharded. The consolidated format uses the regular save_pretrained path.
CHECKPOINT_MODES = {"distributed": True, "consolidated": False}


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


# Workers stay at module scope so multiprocessing.spawn can pickle them.
def _test_save_and_from_pretrained(rank, directory, source_config, destination_config, distributed_checkpoint):
    with _distributed_context(rank, directory):
        # Both models start from the same seed checkpoint so their weights match on every rank.
        reference = LlamaForCausalLM.from_pretrained(f"{directory}/seed")
        distributed = LlamaForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(**source_config)
        )

        # Distributed checkpoint: `from_pretrained` detects it and loads it with DCP.
        # Consolidated checkpoint: `from_pretrained` loads it and converts it to a distributed model.
        # Either way, it is resharded when the destination configuration differs from the source one.
        distributed.save_pretrained(f"{directory}/saved", distributed_checkpoint=distributed_checkpoint)
        restored = LlamaForCausalLM.from_pretrained(
            f"{directory}/saved", distributed_config=DistributedConfig(**destination_config)
        )
        full_state_dict = get_model_state_dict(restored, options=StateDictOptions(full_state_dict=True))
        torch.testing.assert_close(full_state_dict, reference.state_dict())


def _test_load_model_checkpoint_distributed(rank, directory):
    with _distributed_context(rank, directory):
        reference = LlamaForCausalLM.from_pretrained(f"{directory}/seed")
        model = LlamaForCausalLM.from_pretrained(
            f"{directory}/seed", distributed_config=DistributedConfig(tp_size=2, fsdp_size=2)
        )
        model.save_pretrained(f"{directory}/dcp", distributed_checkpoint=True)
        model.save_pretrained(f"{directory}/safetensors", distributed_checkpoint=False, max_shard_size="4KB")

        for checkpoint in ("dcp", "safetensors"):
            # Zero the weights so the check below only passes if the checkpoint was actually loaded.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            load_model_checkpoint_distributed(model, f"{directory}/{checkpoint}")
            full_state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
            torch.testing.assert_close(full_state_dict, reference.state_dict(), msg=f"checkpoint={checkpoint}")


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

    @parameterized.expand(
        [
            (f"{source_name}_to_{destination_name}_{checkpoint_mode}", source, destination, distributed_checkpoint)
            for source_name, source in PARALLEL_CONFIGS.items()
            for destination_name, destination in PARALLEL_CONFIGS.items()
            for checkpoint_mode, distributed_checkpoint in CHECKPOINT_MODES.items()
        ]
    )
    def test_save_and_from_pretrained(self, _, source_config, destination_config, distributed_checkpoint):
        with tempfile.TemporaryDirectory() as directory:
            reference = LlamaForCausalLM(self.config)
            reference.save_pretrained(f"{directory}/seed")
            mp.spawn(
                _test_save_and_from_pretrained,
                args=(directory, source_config, destination_config, distributed_checkpoint),
                nprocs=4,
                join=True,
            )

            # Reload without a process group or distributed configuration.
            torch.testing.assert_close(
                LlamaForCausalLM.from_pretrained(f"{directory}/saved").state_dict(), reference.state_dict()
            )

    def test_load_model_checkpoint_distributed(self):
        with tempfile.TemporaryDirectory() as directory:
            LlamaForCausalLM(self.config).save_pretrained(f"{directory}/seed")
            mp.spawn(_test_load_model_checkpoint_distributed, args=(directory,), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
